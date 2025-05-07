import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
import json
import pickle
import numpy as np

PAD_ID = "<PAD>"

class RankingDataset(Dataset):
    def __init__(self, df, user_last5, prod_feat, user_feat):
        self.df = df.reset_index(drop=True)
        self.user_last5 = user_last5
        self.prod_feat = prod_feat
        self.user_feat = user_feat
        self.warm_users = list(user_last5.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        u, i = row['user_id'], row['parent_asin']
        labels = row[['verified_purchase', 'clicked', 'add_to_cart', 'favorite']].astype(float).values

        # Fallback to a warm user if the current user has no sequence
        seq = self.user_last5.get(u, self.user_last5[random.choice(self.warm_users)])

        seq_feats, seq_mask = [], []
        for item_id in seq:
            is_pad = (item_id == PAD_ID)
            feat = self.prod_feat.get(item_id, self.prod_feat[PAD_ID])
            seq_feats.append(feat)
            seq_mask.append(is_pad)

        return (
            
            torch.from_numpy(np.array(seq_feats, dtype=np.float32)),    # (S, D_item)
            torch.from_numpy(np.array(seq_mask, dtype=np.bool)),        # (S,)
            torch.from_numpy(np.array(self.user_feat[u], dtype=np.float32)),  # (D_user,)
            torch.from_numpy(np.array(self.prod_feat[i], dtype=np.float32)),  # (D_item,)
            torch.from_numpy(np.array(labels, dtype=np.float32))       # (4,)
        )
    


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.ws = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.ws, self.bs):
            x = x0 * w(x) + b + x  # Element-wise x0 * w(x) + b + x
        return x

class PLELayer(nn.Module):
    def __init__(self, input_dim, num_tasks=4, num_shared=2, num_expert=2, expert_hidden=64):
        super().__init__()
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, input_dim)
            ) for _ in range(num_shared)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.ReLU(),
                    nn.Linear(expert_hidden, input_dim)
                ) for _ in range(num_expert)
            ]) for _ in range(num_tasks)
        ])
        self.shared_gate = nn.Linear(input_dim, num_shared)
        self.task_gates = nn.ModuleList([nn.Linear(input_dim, num_expert) for _ in range(num_tasks)])

    def forward(self, x):
        shared_stack = torch.stack([e(x) for e in self.shared_experts], dim=2)  # (B, D, ns)
        g_s = F.softmax(self.shared_gate(x), dim=1).unsqueeze(1)                # (B, 1, ns)
        shared = torch.bmm(shared_stack, g_s.transpose(1, 2)).squeeze(2)        # (B, D)

        outs = []
        for t, experts in enumerate(self.task_experts):
            task_stack = torch.stack([e(x) for e in experts], dim=2)            # (B, D, ne)
            g_t = F.softmax(self.task_gates[t](x), dim=1).unsqueeze(1)          # (B, 1, ne)
            task = torch.bmm(task_stack, g_t.transpose(1, 2)).squeeze(2)        # (B, D)
            outs.append(shared + task)
        return outs

class RecModel(nn.Module):
    def __init__(self, D_user, D_item, cross_layers=2, ple_params=None, attn_heads=4):
        super().__init__()
        self.total_dim = D_user + D_item
        self.cross = CrossNetwork(self.total_dim, num_layers=cross_layers)

        # Learnable projection from total_dim to D_item for attention queries
        self.query_proj = nn.Linear(self.total_dim, D_item)
        self.attn = nn.MultiheadAttention(embed_dim=D_item, num_heads=attn_heads, batch_first=True)
        self.attn_proj = nn.Linear(D_item, self.total_dim)

        if ple_params is None:
            ple_params = dict(num_tasks=4, num_shared=2, num_expert=2, expert_hidden=self.total_dim)
        self.ple = PLELayer(input_dim=self.total_dim, **ple_params)

        self.towers = nn.ModuleList([
            nn.Linear(self.total_dim, 1) for _ in range(4)
        ])

    def forward(self, seq_feats, seq_mask, user_feat, item_feat):
        B, S, D_item = seq_feats.shape

        # Attention query: project [user | item] to D_item
        concat_q = torch.cat([user_feat, item_feat], dim=1)  # (B, total_dim)
        q = self.query_proj(concat_q).unsqueeze(1)           # (B, 1, D_item)

        attn_out, _ = self.attn(q, seq_feats, seq_feats, key_padding_mask=seq_mask)  # (B, 1, D_item)
        attn_out = self.attn_proj(attn_out.squeeze(1))                                # (B, total_dim)

        x = self.cross(concat_q) + attn_out                                            # (B, total_dim)
        ple_outs = self.ple(x)                                                        # list of (B, total_dim)

        logits = torch.cat([self.towers[i](h).view(-1, 1) for i, h in enumerate(ple_outs)], dim=1)  # (B, 4)
        return logits

def train(model, dataset, epochs=5, batch_size=256, lr=1e-3, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for seq_feats, seq_mask, u_f, i_f, labels in loader:
            seq_feats, seq_mask = seq_feats.to(device), seq_mask.to(device)
            u_f, i_f, labels = u_f.to(device), i_f.to(device), labels.to(device)

            logits = model(seq_feats, seq_mask, u_f, i_f)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(seq_feats)

        print(f"Epoch {ep}: loss={total_loss/len(dataset):.4f}")

def evaluate(model, dataset, batch_size=256, device='cuda'):
    from sklearn.metrics import roc_auc_score

    loader = DataLoader(dataset, batch_size=batch_size)
    model.to(device).eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for seq_feats, seq_mask, u_f, i_f, labels in loader:
            seq_feats, seq_mask = seq_feats.to(device), seq_mask.to(device)
            u_f, i_f = u_f.to(device), i_f.to(device)

            logits = model(seq_feats, seq_mask, u_f, i_f)
            preds = torch.sigmoid(logits).cpu()
            all_labels.append(labels)
            all_preds.append(preds)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    names = ['purchase', 'click', 'add_to_cart', 'favorite']
    aucs = []
    for i, name in enumerate(names):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except:
            auc = float('nan')
        print(f"AUC {name}: {auc:.4f}")
        aucs.append(auc)

    print("Avg AUC:", sum(aucs) / len(aucs))
    return aucs

if __name__ == "__main__":
    df = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    uf = pickle.load(open("../RetrievalModel/Data/user_features_IncludeUid.pkl", "rb"))
    pf = pickle.load(open("./Data/product_features_IncludePid_withPad.pkl", "rb"))
    ul5 = json.load(open("./Data/last_5_purchases_withPad.json", "r"))

    D_user = len(next(iter(uf.values())))
    D_item = len(next(iter(pf.values())))
    ds = RankingDataset(df, ul5, pf, uf)

    model = RecModel(D_user, D_item, cross_layers=2, attn_heads=4)
    train(model, ds, epochs=5, batch_size=256, lr=1e-3, device='cuda')

    print("=== EVAL ===")
    evaluate(model, ds)
