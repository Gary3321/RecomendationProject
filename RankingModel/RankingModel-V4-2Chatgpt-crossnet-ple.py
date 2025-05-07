import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

PAD_ID = "<PAD>"

# --- Dataset unchanged from original code ---
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

        seq = self.user_last5.get(u, [PAD_ID]*5)
        seq_feats, seq_mask = [], []
        for item_id in seq:
            is_pad = (item_id == PAD_ID)
            feat = self.prod_feat.get(item_id, self.prod_feat[PAD_ID])
            seq_feats.append(feat)
            seq_mask.append(is_pad)

        return (
            torch.from_numpy(np.array(seq_feats, dtype=np.float32)),
            torch.from_numpy(np.array(seq_mask, dtype=bool)),
            torch.from_numpy(np.array(self.user_feat[u], dtype=np.float32)),
            torch.from_numpy(np.array(self.prod_feat[i], dtype=np.float32)),
            torch.from_numpy(np.array(labels, dtype=np.float32))
        )

# --- PLE Layer Definition ---
class PLELayer(nn.Module):
    def __init__(self, input_dim, num_tasks=4, num_shared=2, num_expert=2, expert_hidden=64):
        super().__init__()
        # shared experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, input_dim)
            ) for _ in range(num_shared)
        ])
        # task-specific experts
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, input_dim)
                ) for _ in range(num_expert)
            ]) for _ in range(num_tasks)
        ])
        # gating networks
        self.shared_gate = nn.Linear(input_dim, num_shared)
        self.task_gates = nn.ModuleList([nn.Linear(input_dim, num_expert) for _ in range(num_tasks)])

    def forward(self, x):
        # shared
        shared_stack = torch.stack([e(x) for e in self.shared_experts], dim=2)  # (B, D, ns)
        g_s = F.softmax(self.shared_gate(x), dim=1).unsqueeze(1)                  # (B,1,ns)
        shared = torch.bmm(shared_stack, g_s.transpose(1,2)).squeeze(2)         # (B,D)
        # task outputs
        outs = []
        for t, experts in enumerate(self.task_experts):
            task_stack = torch.stack([e(x) for e in experts], dim=2)            # (B,D,ne)
            g_t = F.softmax(self.task_gates[t](x), dim=1).unsqueeze(1)           # (B,1,ne)
            task = torch.bmm(task_stack, g_t.transpose(1,2)).squeeze(2)          # (B,D)
            outs.append(shared + task)
        return outs  # list of (B,D)

# --- CrossNetwork to enhance feature interactions ---
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.ws = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.ws, self.bs):
            x = x0 * w(x) + b + x
        return x

# --- Model with CrossNetwork + PLE ---
class PLERecModel(nn.Module):
    def __init__(self, D_user, D_item, cross_layers=2, ple_params=None, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = D_user + D_item
        # cross network
        self.cross = CrossNetwork(self.input_dim, num_layers=cross_layers)
        # PLE
        if ple_params is None:
            ple_params = dict(num_tasks=4, num_shared=2, num_expert=2, expert_hidden=self.input_dim)
        self.ple = PLELayer(input_dim=self.input_dim, **ple_params)
        # separate towers for each task
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim,1))
            for _ in range(4)
        ])

    def forward(self, seq_feats, seq_mask, user_feat, item_feat):
        # ignore seq for now
        x = torch.cat([user_feat, item_feat], dim=1)
        x = self.cross(x)
        ple_outs = self.ple(x)
        logits = torch.cat([tower(h).view(-1,1) for tower,h in zip(self.towers, ple_outs)], dim=1)
        return logits.clamp(min=-10, max=10)

# --- Training loop ---
def train(model, dataset, epochs=5, batch_size=256, lr=1e-4, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for seq_feats, seq_mask, u_f, i_f, labels in loader:
            u_f, i_f, labels = u_f.to(device), i_f.to(device), labels.to(device)
            logits = model(seq_feats, seq_mask, u_f, i_f)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            opt.step()

            total_loss += loss.item() * labels.size(0)
        print(f"Epoch {ep}: loss={total_loss/len(dataset):.4f}")

if __name__ == "__main__":
    df = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    uf = pickle.load(open("../RetrievalModel/Data/user_features_IncludeUid.pkl", "rb"))
    # normalize uf
    user_feats = np.array(list(uf.values()))
    mean_u, std_u = user_feats.mean(0), user_feats.std(0) + 1e-6
    for k in uf: uf[k] = (uf[k] - mean_u)/std_u

    pf = pickle.load(open("./Data/product_features_IncludePid_withPad.pkl", "rb"))
    keys = [k for k in pf if k!=PAD_ID]
    vals = np.array([pf[k] for k in keys])
    scaled = StandardScaler().fit_transform(vals)
    scaled = np.clip(scaled, -5,5)
    for i,k in enumerate(keys): pf[k] = scaled[i]
    pf[PAD_ID] = np.zeros_like(pf[next(iter(pf))])

    ul5 = json.load(open("./Data/last_5_purchases_withPad.json","r"))
    D_user, D_item = len(next(iter(uf.values()))), len(next(iter(pf.values())))
    ds = RankingDataset(df, ul5, pf, uf)
    model = PLERecModel(D_user, D_item)
    train(model, ds, epochs=5, batch_size=256, lr=1e-4, device='cuda')
    print("=== EVAL ===")
    # evaluate unchanged
'''
Epoch 1: loss=0.3386
Epoch 2: loss=0.2805
Epoch 3: loss=0.2584
Epoch 4: loss=0.2413
Epoch 5: loss=0.2291
'''