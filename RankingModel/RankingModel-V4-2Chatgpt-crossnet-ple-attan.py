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

# --- Dataset ---
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
        labels = row[['verified_purchase','clicked','add_to_cart','favorite']].astype(float).values

        seq = self.user_last5.get(u, [PAD_ID]*5)
        seq_feats, seq_mask = [], []
        for item_id in seq:
            is_pad = (item_id == PAD_ID)
            feat = self.prod_feat.get(item_id, self.prod_feat[PAD_ID])
            seq_feats.append(feat)
            seq_mask.append(is_pad)

        return (
            torch.from_numpy(np.array(seq_feats, dtype=np.float32)),
            torch.tensor(seq_mask, dtype=torch.bool),
            torch.tensor(self.user_feat[u], dtype=torch.float32),
            torch.tensor(self.prod_feat[i], dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32)
        )

# --- PLE Layer ---
class PLELayer(nn.Module):
    def __init__(self, input_dim, num_tasks=4, num_shared=2, num_expert=2, expert_hidden=64):
        super().__init__()
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, input_dim))
            for _ in range(num_shared)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(input_dim, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, input_dim))
                for _ in range(num_expert)
            ]) for _ in range(num_tasks)
        ])
        self.shared_gate = nn.Linear(input_dim, num_shared)
        self.task_gates = nn.ModuleList([nn.Linear(input_dim, num_expert) for _ in range(num_tasks)])

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        shared_stack = torch.stack([e(x) for e in self.shared_experts], dim=2)
        g_s = F.softmax(self.shared_gate(x).clamp(-10, 10), dim=1).unsqueeze(1)
        shared = torch.bmm(shared_stack, g_s.transpose(1, 2)).squeeze(2)

        outs = []
        for t, experts in enumerate(self.task_experts):
            task_stack = torch.stack([e(x) for e in experts], dim=2)
            g_t = F.softmax(self.task_gates[t](x).clamp(-10, 10), dim=1).unsqueeze(1)
            task = torch.bmm(task_stack, g_t.transpose(1, 2)).squeeze(2)
            outs.append(shared + task)
        return outs

# --- CrossNetwork ---
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.ws = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.ws, self.bs):
            x = x0 * w(x) + b + x
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        return x

# --- PLE + CrossNet + Attention Model ---
class PLEAttnRecModel(nn.Module):
    def __init__(self, D_user, D_item, cross_layers=2, ple_params=None, ple_hidden=64, attn_heads=2, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.total_dim = D_user + D_item
        self.cross = CrossNetwork(self.total_dim, num_layers=cross_layers)

        if ple_params is None:
            ple_params = dict(num_tasks=4, num_shared=2, num_expert=2, expert_hidden=ple_hidden)
        self.ple_input_dim = self.total_dim * 2  # because attn_out + x_cross are concatenated
        self.ple = PLELayer(input_dim=self.ple_input_dim, **ple_params)

        self.attn_q = nn.Linear(self.total_dim, D_item)
        self.attn = nn.MultiheadAttention(embed_dim=D_item, num_heads=attn_heads, batch_first=True)
        self.attn_proj = nn.Linear(D_item, self.total_dim)
        self.attn_norm = nn.LayerNorm(self.total_dim)

        

        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.total_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for _ in range(4)
        ])
        self.ple_proj = nn.Linear(self.ple_input_dim, self.total_dim)  # project 7950 → 3975
        self.final_norm = nn.LayerNorm(self.total_dim)

    def forward(self, seq_feats, seq_mask, user_feat, item_feat):
        # sanitize input features
        user_feat = torch.nan_to_num(user_feat, nan=0.0, posinf=0.0, neginf=0.0)
        item_feat = torch.nan_to_num(item_feat, nan=0.0, posinf=0.0, neginf=0.0)

        x0 = torch.cat([user_feat, item_feat], dim=1)
        if torch.isnan(x0).any(): 
            print("NaN in x0")

        q = self.attn_q(x0).unsqueeze(1)
        seq_feats = torch.nan_to_num(seq_feats, nan=0.0, posinf=1e3, neginf=-1e3)
        q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
        attn_out,_ = self.attn(q, seq_feats, seq_feats, key_padding_mask=seq_mask)
        attn_out = attn_out.squeeze(1)
        attn_out = self.attn_proj(attn_out)
        attn_out = self.attn_norm(attn_out)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e3, neginf=-1e3)

        x_cross = self.cross(x0)
        combined = torch.cat([attn_out, x_cross], dim=1)
        ple_outs = self.ple(combined)
        if torch.isnan(torch.stack(ple_outs)).any(): 
            print("NaN in PLE")

        x = torch.stack(ple_outs, dim=2).sum(2)  # [batch, 7950]
        x = self.ple_proj(x)  # [batch, 3975]
        x = self.final_norm(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        logits = torch.cat([tower(x).view(-1,1) for tower in self.towers], dim=1)
        if torch.isnan(logits).any():
            print("NaN in logits. Debug info:")
            print("x min:", x.min().item(), "x max:", x.max().item(), "x mean:", x.mean().item())

        logits = logits.clamp(min=-10,max=10)
        if torch.isnan(logits).any():
            raise ValueError("NaN detected in logits")
        return logits



# --- Training Loop ---
def train(model, dataset, epochs=5, batch_size=256, lr=1e-4, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        for seq_feats, seq_mask, u_f, i_f, labels in loader:
            seq_feats = seq_feats.to(device)
            seq_mask = seq_mask.to(device)
            u_f = u_f.to(device)
            i_f = i_f.to(device)
            labels = labels.to(device)

            logits = model(seq_feats, seq_mask, u_f, i_f)

            # sanity check
            if torch.isnan(logits).any():
                raise ValueError("NaN detected in logits")

            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()

            total += loss.item() * labels.size(0)

        print(f"Epoch {ep}: loss={total / len(dataset):.4f}")

if __name__ == "__main__":
    df = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    uf = pickle.load(open("../RetrievalModel/Data/user_features_IncludeUid.pkl", "rb"))

    arr = np.array(list(uf.values()))
    m, s = arr.mean(0), arr.std(0) + 1e-6
    for k in uf:
        uf[k] = np.nan_to_num((uf[k] - m) / s, nan=0.0, posinf=5.0, neginf=-5.0)

    pf = pickle.load(open("./Data/product_features_IncludePid_withPad.pkl", "rb"))
    keys = [k for k in pf if k != PAD_ID]
    arr = np.array([pf[k] for k in keys])
    sc = StandardScaler().fit_transform(arr)
    sc = np.clip(sc, -5, 5)
    for i, k in enumerate(keys):
        pf[k] = np.nan_to_num(sc[i], nan=0.0, posinf=5.0, neginf=-5.0)
    pf[PAD_ID] = np.zeros_like(pf[keys[0]])

    ul5 = json.load(open("./Data/last_5_purchases_withPad.json", "r"))
    D_user, D_item = len(next(iter(uf.values()))), len(next(iter(pf.values())))
    ds = RankingDataset(df, ul5, pf, uf)
    model = PLEAttnRecModel(D_user, D_item)
    train(model, ds, epochs=5, batch_size=256, lr=1e-4, device='cuda')
    print("=== EVAL ===")
