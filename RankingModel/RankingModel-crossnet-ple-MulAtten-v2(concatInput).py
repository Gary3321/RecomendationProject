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

def check_tensor_nan(t, name="tensor"):
    if torch.isnan(t).any():
        print(f"NaN detected in {name}")
    if torch.isinf(t).any():
        print(f"Inf detected in {name}")

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

        # Convert to numpy arrays
        seq_feats = np.array(seq_feats, dtype=np.float32)
        seq_mask = np.array(seq_mask, dtype=bool)

        # --- FIX: Handle case where all are PADs, 
        # they will cause NaN in attention output, then train loss NaN ---
        if seq_mask.all():
            seq_mask[0] = False  # Unmask the first token
            seq_feats[0] = np.zeros_like(seq_feats[0])  # Replace with neutral zero vector

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
    def __init__(self, D_user, D_item, cross_layers=2, ple_params=None, hidden_dim=128, dropout=0.1, attn_heads=2):
        super().__init__()
        self.input_dim = D_user + D_item



        # cross network
        self.cross = CrossNetwork(self.input_dim, num_layers=cross_layers)

        # attention: user embedding as query, seq as kv
        self.attn_input_dim = 512  # any multiple of attn_heads, like 512 or 1024
        self.ln_user = nn.LayerNorm(self.attn_input_dim)
        self.ln_seq = nn.LayerNorm(self.attn_input_dim)       

        self.user_item_proj = nn.Linear(self.input_dim, self.attn_input_dim)
        self.seq_proj = nn.Linear(D_item, self.attn_input_dim)
        self.attn = nn.MultiheadAttention(self.attn_input_dim, attn_heads, batch_first=True)
        #self.attn_proj = nn.Linear(D_user, D_user * 2) # project attn_out to 2D

        self.attn_to_input = nn.Linear(self.attn_input_dim, self.input_dim)

        # PLE
        if ple_params is None:
            ple_params = dict(num_tasks=4, num_shared=2, num_expert=2, expert_hidden=self.input_dim)
        self.ple = PLELayer(input_dim=self.input_dim, **ple_params)
        # separate towers for each task
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim,1))
            for _ in range(4)
        ])

    def forward(self, seq_feats, seq_mask, user_feat, item_feat):
        # ignore seq for now

        
        if torch.isnan(seq_feats).any(): print("NaN in seq_feats")
        if torch.isnan(user_feat).any(): print("NaN in user_feat")
        if torch.isnan(item_feat).any(): print("NaN in item_feat")

        x = torch.cat([user_feat, item_feat], dim=1)
        user_item = torch.cat([user_feat, item_feat], dim=1)
        # print(f"x shape: {x.shape}")
        # attention pooling
        # x_proj = self.user_item_proj(x).unsqueeze(1)           # (B, 1, attn_input_dim)
        # seq_feats_proj = self.seq_proj(seq_feats)              # (B, 5, attn_input_dim)

        x_proj = self.ln_user(self.user_item_proj(x)).unsqueeze(1)       # (B, 1, attn_input_dim)
        seq_feats_proj = self.ln_seq(self.seq_proj(seq_feats))           # (B, 5, attn_input_dim)

        # query: user_embs.unsqueeze(1) (B,1,D)
        # print(f"checking device: {x.device}, {seq_feats.device}, {seq_mask.device}")
        
        # Ensure seq_mask is boolean and True=masked
        # print(f"seq_mask type: {seq_mask.dtype}")
        # print(f"Check seq_mask: {seq_mask[0]}")  # Should be True for PAD tokens, False for valid
        if seq_mask.dtype != torch.bool:
            seq_mask = seq_mask.bool()

        # Manually zero out padding positions to prevent contribution in attention output
        seq_feats_proj = seq_feats_proj.masked_fill(seq_mask.unsqueeze(-1), 0.0)
        
        attn_out, _ = self.attn(query=x_proj, key=seq_feats_proj, value=seq_feats_proj,
                                 key_padding_mask=seq_mask)  # (B,1,attn_input_dim)
        
        if torch.isnan(attn_out).any(): print("NaN in attn_out")
        # print(f"before squeeze attn_out shape: {attn_out.shape}")
        attn_out = attn_out.squeeze(1)  # (B,D)
        #attn_out_proj = self.attn_proj(attn_out) # (B, 2D)
        # print(f"attn_out shape: {attn_out.shape}")

        x = self.cross(x)
        # print(f"after cross x shape: {x.shape}")
        #ple_outs = self.ple(x)
        attn_out = self.attn_to_input(attn_out)

        if torch.isnan(attn_out).any(): print("NaN in attn_out")
        if torch.isnan(x).any(): print("NaN in x before adding attn")

        ple_outs = self.ple(x+attn_out)
        # Ensure that the ple_outs is a tensor:
        #ple_outs = torch.tensor(ple_outs) if isinstance(ple_outs, list) else ple_outs
        ple_outs_concatnated = [torch.cat([ple_out, user_item], dim=1) for ple_out in ple_outs ]# concatnate input features with ple outputs
        logits = torch.cat([tower(h).view(-1,1) for tower,h in zip(self.towers, ple_outs_concatnated)], dim=1)
        return logits.clamp(min=-10, max=10)

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pth', device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
    return epoch



def load_model_for_inference(model_class, model_args, checkpoint_path='best_model.pth', device='cuda'):
    model = model_class(*model_args)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Best model loaded from {checkpoint_path}")
    return model

# --- Training loop ---
def train(model, opt, loss_fn, dataset, epochs=5, batch_size=256,  device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)


    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for seq_feats, seq_mask, u_f, i_f, labels in loader:


            seq_feats = seq_feats.to(device) 
            seq_mask = seq_mask.to(device)
            u_f, i_f, labels = u_f.to(device), i_f.to(device), labels.to(device)
            # print(f"user feat mean: {u_f.mean().item()}, std: {u_f.std().item()}")
            # print(f"item feat mean: {i_f.mean().item()}, std: {i_f.std().item()}")
            # check_tensor_nan(seq_feats, "seq_feats")
            # check_tensor_nan(seq_mask.float(), "seq_mask")
            # check_tensor_nan(u_f, "user_feat")
            # check_tensor_nan(i_f, "item_feat")
            # check_tensor_nan(labels, "labels")

            logits = model(seq_feats, seq_mask, u_f, i_f)
            check_tensor_nan(logits, "logits")

            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            opt.step()

            total_loss += loss.item() * labels.size(0)
            # print(f"seq_feats shape: {seq_feats.shape}, labels shape: {labels.shape}")
            # print(f"loss: {loss.item()}, time labels size {loss.item() * labels.size(0)}, dataset shape: {len(dataset)}")
        print(f"Epoch {ep}: loss={total_loss/len(dataset):.4f}")

    # # ---- Evaluate on validation set here ----
    # aucs = evaluate(model, val_dataset)  # Pass validation dataset
    # avg_auc = sum(aucs) / len(aucs)

    # if avg_auc > best_auc:
    #     best_auc = avg_auc
    #     torch.save(model.state_dict(), best_model_path)
    #     print(f"New best model saved with avg AUC = {avg_auc:.4f}")


def evaluate(model, dataset, batch_size=256, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    model.to(device)
    model.eval()
    from sklearn.metrics import roc_auc_score
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for seq_feats, seq_mask, u_f, i_f, labels in loader:
            seq_feats = seq_feats.to(device) 
            seq_mask = seq_mask.to(device)
            u_f, i_f, labels = u_f.to(device), i_f.to(device), labels.to(device)
            logits = model(seq_feats, seq_mask, u_f, i_f)
            preds = torch.sigmoid(logits).cpu()
            all_labels.append(labels)
            all_preds.append(preds)
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    aucs = []
    for i, action in enumerate(['purchase', 'click', 'add_to_cart', 'favorite']):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
        print(f"AUC {action}: {auc:.4f}")
    print(f"Average AUC: {sum(aucs)/len(aucs):.4f}")
    return aucs

if __name__ == "__main__":
    df = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    # Step 1: extract train user and item IDs
    train_user_ids = set(df['user_id'].unique())
    train_item_ids = set(df['parent_asin'].unique())

    uf = pickle.load(open("../RetrievalModel/Data/user_features_IncludeUid.pkl", "rb"))
    # Step 2: compute normalization stats from training users only 
    train_user_feats = np.array([uf[uid] for uid in train_user_ids if uid in uf])
    #user_feats = np.array(list(uf.values()))
    #mean_u, std_u = user_feats.mean(0), user_feats.std(0) + 1e-6
    mean_u, std_u = train_user_feats.mean(0), train_user_feats.std(0) + 1e-6
    # Step 3: normalize all users
    for k in uf: 
        uf[k] = (uf[k] - mean_u)/std_u

    pf = pickle.load(open("./Data/product_features_IncludePid_withPad.pkl", "rb"))

    # Step 4: compute item normalization from training items only
    train_item_feats = np.array([pf[iid] for iid in train_item_ids if iid in pf and iid != PAD_ID])
    #keys = [k for k in pf if k!=PAD_ID]
    #vals = np.array([pf[k] for k in keys])
    #scaled = StandardScaler().fit_transform(vals)
    scaled = StandardScaler()
    scaled.fit(train_item_feats) ## Compute mean and std based on training set
    keys = [k for k in pf if k!=PAD_ID]
    # Apply the transformation to all item features (including test items)
    for i,k in enumerate(keys): 
        pf[k] = np.clip(scaled.transform([pf[k]]), -5, 5)[0]
    pf[PAD_ID] = np.zeros_like(pf[next(iter(pf))])

    ul5 = json.load(open("./Data/last_5_purchases_withPad.json","r"))
    D_user, D_item = len(next(iter(uf.values()))), len(next(iter(pf.values())))
    print(f"D user: {D_user}, D item: {D_item}")



    ds = RankingDataset(df, ul5, pf, uf)
    model = PLERecModel(D_user, D_item)
    lr=1e-4
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    train(model, opt, loss_fn, ds, epochs=5, batch_size=256,  device='cuda')

    save_checkpoint(model, optimizer=opt, epoch=5, path='./Data/RM_CPM_model_catInput_epoch5.pth')

    print("=== EVAL ===")
    # Load the trained model before evaluation
    #model.load_state_dict(torch.load('./Data/RM_CPM_model_epoch5.pth'))


    df_test = pd.read_excel('./Data/testing_set_ranking9k.xlsx')
    # don't need normalize as all features already normalized
    ds = RankingDataset(df_test, ul5, pf, uf)
    model = PLERecModel(D_user, D_item)
    # evaluate unchanged
    evaluate(model, ds)


'''
D user: 3, D item: 3972
Epoch 1: loss=0.1676
Epoch 2: loss=0.1462
Epoch 3: loss=0.1408
Epoch 4: loss=0.1356
Epoch 5: loss=0.1314
'''

# add early stopping 