import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler

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

        seq = self.user_last5.get(u, [PAD_ID]*5)
        seq_feats, seq_mask = [], []
        for item_id in seq:
            is_pad = (item_id == PAD_ID)
            feat = self.prod_feat.get(item_id, self.prod_feat[PAD_ID])
            seq_feats.append(feat)
            seq_mask.append(is_pad)

        seq_feats = np.array(seq_feats, dtype=np.float32)
        seq_mask = np.array(seq_mask, dtype=bool)

        if seq_mask.all():
            seq_mask[0] = False
            seq_feats[0] = np.zeros_like(seq_feats[0])

        return (
            torch.from_numpy(seq_feats),
            torch.from_numpy(seq_mask),
            torch.from_numpy(np.array(self.user_feat[u], dtype=np.float32)),
            torch.from_numpy(np.array(self.prod_feat[i], dtype=np.float32)),
            torch.from_numpy(labels)
        )

class WideDeepRecModel(nn.Module):
    def __init__(self, D_user, D_item, deep_hidden=[256, 128,64], dropout=0.2):
        super().__init__()
        self.D_user, self.D_item = D_user, D_item

        self.wide = nn.Linear(D_user + D_item + D_item, 1)

        layers = []
        input_dim = D_user + D_item + D_item
        for h in deep_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(input_dim, 4)

    def forward(self, seq_feats, seq_mask, user_feat, item_feat):
        seq_feats = seq_feats.masked_fill(seq_mask.unsqueeze(-1), 0.0)
        counts = (~seq_mask).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        seq_mean = seq_feats.sum(dim=1) / counts

        x = torch.cat([user_feat, item_feat, seq_mean], dim=1)
        wide_logit = self.wide(x)
        deep_h = self.deep(x)
        deep_logit = self.deep_out(deep_h)
        wide_expanded = wide_logit.expand(-1, 4)
        logits = wide_expanded + deep_logit
        return logits

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pth', device='cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #print(f"Loaded checkpoint '{path}' (epoch {ckpt.get('epoch', '?')})")

def train(model, optimizer, loss_fn, dataset, best_path, epochs=5, batch_size=256, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    best_auc = 0
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for seq_f, seq_m, u_f, i_f, labels in loader:
            seq_f, seq_m = seq_f.to(device), seq_m.to(device)
            u_f, i_f, labels = u_f.to(device), i_f.to(device), labels.to(device)
            logits = model(seq_f, seq_m, u_f, i_f)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {ep}: train loss = {avg_loss:.4f}")
        aucs = evaluate(model, dataset, batch_size, device)
        avg_auc = sum(aucs) / len(aucs)
        if avg_auc > best_auc:
            best_auc = avg_auc
            save_checkpoint(model, optimizer, ep, best_path)
            print(f"New best model (avg AUC = {avg_auc:.4f}) saved.")

def evaluate(model, dataset, batch_size=256, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    from sklearn.metrics import roc_auc_score
    all_labels, all_preds = [], []
    with torch.no_grad():
        for seq_f, seq_m, u_f, i_f, labels in loader:
            seq_f, seq_m = seq_f.to(device), seq_m.to(device)
            u_f, i_f = u_f.to(device), i_f.to(device)
            logits = model(seq_f, seq_m, u_f, i_f)
            preds = torch.sigmoid(logits).cpu().numpy()
            #print("Logit range:", logits.min().item(), logits.max().item())
            #print("Prob range:", preds.min(), preds.max())
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    aucs = []
    for i, name in enumerate(['purchase','click','add_to_cart','favorite']):
        try:
            auc = roc_auc_score(all_labels[:,i], all_preds[:,i])
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
        print(f"AUC {name}: {auc:.4f}")
    print(f"Average AUC: {np.nanmean(aucs):.4f}")
    return aucs

if __name__ == "__main__":
    df = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    uf = pickle.load(open("../RetrievalModel/Data/user_features_IncludeUid.pkl","rb"))
    pf = pickle.load(open("./Data/product_features_IncludePid_withPad.pkl","rb"))
    ul5 = json.load(open("./Data/last_5_purchases_withPad.json","r"))

    # Get training user and product IDs
    train_user_ids = df['user_id'].unique()
    train_item_ids = df['parent_asin'].unique()

    # Gather only training user and product features
    user_array = np.stack([uf[u] for u in train_user_ids if u in uf])
    scaler_u = StandardScaler().fit(user_array)
    for k in uf:
        uf[k] = scaler_u.transform([uf[k]])[0]

    prod_array = np.stack([pf[i] for i in train_item_ids if i in pf])
    scaler_p = StandardScaler().fit(prod_array)
    for k in pf:
        pf[k] = scaler_p.transform([pf[k]])[0]

    D_user, D_item = len(next(iter(uf.values()))), len(next(iter(pf.values())))
    ds = RankingDataset(df, ul5, pf, uf)

    model = WideDeepRecModel(D_user, D_item)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_model_path = './Data/WD_model_best.pth'

    #ds = torch.utils.data.Subset(ds, list(range(128))) # use a smaller dataset to check if overfitting
    train(model, optimizer, loss_fn, ds, best_model_path, epochs=5, batch_size=256, device='cuda')

    #load_checkpoint(model, optimizer, best_model_path, device='cuda')
    print("==== Test ===")
    df_test = pd.read_excel('./Data/testing_set_ranking9k.xlsx')
    ds_test = RankingDataset(df_test, ul5, pf, uf)
    evaluate(model, ds_test, batch_size=256, device='cuda')
