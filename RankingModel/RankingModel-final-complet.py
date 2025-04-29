import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
import json

PAD_ID = "<PAD>"

class RankingDataset(Dataset):
    def __init__(self, df_data, user_last5, product_emb_dict, user_emb_dict):
        """
        df_data: DataFrame with columns [user_id, parent_asin, verified_purchase, clicked, add_to_cart, favorite]
        user_last5: dict user_id -> list of last 5 item_ids (with PAD_ID)
        product_emb_dict: dict item_id -> embedding vector (D,)
        user_emb_dict: dict user_id -> embedding vector (D,)
        """
        self.df = df_data.reset_index(drop=True)
        self.user_last5 = user_last5
        self.prod_emb = product_emb_dict
        self.user_emb = user_emb_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        u = row['user_id']
        i = row['parent_asin']
        # positive labels
        labels = row[['verified_purchase', 'clicked', 'add_to_cart', 'favorite']].astype(int).values

        # sequence of last 5 items
        #seq_items = self.user_last5.get(u, [PAD_ID] * 5)
        # Randomly sample a warm user's sequence
        random_user = random.choice(list(self.user_last5.keys()))
        seq_items = self.user_last5.get(u, self.user_last5[random_user])

        seq_embs, seq_mask = [], []
        #print(f"seq_items: {seq_items}")
        for it in seq_items:
            if it == PAD_ID:
                seq_embs.append(self.prod_emb[PAD_ID])
                seq_mask.append(1)
            else:
                seq_embs.append(self.prod_emb[it])
                seq_mask.append(0)
        # print(f"seq_items: {seq_items}")
        # print(f"seq_embs: {seq_embs}")
        seq_embs = torch.tensor(seq_embs, dtype=torch.float32)  # (S, D)
        seq_mask = torch.tensor(seq_mask, dtype=torch.bool)     # (S,)

        user_emb = torch.tensor(self.user_emb[u], dtype=torch.float32)  # (D,)
        item_emb = torch.tensor(self.prod_emb[i], dtype=torch.float32)  # (D,)
        labels = torch.tensor(labels, dtype=torch.int32)  # (4,)

        return u, seq_embs, seq_mask, user_emb, item_emb, labels, i # item_id

def rec_collate_fn(batch, dataset, neg_k=2):
    """
    Batch negative sampling: sample negatives from the positive items present in this batch.
    batch: list of tuples (u, seq_embs, seq_mask, user_emb, pos_item_emb, labels, pos_item_id)
    """
    users, seq_embs_list, seq_masks, user_embs, pos_item_embs, pos_labels, pos_item_ids = zip(*batch)
    B = len(users)

    # stack positives
    seq_embs = torch.stack(seq_embs_list)      # (B, S, D)
    seq_masks = torch.stack(seq_masks)         # (B, S)
    user_embs = torch.stack(user_embs)         # (B, D)
    pos_item_embs = torch.stack(pos_item_embs) # (B, D)
    pos_labels = torch.stack(pos_labels)       # (B, 4)

    # negative sampling from batch's positive items
    neg_seq_embs, neg_seq_masks, neg_user_embs, neg_item_embs, neg_labels = [], [], [], [], []
    for idx in range(B):
        # candidate negatives: all other pos_item_ids in batch
        candidates = [pid for j,pid in enumerate(pos_item_ids) if j != idx]
        for _ in range(neg_k):
            neg_id = random.choice(candidates)
            neg_user_embs.append(user_embs[idx])
            neg_seq_embs.append(seq_embs_list[idx])
            neg_seq_masks.append(seq_masks[idx])
            neg_item_embs.append(torch.tensor(dataset.prod_emb.get(neg_id, dataset.prod_emb[PAD_ID]), dtype=torch.float32))
            neg_labels.append(torch.zeros(4, dtype=torch.float32))

    if neg_item_embs:
        all_seq_embs = torch.cat([seq_embs, torch.stack(neg_seq_embs)], dim=0)
        all_seq_masks = torch.cat([seq_masks, torch.stack(neg_seq_masks)], dim=0)
        all_user_embs = torch.cat([user_embs, torch.stack(neg_user_embs)], dim=0)
        all_item_embs = torch.cat([pos_item_embs, torch.stack(neg_item_embs)], dim=0)
        all_labels = torch.cat([pos_labels, torch.stack(neg_labels)], dim=0)
    else:
        all_seq_embs, all_seq_masks, all_user_embs, all_item_embs, all_labels = seq_embs, seq_masks, user_embs, pos_item_embs, pos_labels

    return all_seq_embs, all_seq_masks, all_user_embs, all_item_embs, all_labels


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.ws = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            xw = self.ws[i](x)  # (B,1)
            x = x0 * xw + self.bs[i] + x  # (B,D)
        return x


class PLELayer(nn.Module):
    def __init__(self, input_dim, num_tasks=4, num_shared=2, num_expert=2, expert_hidden=64):
        super().__init__()
        self.num_tasks = num_tasks
        # experts
        self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, expert_hidden), nn.ReLU(),
                                                          nn.Linear(expert_hidden, input_dim))
                                             for _ in range(num_shared)])
        self.task_experts = nn.ModuleList(
            [nn.ModuleList([nn.Sequential(nn.Linear(input_dim, expert_hidden), nn.ReLU(),
                                          nn.Linear(expert_hidden, input_dim))
                             for _ in range(num_expert)])
             for _ in range(num_tasks)]
        )
        # gates
        self.shared_gate = nn.Linear(input_dim, num_shared)
        self.task_gates = nn.ModuleList([nn.Linear(input_dim, num_expert) for _ in range(num_tasks)])

    def forward(self, x):
        # shared expert outputs
        shared_outs = [expert(x) for expert in self.shared_experts]  # list of (B,D)
        shared_stack = torch.stack(shared_outs, dim=2)  # (B,D,num_shared)
        gate_shared = F.softmax(self.shared_gate(x), dim=1)  # (B,num_shared)
        gate_shared = gate_shared.unsqueeze(1)  # (B,1,num_shared)
        shared_combined = torch.bmm(shared_stack, gate_shared.transpose(1,2)).squeeze(2)  # (B,D)

        #task_outs = []
        results = []
        for t in range(self.num_tasks):
            # task experts
            experts = [expert(x) for expert in self.task_experts[t]]  # list of (B,D)
            experts_stack = torch.stack(experts, dim=2)  # (B,D,num_expert)
            gate_t = F.softmax(self.task_gates[t](x), dim=1).unsqueeze(1)  # (B,1,num_expert)
            task_combined = torch.bmm(experts_stack, gate_t.transpose(1,2)).squeeze(2)  # (B,D)
            # sum shared + task
            results.append(shared_combined + task_combined)
        return results  # list of (B,D) per task


class RecModel(nn.Module):
    def __init__(self, embed_dim, cross_layers=2, ple_params=None, attn_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        # cross network
        self.cross = CrossNetwork(input_dim=embed_dim*2, num_layers=cross_layers)
        # attention: user embedding as query, seq as kv
        self.attn = nn.MultiheadAttention(embed_dim, attn_heads, batch_first=True)
        # PLE
        if ple_params is None:
            ple_params = dict(num_tasks=4, num_shared=2, num_expert=2, expert_hidden=embed_dim)
        self.ple = PLELayer(input_dim=embed_dim*2, **ple_params)

        self.attn_proj = nn.Linear(embed_dim, embed_dim * 2) # project attn_out to 2D
        # final towers
        self.towers = nn.ModuleList([nn.Sequential(
                                        nn.Linear(embed_dim*2, embed_dim), 
                                        nn.ReLU(),
                                        nn.Linear(embed_dim, 1))
                                     for _ in range(4)])

    def forward(self, seq_embs, seq_masks, user_embs, item_embs):
        # seq_embs: (B, S, D), user_embs: (B, D)
        # attention pooling
        # query: user_embs.unsqueeze(1) (B,1,D)
        attn_out, _ = self.attn(query=user_embs.unsqueeze(1), key=seq_embs, value=seq_embs,
                                 key_padding_mask=seq_masks)  # (B,1,D)
        attn_out = attn_out.squeeze(1)  # (B,D)
        attn_out_proj = self.attn_proj(attn_out) # (B, 2D)

        # combine item & user features for cross and PLE
        x = torch.cat([user_embs, item_embs], dim=1)  # (B,2D)
        x_cross = self.cross(x)  # (B,2D)
        # PLE input: x_cross
        ple_outs = self.ple(x_cross)  # list of 4 (B,2D)

        outputs = []
        for t in range(4):
            # optionally incorporate attn
            h = ple_outs[t] + attn_out_proj  # (B,2D)
            out = self.towers[t](h).squeeze(1)  # (B,)
            outputs.append(out)
        # stack logits
        logits = torch.stack(outputs, dim=1)  # (B,4)
        return logits


def train(model, dataset, epochs=5, batch_size=256, lr=1e-3, neg_k=2, device='cuda'):
    # global batch_dataset
    # batch_dataset = dataset  # for collate access
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: rec_collate_fn(b, dataset=dataset, neg_k=neg_k))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    print("Training epoch --------")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for seq_embs, seq_masks, user_embs, item_embs, labels in loader:
            seq_embs = seq_embs.to(device)
            seq_masks = seq_masks.to(device)
            user_embs = user_embs.to(device)
            item_embs = item_embs.to(device)
            labels = labels.to(device)

            logits = model(seq_embs, seq_masks, user_embs, item_embs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq_embs.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")


def evaluate(model, dataset, batch_size=256, device='cuda'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: rec_collate_fn(b, dataset=dataset, neg_k=0))  # no negatives
    model.to(device)
    model.eval()
    from sklearn.metrics import roc_auc_score
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for seq_embs, seq_masks, user_embs, item_embs, labels in loader:
            seq_embs = seq_embs.to(device)
            seq_masks = seq_masks.to(device)
            user_embs = user_embs.to(device)
            item_embs = item_embs.to(device)
            logits = model(seq_embs, seq_masks, user_embs, item_embs)
            preds = torch.sigmoid(logits).cpu()
            all_labels.append(labels)
            all_preds.append(preds)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
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

    df_train = pd.read_excel('./Data/training_set_ranking65k.xlsx')
    with open("../RetrievalModel/Data/user_embeddings.json", "r") as f:
        user_emb_dict = json.load(f)
    with open("./Data/product_embeddings_withPad.json", "r") as f:
        prod_emb_dict = json.load(f)
    with open("./Data/last_5_purchases_withPad.json", "r") as f:
        user_last5 = json.load(f)
    dataset = RankingDataset(df_train, user_last5, prod_emb_dict, user_emb_dict)

    

    # # === Grab first batch and inspect it ===
    # loader = DataLoader(dataset, batch_size=4, shuffle=False,
    #                     collate_fn=lambda b: rec_collate_fn(b, dataset=dataset, neg_k=2))
    # batch = next(iter(loader))
    # seq_embs, seq_masks, user_embs, item_embs, labels = batch

    # print("seq_embs shape: ", seq_embs.shape)     # (B + B*neg_k, S, D)
    # print("seq_masks shape: ", seq_masks.shape)   # (B + B*neg_k, S)
    # print("user_embs shape: ", user_embs.shape)   # (B + B*neg_k, D)
    # print("item_embs shape: ", item_embs.shape)   # (B + B*neg_k, D)
    # print("labels shape: ", labels.shape)         # (B + B*neg_k, 4)

    # # === Optional: print some values to inspect ===
    # print("\nExample sequence embedding [0]:\n", seq_embs[0])
    # print("Example user embedding [0]:\n", user_embs[0])
    # print("Example item embedding [0]:\n", item_embs[0])
    # print("Example label [0]:\n", labels[0])

    model = RecModel(embed_dim=128)
    train(model, dataset)
    print("Evaluation ...........")
    evaluate(model, dataset)
