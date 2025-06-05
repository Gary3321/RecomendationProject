# Response-based knowledge distillation
import argparse
import os
import json
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler 

# step 1. import "teacher" model definitions and utilities
import RankingModelCrossnetPleMulAtten as teacher_model
# Now we have:
# teacher_model.PAD_ID
# teacher_model.RankingDataset
# teacher_model.PLERecModel
# teacher_model.load_model_for_inference

# step 2. define the L1 student model
class L1Ranker(nn.Module):
    """
    A lightweight student model. We simply concatenate
    (user_feat, item_feat) and feed them through a two
    layer MLP to predict the same 4-dim output as teacher
    model.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # 4 tasks
        )
    
    def forward(self, user_feat, item_feat):
        # user_feat: (B, D_user), item_feat: (B, D_item)
        x = torch.cat([user_feat, item_feat], dim=1) #(B, D_user + D_item)
        return self.mlp(x).clamp(min=-10, max=10) # match tearcher model's clamping range
    
# step 3. custom dataset for distillation
class DistillDataset(Dataset):
    """
    Wraps the original inputs(seq_feats, seq_mas, user_feat, item_feat) but
    replace "labels" with the teacher's logits (or probabilities).
    Only feed(user_feat, item_feat) into L1 for simplisity.
    """
    def __init__(self, base_dataset, teacher_logits):
        """
        base_dataset: instance of RankingDataset
        teacher_logists: tensor of shape (N, 4), where N=len(base_dataset)
              these are the raw logits from L2 for each sample.
        """
        self.base = base_dataset
        self.soft_labels = teacher_logits
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        # we only need user_feat and item_feat from base_dataset;
        # ignore seq_parts
        seq_feats, seq_mask, u_f, i_f, _ =self.base[idx]

        # u_f: (D_user,), i_f: (D_item,), teacher_logits: (N, 4)
        return(
            u_f, i_f, 
            self.soft_labels[idx] # (4, ) tensor
            )
    
# step 4. Utility: Collate function for DistillDataset
def distill_collate(batch):
    """
    batch: list of tuples (user_feat, item_feat, soft_label)
    Stack them into tensors.
    """
    users = torch.stack([b[0] for b in batch], dim=0) # (B, D_user)
    items = torch.stack([b[1] for b in batch], dim=0) # (B, D_item)
    labels = torch.stack([b[2] for b in batch], dim=0) # (B, 4)
    return users, items, labels

# Step 5. Main: parse args, load data, generate soft labels, train L1
def train(l2_checkpoint, train_df, user_feats, prod_feats, last5_json, batch_size, 
         epochs, lr, device):
#     parser = argparse.ArgumentParser(description="Distill teacher -> L1")
#     parser.add_argument(
#         "--l2_checkpoint", type=str, required=True,
#         help="Path to the pretrained teacher model checkpoint (e.g. teacher.pth)"
#     )
#     parser.add_argument(
#         "--train_df", type=str, required=True,
#         help="Path to the Excel file of the trainng set"
#     )
#     parser.add_argument(
#         "--user_feats", type=str, required=True,
#         help="Path to user_features_IncludeUid.pkl"
#     )
#     parser.add_argument(
#         "--prod_feats", type=str, required=True,
#         help="Path to product_features_IncludePid_withPad.pkl"
#     )
#     parser.add_argument(
#         "--last5_json", type=str, required=True,
#         help="Path to last_5_purchases_withPad.json"
#     )
#     parser.add_argument(
#     "--batch_size", type=int, default=256, help="Batch size for distillation"
# )
#     parser.add_argument(
#         "--epochs", type=int, default=5, help="Number of training epochs for L1"
#     )
#     parser.add_argument(
#         "--lr", type=float, default=1e-3, help="Learning rate for L1 optimizer"
#     )
#     parser.add_argument(
#         "--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'"
#     )
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Using device: {device}")

    # Load training datafram
    df_train = pd.read_excel(train_df)

    # Load and normalize user & item features
    uf_dict = pickle.load(open(user_feats, "rb"))

    # compute normalization stas based on training data
    train_user_ids = set(df_train["user_id"].unique())
    train_user_feats = np.array([uf_dict[uid] for uid in train_user_ids if uid in uf_dict])
    mean_u = train_user_feats.mean(axis=0)
    std_u = train_user_feats.std(axis=0) + 1e-6
    #Normalize all user features
    for uid in uf_dict:
        uf_dict[uid] = (uf_dict[uid] - mean_u) / std_u
    
    # item features
    pf_dict = pickle.load(open(prod_feats, "rb"))  # {item_id: array(D_item)}, includes PAD_ID
    # Collect training item IDs
    train_item_ids = set(df_train["parent_asin"].unique())
    # Compute normalization stats on training items only (excluding PAD_ID)
    item_matrix = np.array(
        [pf_dict[iid] for iid in train_item_ids if iid in pf_dict and iid != teacher_model.PAD_ID]
    )
    scaler_p = StandardScaler().fit(item_matrix)
    # Apply transform (and clip) to all items
    for pid in pf_dict:
        if pid != teacher_model.PAD_ID:
            pf_dict[pid] = np.clip(scaler_p.transform([pf_dict[pid]]), -5, 5)[0]
    # Ensure PAD_ID is a zero vector
    pf_dict[teacher_model.PAD_ID] = np.zeros_like(next(iter(pf_dict.values())))

    # load last-5 purchase sequences
    with open(last5_json, "r") as f:
        ul5 = json.load(f)  # {user_id: [last_5_item_ids, …]}

    # Dimension sizes
    D_user = len(next(iter(uf_dict.values())))
    D_item = len(next(iter(pf_dict.values())))
    print(f"[INFO] D_user={D_user}, D_item={D_item}")

    # build the Rankingdataset for teacher inference
    train_dataset = teacher_model.RankingDataset(df_train, ul5, pf_dict, uf_dict)

    print(f"Built the Rankingdataset")

    checkpoint = torch.load(l2_checkpoint, weights_only=True, map_location=device)
    l2_model = teacher_model.PLERecModel(D_user, D_item)
    l2_model.load_state_dict(checkpoint['model_state_dict'])

    print(f"loaded model")

    l2_model.to(device)
    l2_model.eval()

    #5.6 generate "soft labels" for every training samples
    all_teacher_logits=[]
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for seq_feats, seq_mask, u_f, i_f, _ in loader:
            seq_feats = seq_feats.to(device)
            seq_mask = seq_mask.to(device)
            u_f = u_f.to(device)
            i_f = i_f.to(device)

            # L2 (Teacher model) forward: returns raw logits (no sigmoid)
            logits = l2_model(seq_feats, seq_mask, u_f, i_f) #(B, 4)
            all_teacher_logits.append(logits.detach().cpu())
    
    all_teacher_logits = torch.cat(all_teacher_logits, dim=0) #(N, 4)
    print(f"[INFO] Collected teacher logits: {all_teacher_logits.shape}")

    # 5.7 Build the Distillation Dataset for L1
    print(f"[INFO] Building DistillDataset for L1 ...")
    distill_dataset = DistillDataset(base_dataset=train_dataset, teacher_logits=all_teacher_logits)
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn = distill_collate
    )

    # 5.8 Instantiate L1 model, loss, optimizer
    print("[INFO] Creating L1 student model ...")
    input_dim = D_user + D_item
    l1_model = L1Ranker(input_dim=input_dim, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(l1_model.parameters(), lr=lr, weight_decay=1e-5)
    # Use MSELoss on raw logits; L1 tries to match L2's logits.
    loss_fn = nn.MSELoss()

    # 5.9 Training loop for L1 student
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        l1_model.train()
        running_loss = 0.0
        total_samples = 0

        for user_feat, item_feat, soft_labels in distill_loader:
            user_feat = user_feat.to(device) #(B, D_user)
            item_feat = item_feat.to(device) #(B, D_item)
            soft_labels = soft_labels.to(device) #(B, 4)

            preds = l1_model(user_feat, item_feat) #(B, 4) # Forward pass
            loss = loss_fn(preds, soft_labels) # pointwise distillation # Compute loss

            optimizer.zero_grad() ## Clear gradients
            loss.backward()  # Backward pass (compute gradients)
            torch.nn.utils.clip_grad_norm_(l1_model.parameters(), max_norm=1)
            optimizer.step()  # Update parameters

            batch_size = soft_labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        epoch_loss = running_loss / total_samples
        print(f"[Epoch {epoch:02d} / {epochs}] L1  distill loss = {epoch_loss:.6f}")

        # Save the best-performing L1 model by validation on the same distillation loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(l1_model.state_dict(), "./Data/l1_student_best.pth")
            print(f"[INFO] New best L1 saved (loss={best_loss:.6f})-> l1_student_best.pth")
    print("[INFO] L1 distillation complete. Best loss: {:.6f}".format(best_loss))
    print("[INFO] Student weights saved to l1_student_best.pth")

def eval_collate(batch):
    # batch: list of tuples (seq_feats, seq_mask, user_feat, item_feat, labels)
    user_feats = torch.stack([b[2] for b in batch], dim=0)  # (B, D_user)
    item_feats = torch.stack([b[3] for b in batch], dim=0)  # (B, D_item)
    labels = torch.stack([b[4] for b in batch], dim=0)      # (B, 4)
    return user_feats, item_feats, labels

# Step 6: Evaluate 
def evaluate(l1_checkpoint, train_df, test_df, user_feats, prod_feats, last5_json, batch_size, 
         device):
    # Load training datafram
    df_train = pd.read_excel(train_df)
    df_test = pd.read_excel(test_df)
    # Load and normalize user & item features
    uf_dict = pickle.load(open(user_feats, "rb"))

    # compute normalization stas based on training data
    train_user_ids = set(df_train["user_id"].unique())
    train_user_feats = np.array([uf_dict[uid] for uid in train_user_ids if uid in uf_dict])
    mean_u = train_user_feats.mean(axis=0)
    std_u = train_user_feats.std(axis=0) + 1e-6
    #Normalize all user features
    for uid in uf_dict:
        uf_dict[uid] = (uf_dict[uid] - mean_u) / std_u
    
    # item features
    pf_dict = pickle.load(open(prod_feats, "rb"))  # {item_id: array(D_item)}, includes PAD_ID
    # Collect training item IDs
    train_item_ids = set(df_train["parent_asin"].unique())
    # Compute normalization stats on training items only (excluding PAD_ID)
    item_matrix = np.array(
        [pf_dict[iid] for iid in train_item_ids if iid in pf_dict and iid != teacher_model.PAD_ID]
    )
    scaler_p = StandardScaler().fit(item_matrix)
    # Apply transform (and clip) to all items
    for pid in pf_dict:
        if pid != teacher_model.PAD_ID:
            pf_dict[pid] = np.clip(scaler_p.transform([pf_dict[pid]]), -5, 5)[0]
    # Ensure PAD_ID is a zero vector
    pf_dict[teacher_model.PAD_ID] = np.zeros_like(next(iter(pf_dict.values())))

    # load last-5 purchase sequences
    with open(last5_json, "r") as f:
        ul5 = json.load(f)  # {user_id: [last_5_item_ids, …]}

    # Dimension sizes
    D_user = len(next(iter(uf_dict.values())))
    D_item = len(next(iter(pf_dict.values())))
    print(f"[INFO] D_user={D_user}, D_item={D_item}")

    # build the Rankingdataset for teacher inference
    test_dataset = teacher_model.RankingDataset(df_test, ul5, pf_dict, uf_dict)

    print(f"Built the test Rankingdataset")

    input_dim = D_user + D_item
    l1_model = L1Ranker(input_dim=input_dim, hidden_dim=64).to(device)
    state_dict = torch.load(l1_checkpoint, map_location=torch.device(device))
    l1_model.load_state_dict(state_dict)

    print(f"loaded model")

    from sklearn.metrics import roc_auc_score

    #loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    distill_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn = eval_collate
    )
    
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for user_feat, item_feat, labels in distill_loader:
                user_feat = user_feat.to(device) #(B, D_user)
                item_feat = item_feat.to(device) #(B, D_item)
                labels = labels.to(device) #(B, 4)
                print(f"user_feat shape: {user_feat.shape}, item_feat shape: {item_feat.shape}")
                logits = l1_model(user_feat, item_feat) #(B, 4) # Forward pass
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2_checkpoint = './Data/RM_CPM_model_best.pth' 
    train_df = './Data/training_set_ranking65k.xlsx'
    user_feats = "../RetrievalModel/Data/user_features_IncludeUid.pkl"
    prod_feats = "./Data/product_features_IncludePid_withPad.pkl"
    last5_json = "./Data/last_5_purchases_withPad.json"
    batch_size =256 
    epochs = 5 
    lr = 1e-4
    # train(l2_checkpoint, train_df, user_feats, prod_feats, last5_json, batch_size, 
    #      epochs, lr, device)
    
    l1_checkpoint = "./Data/l1_student_best.pth"
    test_df = './Data/testing_set_ranking9k.xlsx'
    evaluate(l1_checkpoint, train_df, test_df, user_feats, prod_feats, last5_json, batch_size, 
         device)
