import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity

# Dataset Preparation
class AmazonDataset(Dataset):
    def __init__(self, user_df, product_df, user_feature, product_feature):
        self.user_df = user_df
        self.product_df = product_df

        self.labels = self.user_df['verified_purchase'].astype(int).values

        # Load user and prodcut features
        self.user_features = user_feature
        self.product_features = product_feature


    def __len__(self):
        return len(self.user_df)

    def __getitem__(self, idx):
        row = self.user_df.iloc[idx]

        user_id = row['user_id']
        product_id = row['parent_asin']
        label = self.labels[idx]

        # Convert features to tensors
        user_vec = torch.tensor(self.user_features.get(user_id, np.zeros(3)), dtype=torch.float32)
        product_vec = torch.tensor(self.product_features.get(product_id, np.zeros(4034)), dtype=torch.float32)

        return user_id, user_vec, product_vec, torch.tensor(label, dtype=torch.float), product_id

def rec_collate_fn(batch, neg_k=2):
    """
    Batch negative sampling: sample negatives from the positive items present in this batch.
    batch: list of tuples (u, seq_embs, seq_mask, user_emb, pos_item_emb, labels, pos_item_id)
    """
    user_ids, user_vec, product_vec, labels, product_ids = zip(*batch)
    B = len(user_ids)

    # stack positives
    user_embs = torch.stack(user_vec)         # (B, D)
    pos_item_embs = torch.stack(product_vec) # (B, D)
    pos_labels = torch.stack(labels).view(-1)       # (B, )

    # negative sampling from batch's positive items
    neg_user_embs, neg_item_embs, neg_labels = [], [], []
    for idx in range(B):
        # candidate negatives: all other pos_item_ids in batch
        candidates = [j for j in range(B) if j != idx]
        for _ in range(neg_k):
            neg_idx = random.choice(candidates)
            neg_user_embs.append(user_embs[idx])
            neg_item_embs.append(pos_item_embs[neg_idx])
            neg_labels.append(torch.zeros(1, dtype=torch.float32))

    if neg_item_embs:
        all_user_embs = torch.cat([user_embs, torch.stack(neg_user_embs)], dim=0)
        all_item_embs = torch.cat([pos_item_embs, torch.stack(neg_item_embs)], dim=0)
        all_labels = torch.cat([pos_labels, torch.stack(neg_labels).view(-1)], dim=0)
    else:
        all_user_embs, all_item_embs, all_labels =user_embs, pos_item_embs, pos_labels

    return all_user_embs, all_item_embs, all_labels


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x0):
        x_l = x0
        for i in range(self.num_layers):
            gate = (x_l * self.weights[i]).sum(dim=1, keepdim=True) # (B, 1)
            x_l = x0 * gate + self.biases[i] +x_l  # (B, D)
        return x_l


class TwoTowerModelCrossLayer(nn.Module):
    def __init__(self, users_num, prod_num, cat_num, store_num, user_raw_dim, product_raw_dim, 
                 cat_emb_dim=32, user_embed_dim=128,
                 product_embed_dim=128, deep_hidden_dim=256, cross_layers=2, 
                 use_deep=True):
        super().__init__()
        self.use_deep = use_deep
        # learnable embeddings, convert the following 1D vector to cat_emb_dim D vector
        self.user_id_emb = nn.Embedding(users_num, cat_emb_dim) #
        self.product_id_emb = nn.Embedding(prod_num, cat_emb_dim)
        self.cat_emb = nn.Embedding(cat_num, cat_emb_dim)
        self.store_emb = nn.Embedding(store_num, cat_emb_dim)

        # adjust the user and product raw dimension
        self.user_in_dim = user_raw_dim -1 + cat_emb_dim
        self.product_in_dim = product_raw_dim-3+cat_emb_dim*3

        self.user_tower = nn.Sequential(
            nn.Linear(self.user_in_dim, user_embed_dim),
            nn.ReLU(),
            nn.Linear(user_embed_dim, user_embed_dim),
        )
        self.product_tower = nn.Sequential(
            nn.Linear(self.product_in_dim, product_embed_dim),
            nn.ReLU(),
            nn.Linear(product_embed_dim, product_embed_dim)
        )

        self.cross_input_dim = self.user_in_dim + self.product_in_dim
        self.cross = CrossNetwork(self.cross_input_dim, num_layers=cross_layers)



        if self.use_deep:
            self.deep_mlp = nn.Sequential(
                nn.Linear(self.cross_input_dim, deep_hidden_dim),
                nn.ReLU(),
                nn.Linear(deep_hidden_dim, deep_hidden_dim)
            )
            # Final head
            final_input_dim = self.cross_input_dim + deep_hidden_dim + user_embed_dim + product_embed_dim
        else:
            # Final head
            final_input_dim = self.cross_input_dim  + user_embed_dim + product_embed_dim
        self.pred = nn.Linear(final_input_dim, 1)

    def forward_user(self, x):
        return self.user_tower(x)
    
    def forward_product(self, x):
        return self.product_tower(x)

    def forward(self, user_raw_feats, product_raw_feats):
        # look up
        user_idx = user_raw_feats[:, 0].long() #(B, )
        user_cont = user_raw_feats[:, 1:] #(B, original_user_raw_dim -1)
        user_id_e = self.user_id_emb(user_idx) # (B, cat_emb_dim)
        # new user input
        user_in = torch.cat([user_id_e, user_cont], dim=1) #(B, original_user_dim-1+cat_emb_idm)

        # Item
        p_idx = product_raw_feats[:, 0].long() #(B, )
        cat_idx = product_raw_feats[:, 1].long() #(B, )
        store_idx = product_raw_feats[:, 2].long() #(B, )
        prod_cont = product_raw_feats[:, 3:]  #(B, original -3)

        p_id_e = self.product_id_emb(p_idx)
        cat_e = self.cat_emb(cat_idx)
        store_e = self.store_emb(store_idx)
        # new product input
        product_in = torch.cat([p_id_e, cat_e, store_e, prod_cont], dim=1) # (B, original -3 + 3*cat_emb_dim) 

        # --- pass through towers ---
        # print(f"user_in.shape: {user_in.shape}")
        # print(f"Excepted dim: {self.user_in_dim}")
        user_embed = self.forward_user(user_in)        # (B, user_embed_dim)

        # print(f"user_in.shape: {product_in.shape}")
        # print(f"Excepted dim: {self.product_in_dim}")
        item_embed = self.forward_product(product_in)  # (B, product_embed_dim)

        # --- combine for cross & deep ---
        # Raw input concat
        raw_concat = torch.cat([user_in, product_in], dim=1) #

        # CrossNet
        x_cross = self.cross(raw_concat)

        # print("x_cross:", x_cross.shape)
        
        # print("user_embed:", user_embed.shape)
        # print("item_embed:", item_embed.shape)

        # Optional deep MLP
        if self.use_deep:
            x_deep = self.deep_mlp(raw_concat)
            # print("x_deep:", x_deep.shape if self.use_deep else "not used")
            final_input = torch.cat([x_cross, x_deep, user_embed, item_embed], dim=1)
        else:
            final_input = torch.cat([x_cross, user_embed, item_embed], dim=1)

        score = self.pred(final_input).squeeze(1)
        
        return score, user_embed, item_embed

# Training loop
def train_model(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_feats, product_feats, labels in dataloader:

            optimizer.zero_grad()
            score, _, _ = model(user_feats, product_feats)
            loss = criterion(score, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)  # accumulate total loss (not per batch)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# save embedding
def save_embeddings(model, user_feature_path, product_feature_path, path='./Data'):
    os.makedirs(path, exist_ok=True)

    # Load features
    with open(user_feature_path, 'rb') as f:
        user_features = pickle.load(f)
    with open(product_feature_path, 'rb') as f:
        product_features = pickle.load(f)

    model.eval()

    # compute user embeddings
    user_emb_dict = {}
    for user_id, vec in user_features.items():
        vec = torch.tensor(vec, dtype=torch.float32) # [D]
        user_idx = vec[0].long().unsqueeze(0) # user_id index
        user_cont = vec[1:].unsqueeze(0) # continuous part

        with torch.no_grad():
            user_id_e = model.user_id_emb(user_idx) # [1, cat_emb_dim]
            user_input = torch.cat([user_id_e, user_cont], dim=1) #[1, full_user_dim]
            emb = model.forward_user(user_input).squeeze(0).cpu().numpy()
        user_emb_dict[user_id] = emb
    with open(os.path.join(path, 'user_embeddings_IncludeUid.pkl'), "wb") as f:
        pickle.dump(user_emb_dict, f)

    # compute product embeddings
    product_emb_dict = {}
    for product_id, vec in product_features.items():
        vec = torch.tensor(vec, dtype=torch.float32)
        p_idx = vec[0].long().unsqueeze(0)
        cat_idx = vec[1].long().unsqueeze(0)
        store_idx = vec[2].long().unsqueeze(0)
        prod_cont = vec[3:].long().unsqueeze(0)
        with torch.no_grad():
            p_id_e = model.product_id_emb(p_idx)
            cat_e = model.cat_emb(cat_idx)
            store_e = model.store_emb(store_idx)
            product_input = torch.cat([p_id_e, cat_e, store_e, prod_cont], dim=1)
            emb = model.forward_product(product_input).squeeze(0).cpu().numpy()
        product_emb_dict[product_id] = emb
    with open(os.path.join(path, 'product_embeddings_IncludePid.pkl'), "wb") as f:
        pickle.dump(product_emb_dict, f)



if __name__ == "__main__":
    # full pipline
    # Load and merge data
    user_df = pd.read_excel("Data/top30k_user.xlsx")
    product_df = pd.read_excel("Data/top30k.xlsx")

    user_feature_path = "Data/user_features_IncludeUid.pkl"
    product_feature_path = "Data/product_features_IncludePid.pkl"

    with open(user_feature_path, "rb") as f:
        user_features = pickle.load(f)
    with open(product_feature_path, 'rb') as f:
        product_features = pickle.load(f)

    user_codes = [vec[0] for vec in user_features.values()]
    users_num = len(set(user_codes))

    prod_codes, cat_codes, store_codes = zip(*[(vec[0], vec[1], vec[2]) for vec in product_features.values()])
    prod_num = len(set(prod_codes))
    cat_num = len(set(cat_codes))
    store_num = len(set(store_codes))


    dataset = AmazonDataset(user_df, product_df, user_features, product_features)

    print(np.unique(dataset.labels, return_counts=True))

    missing_users = sum(user_id not in dataset.user_features for user_id in user_df['user_id'])
    missing_products = sum(pid not in dataset.product_features for pid in user_df['parent_asin'])

    print(f"Missing users: {missing_users}/{len(user_df)}")
    print(f"Missing products: {missing_products}/{len(user_df)}")

    batch_size =64
    neg_k =2
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: rec_collate_fn(b,  neg_k=neg_k))

    #Get feature dims from first sample
    user_feat_dim = len(next(iter(dataset))[1])
    product_feat_dim = len(next(iter(dataset))[2])

    # Init model
    model = TwoTowerModelCrossLayer(
        users_num = users_num,
        prod_num = prod_num,
        cat_num =cat_num,
        store_num =store_num,
        user_raw_dim=user_feat_dim,
        product_raw_dim=product_feat_dim
    )

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    print(f"----- Training model -----------")
    train_model(model, dataloader, optimizer, criterion, epochs=10)

    print(f"--------- saving embeddings ---------")
    # Save embeddings
    save_embeddings(model, user_feature_path, product_feature_path)


    # # Evaluate model
    # precision_recall_at_k(model, dataset, dataset.user_encoder, dataset.product_encoder, K=10)

    # # Recommend for a specific user
    # recommend_user = 'A1RPTVW5VEOSI'  # example user_id from your dataset
    # recommendations = recommend_products(
    #     model,
    #     user_id=recommend_user,
    #     user_encoder=dataset.user_encoder,
    #     product_encoder=dataset.product_encoder,
    #     product_df=product_df,
    #     K=5
    # )
    # print(recommendations)
