import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Dataset Preparation
class AmazonDataset(Dataset):
    def __init__(self, user_df, product_df, user_feature_path, product_feature_path):
        self.user_df = user_df
        self.product_df = product_df

        # Label encode user_id and parent_asin
        self.user_encoder = LabelEncoder().fit(self.user_df['user_id'])
        self.product_encoder = LabelEncoder().fit(self.product_df['parent_asin'])

        self.user_df['user_idx'] = self.user_encoder.transform(self.user_df['user_id'])
        self.product_df['product_idx'] = self.product_encoder.transform(self.product_df['parent_asin'])

        self.labels = self.user_df['verified_purchase'].astype(int).values

        # Load user and prodcut features
        with open(user_feature_path, "rb") as f:
            self.user_features = pickle.load(f)
        with open(product_feature_path, 'rb') as f:
            self.product_features = pickle.load(f)


    def __len__(self):
        return len(self.user_df)

    def __getitem__(self, idx):
        row = self.user_df.iloc[idx]

        user_id = row['user_id']
        product_id = row['parent_asin']
        label = self.labels[idx]

        # Convert features to tensors
        user_vec = torch.tensor(self.user_features.get(user_id, np.zeros(2)), dtype=torch.float32)
        product_vec = torch.tensor(self.product_features.get(product_id, np.zeros(4033)), dtype=torch.float32)

        return user_vec, product_vec, torch.tensor(label, dtype=torch.float)

# Two tower model
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, product_dim, hidden_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.product_tower = nn.Sequential(
            nn.Linear(product_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward_user(self, x):
        return self.user_tower(x)
    
    def forward_product(self, x):
        return self.product_tower(x)

    def forward(self, user_feats, product_feats):
        user_vec = self.forward_user(user_feats)
        product_vec = self.forward_product(product_feats)
        score = (user_vec * product_vec).sum(dim=1)
        return score, user_vec, product_vec

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
    user_emb_list = []
    user_ids = []
    for user_id, vec in user_features.items():
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0) # [1, dim]
        with torch.no_grad():
            emb = model.forward_user(x).squeeze(0).cpu().numpy()
        user_emb_list.append(emb)
        user_ids.append(user_id)
    
    user_df = pd.DataFrame(user_emb_list, index=user_ids)

    # compute product embeddings
    product_emb_list = []
    product_ids = []
    for product_id, vec in product_features.items():
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model.forward_product(x).squeeze(0).cpu().numpy()
        product_emb_list.append(emb)
        product_ids.append(product_id)
    
    product_df = pd.DataFrame(product_emb_list, index=product_ids)

    user_df.to_csv(os.path.join(path, 'user_embeddings.csv'))
    product_df.to_csv(os.path.join(path, 'product_embeddings.csv'))



def recommend_top_k_products(user_id, user_emb_path, product_emb_path, k=5):
    """
    1. Loads the saved user and product embeddings from CSV.

    2. Retrieves the embedding vector for a given user_id.

    3. Computes cosine similarity between that user embedding and all product embeddings.

    4. Returns the top 5 most similar product IDs
    """
    # Load embeddings
    user_df = pd.read_csv(user_emb_path, index_col=0)
    product_df = pd.read_csv(product_emb_path, index_col=0)

    # Check if user_id exists
    if user_id not in user_df.index:
        raise ValueError(f"User ID '{user_id}' not found in user embeddings.")

    # Get user embedding
    # [user_id] (single brackets) returns a Series (1D vector).
    # .values converts it to a 1D NumPy array: shape (embedding_dim,)
    # .reshape(1, -1) manually reshapes it into a 2D array: shape (1, embedding_dim)
    # This is more flexible if you want to do additional reshaping or operations manually
    user_emb = user_df.loc[user_id].values.reshape(1, -1)

    # Compute cosine similarity
    sim_scores = cosine_similarity(user_emb, product_df.values)[0] # 2D array (1,N) 1 user, N products, get the row (first row)

    # Get top-k product indices
    # np.argsort(sim_scores) returns the indices that would sort the array in ascending order
    # [::-1] reverses the order to get descending sort
    top_k_idx = np.argsort(sim_scores)[::-1][:k]
    top_k_products = product_df.index[top_k_idx]
    top_k_scores = sim_scores[top_k_idx]

    # Return as DataFrame
    return pd.DataFrame({
        'user_id': [user_id] * k, 
        'product_id': top_k_products,
        'similarity': top_k_scores
    })


def get_top_k_similar_products_for_users(user_ids, k=5, path='./Data'):
    # Load embeddings
    user_df = pd.read_csv(f"{path}/user_embeddings.csv", index_col=0)
    product_df = pd.read_csv(f"{path}/product_embeddings.csv", index_col=0)

    results = []

    for user_id in user_ids:
        if user_id not in user_df.index:
            print(f"Warning: User ID '{user_id}' not found. Skipping.")
            continue

        # Get user embedding
        # [[user_id]] (double brackets) means you’re selecting a DataFrame (not a Series).
        # .values converts the DataFrame to a NumPy array.
        # The result is automatically 2D: shape (1, embedding_dim).
        user_emb = user_df.loc[[user_id]].values  # shape (1, embed_dim)

        # Compute cosine similarity
        sim_scores = cosine_similarity(user_emb, product_df.values)[0]

        # Get top-k product indices
        top_k_idx = np.argsort(sim_scores)[::-1][:k]
        top_k_product_ids = product_df.index[top_k_idx]
        top_k_scores = sim_scores[top_k_idx]

        # Collect results
        for pid, score in zip(top_k_product_ids, top_k_scores):
            results.append({
                'user_id': user_id,
                'product_id': pid,
                'similarity': score
            })

    result_df = pd.DataFrame(results)
    return result_df




# Evaluation
def precision_recall_at_k(model, dataset, user_encoder, product_encoder, K=10):
    model.eval()
    
    user_embeddings = model.user_embedding.weight.data  # [num_users, embed_dim]
    product_embeddings = model.product_embedding.weight.data  # [num_products, embed_dim]
    
    # Normalize for cosine similarity
    user_embeddings = nn.functional.normalize(user_embeddings, dim=1)
    product_embeddings = nn.functional.normalize(product_embeddings, dim=1)
    
    precisions = []
    recalls = []
    
    for user_id in np.unique(dataset.user_df['user_id']):
        user_idx = user_encoder.transform([user_id])[0]
        user_vec = user_embeddings[user_idx]  # [embed_dim]
        
        # Compute similarity with all products
        scores = torch.matmul(product_embeddings, user_vec)
        topk_indices = torch.topk(scores, K).indices.cpu().numpy()
        topk_product_ids = product_encoder.inverse_transform(topk_indices)
        
        # Get actual products this user bought (in dataset)
        actual_bought = dataset.user_df[
            (dataset.user_df['user_id'] == user_id) & 
            (dataset.labels == 1)
        ]['parent_asin'].unique()
        
        if len(actual_bought) == 0:
            continue
        
        # Compute Precision@K and Recall@K
        hits = len(set(actual_bought) & set(topk_product_ids))
        precision = hits / K
        recall = hits / len(actual_bought)
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    print(f"Precision@{K}: {avg_precision:.4f}, Recall@{K}: {avg_recall:.4f}")

# Inference: Recommend Top-K Products for a User
def recommend_products(model, user_id, user_encoder, product_encoder, product_df, K=10):
    model.eval()
    
    if user_id not in user_encoder.classes_:
        print("User not found!")
        return []

    user_idx = torch.tensor(user_encoder.transform([user_id]), dtype=torch.long)
    user_vec = model.user_embedding(user_idx).squeeze(0)
    user_vec = nn.functional.normalize(user_vec, dim=0)

    product_embeddings = model.product_embedding.weight.data
    product_embeddings = nn.functional.normalize(product_embeddings, dim=1)

    scores = torch.matmul(product_embeddings, user_vec)
    topk_indices = torch.topk(scores, K).indices.cpu().numpy()
    topk_product_ids = product_encoder.inverse_transform(topk_indices)

    return product_df[product_df['parent_asin'].isin(topk_product_ids)][['title', 'price', 'parent_asin']]


if __name__ == "__main__":
    # # full pipline
    # # Load and merge data
    # user_df = pd.read_excel("Data/top30k_user.xlsx")
    # product_df = pd.read_excel("Data/top30k.xlsx")

    # user_feature_path = "Data/user_features.pkl"
    # product_feature_path = "Data/product_features.pkl"
    # dataset = AmazonDataset(user_df, product_df, user_feature_path, product_feature_path)

    # print(np.unique(dataset.labels, return_counts=True))

    # missing_users = sum(user_id not in dataset.user_features for user_id in user_df['user_id'])
    # missing_products = sum(pid not in dataset.product_features for pid in user_df['parent_asin'])

    # print(f"Missing users: {missing_users}/{len(user_df)}")
    # print(f"Missing products: {missing_products}/{len(user_df)}")

    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # #Get feature dims from first sample
    # user_feat_dim = len(next(iter(dataset))[0])
    # product_feat_dim = len(next(iter(dataset))[1])

    # # Init model
    # model = TwoTowerModel(
    #     user_dim=user_feat_dim,
    #     product_dim=product_feat_dim,
    #     hidden_dim=128
    # )

    # # Optimizer and Loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCEWithLogitsLoss()

    # # Train
    # train_model(model, dataloader, optimizer, criterion, epochs=10)

    # # Save embeddings
    # save_embeddings(model, user_feature_path, product_feature_path)


    # # Example usage
    top_k = recommend_top_k_products(
        user_id='AHZZLN7P67CN7L4RODX665VBYQXQ',
        user_emb_path='./Data/user_embeddings.csv',
        product_emb_path='./Data/product_embeddings.csv',
        k=5
    )
    top_k.to_excel("Top5SimilarProductsV2.xlsx")
    print(top_k)

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
