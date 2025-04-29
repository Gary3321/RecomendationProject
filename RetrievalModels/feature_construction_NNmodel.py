import torch
import torch.nn as nn 
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# user_reviews_file = "Data/All_Beauty_UserReview.jsonl"
# products_file = "Data/meta_All_Beauty.jsonl"

# # Load datasets
# df_reviews = pd.read_json(user_reviews_file, lines=True)
# df_train = pd.read_json(products_file, lines=True)

# df_reviews["datatime"] = pd.to_datetime(df_reviews["timestamp"])
# df_reviews_agg = df_reviews.groupby("parent_asin").agg(
#  total_counts = ("parent_asin", 'count'),
#  purchase_counts = ("verified_purchase", lambda x: x.sum()),
#  data = ("datatime", "min")
# )
# df_reviews_agg = df_reviews_agg.reset_index()
# df_reviews_agg['purchase_score'] = df_reviews_agg['purchase_counts'] / df_reviews_agg['total_counts']
# #print(df_reviews_agg)

# print(df_train.shape)
# # add purchase_score and datatime to df_train
# df_train = pd.merge(df_train, df_reviews_agg[["parent_asin", "purchase_score", "data"]], on="parent_asin")

# df_train = pd.read_excel("Data/training_set.xlsx")
# df_test = pd.read_excel("Data/test_set.xlsx")

# df_train[['purchase_score', 'average_rating']] =df_train[['purchase_score', 'average_rating']].fillna(0)
# df_test[['purchase_score', 'average_rating']] =df_test[['purchase_score', 'average_rating']].fillna(0)


# scaler = MinMaxScaler()

# scaler.fit(df_train[['purchase_score', 'average_rating']])
# # scaler need 2D
# df_train[['purchase_score', 'average_rating']] = scaler.transform(df_train[['purchase_score', 'average_rating']] )
# df_test[['purchase_score', 'average_rating']] = scaler.transform(df_test[['purchase_score', 'average_rating']] )


# # df_train['composite_score'] = 0.6 * df_train['purchase_score'] + 0.4 * df_train['average_rating']
# # df_train['composite_score'] = scaler.fit_transform(df_train['composite_score'])


# # Load embedder
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_field(text):
    try:
        if not text: # empty to 0s
            return np.zeros(embedder.get_sentence_embedding_dimension())
        return embedder.encode(text, convert_to_numpy=True)
    except Exception:
        return np.zeros(embedder.get_sentence_embedding_dimension())

# for each row in the data
def row_to_text_embeddings(row):
    emb_title = embed_text_field(row['title'])
    emb_features = embed_text_field(" | ".join(row['features']))
    emb_description = embed_text_field(" | ".join(row['description']))
    emb_categories = embed_text_field(" | ".join(row['categories']))
    return np.concatenate([emb_title, emb_features, emb_description, emb_categories])


# Helper function to lookup
def embed_category(row):
    idx = cat2idx.get(row['main_category'], cat2idx['UNK'])
    return cat_emb(torch.tensor(idx)).detach().numpy()

def embed_store(row):
    idx = store2idx.get(row['store'], store2idx['UNK'])
    return store_emb(torch.tensor(idx)).detach().numpy()

# Dic fiels, like details (dict) e.g. {'Package Dimensions': '7.1 x 5.5 x 3 inches; 2.38 Pounds', 'UPC': '617390882781'}    
# just simple text embed
def embed_details(row):
    if not row['details']:
        return np.zeros(embedder.get_sentence_embedding_dimension())
    text = json.dumps(row['details'], ensure_ascii=False)
    return embed_text_field(text)

# Assemble final feature
def build_feature_vector(row, image_emb_dict):
    # 1. textual embeddigns (4 fields x 384 dim)
    text_emb = row_to_text_embeddings(row) # 4 x 384 = 1536 dims

    # 2. categorical embeddigns (32 + 32 dims)
    cat_vec = embed_category(row)  # 32 dims
    store_vec = embed_store(row)   # 32 dims

    # 3. details embedding (384 dims)
    details_vec = embed_details(row) # 384 dims

    # 4. numeric features (1 dim - price)
    numeric_vec = np.array([row['price']])  # 1 dim

    # 5. image embedding
    image_vec = image_emb_dict[row['parent_asin']]
    # concatenate everything
    return np.concatenate([text_emb, cat_vec, store_vec, details_vec, numeric_vec, image_vec]) # 1985 dim + image_vec dim

# Train model
class ContentDataset(Dataset):
    def __init__(self, df, asin_emb_dict):
        self.df = df
        self.asin_emb_dict = asin_emb_dict
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(build_feature_vector(row, self.asin_emb_dict), dtype=torch.float32)
        y = torch.tensor(0.6 * row['purchase_score'] + 0.4 * row['average_rating'], dtype=torch.float32)
        return x, y 

# Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

def train_blip_model():
    blip_image_emb = np.load("Data/image_caption_embeddings.npz")  # 384
    blip_image_emb_dict = {asin : emb for asin, emb in zip(blip_image_emb['image_ids'], blip_image_emb['embeddings'])}

    # Instantiate
    dataset = ContentDataset(df_train, blip_image_emb_dict)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MLP(input_dim=2369) 
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training Loop
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"BLIP Epoch {epoch}: Loss {total_loss / len(loader):.4f}")

    # save the model's state_dict after training
    torch.save(model.state_dict(), "Blip_emb_trained_model.pth")

def train_cnn_model():
    # CNN embedding
    print("CNN embedding ...................................")
    cnn_image_emb = np.load("Data/image_cnn_embeddings.npz")  # 2048
    cnn_image_emb_dict = {asin : emb for asin, emb in zip(cnn_image_emb['parent_asins'], cnn_image_emb['embeddings'])}

    # Instantiate
    dataset = ContentDataset(df_train, cnn_image_emb_dict)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MLP(input_dim=4033)  
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training Loop
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"CNN Epoch {epoch}: Loss {total_loss / len(loader):.4f}")

    # save the model's state_dict after training
    torch.save(model.state_dict(), "CNN_emb_trained_model.pth")


######### Model Evaluation ###########
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return y_pred 

# Evaluate BLIP caption embedding model
def evaluate_blip(df_test):
    blip_image_emb = np.load("Data/image_caption_embeddings.npz")  # 384
    blip_image_emb_dict = {asin : emb for asin, emb in zip(blip_image_emb['image_ids'], blip_image_emb['embeddings'])}

    df_test = df_test[df_test['parent_asin'].isin(blip_image_emb_dict.keys())]
    # build test dataset and loader
    blip_test_dataset = ContentDataset(df_test, blip_image_emb_dict)
    blip_test_loader = DataLoader(blip_test_dataset, batch_size=32)

    # load model
    blip_model = MLP(input_dim=2369)
    blip_model.load_state_dict(torch.load("Blip_emb_trained_model.pth"))

    # Evaluate
    y_pred_blip = evaluate_model(blip_model, blip_test_loader)

    df_test['blip_prediction'] = y_pred_blip
    return df_test

def evaluate_cnn(df_test):
    cnn_image_emb = np.load("Data/image_cnn_embeddings.npz")  # 2048
    cnn_image_emb_dict = {asin : emb for asin, emb in zip(cnn_image_emb['parent_asins'], cnn_image_emb['embeddings'])}

    df_test = df_test[df_test['parent_asin'].isin(cnn_image_emb_dict.keys())]
    # build test dataset and loader
    cnn_test_dataset = ContentDataset(df_test, cnn_image_emb_dict)
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=32)

    # load model
    cnn_model = MLP(input_dim=4033)
    cnn_model.load_state_dict(torch.load("CNN_emb_trained_model.pth"))

    # Evaluate
    y_pred_cnn = evaluate_model(cnn_model, cnn_test_loader)

    df_test['cnn_prediction'] = y_pred_cnn
    return df_test 

if __name__ == "__main__":

    df_train = pd.read_excel("Data/training_set24k.xlsx")
    df_test = pd.read_excel("Data/test_set6k.xlsx")

    df_train[['purchase_score', 'average_rating']] =df_train[['purchase_score', 'average_rating']].fillna(0)
    df_test[['purchase_score', 'average_rating']] =df_test[['purchase_score', 'average_rating']].fillna(0)


    scaler = MinMaxScaler()

    scaler.fit(df_train[['purchase_score', 'average_rating']])
    # scaler need 2D
    df_train[['purchase_score', 'average_rating']] = scaler.transform(df_train[['purchase_score', 'average_rating']] )
    df_test[['purchase_score', 'average_rating']] = scaler.transform(df_test[['purchase_score', 'average_rating']] )

    # Load embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # build vocab for main_category and store fields
    categories = df_train['main_category'].fillna('UNK').unique().tolist()
    if 'UNK' not in categories:
        categories.append('UNK')

    stores = df_train['store'].fillna('UNK').unique().tolist()
    if 'UNK' not in stores:
        stores.append('UNK')

    cat2idx = {c:i for i, c in enumerate(categories)}
    store2idx = {s:i for i, s in enumerate(stores)}

    # Create embedding layers
    cat_emb = nn.Embedding(len(categories), 32)
    store_emb = nn.Embedding(len(stores), 32)

    # Numeric fields - fillna with median -- normalization
    df_train['price'] = df_train['price'].fillna(df_train['price'].median())
    # scaler = MinMaxScaler() 
    # df_train['price'] = scaler.fit_transform(df_train['price']) # data should be 2-dimensional
    # Min-Max scaling
    df_train['price'] = (df_train['price'] - df_train['price'].min()) / (df_train['price'].max() - df_train['price'].min())


    df_test['price'] = df_test['price'].fillna(df_test['price'].median())
    df_test['price'] = (df_test['price'] - df_test['price'].min()) / (df_test['price'].max() - df_test['price'].min())

    print(f"training cnn model ............")
    #train_cnn_model()

    print(f"training blip model ............")
    #train_blip_model()

    print("evaluation ..........")
    df_test = evaluate_blip(df_test)
     

    df_test.sort_values(by="blip_prediction", ascending=False).head(50).to_excel("blip_retrieval_top50.xlsx", index=False)

    df_test = evaluate_cnn(df_test)
    df_test.sort_values(by="cnn_prediction", ascending=False).head(50).to_excel("cnn_retrieval_top50.xlsx", index=False)