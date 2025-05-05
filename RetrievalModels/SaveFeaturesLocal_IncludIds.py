import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json

def generate_user_features():
    # 1. Load the raw DataFrame
    df = pd.read_excel("Data/top30k_user.xlsx") 

    # 2. Select and aggregate features per user
    user_features = df[['user_id', 'rating', 'helpful_vote']].copy()

    # 3. Handle NaNs
    user_features['rating'] = user_features['rating'].fillna(user_features['rating'].median())
    user_features['helpful_vote'] = user_features['helpful_vote'].fillna(0)

    # 4. Group and aggregate: mean rating and total helpful votes per user
    agg_user_features = user_features.groupby('user_id').agg({
        'rating': 'mean',
        'helpful_vote': 'sum'
    }).reset_index()

    # 5. Encode user_id to interger code
    le = LabelEncoder()
    agg_user_features['user_code'] = le.fit_transform(agg_user_features['user_id'])

    # save the encoder for inference
    with open("Data/user_id_encoder.pkl", "wb") as f_enc:
        pickle.dump(le, f_enc)

    # 6. Normalize continuous features
    scaler = StandardScaler()
    agg_user_features[['rating_s', 'helpful_vote_s']] = scaler.fit_transform(agg_user_features[['rating', 'helpful_vote']])
    
    # save this scaler for transforming new users later
    with open("Data/user_feat_scaler.pkl", "wb") as f_scl:
        pickle.dump(scaler, f_scl)


    # 7. Build final feature vector [user_code, rating_s, helpful_vote_s]
    user_feature_dict = {}
    for _, row in agg_user_features.iterrows():
        uid = row['user_id']
        vec = [float(row['user_code']), row['rating_s'], row['helpful_vote_s']]
        user_feature_dict[uid] = vec
    # 8. Persist -- Save to disk
    with open("Data/user_features_IncludeUid.pkl", "wb") as f:
        pickle.dump(user_feature_dict, f)

    print(f"Saved {len(user_feature_dict)} user feature vectors.")

##### Product features
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
    return idx

def embed_store(row):
    idx = store2idx.get(row['store'], store2idx['UNK'])
    return idx

# Dic fiels, like details (dict) e.g. {'Package Dimensions': '7.1 x 5.5 x 3 inches; 2.38 Pounds', 'UPC': '617390882781'}    
# just simple text embed
def embed_details(row):
    if not row['details']:
        return np.zeros(embedder.get_sentence_embedding_dimension())
    text = json.dumps(row['details'], ensure_ascii=False)
    return embed_text_field(text)

# Assemble final feature
def build_feature_vector(row, image_emb_dict):
    # 0) product_code
    prod_code = np.array([row['product_code']], dtype=np.float32)


    # 1. categorical embeddigns (1 + 1 dims)
    cat_vec = np.array([embed_category(row)], dtype= np.float32) # 1 dims
    store_vec = np.array([embed_store(row)], dtype=np.float32)    # 1 dims

    # 1. textual embeddigns (4 fields x 384 dim)
    text_emb = row_to_text_embeddings(row) # 4 x 384 = 1536 dims

    # 3. details embedding (384 dims)
    details_vec = embed_details(row) # 384 dims

    # 4. numeric features (1 dim - price)
    numeric_vec = np.array([row['price']])  # 1 dim

    # 5. image embedding
    image_vec = image_emb_dict[row['parent_asin']]
    # concatenate everything
    return np.concatenate([prod_code, cat_vec, store_vec, text_emb, details_vec, numeric_vec, image_vec]) # 1924 dim + image_vec dim

def generate_product_features(df_meta):

    cnn_image_emb = np.load("Data/image_cnn_embeddings.npz")  # 2048
    cnn_image_emb_dict = {
        asin : emb 
        for asin, emb in zip(cnn_image_emb['parent_asins'], cnn_image_emb['embeddings'])
        }

    # encode parent_asin ---> integer code
    le = LabelEncoder()
    df_meta['product_code'] = le.fit_transform(df_meta['parent_asin'])
    # save encoder for inference
    with open("Data/product_id_encoder.pkl", "wb") as f_enc:
        pickle.dump(le, f_enc)
    
    # build feature vectors
    product_feature_dict = {}
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Generating product features"):
        parent_asin = row['parent_asin']

        # ensure image embedding exist
        if parent_asin not in cnn_image_emb_dict:
            continue

        try:
            product_feature = build_feature_vector(row, cnn_image_emb_dict)
            product_feature_dict[parent_asin] = product_feature
        except Exception as e:
            print(f"Skipped {parent_asin} due to error: {e}")
            continue
    # save to disk
    with open("Data/product_features_IncludePid.pkl", "wb") as f:
        pickle.dump(product_feature_dict, f)
    print(f"Saved {len(product_feature_dict)} product feature vectors.")

if __name__ == "__main__":
    #generate_user_features()
    # Load embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    df_meta = pd.read_excel("Data/top30k.xlsx")
    df_meta = df_meta.dropna(subset=['parent_asin'])
    # build vocab for main_category and store fields
    categories = df_meta['main_category'].fillna('UNK').unique().tolist()
    if 'UNK' not in categories:
        categories.append('UNK')

    stores = df_meta['store'].fillna('UNK').unique().tolist()
    if 'UNK' not in stores:
        stores.append('UNK')

    cat2idx = {c:i for i, c in enumerate(categories)}
    store2idx = {s:i for i, s in enumerate(stores)}


    # Numeric fields - fillna with median -- normalization
    df_meta['price'] = df_meta['price'].fillna(df_meta['price'].median())
    df_meta['price'] = (df_meta['price'] - df_meta['price'].min()) / (df_meta['price'].max() - df_meta['price'].min())

    generate_product_features(df_meta)


