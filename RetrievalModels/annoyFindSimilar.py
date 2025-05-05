from annoy import AnnoyIndex
import pickle
import os

# --- Load product embeddings ---
embedding_path = 'Data/product_embeddings_IncludePid.pkl'
with open(embedding_path, 'rb') as f:
    product_embeddings = pickle.load(f)

# --- Determine embedding dimension ---
sample_embedding = next(iter(product_embeddings.values()))
embedding_dim = len(sample_embedding)

# --- Create Annoy index ---
annoy_index = AnnoyIndex(embedding_dim, 'angular')  # angular is for cosine similarity approximation

# Map from internal Annoy index to product ID
index_to_pid = {}
pid_to_index = {}

# --- Add items to Annoy index ---
for i, (pid, emb) in enumerate(product_embeddings.items()):
    annoy_index.add_item(i, emb)
    index_to_pid[i] = pid
    pid_to_index[pid] = i

# --- Build the index ---
annoy_index.build(n_trees=10)  # More trees = more accuracy

# --- Save the index for later use (optional) ---
annoy_index.save('Data/product_annoy_index.ann')

# --- Query: find top 5 similar products to a given product ID ---
def get_similar_products(query_pid, top_k=5):
    if query_pid not in pid_to_index:
        raise ValueError(f"Product ID {query_pid} not in index.")
    
    query_index = pid_to_index[query_pid]
    nearest_indices = annoy_index.get_nns_by_item(query_index, top_k+1)  # +1 to exclude self

    # Remove the query itself
    nearest_indices = [idx for idx in nearest_indices if idx != query_index]

    similar_pids = [index_to_pid[idx] for idx in nearest_indices]
    return similar_pids

# --- Example usage ---
query_product = 'B0B38YRJKY'  
try:
    similar_products = get_similar_products(query_product)
    print(f"Products similar to {query_product}:")
    for pid in similar_products:
        print(pid)
except ValueError as e:
    print(e)

try:
    query_product = "B0B38YRJKY"  # example product
    idx = pid_to_index.get(query_product)
    if idx is None:
        raise ValueError(f"Product ID {query_product} not found.")

    # Get top 5 similar items and their distances
    similar_indices, distances = annoy_index.get_nns_by_item(idx, 5, include_distances=True)

    print(f"Products similar to {query_product}:")
    for i, dist in zip(similar_indices, distances):
        sim_product = index_to_pid[i]
        similarity = 1 / (1 + dist)  # Convert distance to a rough similarity score
        print(f"{sim_product} (similarity ≈ {similarity:.4f})")

except ValueError as e:
    print(e)
