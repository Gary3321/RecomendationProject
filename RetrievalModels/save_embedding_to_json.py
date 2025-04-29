import pandas as pd
import json

def save_embeddings_to_json_from_csv(user_embeddings_csv, product_embeddings_csv, 
                                     user_json_path, product_json_path):
    user_df = pd.read_csv(user_embeddings_csv, index_col=0)
    product_df = pd.read_csv(product_embeddings_csv, index_col=0)

    # Convert dataframes to dictionaries
    user_embeddings_dict = user_df.apply(lambda row: row.tolist(), axis=1).to_dict() # Convert rows to dictionary with IDs as keys
    product_embeddings_dict = product_df.apply(lambda row: row.tolist(), axis=1).to_dict()

    # save dictionaries as JSON files
    with open(user_json_path, 'w') as user_json_file:
        json.dump(user_embeddings_dict, user_json_file, indent=4)
    
    with open(product_json_path, 'w') as product_json_file:
        json.dump(product_embeddings_dict, product_json_file, indent=4)

if __name__ == "__main__":
    save_embeddings_to_json_from_csv(
    user_embeddings_csv='RetrievalModel/Data/user_embeddings.csv',
    product_embeddings_csv='RetrievalModel/Data/product_embeddings.csv',
    user_json_path='RetrievalModel/Data/user_embeddings.json',
    product_json_path='RetrievalModel/Data/product_embeddings.json'
)