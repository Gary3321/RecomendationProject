
import pandas as pd

user_reviews_file = ["Data/All_Beauty_UserReview.jsonl", "Data/Appliances.jsonl", "Data/Digital_Music.jsonl"]
products_file = ["Data/meta_All_Beauty.jsonl", "Data/meta_Appliances.jsonl", "Data/meta_Digital_Music.jsonl"]

df_top30k = pd.DataFrame()
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for i in range(len(user_reviews_file)):
    # Load datasets
    df_reviews = pd.read_json(user_reviews_file[i], lines=True)
    df_meta = pd.read_json(products_file[i], lines=True)

    df_reviews["datatime"] = pd.to_datetime(df_reviews["timestamp"])
    df_reviews_agg = df_reviews.groupby("parent_asin").agg(
    total_counts = ("parent_asin", 'count'),
    purchase_counts = ("verified_purchase", lambda x: x.sum()),
    data = ("datatime", "min")
    )
    df_reviews_agg = df_reviews_agg.reset_index()
    df_reviews_agg['purchase_score'] = df_reviews_agg['purchase_counts'] / df_reviews_agg['total_counts']
    #print(df_reviews_agg)

    print(df_meta.shape)
    # add purchase_score and datatime to df_meta
    df_meta = pd.merge(df_meta, df_reviews_agg[["parent_asin", "purchase_score", "data"]], on="parent_asin")

    df_meta_sorted = df_meta.sort_values(by="data", ascending=False)

    # select top 10k rows
    df_meta_top_10k = df_meta_sorted.head(10000)

    df_top30k = pd.concat([df_top30k, df_meta_top_10k], ignore_index=True)

    # split into training and test sets
    df_train_set = df_meta_top_10k.iloc[2000:].copy() #8000 for training
    df_train = pd.concat([df_train, df_train_set], ignore_index=True)

    df_test_set = df_meta_top_10k.iloc[:2000].copy() 
    df_test = pd.concat([df_test, df_test_set], ignore_index=True)

    

df_train.to_excel("Data/training_set24k.xlsx", index=False)
df_test.to_excel("Data/test_set6k.xlsx", index=False)

df_top30k.to_excel("Data/top30k.xlsx", index=False)

print(f"Training set shape: {df_train.shape}")
print(f"Test set shape: {df_test.shape}")
print(f"Top-30k set shape: {df_top30k.shape}")

"""
(112590, 14)
(94327, 16)
(70537, 14)
Training set shape: (24000, 18)
Test set shape: (6000, 18)
Top-30k set shape: (30000, 18)
"""