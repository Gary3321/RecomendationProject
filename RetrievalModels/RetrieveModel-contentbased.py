import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


# user_reviews_file = "Data/All_Beauty_UserReview.jsonl"
# products_file = "Data/meta_All_Beauty.jsonl"

# # Load datasets
# df_reviews = pd.read_json(user_reviews_file, lines=True)
# df_meta = pd.read_json(products_file, lines=True)

df_meta = pd.read_excel("Data/top30k.xlsx")

# print(f"User reviews columns: {df_reviews.columns.tolist()}, dimention: {df_reviews.shape}")
print(f"meta columns: {df_meta.columns.tolist()}, dimention: {df_meta.shape}")



### Cohort Retrieval, based on adjusted average rating

def cohort_retrieval(df_meta):
    # Log-transform to deal with skew
    # computes the natural log of (1 + x) for each element. It's safer for small numbers or zeros than np.log(x)
    X = np.log1p(df_meta['rating_number']).values.reshape(-1,1)
    #  reshapes that 1D array into a 2D array. -1 tells NumPy: “Figure out this dimension for me.”
    y = df_meta['average_rating'].values

    # fit regression model
    reg = LinearRegression()
    reg.fit(X, y)

    # predict
    df_meta['predicted_rating'] = reg.predict(X)
    df_meta['adjust_rating'] = 0.7*df_meta['average_rating'] + 0.3 *df_meta['predicted_rating']

    print(f"df_meta shape: {df_meta.shape}")

    # Retrieve top 50 items based on most actions
    top_items = df_meta.sort_values('adjust_rating', ascending=False).head(50)

    print("Top 50 items based on most actions:")
    print(top_items[['parent_asin', 'title', 'main_category', 'average_rating', 'adjust_rating']])

    top_items[['parent_asin', 'title', 'main_category', 'average_rating', 'adjust_rating']].to_excel("cohort_rating_retrieval_top50.xlsx", index=False)

def trending_retrieval(df_reviews):
    # Convert timestamp (milliseconds) to datatime
    df_reviews['datatime'] = pd.to_datetime(df_reviews['timestamp'])

    # Filter reviews from 2023 onwards
    df_reviews = df_reviews[df_reviews['datatime'] >= '2023-01-01']  # 14405

    df_reviews = df_reviews.set_index('datatime')
    # Resample actions weekly
    df_weekly_actions = df_reviews.groupby('parent_asin').resample('W').size().reset_index(name='review_count') #resample('W') (weekly aggregation) only works with a DatetimeIndex

    # Compute the week-over-week percentage change for each product
    df_weekly_actions['pct_change'] = df_weekly_actions.groupby('parent_asin')['review_count'].pct_change()
    # why some pct_change have NaN or inf or -1.0

    # Drop NaN and infinite pct_change values
    df_weekly_actions = df_weekly_actions.dropna(subset=['pct_change'])
    df_weekly_actions = df_weekly_actions[df_weekly_actions['pct_change'] != float('inf')]
    df_weekly_actions = df_weekly_actions[df_weekly_actions['pct_change'] != float('-inf')]

    # Calculate the average percentage change across all products (ignoring NaNs)
    avg_pct_change = df_weekly_actions['pct_change'].mean()
    print(f"average percentage change: {avg_pct_change}")

    threshold = avg_pct_change + 0.5
    print(f"Threshold: {threshold}")

    # Filter products with pct_change above threshold in the most recent week of each product (current trending)
    df_lastest_week = df_weekly_actions.groupby('parent_asin').tail(1)
    # why lastest week contains data from 2014, 2016, 2017, 2018, 2019, 2023? should not be only from 2023?

    df_trending_products = df_lastest_week[df_lastest_week['pct_change'] > threshold]
    # df_trending_products is empty
    print(f"Number of products above threshold: {len(df_trending_products)}")

    df_trending_products = df_trending_products.sort_values(by='pct_change', ascending=False)
    
    print("Trending Products:")
    print(df_trending_products[['parent_asin', 'review_count', 'pct_change']].head(50))
    df_trending_products[['parent_asin', 'review_count', 'pct_change']].head(50).to_excel("trending_retrieval_top50.xlsx", index=False)

# retrieve top itmes based on best sell
def best_buy(df_reviews, df_meta):
    df_aggregated_reviews = df_reviews.groupby('parent_asin').agg(
        verified_purchase_count = ('verified_purchase', lambda x: x.sum())
    ).reset_index()

    # Join aggregated reviews with product data
    df_merge = pd.merge(df_aggregated_reviews, df_meta, on='parent_asin')

    print(f"df_merge shape: {df_merge.shape}")

    # Retrieve top 50 items based on most actions
    top_items = df_merge.sort_values('verified_purchase_count', ascending=False).head(50)

    print("Top 50 items based on most actions:")
    print(top_items[['parent_asin', 'title', 'main_category', 'average_rating', 'verified_purchase_count']])

    top_items[['parent_asin', 'title', 'main_category', 'average_rating', 'verified_purchase_count']].to_excel("bestbuy_retrieval_top50.xlsx", index=False)

# trending_retrieval(df_reviews)

cohort_retrieval(df_meta)

# best_buy(df_reviews, df_meta)

# ### Cohort Retrieval
# def cohort_retrieval(reviews_df, cohort_column="user_id"):
#     """Group products into cohorts based on user_id and retrieve top items."""
    
#     cohort_groups = reviews_df.groupby(cohort_column)
#     top_items = {}
    
#     for cohort, group in cohort_groups:
#         # Compute a score: weighted sum of clicks, purchases, etc.
#         group["score"] = 0.5 * group["helpful_vote"] + 0.5 * group["verified_purchase"].astype(int)
#         top_items[cohort] = group.sort_values("score", ascending=False).head(10)
    
#     return top_items

# ### Trending Retrieval
# def trending_retrieval(reviews_df):
#     """Identify trending products based on action rate increase."""
    
#     # Convert timestamp to datetime
#     reviews_df["datetime"] = pd.to_datetime(reviews_df["timestamp"], unit='ms')
    
#     # Resample actions weekly
#     weekly_actions = reviews_df.groupby("asin").resample("W", on="datetime").count().reset_index()
    
#     # Compute percentage change
#     weekly_actions["pct_change"] = weekly_actions.groupby("asin")["user_id"].pct_change()
    
#     # Filter trending products (those with high percentage change)
#     average_change = weekly_actions["pct_change"].mean()
#     trending_products = weekly_actions[weekly_actions["pct_change"] > (average_change + 0.5)]["asin"].unique()
    
#     return trending_products

# ### Best Seller Retrieval
# def best_seller_retrieval(reviews_df):
#     """Retrieve top-selling products based on purchases."""
    
#     purchase_summary = reviews_df.groupby("asin")["verified_purchase"].sum().reset_index()
    
#     # Sort and retrieve top products
#     best_sellers = purchase_summary.sort_values("verified_purchase", ascending=False).head(10)
    
#     return best_sellers

# # Execute all retrieval strategies
# top_cohorts = cohort_retrieval(reviews_df)
# trending_products = trending_retrieval(reviews_df)
# best_sellers = best_seller_retrieval(reviews_df)

# # Print results
# print("Top Cohorts:")
# for cohort, items in list(top_cohorts.items())[:3]:  # Display first 3 cohorts
#     print(f"Cohort {cohort}: {items[['asin', 'score']].to_dict('records')}")

# print("\nTrending Products:", trending_products)
# print("\nBest Sellers:")
# print(best_sellers)
