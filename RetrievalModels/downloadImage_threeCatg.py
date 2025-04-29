import requests
import pandas as pd
import os
import ast

# Directories for saving images
IMAGE_SAVE_DIR = "Data/images"

image_process_counter = 0

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# Function to download the 'large' image from the images field
def download_large_image(images_info, parent_asin):
    """
    Download the 'large' image for a give product
    """

    #df.images[0]: [{'thumb': 'https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg', 'large': 'https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg', 'variant': 'MAIN', 'hi_res': None}, {'thumb': 'https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg', 'large': 'https://m.media-amazon.com/images/I/41w2yznfuZL.jpg', 'variant': 'PT01', 'hi_res': 'https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg'}]
    # only download the first image for each project because other images may contain a set of products
    global image_process_counter
    image_process_counter += 1
    print(f"Processed image {image_process_counter}")
    file_path = os.path.join(IMAGE_SAVE_DIR, f"{parent_asin}.jpg")
    if os.path.exists(file_path):
        return file_path 

    
    images_info = ast.literal_eval(images_info)
    for img in images_info:
        if 'large' in img:
            url = img['large']
            try:
                response = requests.get(url, timeout=100)
                if response.status_code == 200:
                    
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    return file_path 
            except Exception as e:
                print(f"Errof downloading image for {parent_asin}: {e}")
    return None



def download_images_with_process(row):
    row_idx = row.name # row index
    if row_idx % 100 == 0:
        print(f"Processing row {row_idx + 1} of {total_rows} ...")
    return download_large_image(row['images'], row['parent_asin'])

# products_file = "Data/meta_All_Beauty.jsonl"

# # Load datasets
# df_meta = pd.read_json(products_file, lines=True)

df_meta = pd.read_excel("Data/top30k.xlsx")
total_rows = len(df_meta)

#df_meta['image_path'] = df_meta.apply(lambda row: download_large_image(row['images'], row['parent_asin']), axis=1)

df_meta['image_path'] = df_meta.apply(download_images_with_process, axis=1)

df_meta = df_meta[df_meta['image_path'].notna()]
df_meta.to_excel("Data/top30k.xlsx", index=False)
print("Image downlowd finished")