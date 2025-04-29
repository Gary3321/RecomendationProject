import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from transformers import AutoTokenizer, AutoModel

import torchvision.transforms as T
from torchvision.models import resnet50
import torch.nn as nn
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

########### Using BLIP2 generating Image embeddigns
# load processor and model
model_path = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype= torch.float16).to(device)

# function to generate cation
def generate_caption(row):
    print(row.name) # prints the row index (row number)
    image = Image.open(row['image_path']).convert("RGB")
    # preprocess image with the processor
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs)
    
    # decode the output to get the final caption
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption 

# Use sentence-transformer to generate embedding
class SentenceTransformerEmbedding:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize with a pretrained transformer model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode(self, text):
        """
        Tokenize the input text and return the mean-pooled embedding

        Returns:
            torch.Tensor: Embedding tensor of shape (1, embed_dim)
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling over the token embeddigns (dim 1 is the sequence length)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings 
    

def GenerateEmb_Caption():
    
    df = pd.read_excel("Data/top30k.xlsx")
    
    # Initialize the sentence embedder
    sentence_embedder = SentenceTransformerEmbedding()

    # List to store embeddings and corresponding image IDs.
    embeddings_list = []
    parent_asins = []

    # Iterate through each row in the DataFrame.
    for index, row in df.iterrows():
        caption = row["caption"]
        parent_asin = row['parent_asin']
        # Generate embedding (output shape: [1, embed_dim])
        embedding = sentence_embedder.encode(caption)

        # Convert tensor to numpy array and squeeze the batch dimension
        embeddings_list.append(embedding.squeeze(0).cpu().numpy())
        parent_asins.append(parent_asin)
        print(index)
    
    # stack all embeddings into a single NumPy array of shape [num_captions, embed_dim]
    embeddings_array = np.stack(embeddings_list)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Save the embeddings and corresponding image IDs to disk.
    # we save as a dictionary using NumPy's savez function
    np.savez("Data/image_caption_embeddings.npz", image_ids=parent_asins, embeddings=embeddings_array)

########## Use CNN (resnet50) generating Image embeddings


def get_image_embedding_cnn():

    # simple resnet backbone
    resnet = resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
    resnet.eval()

    # Image transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])
    
    df = pd.read_excel("Data/top30k.xlsx")
    embeddings_list = []
    parent_asins = []

    for index, row in df.iterrows():
        print(f"index: {index}")
        image_path = row['image_path']
        parent_asin = row['parent_asin']
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            embedding = resnet(img_tensor).squeeze() # shape: [2048]
        
        # Convert tensor to numpy array and squeeze the batch dimension
        embeddings_list.append(embedding.numpy())
        parent_asins.append(parent_asin)
        
    embeddings_array = np.stack(embeddings_list)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Save the embeddings and corresponding image IDs to disk.
    # we save as a dictionary using NumPy's savez function
    np.savez("Data/image_cnn_embeddings.npz", parent_asins=parent_asins, embeddings=embeddings_array)

if __name__ == "__main__":
    get_image_embedding_cnn()

    df = pd.read_excel("Data/top30k.xlsx")
    df['caption'] = df.apply(generate_caption, axis=1)
    df.to_excel("Data/top30k.xlsx", index=False)

    GenerateEmb_Caption()
