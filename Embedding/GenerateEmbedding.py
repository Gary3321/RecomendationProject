import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy

def generate_caption(model_path, device):
    df1 = pd.read_csv("Data/test_sklearn1241.csv")
    df2 = pd.read_csv("Data/train_sklearn3217.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)

    for index, row in df.iterrows():
        print(index)
        image = Image.open(f"../Data/flickr30k-images/{row['image']}").convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        generated_caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        df.loc[index, "generated_caption"] = generated_caption
    
    df.to_csv("GeneratedCaptions.csv", index=False)

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
        Tokenize the input text and return the mean-pooled embedding.

        Parameters:
            text (str): Input sentence/caption
        Returns:
            torch.Tensor: Embedding tensor of shape (1, embed_dim)
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        

        # Mean pooling over the token embeddings (dim 1 is the sequence length)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        """
        When you pass input text through the model, its output—specifically outputs.last_hidden_state—is a tensor with three dimensions:

        Dimension 0: The batch size (number of input sentences or captions).

        Dimension 1: The sequence length (number of tokens in each sentence).

        Dimension 2: The hidden size (embedding dimension for each token).

        For example, if you have a batch of 2 sentences where each sentence is tokenized into 10 tokens, and the hidden size is 768, 
        the shape of outputs.last_hidden_state would be [2, 10, 768].

        embeddings = outputs.last_hidden_state.mean(dim=1) computes the mean of the token embeddings along dimension 1 (the sequence length). 
        In other words, it averages the embeddings of all tokens in each sentence to produce a single fixed-size vector per sentence. 
        This results in a tensor of shape [batch_size, hidden_size].
        By stating "dim 1 is the sequence length," the comment explains that dimension 1 of the tensor corresponds to the number of tokens 
        in the input sequence, and that's the dimension over which the mean pooling is performed.
        """


        return embeddings 


# ---------------------------
# Method 2: Word-Level Embedding with NER Extraction
# ---------------------------


class WordLevelEmbeddingWithNER:
    def __init__(self, vocab_size=1250, embedding_dim=200):
        """
        Create an embedding layer and initialize spaCy's small English model for NER.
        """
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # build a vocabulary 
        self.word_to_idx = {}

        # Load spaCy's small English model (make sure to run: python -m spacy download en_core_web_sm)
        self.nlp = spacy.load("en_core_web_sm")

    def get_word_indices(self, words):
        """
        Convert a list of words into indices.
        """
        indices = []
        for word in words:
            key = word.lower()
            if key not in self.word_to_idx:
                self.word_to_idx[key] = len(self.word_to_idx) % self.vocab_size
            indices.append(self.word_to_idx[key])
        return torch.tensor(indices)
    
    def encode(self, text):
        """
        Process the text with spaCy to extract named entities.
        """
        doc = self.nlp(text)

        # Extract named entities 
        entities = [ent.text for ent in doc.ents]
        # Nouns, Proper Nouns, Verbs, Adjectives
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']]
        entities.extend(keywords)

        if not entities:
            # split on Whitespace
            entities = text.split()
        
        indices = self.get_word_indices(entities)
        token_embeddings = self.embedding(indices)
        # Aggregate embeddings (using mean pooling) (sum embeddings / total keywords)
        #To remove grad_fn, use .detach(), with torch.no_grad(), or ensure that token_embeddings.requires_grad=False.
        final_embedding = token_embeddings.mean(dim=0, keepdim=True).detach() 
        return final_embedding

def DecideVocbSize():
    import pandas as pd
    from collections import Counter 
    import nltk
    from nltk.tokenize import word_tokenize
    # Ensure the NLTK tokenizer data is downloaded (if not already)
    nltk.download('punkt_tab')

    df = pd.read_csv("C:/Users/GaryBao/Documents/Work/ML/Recommend/Embedding/GeneratedCaptions_unique.csv")
    
    # Ensure that there are no missing values and convert to list
    captions = df['generated_caption'].dropna().tolist()

    # Initialize a counter to track word frequencies
    word_counter = Counter()

    # Tokenize each caption and update the word counter
    for caption in captions:
        # Convert to Lowercase for consistency and tokenize
        tokens = word_tokenize(caption.lower())
        word_counter.update(tokens)
    
    # The total vocabulary size (all unique tokens)
    total_vocab_size = len(word_counter)

    # Optionally, filter out words that appear less than a threshold (e.g. frequence < 2)
    min_freq = 2
    filtered_vocab = {word: count for word, count in word_counter.items() if count >=min_freq}
    filtered_vocab_size = len(filtered_vocab)

    print(f"Total vocabulary size (all unique tokens): {total_vocab_size}")
    print(f"Filtered vocabulary size (frequence >= {min_freq}): {filtered_vocab_size}")


def GenerateEmb_wordLevel():
    
    df = pd.read_csv("C:/Users/GaryBao/Documents/Work/ML/Recommend/Embedding/GeneratedCaptions_unique.csv")
    #sample_caption = "A man riding a horse in a beautiful sunset over the mountains."
    
    # Initialize the sentence embedder
    word_ner_embedder = WordLevelEmbeddingWithNER()
    

    # List to store embeddings and corresponding image IDs.
    embeddings_list = []
    image_ids = []

    # Iterate through each row in the DataFrame.
    for index, row in df.iterrows():
        caption = row["generated_caption"]
        image_id = row["image"]
        # Generate embedding (output shape: [1, embed_dim])
        embedding = word_ner_embedder.encode(caption)

        # Convert tensor to numpy array and squeeze the batch dimension
        embeddings_list.append(embedding.squeeze(0).numpy())

        # squeeze(0) removes the batch dimension if it's of size 1.
        # the  embedding is of shape [1, embedding_dim] because of token_embeddings.mean(dim=0, keepdim=True)
        # After squeeze(0), it becomes [embedding_dim], which is a standard vector format
        # .numpy() converts the tensor from a PyTorch tensor to a NumPy array

        image_ids.append(image_id)
        print(index)
    
    # stack all embeddings into a single NumPy array of shape [num_captions, embed_dim]
    embeddings_array = np.stack(embeddings_list)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Save the embeddings and corresponding image IDs to disk.
    # we save as a dictionary using NumPy's savez function
    np.savez("C:/Users/GaryBao/Documents/Work/ML/Recommend/Embedding/caption_embeddings_wordLevel.npz", image_ids=image_ids, embeddings=embeddings_array)



def GenerateEmb():
    
    df = pd.read_csv("Embedding/GeneratedCaptions_unique.csv")
    #sample_caption = "A man riding a horse in a beautiful sunset over the mountains."
    
    # Initialize the sentence embedder
    sentence_embedder = SentenceTransformerEmbedding()

    # List to store embeddings and corresponding image IDs.
    embeddings_list = []
    image_ids = []

    # Iterate through each row in the DataFrame.
    for index, row in df.iterrows():
        caption = row["generated_caption"]
        image_id = row["image"]
        # Generate embedding (output shape: [1, embed_dim])
        embedding = sentence_embedder.encode(caption)

        # Convert tensor to numpy array and squeeze the batch dimension
        embeddings_list.append(embedding.squeeze(0).cpu().numpy())
        image_ids.append(image_id)
        print(index)
    
    # stack all embeddings into a single NumPy array of shape [num_captions, embed_dim]
    embeddings_array = np.stack(embeddings_list)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Save the embeddings and corresponding image IDs to disk.
    # we save as a dictionary using NumPy's savez function
    np.savez("Embedding/caption_embeddings.npz", image_ids=image_ids, embeddings=embeddings_array)

def Open_npz_file(file):
    # Load the .npz file
    data = np.load(file)

    # List all the keys (arrays) inside the file
    print(f"Available keys: {data.files}")

    # Access data using keys
    image_ids = data['image_ids']
    embeddings = data['embeddings']

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings[0].shape}")
    print(f"First Image ID: {image_ids[0]}")
    print(f"First Embedding: {embeddings[0]}")
    return embeddings

def pca_visualize(embedding):
    pca = PCA(n_components=2) # we want to reduce the embeddings to 2 principal components (i.e., from 384 dimensions to 2 dimensions).
    reduced_embeddings = pca.fit_transform(embedding) # Projects the embeddings onto the 2D space defined by those principal components.

    # After applying fit_transform(), Each of the 4448 embeddings is now represented by only 2 values (the coordinates in the reduced 2D space)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10) # s--> size
    plt.title("PCA Visualization of Image Caption Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    """
    Let’s say embedding contains 4448 sentence embeddings, each with 384 values representing the sentence meaning. After applying PCA:

    If two sentences are semantically similar, their 2D coordinates will be close.

    If they are different, their coordinates will be farther apart.

    You can plot the result using plt.scatter() for easy visualization.
    """

def cosin_similarity_eva_wordLevel():
    from sklearn.metrics.pairwise import cosine_similarity

    data = np.load("C:/Users/GaryBao/Documents/Work/ML/Recommend/Embedding/caption_embeddings_wordLevel.npz")
    image_ids = data['image_ids']
    embeddings = data['embeddings']
    #Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings)  # calculates the cosine similarity between every pair of embeddings, shape (4448, 4448)

    # similarity between the first and second caption
    print(f"Cosine similarity (caption 0 vs caption 1): {similarity_matrix[0, 1]}")

    # get top 5 most similar captions for caption 0
    # np.argsort() returns the indices of the similarity scores in ascending order (from the least similar to the most similar)
    # [::-1] This reverses the array to get the indices in descending order
    top5_similar = np.argsort(similarity_matrix[0])[::-1][1:6]
    print(f"Top 5 Similar scores to Caption 0: {similarity_matrix[0][top5_similar]}")
    print(f"Top 5 Similar Captions to Caption 0: {top5_similar}")
    print(f"Top 5 Similar images to Caption 0: {image_ids[top5_similar]}")

def cosin_similarity_eva():
    from sklearn.metrics.pairwise import cosine_similarity

    data = np.load("Embedding/caption_embeddings.npz")
    image_ids = data['image_ids']
    embeddings = data['embeddings']
    #Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings)  # calculates the cosine similarity between every pair of embeddings, shape (4448, 4448)

    # similarity between the first and second caption
    print(f"Cosine similarity (caption 0 vs caption 1): {similarity_matrix[0, 1]}")

    # get top 5 most similar captions for caption 0
    # np.argsort() returns the indices of the similarity scores in ascending order (from the least similar to the most similar)
    # [::-1] This reverses the array to get the indices in descending order
    top5_similar = np.argsort(similarity_matrix[0])[::-1][1:6]
    print(f"Top 5 Similar scores to Caption 0: {similarity_matrix[0][top5_similar]}")
    print(f"Top 5 Similar Captions to Caption 0: {top5_similar}")
    print(f"Top 5 Similar images to Caption 0: {image_ids[top5_similar]}")
    # # get top 10 most similar images
    # num_images = similarity_matrix.shape[0]
    # top_similar_images = []

    # for i in range(num_images):
    #     # get similarity scores for image i
    #     similarities = similarity_matrix[i]

    #     # Sort in descending order and exlude self (index 0 is the most similar which is itself)
    #     top_indices = np.argsort(similarities)[::-1][1:10+1]

    #     # Store results as (image_index, top_similar_images)
    #     top_similar_images.append((i, top_indices))
    #     break
    # print(top_similar_images)

def tsne_eva(embeddings):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE


    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)



    # Plot the embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10)
    plt.title("t-SNE Visualization of Image Caption Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def kmean_eva(embeddings):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=7, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Evaluate clustering using Silhouette Score (higher is better)
    score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {score}")

    # Visualize clusters using PCA for simplicity
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=10)
    plt.title("PCA Visualization of Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster Label")
    plt.show()


if __name__ == "__main__":
    # model_path = f"../blip-finetuned2"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # generate_caption(model_path, device)
    # df = pd.read_csv("Embedding/GeneratedCaptions.csv")
    # df = df[["image", "generated_caption"]].drop_duplicates()
    # df.to_csv("Embedding/GeneratedCaptions_unique.csv", index=False)
    #GenerateEmb()
    # file = "Embedding/caption_embeddings.npz"
    # embedding = Open_npz_file(file)
    # #pca_visualize(embedding)
    # #cosin_similarity_eva()
    # #tsne_eva(embeddings= embedding)
    # kmean_eva(embeddings=embedding)
    # DecideVocbSize()

    # GenerateEmb_wordLevel()

    file_wordLevel = "C:/Users/GaryBao/Documents/Work/ML/Recommend/Embedding/caption_embeddings_wordLevel.npz"
    embedding = Open_npz_file(file_wordLevel)
    #pca_visualize(embedding)
    # cosin_similarity_eva_wordLevel()
    #tsne_eva(embeddings= embedding)
    kmean_eva(embeddings=embedding)
