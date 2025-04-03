import torch
from PIL import Image 
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import nltk 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from evaluate import load
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import spacy 

"""
sentence_bleu() expects lists of tokens, not raw strings.

RougeScorer automatically tokenizes internally.

meteor_score() requires tokenized input, so we must use .split().

cider.compute_score() expects full sentences, and internally tokenizes.

spice.compute() also expects full sentences and handles tokenization internally.
"""

# Ensure nltk and spacy models are downloaded
#nltk.download("wordnet")
#spacy.cli.download("en_core_web_sm") # what is cli ?

# load fine-tuned BLIP model
model_path = "blip-finetuned2"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)

# BLEU Score
def compute_bleu(reference, prediction):
    # Smoothes to avoid zero n-gram scores
    smooth = SmoothingFunction().method4
    #return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    return corpus_bleu([[ref.split() for ref in reference]], [prediction.split()], smoothing_function=smooth)

# ROUGE Score
def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    #scores = scorer.score(reference, prediction)
    #return scores["rougeL"].fmeasure  # ROUGE-L F1 score
    scores = [scorer.score(ref, prediction)["rougeL"].fmeasure for ref in reference]   
    return sum(scores) / len(scores)  # Average ROUGE-L across all references

# METEOR Score
def compute_meteor(reference, prediction):
    #return meteor_score([reference.split()], prediction.split())
    return sum(meteor_score([ref.split()], prediction.split()) for ref in reference) / len(reference)  # Average METEOR

# CIDEr Score
cider = Cider()
def compute_cider(references, predictions):
    # Format input as dictionaries with image IDs as keys
    # Ensure references are in list format
 
    if isinstance(references, str):  
        references = [references]  # Convert a single reference into a list
    gts = {0: references}  # Ground truth (gts) -> Multiple references per image
    res = {0: [predictions]}  # Predictions (res) -> Single generated caption per image
    score, _ = cider.compute_score(gts, res)
    return score

# SPICE Score
spice = Spice()
def compute_spice(reference, prediction):
    gts = {0: [reference]}  # Ground truth (gts) -> Multiple references per image
    res = {0: [prediction]}  # Predictions (res) -> Single generated caption per image
    score, _ = spice.compute_score(gts, res)
    return score

# load test dataset
df = pd.read_csv("testset_forEvaluation1334.csv")

# Evaluate model on test dataset
# Group by image to get all 5 captions per image
grouped_df = df.groupby("image")["caption"].apply(list).reset_index()

all_bleu, all_rouge, all_meteor, all_cider, all_spice = [], [], [], [], []

for _, row in grouped_df.iterrows():
    image = Image.open(f"Data/flickr30k-images/{row['image']}").convert("RGB")
    reference_caption = row["caption"]

    # Generate caption using BLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    generated_caption = processor.batch_decode(output, skip_special_tokens=True)[0]

    # Compute metrics
    bleu = compute_bleu(reference_caption, generated_caption)
    rouge = compute_rouge(reference_caption, generated_caption)
    meteor = compute_meteor(reference_caption, generated_caption)
    #cider = compute_cider(reference_caption, generated_caption)
    # SPICE requires the Java Runtime Environment (JRE) to be installed 
    # # and available in the system's environment variables 
    #spice = compute_spice(reference_caption, generated_caption)

    all_bleu.append(bleu)
    all_rouge.append(rouge)
    all_meteor.append(meteor)
    #all_cider.append(cider)
    #all_spice.append(spice)

# Compute average scores
print(f"Avg BLEU: {sum(all_bleu) / len(all_bleu):.4f}")
print(f"Avg ROUGE-L: {sum(all_rouge) / len(all_rouge):.4f}")
print(f"Avg METEOR: {sum(all_meteor) / len(all_meteor):.4f}")
#print(f"Avg CIDEr: {sum(all_cider) / len(all_cider):.4f}")
#print(f"Avg SPICE: {sum(all_spice) / len(all_spice):.4f}")

