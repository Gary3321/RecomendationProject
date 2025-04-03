import pandas as pd
from sklearn.model_selection import train_test_split
import os 

from PIL import Image
from torch.utils.data import Dataset
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

from transformers import TrainingArguments, Trainer

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor, image_dir = "Data/flickr30k-images/"):
        self.dataset = dataset
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Construct full image path
        image_path = os.path.join(self.image_dir, self.dataset[idx]["image"])
        caption = self.dataset[idx]["caption"]

        # Open image
        image = Image.open(image_path).convert("RGB")
        # Tokenize image & text
        encoding = self.processor(images = image, text=caption, return_tensors = "pt",
                                  padding="max_length", truncation=True, max_length=50)
        
        # Move to tensor format
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Add Labels for loss computation
        encoding["labels"] = encoding["input_ids"]
        return encoding




dataset_path = "Data/image30k_captions.txt"

def SaveDataCsv(dataset_path):
    # Directionary to store image-caption pairs
    image_caption_dict = {}

    # Read the text file
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1) # Split into image name and caption, based on the 1st space
            if len(parts) < 2:
                continue # Skip any malformed lines

            image_name, caption = parts
            image_name = image_name.split("#")[0]

            # Store captions in a list for each image
            if image_name not in image_caption_dict:
                image_caption_dict[image_name] = []
            
            image_caption_dict[image_name].append(caption)
        
    # # Print sample output
    # for img, captions in list(image_caption_dict.items())[:2]: # Show first 2 images
    #     print(f"Image: {img}")
    #     for i, caption in enumerate(captions):
    #         print(f"  Caption {i}: {caption}")

    # Convert dictionary to a Dataframe
    data = []
    for image, captions in image_caption_dict.items():
        for caption in captions:
            data.append({"image": image, "caption": caption})

    df = pd.DataFrame(data)

    # Save as CSV
    df.to_csv("image_captions.csv", index = False)
    print(df.head())

def SplitTrainTest(df):

    # Get unique image filenames
    unique_images = df["image"].unique()

    # Split images into train and test sets
    train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state = 42)

    # Select rows where the image is in the train / test set
    train_df = df[df["image"].isin(train_images)]
    test_df = df[df["image"].isin(test_images)]

    unique_images = test_df["image"].unique()
    eval_images, test_iamges = train_test_split(unique_images, test_size=0.2, random_state = 42)
    eval_df = test_df[test_df["image"].isin(eval_images)]
    test_df = test_df[test_df["image"].isin(test_iamges)]
    
    #Save split datasets
    train_df.to_csv("train_sklearn.csv", index=False)
    eval_df.to_csv("eval_sklearn1.csv", index=False)
    test_df.to_csv("test_sklearn1.csv", index=False)

    print(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}, Test size: {len(test_df)}")

if __name__ == "__main__":
    df = pd.read_csv("image_captions.csv")
    #SplitTrainTest(df=df)

    # Load Dataset
    # train_dataset = load_dataset("csv", data_files={"train": "train_sklearn.csv"})["train"]
    # eval_dataset = load_dataset("csv", data_files={"eval": "eval_sklearn.csv"})["eval"]
    train_dataset = load_dataset("csv", data_files={"train": "train_sklearn3101.csv"})["train"]
    eval_dataset = load_dataset("csv", data_files={"eval": "test_sklearn1241.csv"})["eval"]

    print(train_dataset[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Create dataset
    train_data = ImageCaptionDataset(train_dataset, processor)

    eval_data = ImageCaptionDataset(eval_dataset, processor) 
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./blip-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        eval_strategy="epoch", # Evaluate after every epoch
        save_strategy="epoch",
        #logging_strategy = "epoch",
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=True, # Use mixed precision if available
        save_total_limit=2
    )

    # Define Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    # Train the model
    trainer.train()

    # Save the Model
    model.save_pretrained("./blip-finetuned1")
    processor.save_pretrained("./blip-finetuned1")

################################

    train_dataset = load_dataset("csv", data_files={"train": "train_sklearn3217.csv"})["train"]
    eval_dataset = load_dataset("csv", data_files={"eval": "test_sklearn1241.csv"})["eval"]

    print(train_dataset[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Create dataset
    train_data = ImageCaptionDataset(train_dataset, processor)

    eval_data = ImageCaptionDataset(eval_dataset, processor) 
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./blip-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        eval_strategy="epoch", # Evaluate after every epoch
        save_strategy="epoch",
        #logging_strategy = "epoch",
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=True, # Use mixed precision if available
        save_total_limit=2
    )
 

    # Define Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    # Train the model
    trainer.train()

    # Save the Model
    model.save_pretrained("./blip-finetuned2")
    processor.save_pretrained("./blip-finetuned2")