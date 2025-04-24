import json 
import pandas as pd 
import torch 
from PIL import Image
from tqdm import tqdm
from datasets import Dataset

def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["label"] = df["label"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return Dataset.from_pandas(df[["image_path", "label"]])


def get_preprocessing_fn(tokenizer, feature_extractor):
    def process_data(example):
        image = Image.open(example["image_path"]).convert("RGB")
        pixel_values = feature_extractor(image, return_tensors="pt")["pixel_values"][0]

        tokens = tokenizer.encode(example["label"], add_special_tokens=False)
        tokens = tokens[:tokenizer.model_max_length - 1]
        tokens.append(tokenizer.eos_token_id)
        tokens += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(tokens))
        labels = [tok if tok != tokenizer.pad_token_id else -100 for tok in tokens]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }

    return process_data



def preprocess_dataset(dataset, tokenizer, feature_extractor, max_length):
    images, labels = [], []
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        image = Image.open(item["image_path"]).convert("RGB")
        pixel = feature_extractor(images=image, return_tensors = "pt", size=(384,384)).pixel_values[0]

        token_ids = tokenizer.encode(item["label"], add_special_tokens=False)
        token_ids = token_ids[:max_length] + [tokenizer.eos_token_id]
        token_ids += [tokenizer.pad_token_id] * (max_length - len(token_ids))
        labels_tensor = torch.tensor([i if i != tokenizer.pad_token_id else -100 for i in token_ids])

        images.append(pixel)
        labels.append(labels_tensor)

    return images, labels 


class OCRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            "pixel_values": self.images[idx], 
            "labels": self.labels[idx]
        }
    

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]), 
        "labels": torch.stack([b["labels"] for b in batch])
    }



