import json 
import pandas as pd 
import torch 
from PIL import Image
from tqdm import tqdm
from datasets import Dataset


def load_dataset(json_path, add_sep_token: bool = False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"] = df["text"].astype(str)
    if add_sep_token:
        df["text"] += " [SEP]"
    df["image_path"] = df["image_path"].astype(str)
    return Dataset.from_pandas(df[["image_path", "text"]])


def preprocess_dataset(dataset, tokenizer, feature_extractor, max_length):
    images, labels = [], []
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        image = Image.open(item["image_path"]).convert("RGB") # rgb only required for trocr
        pixel = feature_extractor(images=image, return_tensors = "pt", size=(384,384)).pixel_values[0] # only required for trocr
        # token_ids = tokenizer.encode(item["text"], add_special_tokens=False) # converting labels to token ids
        # token_ids = token_ids[:max_length] + [tokenizer.eos_token_id] # [SEP]
        # token_ids += [tokenizer.pad_token_id] * (max_length - len(token_ids)) # padding
        token_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if len(token_ids) >= max_length:
            token_ids = token_ids[:max_length-1]  # reserve space for EOS
        token_ids = token_ids + [tokenizer.eos_token_id] 
        token_ids += [tokenizer.pad_token_id] * (max_length - len(token_ids))  # pad to max_length

        labels_tensor = torch.tensor([i if i != tokenizer.pad_token_id else -100 for i in token_ids]) # -100 for padding and convert to tensor

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
    



class OCRLazyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, feature_extractor, max_length=100):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel = self.feature_extractor(images=image, return_tensors="pt", size=(384, 384)).pixel_values[0]

        token_ids = self.tokenizer.encode(item["text"], add_special_tokens=False)
        if len(token_ids) >= self.max_length:
            token_ids = token_ids[:self.max_length - 1]
        token_ids = token_ids + [self.tokenizer.eos_token_id]
        token_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        labels_tensor = torch.tensor([i if i != self.tokenizer.pad_token_id else -100 for i in token_ids])

        return {"pixel_values": pixel, "labels": labels_tensor}

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]), 
        "labels": torch.stack([b["labels"] for b in batch])
    }
