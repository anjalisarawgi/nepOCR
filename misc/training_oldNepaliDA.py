import os
import torch
from PIL import Image
import wandb
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback
)
from jiwer import wer, cer
from typing import Any, Dict, List
from torchmetrics import CharErrorRate

set_seed(42)
wandb.init(project="oldNepali-ocr-logs", name="swinbert-oldNepaliSynth105kvNoisy-nagari-oldNepali-cbpe-200")

model_path = "models/swinbert-oldNepaliSynth105kvNoisy-nagari-cbpe-200"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

vocab = tokenizer.get_vocab()
safe_tokens = ["[PAD]", "[CLS]", "[SEP]"]

for tok in safe_tokens:
    if tok not in vocab:
        raise ValueError(f"Token {tok} not found in vocab! Please use an existing EOS token like </s>")

tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"
tokenizer.eos_token = "[SEP]"

model.decoder.resize_token_embeddings(len(tokenizer))
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 100
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 5
model.config.num_beams = 5


# from transformers import TrOCRProcessor
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# feature_extractor = processor.feature_extractor  
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")


# dataset = load_dataset("json", data_files={"train": "oldNepaliDataCombinedAugmented/train/labels_processed.json"})
# new_dataset = dataset["train"]

import json
import pandas as pd
from datasets import Dataset

def load_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # df["text"] = df["text"].astype(str)
    df["text"] = df["text"].astype(str) + " [SEP]"
    return Dataset.from_pandas(df[["image_path", "text"]])



train_dataset = load_from_json("oldNepaliDataCombinedAugmented_3/train/labels.json")
test_dataset = load_from_json("oldNepaliDataCombinedAugmented_3/test/labels_processed_new.json")
val_dataset = test_dataset
eval_dataset = val_dataset

def process_data(example):
    
    image_path = example["image_path"]
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt",  size=(384, 384)).pixel_values[0]
    text_with_eos = example["text"] + " " + tokenizer.eos_token

    tokenized = tokenizer(
        text_with_eos,
        padding="max_length",
        max_length=100, 
        return_tensors="pt",
        truncation=True
    )

    labels = tokenized.input_ids[0]
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]

    example["pixel_values"] = pixel_values
    example["labels"] = torch.tensor(labels)

#     print("🔍 [DEBUG] Label decode check")
#     print("Text+EOS:", text_with_eos)
#     decoded_label = tokenizer.decode(
#     [token if token != -100 else tokenizer.pad_token_id for token in labels],
#     skip_special_tokens=False
# )
#     print("Decoded label:", decoded_label)
#     print("[SEP] token ID:", tokenizer.eos_token_id)
#     print("Label IDs:", labels)
#     print("-" * 60)
    return example

# new_dataset = new_dataset.map(process_data, remove_columns=new_dataset.column_names, num_proc=2 )
# new_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# split_dataset = new_dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset = split_dataset["train"]
# eval_dataset = split_dataset["test"]


train_dataset = train_dataset.map(process_data, remove_columns=train_dataset.column_names)
train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

eval_dataset = eval_dataset.map(process_data, remove_columns=eval_dataset.column_names)
eval_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

test_dataset = test_dataset.map(process_data, remove_columns=test_dataset.column_names)
test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

val_dataset = test_dataset


class ImageToTextCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = [feature["labels"] for feature in features]
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )["input_ids"]
        return {
            "pixel_values": pixel_values,
            "labels": padded_labels,
        }

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    print("\n--- EOS token tracking ---")
    for pred, label in zip(pred_str[:3], label_str[:3]):
        print(f"Pred:  {pred}")
        print(f"Label: {label}")
        print(f"[SEP] in pred? {'[SEP]' in pred}")
    print("--------------------------\n")

    metric = CharErrorRate()
    cer = metric(pred_str, label_str)

    print("🔁 [DEBUG] Prediction tracking:")
    for pred, label in zip(pred_str[:3], label_str[:3]):
        print(f"Pred:  {pred}")
        print(f"Label: {label}")
        print(f"[SEP] in pred? {'[SEP]' in pred}")
        print(f"Ends with [SEP]? {pred.strip().endswith('[SEP]')}")
        print("-" * 40)
    
    return {"cer": cer}


# not sure how this works -- chatgpt 
class PrintPredictionsCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, num_samples=5, print_every=100):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            print(f"\n--- Predictions at step {state.global_step} ---")
            samples = [self.eval_dataset[i] for i in range(self.num_samples)]
            pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
            
            true_labels_ids = []
            for sample in samples:
                labels = sample["labels"]
                labels_fixed = [
                    token if token != -100 else self.tokenizer.pad_token_id 
                    for token in labels.tolist()
                ]
                true_labels_ids.append(labels_fixed)
            
            model = kwargs.get("model", None)
            if model is None:
                model = kwargs["trainer"].model
            device = model.device
            pixel_values = pixel_values.to(device)
            
            model.eval()  
            with torch.no_grad():
                # generated_ids = model.generate(pixel_values)
                generated_ids = model.generate(
                    pixel_values,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=5,
                    # repetition_penalty=2.0,
                    # length_penalty=2.0
                )
            model.train()  # switch back to training mode
            
            generated_ids_list = generated_ids.cpu().tolist()
            predictions = self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True)
            true_texts = self.tokenizer.batch_decode(true_labels_ids, skip_special_tokens=True)
            for idx, (pred, true_text) in enumerate(zip(predictions, true_texts)):
                print(f"Image {idx+1}:")
                print(f"  Prediction   : {pred}")
                print(f"  Ground Truth : {true_text}")
            print("\n")

        return control
    
print_callback = PrintPredictionsCallback(
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    num_samples=5,
    print_every=100
)

data_collator = ImageToTextCollator(tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/swinbert-oldNepaliSynth105kvNoisy-nagari-oldNepali-cbpe-200",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="no",
    # save_steps=5000,
    logging_steps=100,
    warmup_steps=500,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    # save_total_limit=2,
    report_to=["wandb"],
    max_grad_norm=0.5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[print_callback]
)

trainer.train()
trainer.save_model("models/swinbert-oldNepaliSynth105kvNoisy-nagari-oldNepali-cbpe-200")
tokenizer.save_pretrained("models/swinbert-oldNepaliSynth105kvNoisy-nagari-oldNepali-cbpe-200")

wandb.finish()