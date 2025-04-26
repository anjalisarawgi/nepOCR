import os
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    SwinModel,
    BertLMHeadModel,
    BertConfig,
    set_seed, 
    TrOCRProcessor
)
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer
import pandas as pd
import json
import wandb
from torchmetrics import CharErrorRate
from tqdm import tqdm

set_seed(42)
wandb.init(project="nepOCR-logs", name="trocr-BERT-oldNepaliSynth-cbpe-1000")

def load_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"] = df["text"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return Dataset.from_pandas(df[["image_path", "text"]])

json_path = "data/oldNepaliSynthetic/10k/labels_processed_new.json"
full_dataset = load_from_json(json_path)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]



### Char BPE 
tokenizer_obj = CharBPETokenizer()
with open("corpus/oldNepaliSynthetic_nagari_oldNepali.txt", "r", encoding="utf-8") as f:
    tokenizer_obj.train_from_iterator(f, vocab_size=1000)

tokenizer_dir = "tokenizer/charBPE/trocr_BERT_oldNepaliSynth_1000/"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer_obj.save(os.path.join(tokenizer_dir, "tokenizer.json"))


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
    pad_token="[PAD]",
    cls_token="[CLS]",
    eos_token="[SEP]",
    unk_token="[UNK]"
)
tokenizer.model_max_length = 100

trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
encoder = trocr_model.encoder 

decoder_config = BertConfig(
    is_decoder=True,
    add_cross_attention=True,
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
decoder = BertLMHeadModel(decoder_config)

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 100
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 5
model.config.num_beams = 5


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
feature_extractor = processor.feature_extractor  

# Preprocess all image-label pairs before training
def preprocess_dataset(dataset, tokenizer, feature_extractor):
    images, labels = [], []
    for item in tqdm(dataset, desc="Preprocessing dataset"):
        image = Image.open(item["image_path"]).convert("RGB")
        pixel = feature_extractor(images=image, return_tensors="pt", size=(384, 384)).pixel_values[0]

        tokenized = tokenizer.encode(item["text"], add_special_tokens=False)
        tokenized.append(tokenizer.eos_token_id)
        tokenized = tokenized[:100] + [tokenizer.pad_token_id] * (100 - len(tokenized))
        label_tensor = torch.tensor([t if t != tokenizer.pad_token_id else -100 for t in tokenized])

        images.append(pixel)
        labels.append(label_tensor)
    return images, labels

train_images, train_labels = preprocess_dataset(train_dataset, tokenizer, feature_extractor)
eval_images, eval_labels = preprocess_dataset(eval_dataset, tokenizer, feature_extractor)

# Torch Dataset
class OCRTorchDataset(TorchDataset):
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

train_ds = OCRTorchDataset(train_images, train_labels)
eval_ds = OCRTorchDataset(eval_images, eval_labels)

# DataLoader collate function
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

class PrintPredictionsCallback(TrainerCallback):
    def __init__(self, sample_batch, tokenizer, print_every=100):
        self.samples = sample_batch
        self.tokenizer = tokenizer
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            print(f"\n--- Predictions at step {state.global_step} ---")
            pixel_values = torch.stack([s["pixel_values"] for s in self.samples])
            labels = [s["labels"] for s in self.samples]

            true_labels_ids = [
                [token if token != -100 else self.tokenizer.pad_token_id for token in label.tolist()]
                for label in labels
            ]

            model = kwargs.get("model") or kwargs["trainer"].model
            device = model.device
            pixel_values = pixel_values.to(device)

            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            model.train()

            predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            true_texts = self.tokenizer.batch_decode(true_labels_ids, skip_special_tokens=True)

            for idx, (pred, true_text) in enumerate(zip(predictions, true_texts)):
                print(f"Image {idx+1}:")
                print(f"  Prediction   : {pred}")
                print(f"  Ground Truth : {true_text}")
            print("\n")

        return control
    
sample_batch = [eval_ds[i] for i in range(5)]  # just 5 examples
print_callback = PrintPredictionsCallback(
    sample_batch=sample_batch,
    tokenizer=tokenizer,
    print_every=100
)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="models/trocr-BERT-oldNepaliSynth-cbpe-1000",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    warmup_steps=500,
    num_train_epochs=20,
    learning_rate=1e-4,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to=["wandb"],
    max_grad_norm=0.5,
    save_strategy="no",
    gradient_accumulation_steps=2 if torch.cuda.is_available() else 1,
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)

# Metric
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = CharErrorRate()(pred_str, label_str)
    return {"cer": cer}

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[print_callback]
)

# Train and save
trainer.train()
trainer.save_model("models/trocr-BERT-oldNepaliSynth-cbpe-1000")
tokenizer.save_pretrained("models/trocr-BERT-oldNepaliSynth-cbpe-1000")

# Final eval
test_results = trainer.evaluate(eval_ds)
print("Test results:", test_results)

wandb.finish()