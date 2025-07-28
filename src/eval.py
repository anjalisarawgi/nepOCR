import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
from torchmetrics.functional.text import char_error_rate
import unicodedata

MODEL_DIR = "models/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_fullset_aug8"
TEST_LABELS_PATH = "data/oldNepali_fullset/labels_normalized_final/labels_test.json"
OUTPUT_CSV = "results/trocr_base_bert_byteBPE/predictions.csv"

MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
# def clean_text(text: str) -> str:
#     text = unicodedata.normalize("NFC", text)
#     text = CLEANUP.sub("", text)
#     return re.sub(r"\s+", "", text)

# model and processor
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
processor = TrOCRProcessor(
    image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-large-handwritten"),
    tokenizer=tokenizer
)
model.config.update({
    "decoder_start_token_id": tokenizer.cls_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_length": MAX_LENGTH,
    "num_beams": NUM_BEAMS,
    "early_stopping": True,
    "no_repeat_ngram_size": 0
})

with open(TEST_LABELS_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

def predict_text(image_path):
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


results = []
for sample in tqdm(test_data, desc="Evaluating"):
    page = sample.get("page", "NA") 
    image_path = sample["image_path"]
    gt = sample["text"]

    try:
        pred = predict_text(image_path)
    except Exception as e:
        print(f"[!] Error on {image_path}: {e}")
        pred = ""

    # gt = clean_text(gt)
    # pred = clean_text(pred)

    cer  = char_error_rate([pred],[gt]).item()

    results.append({
        "page": page,
        "image_path": image_path,
        "ground_truth": gt,
        "prediction": pred,
        "cer": cer
    })


df = pd.DataFrame(results)

# weighted CER
df["gt_length"] = df["ground_truth"].apply(lambda txt: len(txt)) # length of ground truth text
df["weighted_errors"] = df["cer"] * df["gt_length"]
total_chars = df["gt_length"].sum()
total_errs = df["weighted_errors"].sum()
weighted_cer = total_errs/total_chars

df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to{OUTPUT_CSV}")
print(f"Overall length-weighted CER:   {weighted_cer:.4f}") 