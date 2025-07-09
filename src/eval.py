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

# --- Config ---
MODEL_DIR = "models/trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_aug16"
TEST_LABELS_PATH = "data/oldNepali/processed/labels_test.json"
OUTPUT_CSV = "results/eval_oldNepali_aug16.csv"
MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cleanup regex for zero-width and whitespace ---
CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

# --- Load model & processor ---
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
processor = TrOCRProcessor(
    image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
    tokenizer=tokenizer
)
model.config.update({
    "decoder_start_token_id": tokenizer.cls_token_id,
    "pad_token_id":           tokenizer.pad_token_id,
    "eos_token_id":           tokenizer.eos_token_id,
    "max_length":             MAX_LENGTH,
    "num_beams":              NUM_BEAMS,
    "early_stopping":         True,
    "no_repeat_ngram_size":   0
})

# --- Load test data ---
with open(TEST_LABELS_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# --- Prediction function ---
def predict_text(image_path):
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- Run evaluation ---
results = []
for sample in tqdm(test_data, desc="Evaluating"):
    page = sample.get("page", "NA") 
    image_path = sample["image_path"]
    gt         = sample["text"]

    try:
        pred = predict_text(image_path)
    except Exception as e:
        pred = ""
        print(f"[!] Error on {image_path}: {e}")

    # Clean both predictions and labels
    gt_clean   = clean_text(gt)
    pred_clean = clean_text(pred)

    # Sample-level CER
    raw_cer     = char_error_rate([pred],      [gt]).item()
    cleaned_cer = char_error_rate([pred_clean],[gt_clean]).item()

    results.append({
        "page":         page,
        "image_path":   image_path,
        "ground_truth": gt,
        "prediction":   pred,
        "raw_cer":      raw_cer,
        "cleaned_cer":  cleaned_cer
    })

# --- Save predictions and CERs ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to: {OUTPUT_CSV}")

# --- Corpus-level CER (cleaned) ---
all_preds_clean = [clean_text(r["prediction"]) for r in results]
all_gts_clean   = [clean_text(r["ground_truth"]) for r in results]
corpus_cer = char_error_rate(all_preds_clean, all_gts_clean).item()

# --- Summary stats ---
summary = df[["raw_cer", "cleaned_cer"]].agg(["mean", "median", "min", "max"])
print("\n📊 Per-sample CER summary:")
print(summary.to_string())

print(f"\n📈 Corpus-level CER (cleaned): {corpus_cer:.4f}")

# --- Optional: Save summary to TXT ---
summary_path = OUTPUT_CSV.replace(".csv", "_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sample-level CER Summary:\n")
    f.write(summary.to_string())
    f.write(f"\n\nCorpus-level CER (cleaned): {corpus_cer:.4f}")
print(f"📝 Summary saved to: {summary_path}")