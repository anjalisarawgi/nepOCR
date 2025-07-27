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
OUTPUT_CSV = "results/predictions_trocr_large_bert_byteBPE_aug8.csv"

MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

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

    gt_clean = clean_text(gt)
    pred_clean = clean_text(pred)

    cer_orig   = char_error_rate([pred],      [gt]).item()
    cer_clean  = char_error_rate([pred_clean],[gt_clean]).item()

    results.append({
        "page": page,
        "image_path": image_path,
        "ground_truth": gt,
        "prediction": pred,
        "cer": cer_orig,
        "cleaned_cer": cer_clean
    })


# build dataframe
df = pd.DataFrame(results)

# ← NEW: compute a length column on cleaned ground-truth
df["gt_length"] = df["ground_truth"].apply(lambda txt: len(clean_text(txt)))

# ← NEW: compute weighted-errors per line
df["weighted_errors"]   = df["cer"] * df["gt_length"]
df["weighted_errors_clean"] = df["cleaned_cer"] * df["gt_length"]

# ← NEW: overall length-weighted CER
total_chars = df["gt_length"].sum()
total_errs  = df["weighted_errors"].sum()
total_errs_clean = df["weighted_errors_clean"].sum()

weighted_cer       = total_errs  / total_chars
weighted_cer_clean = total_errs_clean / total_chars

# ← NEW: append to summary
summary = df[["cer", "cleaned_cer"]].agg(["mean", "median", "min", "max"])
summary_path = OUTPUT_CSV.replace(".csv", "_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sample-level CER Summary ===\n")
    f.write(summary.to_string())
    f.write(f"\nCorpus-level CER (cleaned): {char_error_rate(df['prediction'].apply(clean_text).tolist(), df['ground_truth'].apply(clean_text).tolist()).item():.4f}\n")
    f.write(f"Overall length-weighted CER (raw):    {weighted_cer:.4f}\n")           # ← NEW
    f.write(f"Overall length-weighted CER (clean):  {weighted_cer_clean:.4f}\n")     # ← NEW

# save full results with weights if needed
df.to_csv(OUTPUT_CSV, index=False)

print(f"Results saved to         {OUTPUT_CSV}")
print(f"Sample summary saved to  {summary_path}")
print(f"Overall length-weighted CER (raw):   {weighted_cer:.4f}")
print(f"Overall length-weighted CER (clean): {weighted_cer_clean:.4f}")