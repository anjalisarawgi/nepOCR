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

# MODEL_DIR = "models/trained/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_aug16" # previous set
# TEST_LABELS_PATH = "data/oldNepali/processed/normalized_labels/labels_test.json"
# OUTPUT_CSV = "results/oldNepali_aug16.csv"

## on full set (still has the mistakes so it is not a true representation)

MODEL_DIR = "models/trained/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_181_aug8" 
TEST_LABELS_PATH = "data/oldNepali_fullset/labels/labels_test.json"
OUTPUT_CSV = "results/oldNepali_181_aug8.csv"

MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# clleaning up with invisible characters incase they are present
CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

# model and the processor
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


with open(TEST_LABELS_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

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

    # cleaning both ground truth and prediction
    gt_clean   = clean_text(gt)
    pred_clean = clean_text(pred)

    cer     = char_error_rate([pred],      [gt]).item()
    # cleaned_cer = char_error_rate([pred_clean],[gt_clean]).item()

    results.append({
        "page":         page,
        "image_path":   image_path,
        "ground_truth": gt,
        "prediction":   pred,
        "cer":      cer,
        # "cleaned_cer":  cleaned_cer
    })


df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

all_predictions = [clean_text(r["prediction"]) for r in results]
all_groundtruths   = [clean_text(r["ground_truth"]) for r in results]
corpus_cer = char_error_rate(all_predictions, all_groundtruths).item()

summary = df[["cer"]].agg(["mean", "median", "min", "max"])
print(summary.to_string())

# saving
summary_path = OUTPUT_CSV.replace(".csv", "_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sample level CER Summary:")
    f.write(summary.to_string())
    f.write(f"Corpus level CER: {corpus_cer:.4f}")
print(f"Results saved to {OUTPUT_CSV} and summary to {summary_path}")
print(f"Corpus-level CER: {corpus_cer:.4f}")
