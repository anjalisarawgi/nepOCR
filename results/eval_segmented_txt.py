import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import re

# === CONFIG ===
BASE_MODEL = "google/byt5-base"
MODEL_DIR  = "post_processing/byt5_ocr_finetuned_noCorruption"
MAX_LEN    = 512
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILE = "corpus/oldNepali_test_ground_truth.txt"
OUTPUT_CSV = "post_processing/oldNepali_test_ground_truth_segmented.csv"
FILE_PATH_BASE = "data/images/oldNepali_test_ground_truth/line_{:05d}.png"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)

# === Byte decoding helper ===
offset = getattr(tokenizer, "offset", None)
def byt5_decode(id_list):
    byte_vals = [i - offset for i in id_list if i >= offset]
    return bytes(byte_vals).decode("utf-8", errors="ignore")

# === Cleaning function ===
def clean_line(text: str) -> str:
    text = text.replace(".", " ").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()

# === Load lines ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# === Inference loop ===
results = []

for i, line in enumerate(tqdm(lines, desc="Correcting")):
    file_path = FILE_PATH_BASE.format(i + 1)
    cleaned = clean_line(line)
    no_space = cleaned.replace(" ", "")

    batch = tokenizer([no_space], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(
            **batch,
            max_length=MAX_LEN,
            num_beams=4,
            early_stopping=True
        )
    corrected = byt5_decode(gen_ids[0].cpu().tolist())
    results.append((file_path, line, cleaned, corrected))

# === Save to CSV ===
df = pd.DataFrame(results, columns=["file_path", "raw_input", "cleaned_input", "segmented_input"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ Done! Saved to {OUTPUT_CSV}")