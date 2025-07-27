import os
import csv
from editdistance import eval as edit_distance  # Install via: pip install editdistance

input_file = "test_samples/predictions_model2.csv"
image_root = "test_samples/processed/images"
output_dir = "test_samples/individual_predictions"
os.makedirs(output_dir, exist_ok=True)

image_folders = os.listdir(image_root)

# Read CSV
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Process each folder
for folder in image_folders:
    folder_rows = [row for row in rows if folder in row["image_path"]]
    if not folder_rows:
        print(f"⚠️ No matches for: {folder}")
        continue

    out_path = os.path.join(output_dir, f"{folder}.txt")
    total_edits = 0
    total_chars = 0
    total_lines = len(folder_rows)
    correct_lines = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for row in folder_rows:
            image_name = os.path.basename(row["image_path"])
            gt = row["ground_truth"].strip()
            pred = row["prediction"].strip()

            # CER as edit distance
            edits = edit_distance(gt, pred)
            total_edits += edits
            total_chars += len(gt)

            if gt == pred:
                correct_lines += 1

            cer = edits / len(gt) if len(gt) > 0 else 0.0

            out.write(f"Image: {image_name}\n")
            out.write(f"Ground Truth: {gt}\n")
            out.write(f"Prediction   : {pred}\n")
            out.write(f"CER          : {cer:.4f}\n")
            out.write("----------------------------------------\n")

        corpus_cer = total_edits / total_chars if total_chars > 0 else 0.0
        accuracy_pct = (correct_lines / total_lines * 100) if total_lines > 0 else 0.0

        out.write("\n")
        out.write(f"Total lines            : {total_lines}\n")
        out.write(f"Correct predictions    : {correct_lines}\n")
        out.write(f"Perfect prediction %   : {accuracy_pct:.2f}\n")
        out.write(f"Corpus-level CER       : {corpus_cer:.4f}\n")

    print(f"✅ Saved: {folder}.txt")