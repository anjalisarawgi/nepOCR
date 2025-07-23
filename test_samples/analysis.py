import os
import csv

input_file = "test_samples/predictions_model2.csv"
image_root = "test_samples/processed/images"
output_dir = "test_samples/individual_predictions"
os.makedirs(output_dir, exist_ok=True)

image_folders = os.listdir(image_root)

# Read the full CSV
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Folder-wise processing
for folder in image_folders:
    folder_rows = [row for row in rows if folder in row["image_path"]]
    if not folder_rows:
        print(f"⚠️ No matches for: {folder}")
        continue

    out_path = os.path.join(output_dir, f"{folder}.txt")
    total_cer = 0.0
    total_lines = len(folder_rows)
    correct_lines = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for row in folder_rows:
            image_name = os.path.basename(row["image_path"])
            gt = row["ground_truth"]
            pred = row["prediction"]
            cer = float(row["cer"])
            total_cer += cer
            if gt.strip() == pred.strip():
                correct_lines += 1

            out.write(f"Image: {image_name}\n")
            out.write(f"Ground Truth: {gt}\n")
            out.write(f"Prediction   : {pred}\n")
            out.write(f"CER          : {cer:.4f}\n")
            out.write("----------------------------------------\n")

        avg_cer = total_cer / total_lines if total_lines > 0 else 0.0
        accuracy_pct = (correct_lines / total_lines * 100) if total_lines > 0 else 0.0

        out.write("\n")
        out.write(f"Total lines            : {total_lines}\n")
        out.write(f"Correct predictions    : {correct_lines}\n")
        out.write(f"Perfect prediction %   : {accuracy_pct:.2f}\n")
        out.write(f"Final Avg CER          : {avg_cer:.4f}\n")

    print(f"Saved: {folder}.txt")