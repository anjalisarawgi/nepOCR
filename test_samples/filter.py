import os
import csv

input_file = "test_samples/predictions_model2.csv"

# The directory that has your image folders
image_root = "test_samples/processed/images"
image_folders = os.listdir(image_root)

# Where to save final .txt files
output_dir = "test_samples/individual_predictions"

os.makedirs(output_dir, exist_ok=True)

# Get all folder names in the image directory
image_folders = os.listdir(image_root)

# Read the full CSV data
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Go folder-by-folder
for folder in image_folders:
    folder_rows = [row for row in rows if folder in row["image_path"]]
    if not folder_rows:
        print(f"⚠️ No matches for: {folder}")
        continue

    out_path = os.path.join(output_dir, f"{folder}.txt")
    total_cer = 0.0

    with open(out_path, "w", encoding="utf-8") as out:
        for row in folder_rows:
            image_name = os.path.basename(row["image_path"])
            gt = row["ground_truth"]
            pred = row["prediction"]
            cer = float(row["cer"])
            total_cer += cer

            out.write(f"Image: {image_name}\n")
            out.write(f"Ground Truth: {gt}\n")
            out.write(f"Prediction   : {pred}\n")
            out.write(f"CER          : {cer:.4f}\n")
            out.write("-" * 40 + "\n")

        avg_cer = total_cer / len(folder_rows)
        out.write(f"\n📊 Final Avg CER for `{folder}`: {avg_cer:.4f}\n")

    print(f"✅ Saved: {folder}.txt with CER {avg_cer:.4f}")

print(f"\n🏁 All folder reports saved to `{output_dir}/`")