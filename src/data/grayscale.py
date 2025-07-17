import os
import json
from PIL import Image

# 1️⃣ Set up your dirs
base_dir    = "data/nagari/original/train"
input_dir   = os.path.join(base_dir, "images")
output_dir  = os.path.join(base_dir, "images_grayscaled")
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ Load your original labels.json
labels_path = os.path.join(base_dir, "labels.json")
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

# 3️⃣ Convert each image → grayscale + build new labels list
grayscale_labels = []
for item in labels:
    # a) get filename
    orig_rel = item["image_path"]
    fname    = os.path.basename(orig_rel)

    # b) open & convert
    img = Image.open(os.path.join(input_dir, fname)).convert("L")

    # c) save to grayscale folder
    new_rel = os.path.join("images_grayscaled", fname)
    img.save(os.path.join(base_dir, new_rel))

    # d) copy label entry and point to new path
    new_item = item.copy()
    new_item["image_path"] = new_rel
    grayscale_labels.append(new_item)

# 4️⃣ Dump your new labels JSON
out_labels = os.path.join(base_dir, "labels_grayscaled.json")
with open(out_labels, "w", encoding="utf-8") as f:
    json.dump(grayscale_labels, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Processed {len(grayscale_labels)} images. New labels at:\n  {out_labels}")