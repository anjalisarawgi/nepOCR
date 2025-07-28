import os
import json
from PIL import Image

base_dir    = "data/nagari/original/train"
input_dir   = os.path.join(base_dir, "images")
output_dir  = os.path.join(base_dir, "images_grayscaled")
os.makedirs(output_dir, exist_ok=True)

labels_path = os.path.join(base_dir, "labels.json")
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

# for convert images to grayscale and also save the labels.json with updated paths
grayscale_labels = []
for item in labels:
    orig_rel = item["image_path"]
    fname    = os.path.basename(orig_rel)

    img = Image.open(os.path.join(input_dir, fname)).convert("L") # convert to grayscale
    new_rel = os.path.join("images_grayscaled", fname)
    img.save(os.path.join(base_dir, new_rel))
    new_item = item.copy()
    new_item["image_path"] = new_rel
    grayscale_labels.append(new_item)

out_labels = os.path.join(base_dir, "labels_grayscaled.json")
with open(out_labels, "w", encoding="utf-8") as f:
    json.dump(grayscale_labels, f, ensure_ascii=False, indent=2)

print("Grayscale conversion complete.")