import json
import os

input_path = "data/oldNepali_fullset_aug8/labels.json"
output_path = "data/oldNepali_fullset_aug8/labels_no_space.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data_no_space = []
for entry in data:
    new_text = entry["text"].replace(" ", "")
    data_no_space.append({
        "text": new_text,
        "image_path": entry["image_path"]
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_no_space, f, indent=2, ensure_ascii=False)

print("Processed data saved to:", output_path)