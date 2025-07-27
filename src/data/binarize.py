import os
import json
import cv2


with open("data/oldNepali_fullset/labels_raw/labels_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_dir = "data/oldNepali_fullset_binarized"

for item in data:
    original_path = item["image_path"]
    img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    relative_path = os.path.relpath(original_path, "data/oldNepali_fullset")
    binarized_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(binarized_path), exist_ok=True)
    cv2.imwrite(binarized_path, binary)
    item["image_path"] = binarized_path

out_json_path = os.path.join(output_dir, "labels_binarized.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Binarization with Otsu completed!")