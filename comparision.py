import json
import re

# Step 1: Load both JSON files
with open('data/oldNepali/processed/normalized_labels/labels_train.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('data/oldNepali/processed/labels_train.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# Utility to normalize both texts by replacing all digits with a placeholder
def normalize_digits(text):
    # Replace both ASCII 0-9 and Devanagari ०-९ with "*"
    return re.sub(r'[0-9०-९]', '*', text)

# Index file2 by image_path for fast lookup
data2_dict = {entry['image_path']: entry['text'] for entry in data2}

# Compare entries ignoring digit mismatches
unmatched = []

for entry in data1:
    img_path = entry['image_path']
    text1 = entry['text']
    text2 = data2_dict.get(img_path)

    if text2 is None:
        unmatched.append({
            "image_path": img_path,
            "reason": "missing_in_file2",
            "text_file1": text1,
            "text_file2": None
        })
    elif normalize_digits(text1) != normalize_digits(text2):
        unmatched.append({
            "image_path": img_path,
            "text_file1": text1,
            "text_file2": text2
        })

# Save the result
with open('unmatched_ignore_digits.json', 'w', encoding='utf-8') as f:
    json.dump(unmatched, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Saved {len(unmatched)} unmatched entries (ignoring digit differences) to unmatched_ignore_digits.json")