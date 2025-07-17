import json
import random

# Load your JSON data
with open("data/nagari/original/train/labels_grayscaled_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle the data to randomize the split
random.shuffle(data)

# Use fixed validation size
val_size = 514
train_data = data[:-val_size]
val_data = data[-val_size:]

# Save the splits into separate files
with open("labels_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("labels_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_data)} samples, Validation: {len(val_data)} samples")