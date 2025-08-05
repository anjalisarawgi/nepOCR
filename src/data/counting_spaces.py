import json

with open("data/oldNepali_fullset/labels_normalized_final/labels_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total_spaces = 0
lines_with_spaces = 0
total_chars = 0

for entry in data:
    text = entry.get("text", "")
    space_count = text.count(" ")
    total_spaces += space_count
    total_chars += len(text)
    if space_count > 0:
        lines_with_spaces += 1

print("Total number of space characters:",total_spaces)
print("Number of lines containing at least one space:", lines_with_spaces)
print("Total number of lines:", len(data))
print("Total number of characters (including spaces):", total_chars)
