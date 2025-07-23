import json

# Mapping from ASCII digits to Nepali (Devanagari) digits
ascii_to_nepali_digits = str.maketrans({
    '0': '०',
    '1': '१',
    '2': '२',
    '3': '३',
    '4': '४',
    '5': '५',
    '6': '६',
    '7': '७',
    '8': '८',
    '9': '९',
})

# Load your labels file
with open('data/oldNepali_aug16/labels_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Replace ASCII digits with Nepali ones in the "text" field
for entry in data:
    if 'text' in entry:
        entry['text'] = entry['text'].translate(ascii_to_nepali_digits)

# Save the modified version to a new file (or overwrite if you want)
with open('data/oldNepali_aug16/labels_train_converted.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ All ASCII digits replaced with Nepali digits and saved to labels_nepali_digits.json")