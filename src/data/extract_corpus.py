import json

with open( 'data/oldNepali_fullset/labels_normalized_final/labels_train.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

with open('labels_train.txt', 'w', encoding='utf-8') as outfile:
    for entry in data:
        text = entry.get('text', '').strip()
        if text:
            outfile.write(text + '\n')

print("completed")
