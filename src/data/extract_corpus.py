import json

input_json_path = 'data/oldNepali_fullset/labels_normalized_final/labels_train.json'
output_txt_path = 'labels_train.txt'

with open(input_json_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

with open(output_txt_path, 'w', encoding='utf-8') as outfile:
    for entry in data:
        text = entry.get('text', '').strip()
        if text:
            outfile.write(text + '\n')

print("completed")
