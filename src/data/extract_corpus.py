import json

# Path to your input JSON and output TXT
input_json_path = 'data/oldNepali_fullset/labels_normalized_final/labels_train.json'
output_txt_path = 'labels_train.txt'

# Load the JSON data
with open(input_json_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# Extract "text" fields and write them line by line
with open(output_txt_path, 'w', encoding='utf-8') as outfile:
    for entry in data:
        text = entry.get('text', '').strip()
        if text:
            outfile.write(text + '\n')

print(f"Wrote {len(data)} lines to {output_txt_path}")


# import pandas as pd

# # 1) Load your CSV
# df = pd.read_csv(
#     "data/evaluation_results_cleaned_bytebpe500_test_new8.csv",
#     encoding="utf-8"
# )

# # 2) Extract the prediction column
# preds = df["prediction"].astype(str).tolist()

# # 3) Write them out to a .txt (one line per prediction)
# with open("corpus/oldNepali_test_ground_truth.txt", "w", encoding="utf-8") as f:
#     for line in preds:
#         f.write(line + "\n")

# print(f"✅ Saved {len(preds)} lines to predictions_corpus.txt")