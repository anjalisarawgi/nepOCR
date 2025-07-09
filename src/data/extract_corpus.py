# import json

# # Path to your input JSON and output TXT
# input_json_path = 'data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json'
# output_txt_path = 'corpus/oldNepali_test_v12.txt'

# # Load the JSON data
# with open(input_json_path, 'r', encoding='utf-8') as infile:
#     data = json.load(infile)

# # Extract "text" fields and write them line by line
# with open(output_txt_path, 'w', encoding='utf-8') as outfile:
#     for entry in data:
#         text = entry.get('text', '').strip()
#         if text:
#             outfile.write(text + '\n')

# print(f"Wrote {len(data)} lines to {output_txt_path}")


import pandas as pd

# 1) Load your CSV
df = pd.read_csv(
    "results/evaluation_results_cleaned_bytebpe500_test_new8.csv",
    encoding="utf-8"
)

# 2) Extract the prediction column
preds = df["prediction"].astype(str).tolist()

# 3) Write them out to a .txt (one line per prediction)
with open("corpus/oldNepali_test_ground_truth.txt", "w", encoding="utf-8") as f:
    for line in preds:
        f.write(line + "\n")

print(f"✅ Saved {len(preds)} lines to predictions_corpus.txt")