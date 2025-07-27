import pandas as pd
from collections import Counter

input_file = "oldNepali_fullset_labels.txt" 
output_csv = "char_frequency_full.csv"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

counter = Counter(text)
total = sum(counter.values())

filtered_counter = {c: freq for c, freq in counter.items() if not c.isspace() and c.isprintable()}

rows = []
for i, (char, freq) in enumerate(sorted(filtered_counter.items(), key=lambda x: -x[1]), start=1):
    rows.append({
        "#": i,
        "Character": char,
        "Frequency": freq,
        "Relative Frequency (%)": round((freq / total) * 100, 4)
    })


df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False, encoding="utf-8")
