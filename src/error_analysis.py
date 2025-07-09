import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.text import CharErrorRate  # pip install torchmetrics

# === Load your CSV ===
# df = pd.read_csv("results/evaluation_results_cleaned_bytebpe500_test_new8.csv")
df = pd.read_csv("results/evaluation_results_cleaned_bytebpe500_test_new8.csv")

# === Plot histogram for raw CER ===
plt.figure(figsize=(10, 5))
plt.hist(df['raw_cer'], bins=20, edgecolor='black')
plt.title("Distribution of Raw CER")
plt.xlabel("Character Error Rate")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("results/raw_cer_histogram.png")

# === Totals & means by prefix (DNA / CA / NA) ===
dna_total = df['page'].str.startswith("DNA").sum()
ca_total  = df['page'].str.startswith("CA").sum()
na_total  = df['page'].str.startswith("NA").sum()

mean_cer_dna = df[df['page'].str.startswith("DNA")]['raw_cer'].mean()
mean_cer_ca  = df[df['page'].str.startswith("CA")]['raw_cer'].mean()
mean_cer_all = df['raw_cer'].mean()
median_cer_all = df['raw_cer'].median()

print(f"Total DNA pages: {dna_total}")
print(f"Total CA pages: {ca_total}")
print(f"Total NA pages: {na_total}")
print(f"Mean CER for DNA pages: {mean_cer_dna:.4f}")
print(f"Mean CER for CA pages: {mean_cer_ca:.4f}")
print(f"Mean CER for all pages: {mean_cer_all:.4f}")
print(f"Median CER for all pages: {median_cer_all:.4f}")

# === True Corpus-level CER ===
cer = CharErrorRate()
corpus_level_cer = cer(df['prediction'].astype(str).tolist(), df['ground_truth'].astype(str).tolist()).item()
print(f"Corpus-level CER (true total edits / total chars): {corpus_level_cer:.4f}")


# perfect predictions 
perfect_predictions = df[df['raw_cer'] == 0]
print(f"percentage of perfect predictions: {len(perfect_predictions) / len(df) * 100:.2f}%")

# === Dotless CER ===
from torchmetrics.text import CharErrorRate

def strip_dots(text):
    return str(text).replace(".", "").replace("।", "")

# Create dot-stripped versions of GT and predictions
gt_nodots = df['ground_truth'].apply(strip_dots).tolist()
pred_nodots = df['prediction'].apply(strip_dots).tolist()

# Calculate CER on dot-stripped data
cer = CharErrorRate()
dotless_cer = cer(pred_nodots, gt_nodots).item()

print(f"CER with all dots removed from GT and Prediction: {dotless_cer:.4f}")
original_cer = cer(df['prediction'].astype(str).tolist(), df['ground_truth'].astype(str).tolist()).item()
print(f"Original CER: {original_cer:.4f}")
print(f"Dotless CER:  {dotless_cer:.4f}")
print(f"Δ CER (reduction): {original_cer - dotless_cer:.4f}")
