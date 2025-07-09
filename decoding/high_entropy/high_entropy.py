import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

JSON_PATH = "decoding/results/beam_search_detailed/results_with_tokens.json"
ENTROPY_THRESHOLD = 1.0

def align_pred_with_truth(pred, truth):
    aligned = []
    min_len = min(len(pred), len(truth))
    for i in range(min_len):
        aligned.append(pred[i] == truth[i])
    aligned.extend([False] * (len(pred) - min_len))
    return aligned

with open(JSON_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

rows = []

for r in tqdm(results):
    pred = r["pred_text"]
    truth = r["ground_truth"]
    tokens = r["token_probs"]

    correctness = align_pred_with_truth(pred, truth)

    for i, token in enumerate(tokens):
        is_wrong = True if i >= len(correctness) else not correctness[i]
        entropy = token["entropy"]
        rows.append({
            "image_path": r["image_path"],
            "token_index": i,
            "pred_token": token["pred_token"],
            "entropy": entropy,
            "is_wrong": is_wrong
        })

df = pd.DataFrame(rows)

high_entropy_df = df[df["entropy"] > ENTROPY_THRESHOLD]
high_entropy_df = high_entropy_df[high_entropy_df["is_wrong"].isin([True, False])]

wrong_count = high_entropy_df["is_wrong"].sum()
total = len(high_entropy_df)

print(f" High-entropy tokens (> {ENTROPY_THRESHOLD}): {total}")
print(f" Wrong predictions among them: {wrong_count}")

df["prob_bin"] = pd.cut(
    df["top_prob"],
    bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
)

bin_stats = df.groupby("prob_bin")["is_wrong"].agg(["mean", "count"]).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=bin_stats, x="prob_bin", y="mean", palette="mako")
plt.title("❌ Wrong Prediction Rate per Top-1 Probability Bin")
plt.xlabel("Top-1 Probability Bin")
plt.ylabel("Fraction of Wrong Predictions")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("decoding/high_entropy/wrong_pred_rate_prob_bins.png")

print("\n📊 Probability bin stats:")
print(bin_stats)

sns.set(style="whitegrid")


# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=high_entropy_df, x="token_index", y="entropy", hue="is_wrong", palette={True: "red", False: "green"}, alpha=0.6)
# plt.axhline(y=ENTROPY_THRESHOLD, color="blue", linestyle="--", label=f"Threshold = {ENTROPY_THRESHOLD}")
# plt.title("Entropy vs Token Index (High Entropy Tokens Only)")
# plt.xlabel("Token Index")
# plt.ylabel("Entropy")
# plt.legend()
# plt.tight_layout()
# plt.savefig("decoding/high_entropy/scatter_entropy_vs_wrongness_high_entropy_only.png")

plt.figure(figsize=(10, 6))
sns.kdeplot(high_entropy_df["entropy"], label="High-Entropy Tokens", fill=True)
sns.kdeplot(high_entropy_df[high_entropy_df["is_wrong"] == True]["entropy"], label="Wrong High-Entropy Tokens", fill=True, color="red")
plt.axvline(x=ENTROPY_THRESHOLD, color="blue", linestyle="--", label="Entropy Threshold")
plt.title("Entropy Distribution: High-Entropy Zone Only (Entropy > 1.0)")
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("decoding/high_entropy/high_entropy_distribution.png")

total_wrong_preds = df["is_wrong"].sum()
wrong_with_high_entropy = high_entropy_df["is_wrong"].sum()

print(f"Total wrong predictions: {total_wrong_preds}")
print(f"Wrong predictions with high entropy (> {ENTROPY_THRESHOLD}): {wrong_with_high_entropy}")
print(f"Fraction of wrong preds that are high-entropy: {100 * wrong_with_high_entropy / total_wrong_preds:.2f}%")