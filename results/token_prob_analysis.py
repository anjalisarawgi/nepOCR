import pandas as pd
import ast
import numpy as np

CSV_PATH = "results/trocr_large_bert_byteBPE/predictions_with_logits.csv"

df = pd.read_csv(CSV_PATH, converters={
    "pred_tokens": ast.literal_eval,
    "pred_token_probs": ast.literal_eval,
    "gt_tokens": ast.literal_eval
})

def analyze_token_level_confidence(row):
    pred_tokens = row["pred_tokens"]
    gt_tokens = row["gt_tokens"]
    probs = row["pred_token_probs"]

    min_len = min(len(pred_tokens), len(gt_tokens), len(probs))
    return [
        {"token": p, "gt": g, "prob": prob, "correct": p == g}
        for p, g, prob in zip(pred_tokens[:min_len], gt_tokens[:min_len], probs[:min_len])
    ]

df["token_analysis"] = df.apply(analyze_token_level_confidence, axis=1)
token_df = pd.DataFrame([t for sublist in df["token_analysis"] for t in sublist])
bins = np.arange(0.0, 1.01, 0.2)
labels = [f"{round(bins[i],1)}–{round(bins[i+1],1)}" for i in range(len(bins)-1)]
token_df["prob_bin"] = pd.cut(token_df["prob"], bins=bins, labels=labels, include_lowest=True)

bin_counts = token_df.groupby(["prob_bin", "correct"]).size().unstack(fill_value=0)

bin_counts.to_csv("results/trocr_large_bert_byteBPE/token_prob_analysis.csv")
print(bin_counts)