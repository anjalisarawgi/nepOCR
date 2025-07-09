import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein

INPUT_PATH   = "decoding/results/beam_search_detailed/results_with_tokens.json"
OUTPUT_DIR   = 'decoding/counts'
OUTPUT_FILE  = 'confusion_matrix_5x2.png'

bucket_names = ['vlow', 'low', 'mid', 'high', 'vhigh']
counts = {b: {'correct': 0, 'error': 0} for b in bucket_names}

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    records = json.load(f)

for rec in records:
    gt   = rec['ground_truth']
    pred = rec['pred_text']
    probs = [tok['topk'][0]['prob'] for tok in rec['token_probs']]
    opcodes = Levenshtein.opcodes(gt, pred)
    prob_idx = 0

    for tag, i1, i2, j1, j2 in opcodes:
        steps = max(i2 - i1, j2 - j1) if tag == 'replace' else j2 - j1 if tag == 'insert' else i2 - i1
        for _ in range(steps):
            if tag == 'delete':
                counts['vlow']['error'] += 1  # assume lowest confidence when deleted
            elif prob_idx < len(probs):
                prob = probs[prob_idx]
                if prob <= 0.2:
                    bucket = 'vlow'
                elif prob <= 0.4:
                    bucket = 'low'
                elif prob <= 0.6:
                    bucket = 'mid'
                elif prob <= 0.8:
                    bucket = 'high'
                else:
                    bucket = 'vhigh'

                if tag == 'equal':
                    counts[bucket]['correct'] += 1
                else:
                    counts[bucket]['error'] += 1

                prob_idx += 1


data = np.array([
    [counts['vlow']['correct'],  counts['vlow']['error']],
    [counts['low']['correct'],   counts['low']['error']],
    [counts['mid']['correct'],   counts['mid']['error']],
    [counts['high']['correct'],  counts['high']['error']],
    [counts['vhigh']['correct'], counts['vhigh']['error']],
])
labels = np.array([[f"{v:,}" for v in row] for row in data])

data_reversed = data[::-1]
labels_reversed = labels[::-1]
yticklabels_reversed = [
    'Very High (>0.8)',
    'High (0.6–0.8)',
    'Mid (0.4–0.6)',
    'Low (0.2–0.4)',
    'Very Low (≤0.2)'
]

plt.figure(figsize=(7, 6))
ax = sns.heatmap(data_reversed,
                 annot=labels_reversed,
                 fmt='',
                 cmap='Blues',
                 xticklabels=['Correct', 'Error'],
                 yticklabels=yticklabels_reversed,
                 cbar=False,
                 linewidths=1,
                 linecolor='lightgray')

plt.title("5×2 Confusion Matrix: Confidence vs. Correctness", pad=15)
plt.xlabel("Correctness")
plt.ylabel("Confidence Level")
plt.tight_layout()

# ——— SAVE HEATMAP ———
os.makedirs(OUTPUT_DIR, exist_ok=True)
outpath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(outpath, dpi=200)
plt.close()
