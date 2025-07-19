import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# === CONFIG ===
json_path = "data/oldNepali/processed/raw_labels/labels_raw.json"
image_base = "data/oldNepali/processed/images"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# === LOAD JSON DATA ===
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === TEXT STATS (global) ===
text_lengths = [len(entry["text"]) for entry in data if "text" in entry]

print("📜 TEXT STATISTICS")
print("=" * 50)
print(f"Total text entries       : {len(text_lengths)}")
print(f"Average text length      : {np.mean(text_lengths):.2f} characters")
print(f"Shortest text length     : {np.min(text_lengths)} characters")
print(f"Longest text length      : {np.max(text_lengths)} characters")
print(f"Standard deviation       : {np.std(text_lengths):.2f} characters")
print("=" * 50 + "\n")

# === IMAGE STATS (global) ===
image_sizes = []
dpi_values = []
missing_images = []

for entry in data:
    rel_path = entry["image_path"]
    full_path = rel_path  # or use os.path.join(image_base, rel_path)

    if os.path.exists(full_path):
        try:
            with Image.open(full_path) as img:
                image_sizes.append(img.size)
                dpi = img.info.get("dpi", (None, None))
                dpi_values.append(dpi)
        except Exception as e:
            print(f"⚠️ Error reading image: {full_path} — {e}")
    else:
        missing_images.append(full_path)

if image_sizes:
    dims = np.array(image_sizes)
    widths, heights = dims[:, 0], dims[:, 1]
    aspect_ratios = widths / heights

    dpi_x_vals = [x[0] for x in dpi_values if isinstance(x[0], (int, float))]
    dpi_y_vals = [x[1] for x in dpi_values if isinstance(x[1], (int, float))]

    avg_dpi_x = np.mean(dpi_x_vals) if dpi_x_vals else "?"
    avg_dpi_y = np.mean(dpi_y_vals) if dpi_y_vals else "?"

    print("🖼️ IMAGE SIZE STATISTICS")
    print("=" * 50)
    print(f"Images found             : {len(image_sizes)} / {len(data)}")
    print(f"Average width × height   : {widths.mean():.1f} × {heights.mean():.1f} pixels")
    print(f"Smallest image           : {widths.min()} × {heights.min()} pixels")
    print(f"Largest image            : {widths.max()} × {heights.max()} pixels")
    print(f"Average aspect ratio     : {aspect_ratios.mean():.2f} (W/H)")
    print(f"Average DPI X            : {avg_dpi_x}")
    print(f"Average DPI Y            : {avg_dpi_y}")
    print(f"Missing or unreadable    : {len(missing_images)}")
    print("=" * 50 + "\n")

    # === Global Image Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.5, edgecolors='k')
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Scatter Plot of Image Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "oldNepali_image_descriptive.png"))

    # === Text Length Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(text_lengths, bins=30, color="teal", edgecolor="black")
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "oldNepali_text_length_histogram.png"))

else:
    print("⚠️ No images could be opened or found.\n")