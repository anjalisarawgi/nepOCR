import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
json_path = "data/oldNepali/processed/raw_labels/labels_raw.json"
image_base = "data/oldNepali/processed/images"

# === LOAD JSON DATA ===
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === TEXT STATS ===
text_lengths = [len(entry["text"]) for entry in data if "text" in entry]

print("📜 TEXT STATISTICS")
print(f"Total samples: {len(text_lengths)}")
print(f"Average length: {np.mean(text_lengths):.2f} characters")
print(f"Min length: {np.min(text_lengths)}")
print(f"Max length: {np.max(text_lengths)}")
print(f"Std deviation: {np.std(text_lengths):.2f}\n")

# === IMAGE STATS ===
image_sizes = []
missing_images = []

for entry in data:
    rel_path = entry["image_path"]
    full_path = rel_path

    if os.path.exists(full_path):
        try:
            with Image.open(full_path) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"⚠️ Error reading image: {full_path} — {e}")
    else:
        missing_images.append(full_path)

if image_sizes:
    dims = np.array(image_sizes)
    widths, heights = dims[:, 0], dims[:, 1]
    aspect_ratios = widths / heights

    print("🖼️ IMAGE SIZE STATISTICS")
    print(f"Images found: {len(image_sizes)} / {len(data)}")
    print(f"Avg width x height: {widths.mean():.1f} x {heights.mean():.1f}")
    print(f"Min size: {widths.min()} x {heights.min()}")
    print(f"Max size: {widths.max()} x {heights.max()}")
    print(f"Avg aspect ratio (W/H): {aspect_ratios.mean():.2f}")
    print(f"Images missing or not found: {len(missing_images)}\n")

    # === Optional Visualization ===
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.5, edgecolors='k')
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Scatter Plot of Image Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/oldNepali_image_descriptive.png")

    # === Optional: Aspect Ratio Histogram ===
    plt.figure(figsize=(8, 4))
    plt.hist(aspect_ratios, bins=20, color="purple", edgecolor="black")
    plt.title("Aspect Ratio Distribution (Width / Height)")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/oldNepali_aspect_ratio_histogram.png")

    # === TEXT LENGTH HISTOGRAM ===
    plt.figure(figsize=(8, 4))
    plt.hist(text_lengths, bins=30, color="teal", edgecolor="black")
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/oldNepali_text_length_histogram.png")

else:
    print("⚠️ No images could be opened or found.")