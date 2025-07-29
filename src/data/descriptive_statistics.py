import json
import numpy as np
from PIL import Image
from collections import defaultdict
import re

with open("data/oldNepali_fullset/labels_raw/labels_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#### text statistics
line_lengths = [len(entry["text"]) for entry in data]
line_lengths_np = np.array(line_lengths)

total_lines = len(line_lengths)
total_chars = int(line_lengths_np.sum())
average_length = float(line_lengths_np.mean())
min_length = int(line_lengths_np.min())
max_length = int(line_lengths_np.max())
median_length = float(np.median(line_lengths_np))
std_dev = float(line_lengths_np.std())

print("Total lines:", total_lines)
print("Total characters:", total_chars)
print("Average line length:", average_length)
print("Minimum line length:", min_length)
print("Maximum line length:", max_length)
print("Median line length:", median_length)
print("Standard deviation of line lengths:", std_dev)

long_lines = [entry for entry in data if len(entry["text"]) > 100]
print("Number of lines with length > 120:", len(long_lines))



#### image statisticss
widths = []
heights = []
resolutions = []
for entry in data:
    img = Image.open(entry["image_path"])
    w, h = img.size
    widths.append(w)
    heights.append(h)

    dpi = img.info.get("dpi")
    if dpi:
        resolutions.append(dpi) 

avg_width = np.mean(widths)
avg_height = np.mean(heights)
print("Average image width:", avg_width)
print("Average image height:", avg_height)

