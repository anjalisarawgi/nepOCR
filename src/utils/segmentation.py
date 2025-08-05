import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# img_path = "dataset1/images/DNA_0014_0296_bw.png"

# change 1
img = Image.open("dataset1/images/DNA_0014_0296_bw.png").convert("RGB")
draw = ImageDraw.Draw(img)

# change 2
with open("output.json" , "r") as f:
    data = json.load(f)

for line in data["lines"]:
    if "boundary" in line:
        boundary = [tuple(pt) for pt in line["boundary"]]
        draw.polygon(boundary, outline="red")
        
    if "baseline" in line:
        baseline = [tuple(pt) for pt in line["baseline"]]
        draw.line(baseline, fill="green", width=2)

plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.title("Kraken Line Segmentation: Red = Polygon, Green = Baseline")
plt.axis("off")
# plt.show()
