import os
from PIL import Image
import pandas as pd

# === CONFIG ===
base_dir = "data/oldNepali_full"  # Adjust if needed
image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

# === COLLECT IMAGE INFO ===
image_info = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    dpi = img.info.get("dpi", ("?", "?"))
                    
                    # Handle missing DPI gracefully
                    dpi_x = dpi[0] if isinstance(dpi[0], (int, float)) else None
                    dpi_y = dpi[1] if isinstance(dpi[1], (int, float)) else None
                    
                    image_info.append({
                        "Image Path": path,
                        "Width (px)": width,
                        "Height (px)": height,
                        "DPI X": dpi_x,
                        "DPI Y": dpi_y
                    })
            except Exception as e:
                print(f"⚠️ Error reading {path}: {e}")

# === DISPLAY RESULTS ===
df = pd.DataFrame(image_info)
pd.set_option("display.max_colwidth", None)

print("\n📊 Image Dimension & DPI Info:\n")
print(df.to_string(index=False))

# === DISPLAY AVERAGES ===
print("\n📈 AVERAGES ACROSS ALL IMAGES")
print("=" * 50)
print(f"Average Width (px)   : {df['Width (px)'].mean():.2f}")
print(f"Average Height (px)  : {df['Height (px)'].mean():.2f}")

if df['DPI X'].notna().any():
    print(f"Average DPI X        : {df['DPI X'].dropna().astype(float).mean():.2f}")
else:
    print("Average DPI X        : ?")

if df['DPI Y'].notna().any():
    print(f"Average DPI Y        : {df['DPI Y'].dropna().astype(float).mean():.2f}")
else:
    print("Average DPI Y        : ?")
