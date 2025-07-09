import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# === CONFIG ===
IMG_DIR   = 'data/oldNepali/raw/images'
XML_DIR   = 'data/oldNepali/raw/groundTruth'
OUT_BASE  = 'data/oldNepali/processed/images'
JSON_PATH = 'data/oldNepali/processed/labels.json'
os.makedirs(OUT_BASE, exist_ok=True)

# ALTO namespace
ns = {'a': 'http://www.loc.gov/standards/alto/ns-v4#'}

# Helper to match image by basename
def find_image(basename):
    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        path = os.path.join(IMG_DIR, basename + ext)
        if os.path.exists(path):
            return path
    return None

all_labels = {}
total_line_images = 0


for xml_fname in os.listdir(XML_DIR):
    if not xml_fname.lower().endswith('.xml'):
        continue

    basename = os.path.splitext(xml_fname)[0]
    xml_path = os.path.join(XML_DIR, xml_fname)
    img_path = find_image(basename)

    if img_path is None:
        print(f"[!] No image for {basename}, skipping")
        continue

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[✗] Couldn't load {img_path}, skipping")
        continue
    h, w = img.shape[:2]

    # Output dir for cropped lines
    out_dir = os.path.join(OUT_BASE, basename)
    os.makedirs(out_dir, exist_ok=True)

    # Output label list for this page
    page_labels = []

    # Find text lines
    lines = root.findall('.//a:TextLine', ns)
    print(f"⏳ Processing {basename}: {len(lines)} lines")

    for idx, line in enumerate(lines, start=1):
        lid = line.attrib.get('ID', f'line{idx}')
        poly_el = line.find('.//a:Polygon', ns)
        if poly_el is None:
            continue

        coords = list(map(int, poly_el.attrib['POINTS'].split()))
        pts = np.array(coords, dtype=np.int32).reshape(-1, 2)

        # Skip tiny lines
        x, y, ww, hh = cv2.boundingRect(pts)
        if ww < 5 or hh < 5:
            continue

        # Build mask
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Crop region
        crop_img = img[y:y+hh, x:x+ww]
        crop_mask = mask[y:y+hh, x:x+ww]

        # White background composite
        out = np.full_like(crop_img, 255)
        out[crop_mask == 255] = crop_img[crop_mask == 255]

        # Output path
        out_fname = f"{lid}.png"
        out_path = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_path, out)
        total_line_images += 1

        # Get transcription (join all String nodes)
        texts = [s.attrib.get('CONTENT', '') for s in line.findall('.//a:String', ns)]
        transcription = ' '.join(texts)

        # Save metadata
        page_labels.append({
            "text": transcription,
            "image_path": os.path.join('data/oldNepali/processed/images', basename, out_fname)  # relative to processed/
        })

    all_labels[basename] = page_labels
    print(f"[✓] Saved {len(page_labels)} crops to {out_dir}")
    print(f"Total line images saved: {total_line_images}")


# Flatten all page_labels into one big list
flat_labels = [entry for lines in all_labels.values() for entry in lines]

# Save flat structure
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(flat_labels, f, ensure_ascii=False, indent=2)
    
print(f"[🎉] Written all labels to {JSON_PATH}")
print("✅ Done!")
