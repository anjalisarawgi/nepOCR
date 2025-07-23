import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# === CONFIG ===
IMG_DIR  = 'test_samples/raw/images'
XML_DIR  = 'test_samples/raw/ground_truth'
OUT_BASE = 'test_samples/processed/images'
JSON_PATH = 'test_samples/processed/labels.json'

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

# ALTO namespace
ns = {'a': 'http://www.loc.gov/standards/alto/ns-v4#'}

# helper to find matching image file
def find_image(basename):
    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        p = os.path.join(IMG_DIR, basename + ext)
        if os.path.exists(p):
            return p
    return None

all_labels = {}

# === MAIN LOOP ===
for xml_fname in os.listdir(XML_DIR):
    if not xml_fname.lower().endswith('.xml'):
        continue
    basename = os.path.splitext(xml_fname)[0]
    xml_path = os.path.join(XML_DIR, xml_fname)
    img_path = find_image(basename)
    if img_path is None:
        print(f"[!] No image for {basename}, skipping")
        continue

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[✗] Couldn't load image {img_path}")
        continue
    h, w = img.shape[:2]

    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[!] Failed to parse {xml_path}: {e}")
        continue

    lines = root.findall('.//a:TextLine', ns)
    if not lines:
        print(f"[!] No lines found in {basename}")
        continue

    out_dir = os.path.join(OUT_BASE, basename)
    os.makedirs(out_dir, exist_ok=True)
    page_labels = []

    print(f"⏳ Processing {basename}: {len(lines)} lines")

    for idx, line in enumerate(lines, start=1):
        lid = line.attrib.get('ID', f'line{idx}')
        poly_el = line.find('.//a:Polygon', ns)
        if poly_el is None:
            continue

        coords = list(map(int, poly_el.attrib['POINTS'].split()))
        if len(coords) < 6:
            continue  # Skip malformed polygons
        pts = np.array(coords, dtype=np.int32).reshape(-1, 2)

        # Create mask
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Bounding box crop
        x, y, ww, hh = cv2.boundingRect(pts)
        if ww < 5 or hh < 5:
            continue
        crop_img = img[y:y+hh, x:x+ww]
        crop_mask = mask[y:y+hh, x:x+ww]

        # Mask background to white
        out = np.full_like(crop_img, 255)
        out[crop_mask == 255] = crop_img[crop_mask == 255]

        # Save cropped line
        out_path = os.path.join(out_dir, f"{lid}.png")
        cv2.imwrite(out_path, out)

        # Collect text
        texts = [s.attrib.get('CONTENT', '') for s in line.findall('.//a:String', ns)]
        transcription = ' '.join(texts)

        page_labels.append({
            "text": transcription,
            "image_path": os.path.join('oldNepali_100', 'processed', 'images', basename, f"{lid}.png")
        })

    # all_labels[basename] = page_labels
    all_labels.setdefault("data", []).extend(page_labels)
    print(f"[✓] Saved {len(page_labels)} crops and labels for {basename}")

# Save JSON
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    # json.dump(all_labels, f, ensure_ascii=False, indent=2)
    json.dump(all_labels["data"], f, ensure_ascii=False, indent=2)
print(f"[🎉] Written all labels to {JSON_PATH}")