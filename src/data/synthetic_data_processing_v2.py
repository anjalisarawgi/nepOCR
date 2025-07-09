#!/usr/bin/env python3
"""
synthetic_data_processing.py: Read Nepali lines, randomly scramble spaces,
apply per-character distortions (rotation/scale), render to sized PNGs,
and save metadata to labels.json.
Improved version with fixed character clipping issues.
"""

import os
import re
import regex
import json
import random
from PIL import Image, ImageDraw, ImageFont

# === CONFIGURATION ===
TEXT_FILE = "corpus/top5_confusion_lines_1000.txt"
OUTPUT_DIR = "data/top5_confusion_lines_1000"
IMAGE_SUBDIR = "images"
FONT_DIR = "fonts"

# Scramble settings
REMOVE_PROB = 0.2
DOT_PROB = 0.2

# Distortion settings (tuned for legibility)
DISTORT_PROP_MIN = 0.05   # 5% of characters
DISTORT_PROP_MAX = 0.20   # up to 20%
ANGLE_RANGE = (-15, 15)   # max ±15° rotation
SCALE_RANGE = (0.9, 1.1)  # ±10% scale
MIN_CLUSTER_SIZE = 12     # skip distort for tiny clusters
DISTORTION_PADDING = 0.5  # 50% padding around distorted characters

# Font size & padding
FONT_SIZE_MIN = 28
FONT_SIZE_MAX = 36
PADDING = 20
BOTTOM_PADDING = 5

# Seed for reproducibility (None for fully random)
SEED = 42

# Create output dirs
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_dir = os.path.join(OUTPUT_DIR, IMAGE_SUBDIR)
os.makedirs(image_dir, exist_ok=True)

# Load font paths
font_paths = [os.path.join(FONT_DIR, f)
              for f in os.listdir(FONT_DIR) if f.lower().endswith('.ttf')]
if not font_paths:
    raise RuntimeError(f"No .ttf fonts found in '{FONT_DIR}'")

# Read lines
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# Prepare regex for space scrambling
space_pattern = re.compile(r'(?<=\S) (?=\S)')
def randomly_scramble_spaces(text, remove_prob, dot_prob, seed_val=None):
    if seed_val is not None:
        random.seed(seed_val)
    text = re.sub(r'[,\|]+', '', text)
    def repl(m):
        if random.random() < remove_prob:
            return '.' if random.random() < dot_prob else ''
        return ' '
    return space_pattern.sub(repl, text)

def calculate_initial_canvas_size(clusters, sizes, font_size, rand):
    """Calculate initial canvas size with some randomness"""
    total_w = sum(w for w,_ in sizes) + PADDING*2
    max_h = max(h for _,h in sizes) + PADDING*2 + BOTTOM_PADDING
    
    # Add some random extra space
    extra_w = rand.randint(0, int(total_w * 0.2))  # Up to 20% extra width
    extra_h = rand.randint(0, int(max_h * 0.2))    # Up to 20% extra height
    
    return total_w + extra_w, max_h + extra_h

def render_text_with_distortions(draw, clusters, sizes, to_distort, font, rand):
    """Render text with distortions, returning positions and dimensions"""
    x_cursor, y_base = PADDING, PADDING
    positions = []
    
    for i, (cl, (w_cl, h_cl)) in enumerate(zip(clusters, sizes)):
        if i in to_distort and w_cl >= MIN_CLUSTER_SIZE:
            # Apply transformations with padding
            angle = rand.uniform(*ANGLE_RANGE)
            scale_x = rand.uniform(*SCALE_RANGE)
            scale_y = rand.uniform(*SCALE_RANGE)
            
            # Create padded layer for the character
            pad = int(max(w_cl, h_cl) * DISTORTION_PADDING)
            layer_w = w_cl + pad * 2
            layer_h = h_cl + pad * 2
            layer = Image.new('RGBA', (layer_w, layer_h), (255, 255, 255, 0))
            dl = ImageDraw.Draw(layer)
            dl.text((pad, pad), cl, font=font, fill=(0, 0, 0, 255))
            
            # Apply scaling
            new_w = max(1, int(w_cl * scale_x))
            new_h = max(1, int(h_cl * scale_y))
            layer = layer.resize((new_w + pad * 2, new_h + pad * 2), 
                               resample=Image.BICUBIC)
            
            # Apply rotation
            layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Calculate position
            x_pos = x_cursor + (w_cl - layer.width) // 2
            y_pos = y_base + (h_cl - layer.height) // 2
            
            positions.append({
                'type': 'distorted',
                'x': x_pos,
                'y': y_pos,
                'width': layer.width,
                'height': layer.height,
                'layer': layer,
                'advance': max(w_cl, layer.width // 2)
            })
            
            x_cursor += positions[-1]['advance']
        else:
            # Regular character
            positions.append({
                'type': 'normal',
                'x': x_cursor,
                'y': y_base,
                'width': w_cl,
                'height': h_cl,
                'advance': w_cl
            })
            x_cursor += w_cl
    
    return positions

def text_to_image(text, font_path, out_path, line_seed):
    # deterministic seed per line
    rand = random.Random(line_seed)

    # Scramble spaces
    scrambled = randomly_scramble_spaces(text, REMOVE_PROB, DOT_PROB, line_seed)

    # Pick font & size
    size = rand.randint(FONT_SIZE_MIN, FONT_SIZE_MAX)
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        font = ImageFont.load_default()

    # Grapheme clusters
    clusters = regex.findall(r"\X", scrambled)
    valid_idxs = [i for i, c in enumerate(clusters) if not c.isspace()]
    prop = rand.uniform(DISTORT_PROP_MIN, DISTORT_PROP_MAX)
    n_distort = max(1, int(len(valid_idxs) * prop))
    to_distort = set(rand.sample(valid_idxs, n_distort))

    # Measure clusters
    dummy = Image.new('RGB', (1, 1))
    d0 = ImageDraw.Draw(dummy)
    sizes = []
    for c in clusters:
        try:
            bbox = d0.textbbox((0, 0), c, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = font.getsize(c)
        sizes.append((w, h))

    # Calculate initial canvas size
    img_w, img_h = calculate_initial_canvas_size(clusters, sizes, size, rand)
    canvas = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Render text with distortions
    positions = render_text_with_distortions(draw, clusters, sizes, to_distort, font, rand)

    # Determine final dimensions needed
    final_width = max(pos['x'] + pos['width'] for pos in positions) + PADDING
    final_height = max(pos['y'] + pos['height'] for pos in positions) + BOTTOM_PADDING

    # Create final canvas if needed
    if final_width > img_w or final_height > img_h:
        new_canvas = Image.new('RGB', (final_width, final_height), color=(255, 255, 255))
        new_canvas.paste(canvas, (0, 0))
        canvas = new_canvas
        draw = ImageDraw.Draw(canvas)

    # Draw all elements
    for pos in positions:
        if pos['type'] == 'distorted':
            canvas.paste(pos['layer'], (pos['x'], pos['y']), pos['layer'])
        else:
            c = clusters[positions.index(pos)]
            draw.text((pos['x'], pos['y']), c, font=font, fill=(0, 0, 0))

    # Crop to content if desired (remove excess whitespace)
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    
    canvas.save(out_path)

# Main loop
labels = []
for idx, orig in enumerate(lines, start=1):
    seed_line = None if SEED is None else SEED + idx
    font_path = random.choice(font_paths)
    fname = f"img_{idx:06d}.png"
    out_path = os.path.join(image_dir, fname)
    text_to_image(orig, font_path, out_path, seed_line)
    labels.append({"image_path": out_path, "label": orig})
    print(f"✅ Saved {out_path}")

# Save metadata
labels_path = os.path.join(OUTPUT_DIR, 'labels.json')
with open(labels_path, 'w', encoding='utf-8') as lf:
    json.dump(labels, lf, ensure_ascii=False, indent=2)

print(f"\n🎨 Created {len(labels)} images + metadata at '{labels_path}'")