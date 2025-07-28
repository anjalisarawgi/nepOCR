#!/usr/bin/env python3

import os
import json
import random
import re
import regex
from PIL import Image, ImageDraw, ImageFont

TEXT_FILE = "corpus/oldNepaliSynth_105k.txt"
OUTPUT_DIR = "data/oldNepaliSynth_105k"
FONT_DIR = "fonts"
IMAGE_SUBDIR = "images"
SEED = 42

REMOVE_PROB = 0.2
DOT_PROB = 0.2
DISTORT_PROP = (0.05, 0.2)
ANGLE = (-15, 15)
SCALE = (0.9, 1.1)
FONT_SIZE = (28, 36)
PADDING = 20

os.makedirs(os.path.join(OUTPUT_DIR, IMAGE_SUBDIR), exist_ok=True)
font_paths = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]
assert font_paths, f"No .ttf fonts found in '{FONT_DIR}'"

with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f if ln.strip()]

space_pattern = re.compile(r'(?<=\S) (?=\S)')

def scramble_spaces(text, seed=None):
    if seed is not None:
        random.seed(seed)
    def repl(m):
        return '.' if random.random() < DOT_PROB else '' if random.random() < REMOVE_PROB else ' '
    return space_pattern.sub(repl, re.sub(r'[,\|]+', '', text))

def render_line(text, font_path, seed, out_path):
    rand = random.Random(seed)
    text = scramble_spaces(text, seed)
    font_size = rand.randint(*FONT_SIZE)
    font = ImageFont.truetype(font_path, font_size)
    clusters = regex.findall(r'\X', text)

    valid_idxs = [i for i, c in enumerate(clusters) if not c.isspace()]
    distort_count = max(1, int(len(valid_idxs) * rand.uniform(*DISTORT_PROP)))
    distort_idxs = set(rand.sample(valid_idxs, distort_count))
    dummy_img = Image.new('RGB', (1, 1))
    d = ImageDraw.Draw(dummy_img)

    sizes = [d.textbbox((0, 0), c, font=font)[2:] for c in clusters]
    canvas_w = sum(w for w, _ in sizes) + 2 * PADDING
    canvas_h = max(h for _, h in sizes) + 2 * PADDING
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    x_cursor = PADDING

    for i, (c, (w, h)) in enumerate(zip(clusters, sizes)):
        if i in distort_idxs:
            angle = rand.uniform(*ANGLE)
            scale = rand.uniform(*SCALE)
            layer = Image.new('RGBA', (int(w * 2), int(h * 2)), (255, 255, 255, 0))
            draw_layer = ImageDraw.Draw(layer)
            draw_layer.text((int(w * 0.5), int(h * 0.5)), c, font=font, fill=(0, 0, 0, 255))
            scaled = layer.resize((int(layer.width * scale), int(layer.height * scale)))
            rotated = scaled.rotate(angle, expand=True)
            canvas.paste(rotated, (x_cursor, PADDING), rotated)
            x_cursor += int(w * 0.8)
        else:
            draw.text((x_cursor, PADDING), c, font=font, fill=(0, 0, 0))
            x_cursor += w
    canvas = canvas.crop(canvas.getbbox())
    canvas.save(out_path)

labels = []
for idx, line in enumerate(lines, 1):
    seed = None if SEED is None else SEED + idx
    font_path = random.choice(font_paths)
    img_name = f"img_{idx:06d}.png"
    out_path = os.path.join(OUTPUT_DIR, IMAGE_SUBDIR, img_name)
    render_line(line, font_path, seed, out_path)
    labels.append({'image_path': out_path, 'label': line})
    print(f"Saved {img_name}")

with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print(f"Created {len(labels)} images + metadata at '{OUTPUT_DIR}/labels.json'")