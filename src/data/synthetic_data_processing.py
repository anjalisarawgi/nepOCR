#!/usr/bin/env python3

import os
import json
import random
import re
import regex
from PIL import Image, ImageDraw, ImageFont


output_dir = "data/oldNepaliSynth_105k"

# to randomize the distortions we make ranges
DISTORT_PROP = (0.05, 0.2)
ANGLE = (-15, 15)
SCALE = (0.9, 1.1)
FONT_SIZE = (28, 36) 
# padding = 20

# os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
font_paths = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.endswith('.ttf')]

with open("corpus/oldNepaliSynth_105k.txt", 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# we do this because we want to match it to how old Nepali data is with less spaces and dots
# 20 percent chance to replace it with dots, 20% for no spaces and 60% to keep it as it is
def scramble_spaces(text, seed=42):
    space_pattern = re.compile(r'(?<=\S) (?=\S)')
    def repl(m):
        return '.' if random.random() < 0.2 else '' if random.random() < 0.2 else ' '
    return space_pattern.sub(repl, re.sub(r'[,\|]+', '', text))


def render_line(text, font_path, seed, out_path):
    rand = random.Random(seed)
    text = scramble_spaces(text, seed) # space - dots - no space
    font_size = rand.randint(*FONT_SIZE) # varying fonts size
    font = ImageFont.truetype(font_path, font_size) # applying fonts with varying sizes 

    # we apply random distorctions - eg character angle rotation / or eg character scaling to try to resemble hadnwritten text
    clusters = regex.findall(r'\X', text) # to fix for devanagari rendering
    valid_idxs = [i for i, c in enumerate(clusters) if not c.isspace()]
    distort_count = max(1, int(len(valid_idxs) * rand.uniform(*DISTORT_PROP))) # random distoritions
    distort_idxs = set(rand.sample(valid_idxs, distort_count))
    dummy_img = Image.new('RGB', (1, 1))
    d = ImageDraw.Draw(dummy_img)

    # randomizing image sizes also - we make sure all text is captured in the image 
    sizes = [d.textbbox((0, 0), c, font=font)[2:] for c in clusters]
    canvas_w = sum(w for w, _ in sizes) + 2 * 20
    canvas_h = max(h for _, h in sizes) + 2 * 20 
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    x_cursor = 20 # also the same as padding 

    for i, (c, (w, h)) in enumerate(zip(clusters, sizes)):
        if i in distort_idxs:
            angle = rand.uniform(*ANGLE)
            scale = rand.uniform(*SCALE)
            layer = Image.new('RGBA', (int(w * 2), int(h * 2)), (255, 255, 255, 0))
            draw_layer = ImageDraw.Draw(layer)
            draw_layer.text((int(w * 0.5), int(h * 0.5)), c, font=font, fill=(0, 0, 0, 255))
            scaled = layer.resize((int(layer.width * scale), int(layer.height * scale)))
            rotated = scaled.rotate(angle, expand=True)
            canvas.paste(rotated, (x_cursor, 20), rotated)
            x_cursor += int(w * 0.8)
        else:
            draw.text((x_cursor, 20), c, font=font, fill=(0, 0, 0))
            x_cursor += w
    canvas = canvas.crop(canvas.getbbox())
    canvas.save(out_path)

labels = []
for idx, line in enumerate(lines, 1):
    seed = 42
    font_path = random.choice(font_paths)
    img_name = f"img_{idx:06d}.png"
    out_path = os.path.join(output_dir, "images", img_name)
    render_line(line, font_path, 42, out_path)
    
    labels.append({'image_path': out_path, 'label': line})
    print(f"Saved {img_name}")

with open(os.path.join(output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("completed")
