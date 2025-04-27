from PIL import Image, ImageDraw, ImageFont
import os
import json
import random

output_dir = "oldNepaliSynthetic_105k"
os.makedirs(output_dir, exist_ok=True)

with open("corpus/oldNepaliSynth_105k.txt", "r", encoding="utf-8") as f:
    text_lines = [line.strip() for line in f if line.strip()]
    # text_lines = text_lines[:20]

font_dir = "fonts"
font_paths = [os.path.join(font_dir, file) for file in os.listdir(font_dir) if file.endswith(".ttf")]
if not font_paths:
    raise Exception("⚠️ No .ttf fonts found in 'fonts/' folder.")

def text_to_image(text, filename, font_path):
    font_size = random.randint(28, 36)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy_img = Image.new('RGB', (1000, 100), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = 20
    bottom_padding = 5
    extra_w = random.randint(0, 40)
    extra_h = random.randint(0, 20)
    img_w = text_width + padding + extra_w
    img_h = text_height + padding + extra_h + bottom_padding

    image = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))
    image.save(filename)

labels = []
image_dir = "oldNepaliSynthetic_105k/images"
os.makedirs(image_dir, exist_ok=True)

for i, line in enumerate(text_lines):
    font_path = random.choice(font_paths)
    img_filename = f"img_{i+1}.png"
    img_path = os.path.join(image_dir, img_filename)
    text_to_image(line, img_path, font_path)
    labels.append({
        "image_path": img_path,
        "label": line
    })
    print(f"Saved {img_path}")

with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("\n All diverse images created and metadata saved to labels.json.")