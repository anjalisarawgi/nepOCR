from PIL import Image, ImageDraw, ImageFont
import os, random
import regex  # pip install regex

# ─── 1) CONFIG ─────────────────────────────────────────────────────────────

sentence = "प्राचीन श्लको मा सत्य छ ।"
font_dir = "fonts"
pad      = 10

# pick random font & size
font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir)
              if f.lower().endswith(".ttf")]
if not font_paths:
    raise RuntimeError("No .ttf fonts found in 'fonts/'")
font_path = random.choice(font_paths)
font_size = random.randint(28, 36)
font      = ImageFont.truetype(font_path, font_size)

# ─── 2) CLUSTER & PROPORTIONAL PICK ────────────────────────────────────────

clusters   = regex.findall(r'\X', sentence)
valid_idxs = [i for i, c in enumerate(clusters) if not c.isspace()]

# pick a random proportion between 5% and 30%
prop = random.uniform(0.0, 0.40)
# compute how many clusters to transform (at least one)
n_rot = max(1, int(len(valid_idxs) * prop))
to_transform = random.sample(valid_idxs, n_rot)

print(f"Transforming {n_rot} of {len(valid_idxs)} segments "
      f"({prop*100:.1f}%):", [clusters[i] for i in to_transform])

# ─── 3) PRE-MEASURE ────────────────────────────────────────────────────────

dummy = Image.new("RGB", (1,1))
draw0 = ImageDraw.Draw(dummy)
sizes = [draw0.textbbox((0,0), cl, font=font)[2:] for cl in clusters]
widths, heights = zip(*sizes)
total_w = sum(widths) + pad*2
max_h   = max(heights) + pad*2

# ─── 4) RENDER CANVAS ──────────────────────────────────────────────────────

canvas = Image.new("RGB", (total_w, max_h), (255,255,255))
draw   = ImageDraw.Draw(canvas)
x_cursor = pad

for i, cl in enumerate(clusters):
    w_cl, h_cl = sizes[i]
    y = pad

    if i in to_transform:
        # random angle & scale
        angle   = random.uniform(-15, 15)
        scale_x = random.uniform(0.8, 1.2)
        scale_y = random.uniform(0.8, 1.2)

        layer = Image.new("RGBA", (w_cl, h_cl), (255,255,255,0))
        dl    = ImageDraw.Draw(layer)
        dl.text((0,0), cl, font=font, fill=(0,0,0))

        # apply stretch/compress
        new_w = int(w_cl * scale_x)
        new_h = int(h_cl * scale_y)
        layer = layer.resize((new_w, new_h), resample=Image.BICUBIC)

        # apply rotation
        layer = layer.rotate(angle, expand=True)

        # center over original
        x_off = x_cursor - (layer.width - new_w)//2
        y_off = y       - (layer.height - new_h)//2
        canvas.paste(layer, (int(x_off), int(y_off)), layer)
    else:
        draw.text((x_cursor, y), cl, font=font, fill=(0,0,0))

    x_cursor += w_cl

# ─── 5) SAVE ───────────────────────────────────────────────────────────────

out = "rotated_and_stretched_prop.png"
canvas.save(out)
print(f"✅ Saved: {out}")