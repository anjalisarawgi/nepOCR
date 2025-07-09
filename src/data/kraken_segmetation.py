import os
import subprocess

BIN_DIR = "data/oldNepali/raw/images"
SEG_OUT_DIR = "data/oldNepali/processed/jsons"

os.makedirs(SEG_OUT_DIR, exist_ok=True)

def segment_image(image_path, json_path):
    try:
        subprocess.run(["kraken", "-i", image_path, json_path, "segment", "-bl"], check=True)
        print(f"[✓] Segmented: {os.path.basename(image_path)}")
    except subprocess.CalledProcessError:
        print(f"[✗] Failed to segment: {os.path.basename(image_path)}")

def batch_segment():
    for fname in os.listdir(BIN_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(BIN_DIR, fname)
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(SEG_OUT_DIR, json_name)

        if os.path.exists(json_path):
            print(f"[→] Skipping {fname}, already segmented")
            continue

        segment_image(input_path, json_path)

if __name__ == "__main__":
    batch_segment()