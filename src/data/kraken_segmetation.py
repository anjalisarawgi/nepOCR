import os
import subprocess

BIN_DIR = "data/oldNepali/raw/images"
SEG_OUT_DIR = "data/oldNepali/processed/jsons"

os.makedirs(SEG_OUT_DIR, exist_ok=True)

def segment_image(image_path, json_path):
        subprocess.run(["kraken", "-i", image_path, json_path, "segment", "-bl"], check=True)

def batch_segment():
    for fname in os.listdir(BIN_DIR):
        input_path = os.path.join(BIN_DIR, fname)
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(SEG_OUT_DIR, json_name)

        segment_image(input_path, json_path)

if __name__ == "__main__":
    batch_segment()
