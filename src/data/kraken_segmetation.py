import os
import subprocess


# os.makedirs("data/oldNepali/processed/jsons", exist_ok=True)

# def segment_image(image_path, json_path):
#         subprocess.run(["kraken", "-i", image_path, json_path, "segment", "-bl"], check=True)

def segmenting_image():
    for fname in os.listdir("data/oldNepali/raw/images"):
        input_path = os.path.join(BIN_DIR, fname)
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(SEG_OUT_DIR, json_name)

        subprocess.run(["kraken", "-i", input_path, json_path, "segment", "-bl"], check=True)
        # segment_image(input_path, json_path)

if __name__ == "__main__":
    segmenting_image()
