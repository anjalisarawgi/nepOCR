import os
import json
import random
import numpy as np
import imageio
import cv2
from PIL import Image
import imgaug.augmenters as iaa
from tqdm import tqdm
import numpy as np
np.bool = bool 
# --- CONFIGURABLE SETTINGS ---
INPUT_DIR = "data/top5_confusion_lines_1000"
OUTPUT_DIR = "data/top5_confusion_lines_1000_vnoisy/images"
LABEL_FILE_IN = "data/top5_confusion_lines_1000/labels.json"
LABEL_FILE_OUT = "data/top5_confusion_lines_1000_vnoisy/labels.json"


# --- THICKNESS CONTROL ---
def apply_random_thickness(image, min_dilate=1, max_dilate=2, min_erode=0, max_erode=1):
    image = np.array(image)
    if random.random() < 0.7:
        ksize = random.randint(min_dilate, max_dilate)
        if ksize > 0:
            kernel = np.ones((ksize, ksize), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
    else:
        ksize = random.randint(min_erode, max_erode)
        if ksize > 0:
            kernel = np.ones((ksize, ksize), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
    return image

# --- STRONG AUGMENTATION PIPELINE ---

# def get_augmenter():
#     return iaa.Sequential([
#         iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.003, 0.01), mode="reflect")),
#         iaa.Sometimes(0.4, iaa.ElasticTransformation(alpha=(1, 2), sigma=(0.5, 0.8), mode="reflect")),
#         iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(5, 10))),  # small ink shake
#         iaa.Sometimes(0.3, iaa.MotionBlur(k=3)),                      # just enough blur
#         iaa.Sometimes(0.3, iaa.Dropout(p=(0.01, 0.02))),                # slight dropout
#         iaa.Sometimes(0.3, iaa.Affine(
#             shear=(-1, 1),
#             rotate=(-1.5, 1.5),
#             scale=(0.97, 1.03),
#             mode="reflect"
#         )),
#         iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.3, 0.6)))
#     ])


# def get_augmenter():
#     return iaa.Sequential([
#         iaa.Sometimes(0.6, iaa.PiecewiseAffine(scale=(0.005, 0.01))),  # was 0.01–0.03
#         iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(1.5, 2.5), sigma=(0.6, 0.9))),  # slightly relaxed

#         # More aggressive Gaussian noise
#         iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(10, 20))),

#         # Slight to moderate motion blur (simulate writing speed)
#         iaa.Sometimes(0.5, iaa.MotionBlur(k=(3, 5))),

#         # More visible dropout to simulate missing ink
#         iaa.Sometimes(0.4, iaa.Dropout(p=(0.02, 0.05))),

#         # Stronger affine transformations
#         iaa.Sometimes(0.4, iaa.Affine(
#             shear=(-2, 2),           # was (-5, 5)
#             rotate=(-2, 2),          # was (-5, 5)
#             scale=(0.97, 1.03),
#             translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
#             mode="reflect"
#         )),

#         # Add gaussian blur to simulate ink bleed
#         iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 1.0))),

#         # Add contrast variations to simulate faded/strong ink
#         iaa.Sometimes(0.4, iaa.LinearContrast((0.7, 1.4))),

#         # Add brightness jitter
#         iaa.Sometimes(0.4, iaa.Multiply((0.8, 1.2)))
#     ])



def get_augmenter():
    return iaa.Sequential([
        iaa.Sometimes(0.6, iaa.PiecewiseAffine(scale=(0.005, 0.015))),
        iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(1.5, 3.0), sigma=(0.6, 1.0))),
        iaa.Sometimes(0.4, iaa.MotionBlur(k=(3, 5))),
        iaa.Sometimes(0.4, iaa.Dropout(p=(0.01, 0.03))),
        iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0.5, 0.9))),
        iaa.Sometimes(0.3, iaa.LinearContrast((0.7, 1.4))),
        iaa.Sometimes(0.3, iaa.Multiply((0.85, 1.15))),
        iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise(scale=0.01)),
        iaa.Sometimes(0.5, iaa.TranslateY(percent=(-0.015, 0.015))),
        iaa.Convolve(np.array([
            [0.01, 0.02, 0.01],
            [0.02, 0.88, 0.02],
            [0.01, 0.02, 0.01]
        ]))
    ])


# --- PROCESS SINGLE IMAGE ---

def apply_variable_thickness(image):
    image = np.array(image)
    mask = np.random.randint(0, 2, image.shape, dtype=np.uint8)
    kernel1 = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel1, iterations=1)
    image[mask == 0] = cv2.erode(image, kernel2, iterations=1)[mask == 0]
    return image

def process_image(img_path, augmenter, save_path):
    image = imageio.imread(img_path)

    # Convert grayscale to RGB (needed for imgaug)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Apply heavy augmentation
    augmented = augmenter(image=image)

    # Apply thickness control
    # thicker = apply_variable_thickness(augmented, min_dilate=1, max_dilate=2)
    thicker = apply_variable_thickness(augmented)

    # Convert to grayscale before saving
    final = cv2.cvtColor(thicker, cv2.COLOR_RGB2GRAY)

    # Save final image
    imageio.imwrite(save_path, final)

# --- BATCH PROCESSING ---
def process_all_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(LABEL_FILE_IN, "r", encoding="utf-8") as f:
        original_labels = json.load(f)

    noisy_labels = []
    augmenter = get_augmenter()

    for entry in tqdm(original_labels, desc="Augmenting images", ncols=80):  # ⬅️ wrapped with tqdm
        input_path = entry["image_path"]
        label = entry["label"]
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            process_image(input_path, augmenter, output_path)
            noisy_labels.append({"image_path": output_path, "text": label})
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

            
    with open(LABEL_FILE_OUT, "w", encoding="utf-8") as f:
        json.dump(noisy_labels, f, indent=2, ensure_ascii=False)

# --- MAIN ENTRY ---
if __name__ == "__main__":
    process_all_images()