import cv2
import numpy as np
import random
import os
import json
from glob import glob


# all augmentation functions
def grayscale_jitter(img, strength=30):
    jitter = np.random.randint(-strength, strength, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

def horizontal_stretch(img, scale_x=1.2):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale_x), h), interpolation=cv2.INTER_LINEAR)

def vertical_stretch(img, scale_y=1.2):
    h, w = img.shape[:2]
    return cv2.resize(img, (w, int(h * scale_y)), interpolation=cv2.INTER_LINEAR)

def blur_image(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# def add_speckle_noise(img):
#     noise = np.random.normal(0, 0.2, img.shape)
#     noisy = img + img * noise
#     return np.clip(noisy, 0, 255).astype(np.uint8)

def add_stripe_noise(img, thickness=1, intensity=10):
    noisy = img.copy()
    h, w = noisy.shape[:2]

    for i in range(0, h, random.randint(20, 50)):
        color = random.randint(200 - intensity, 255)
        line_color = color if len(noisy.shape) == 2 else (color, color, color)
        cv2.line(noisy, (0, i), (w, i), line_color, thickness)
    return noisy

def add_jpeg_artifacts(img, quality=20):
    _, enc_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)

def reduce_contrast(img, factor=0.75):
    img = img.astype(np.float32)
    mean = np.mean(img)
    reduced = mean + factor * (img - mean)
    return np.clip(reduced, 0, 255).astype(np.uint8)

def rotate_image(img, angle=15):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def thin_letters(img, kernel_size=3):
    inverted = 255 - img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(inverted, kernel, iterations=1)
    return 255 - eroded

def fatten_letters(img, kernel_size=3):
    inverted = 255 - img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    return 255 - dilated

AUGMENTATIONS = {
    'jitter': lambda img: grayscale_jitter(img, strength=random.randint(10, 30)), 
    'hstretch': lambda img: horizontal_stretch(img, scale_x=random.uniform(1.1, 1.3)), 
    'vstretch': lambda img: vertical_stretch(img, scale_y=random.uniform(1.1, 1.3)),   
    'blur': lambda img: blur_image(img, kernel_size=random.choice([3, 5, 7])),  
    'jpeg': lambda img: add_jpeg_artifacts(img, quality=random.randint(15, 30)),
    'gaussiannoise': lambda img: add_gaussian_noise(img, std=random.randint(5, 15)), 
    # 'specklenoise': add_speckle_noise,
    # 'stripenoise': lambda img: add_stripe_noise(img, thickness=1, intensity=10),
    # 'lowcontrast': reduce_contrast, 
    'rotate': lambda img: rotate_image(img, angle=random.uniform(-3, 3)),  
    'thin': lambda img: thin_letters(img, kernel_size=random.choice([2, 3, 4])), 
    'fat': lambda img: fatten_letters(img, kernel_size=random.choice([2, 3, 4])),
}


def augment_and_create_labels(input_folder, output_folder, original_label_path, final_label_path, num_augmentations=3):
    os.makedirs(output_folder, exist_ok=True)
    # loading original labels
    with open(original_label_path, 'r', encoding='utf-8') as f:
        original_labels = json.load(f)
    filename_to_text = {
        os.path.basename(entry['image_path']): entry['text']
        for entry in original_labels
    }

    image_paths = glob(os.path.join(input_folder, '*'))
    new_labels = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # saving the original image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)
        new_labels.append({
            "image_path": save_path,
            "text": filename_to_text.get(filename, "")
        })

        # applying augmentations
        chosen_augs = random.sample(list(AUGMENTATIONS.items()), k=num_augmentations)
        for suffix, aug_func in chosen_augs:
            aug_img = aug_func(img.copy())
            aug_filename = f"{name}_{suffix}.png"
            aug_save_path = os.path.join(output_folder, aug_filename)
            cv2.imwrite(aug_save_path, aug_img)

            new_labels.append({
                "image_path": aug_save_path,
                "text": filename_to_text.get(filename, "")
            })

        print(f"✅ Processed {filename} with: {', '.join(name for name, _ in chosen_augs)}")

    # saving new labels
    for label in new_labels:
        label['image_path'] = os.path.relpath(label['image_path'], os.path.dirname(final_label_path))

    with open(final_label_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=4, ensure_ascii=False)

    print(f"All labels saved at: {final_label_path}")



if __name__ == "__main__":
    input_dir = "data/nagari/binarized/test/images"
    output_dir = "data/nagari/binarized/augmented_test/images"
    original_label_path = "data/nagari/binarized/test/labels_processed.json"
    final_label_path = "data/nagari/binarized/augmented_test/labels.json"

    augment_and_create_labels(input_dir, output_dir, original_label_path, final_label_path, num_augmentations=6)