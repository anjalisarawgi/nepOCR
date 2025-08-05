import cv2
import numpy as np
import random
import os
from glob import glob
import json

def apply_small_random_rotation(img, max_angle=5):
    height, width = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def add_random_blur_marks(img, num_blurs=3, max_radius=5):
    img_blurred = img.copy()
    height, width = img.shape[:2]

    if height < max_radius * 2 or width < max_radius * 2:
        return img_blurred

    for _ in range(num_blurs):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(3, max_radius)

        # Extracting the patch and apply Gaussian blur
        x1 = max(0, x - radius)
        x2 = min(width, x + radius)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius)

        patch = img_blurred[y1:y2, x1:x2]
        if patch.size > 0:
            blurred_patch = cv2.GaussianBlur(patch, (radius | 1, radius | 1), 0)
            img_blurred[y1:y2, x1:x2] = blurred_patch

    return img_blurred


def elastic_distortion(img, alpha=34, sigma=4):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    random_state = np.random.RandomState(None)
    shape = img.shape

    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return distorted



def grayscale_jitter(img, strength=30):
    jitter = np.random.randint(-strength, strength, size=img.shape, dtype=np.int16)
    jittered = np.clip(img.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    return jittered



def horizontal_stretch(img, scale_x=1.2):
    h, w = img.shape[:2]
    new_w = int(w * scale_x)
    stretched = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
    return stretched

def vertical_stretch(img, scale_y=1.2):
    h, w = img.shape[:2]
    new_h = int(h * scale_y)
    stretched = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
    return stretched

def blur_image(img, kernel_size=3):
    if kernel_size % 2 == 0:
        kernel_size += 1 
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred


def add_gaussian_noise(img, mean=0, std=15):
    gaussian = np.random.normal(mean, std, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + gaussian, 0, 255).astype(np.uint8)
    return noisy

def add_multiplicative_noise(img, scale_range=(0.9, 1.1)):
    noise = np.random.uniform(*scale_range, size=img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) * noise, 0, 255).astype(np.uint8)
    return noisy

def add_jpeg_artifacts(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode('.jpg', img, encode_param)
    dec_img = cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)
    return dec_img

def increase_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    return enhanced

# img = cv2.imread('11_nbhv_textline_1.png', cv2.IMREAD_GRAYSCALE)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    h, s, v = cv2.split(hsv)
    
    # Clip to avoid overflow
    v = np.clip(v + value, 0, 255)
    
    brighter_hsv = cv2.merge((h, s, v))
    brighter_img = cv2.cvtColor(brighter_hsv, cv2.COLOR_HSV2BGR)
    return brighter_img

# elastic distortion + blur
def elastic_blur(img): 
    distorted = elastic_distortion(img, alpha=10, sigma=2)
    blurred = blur_image(distorted, kernel_size=5)
    return blurred


def perspective_warp(img, shift=40):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([[shift, 0], [w - 1 - shift, 0], [0, h - 1], [w - 1, h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

def sine(img, amplitude=5, wavelength=100):
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            offset = amplitude * np.sin(2 * np.pi * i / wavelength)
            map_x[i, j] = j + offset
            map_y[i, j] = i
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def horizontal(img, strength=0.002):
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        offset = int(strength * (i - h / 2) ** 2)
        for j in range(w):
            new_x = j + offset - strength * (h / 2) ** 2 / 2
            map_x[i, j] = min(max(new_x, 0), w - 1)
            map_y[i, j] = i
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))



def add_jpeg_artifacts(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode('.jpg', img, encode_param)
    dec_img = cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)
    return dec_img


def random_shift(img, min_dx=5, max_dx=20, min_dy=5, max_dy=20):
    h, w = img.shape[:2]
    dx = random.choice(
        [random.randint(min_dx, max_dx), -random.randint(min_dx, max_dx)]
    )
    dy = random.choice(
        [random.randint(min_dy, max_dy), -random.randint(min_dy, max_dy)]
    )

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    is_color = len(img.shape) == 3
    border_val = (255, 255, 255) if is_color else 255

    return cv2.warpAffine(img, M, (w, h), borderValue=border_val)


def random_morph(img, min_kernel=2, max_kernel=4):
    op = random.choice([cv2.MORPH_ERODE, cv2.MORPH_DILATE])
    k = random.randint(min_kernel, max_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.morphologyEx(gray, op, kernel)

def salt_and_pepper(img, amount=0.005):
    out = img.copy()
    h, w = img.shape[:2]
    num_salt = np.ceil(amount * h * w * 0.5)
    num_pepper = np.ceil(amount * h * w * 0.5)

    # Salt (white)  
    coords = [np.random.randint(0, i, int(num_salt)) for i in (h, w)]
    out[coords[0], coords[1]] = 255

    # Pepper (black)  
    coords = [np.random.randint(0, i, int(num_pepper)) for i in (h, w)]
    out[coords[0], coords[1]] = 0

    return out


def random_shear(img, shear_range=0.1):
    h, w = img.shape[:2]
    dx = random.uniform(-shear_range, shear_range) * w
    pts1 = np.float32([[0,0], [w,0], [0,h]])
    pts2 = np.float32([[dx,0], [w+dx,0], [0,h]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) 
    )


def motion_blur(img, degree=5, angle=None):
    if angle is None: angle = random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((degree/2,degree/2), angle, 1)
    kernel = np.diag(np.ones(degree, dtype=np.float32))
    kernel = cv2.warpAffine(kernel, M, (degree,degree))
    kernel = kernel / degree
    return cv2.filter2D(img, -1, kernel)



def median_blur(img, k=3):
    # k must be odd
    if k % 2 == 0: k += 1
    return cv2.medianBlur(img, k)

def sharpen(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


AUGMENTATIONS = {
    'rotation': lambda img: apply_small_random_rotation(img, max_angle=3),
    'blurredpatches': lambda img: add_random_blur_marks(img, 5, 3),
    'elastic': lambda img: elastic_distortion(img, alpha=10, sigma=3),
    'jitter': lambda img: grayscale_jitter(img, strength=20), 
    'hstretch': lambda img: horizontal_stretch(img, scale_x=1.2), 
    'vstretch': lambda img: vertical_stretch(img, scale_y=1.2), 
    'blur': lambda img: blur_image(img, kernel_size=5), 
    'elasticblur': elastic_blur, 
    "perspective": perspective_warp, 
    'sine': lambda img: sine(img, amplitude=5, wavelength=100),
    'horizontal': lambda img: horizontal(img, strength=0.002),
    'jpeg': lambda img: add_jpeg_artifacts(img, quality=20), 
    'shift': lambda img: random_shift(img, min_dx=5, max_dx=20, min_dy=5, max_dy=20),
    'morph': lambda img: random_morph(img, min_kernel=2, max_kernel=4),
    'saltpepper': salt_and_pepper,
    "shear": lambda img: random_shear(img, shear_range=0.05), 
    'motion_blur': lambda img: motion_blur(img, degree=5), 
    'med_blur': lambda img: median_blur(img, k=3),
    'gaussian_noise': lambda img: add_gaussian_noise(img, mean=0, std=6),
    'sharpen': sharpen,
}



def apply_random_augmentations(input_folder, output_folder, num_augmentations=7):
    os.makedirs(output_folder, exist_ok=True)
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for e in exts:
        image_paths += glob(os.path.join(input_folder, e))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, ext   = os.path.splitext(filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # save original
        original_out = os.path.join(output_folder, filename)
        cv2.imwrite(original_out, img)

        # save augmentations
        chosen_augs = random.sample(list(AUGMENTATIONS.items()), k=num_augmentations)
        for suffix, aug_func in chosen_augs:
            aug_img = aug_func(img.copy())
            out_name = f"{name}_{suffix}.png"
            cv2.imwrite(os.path.join(output_folder, out_name), aug_img)

        print(f" Processed {filename}: {', '.join(s for s,_ in chosen_augs)}")


# for labels.json
# we cna make this more compact - check later
def augment_labels_json(
    labels_json_path,
    input_base_folder,      
    output_base_folder,     
    output_json_path,
    num_augmentations=5
):
    os.makedirs(output_base_folder, exist_ok=True)

    with open(labels_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    augmented_records = []
    for rec in records:
        orig_path = rec['image_path']
        img_path = os.path.join(input_base_folder, os.path.relpath(orig_path, input_base_folder))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rel_save_dir = os.path.dirname(os.path.relpath(orig_path, input_base_folder))
        full_save_dir = os.path.join(output_base_folder, rel_save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        orig_filename = os.path.basename(orig_path)
        cv2.imwrite(os.path.join(full_save_dir, orig_filename), img)

        new_orig_rec = dict(rec)
        new_orig_rec['image_path'] = os.path.join(output_base_folder, os.path.relpath(orig_path, input_base_folder))

        
        augmented_records.append(new_orig_rec)

        choices = random.sample(list(AUGMENTATIONS.items()), k=num_augmentations)
        for suffix, aug_fn in choices:
            aug_img = aug_fn(img.copy())
            name, ext = os.path.splitext(orig_filename)
            new_filename = f"{name}_{suffix}{ext}"
            out_path = os.path.join(full_save_dir, new_filename)
            cv2.imwrite(out_path, aug_img)

            aug_rec = {
                'text': rec['text'],
                # 'page': rec['page'],
                'image_path': os.path.join(output_base_folder, rel_save_dir, new_filename)
            }
            augmented_records.append(aug_rec)

        # print(f"{orig_filename} : {[s for s,_ in choices]}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_records, f, ensure_ascii=False, indent=2)

    
    print("complete")


if __name__ == '__main__':
    augment_labels_json(
        labels_json_path   = 'data/oldNepali_fullset/labels_normalized/labels_train.json',
        input_base_folder  = 'data/oldNepali_fullset/',
        output_base_folder = 'data/oldNepali_fullset_aug12/images',
        output_json_path   = 'data/oldNepali_fullset_aug12/labels.json',
        num_augmentations  = 12
    )


