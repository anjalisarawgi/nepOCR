import cv2
import numpy as np
import random

def add_random_white_lines(img, num_vertical=2, num_horizontal=2, thickness=2):
    """
    Adds random vertical and horizontal white lines to the image.

    Args:
        img (np.array): Input image in grayscale or RGB.
        num_vertical (int): Number of vertical lines.
        num_horizontal (int): Number of horizontal lines.
        thickness (int): Line thickness in pixels.

    Returns:
        np.array: Augmented image.
    """
    img_aug = img.copy()
    height, width = img.shape[:2]

    # Add vertical white lines
    for _ in range(num_vertical):
        x = random.randint(0, width - 1)
        cv2.line(img_aug, (x, 0), (x, height), (255, 255, 255), thickness)

    # Add horizontal white lines
    for _ in range(num_horizontal):
        y = random.randint(0, height - 1)
        cv2.line(img_aug, (0, y), (width, y), (255, 255, 255), thickness)

    return img_aug

def thin_letters(img):
    """
    Applies morphological erosion to thin the letters.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(inverted, kernel, iterations=2)
    thinned = cv2.bitwise_not(eroded)
    return thinned

def fatten_letters(img):
    """
    Applies morphological dilation to fatten the letters.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    fattened = cv2.bitwise_not(dilated)
    return fattened

def apply_small_random_rotation(img, max_angle=5):
    """
    Rotates the image by a small random angle between -max_angle and +max_angle.

    Args:
        img (np.array): Input image.
        max_angle (float): Maximum absolute rotation angle in degrees.

    Returns:
        np.array: Rotated image.
    """
    height, width = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def add_random_blur_marks(img, num_blurs=3, max_radius=5):
    """
    Adds blurred spots to simulate smudges or pen pressure variations.
    """
    img_blurred = img.copy()
    height, width = img.shape[:2]

    if height < max_radius * 2 or width < max_radius * 2:
        return img_blurred

    for _ in range(num_blurs):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(3, max_radius)

        # Extract patch and apply Gaussian blur
        x1 = max(0, x - radius)
        x2 = min(width, x + radius)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius)

        patch = img_blurred[y1:y2, x1:x2]
        if patch.size > 0:
            blurred_patch = cv2.GaussianBlur(patch, (radius | 1, radius | 1), 0)
            img_blurred[y1:y2, x1:x2] = blurred_patch

    return img_blurred


def add_random_gray_patches(img, num_patches=10, max_size=10):
    """
    Adds random gray patches to mimic ink variations or paper artifacts.
    """
    img_patched = img.copy()
    height, width = img.shape[:2]
    
    if height < max_size or width < max_size:
        return img_patched  # Skip this augmentation if image is too small


    for _ in range(num_patches):
        x = random.randint(0, width - max_size)
        y = random.randint(0, height - max_size)
        w = random.randint(5, max_size)
        h = random.randint(5, max_size)
        gray_value = random.randint(180, 230)  # Lighter gray (faded)

        img_patched[y:y+h, x:x+w] = gray_value

    return img_patched


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

def random_cutouts(img, num_rects=5, max_size=20):
    img_cut = img.copy()
    # h, w = img.shape 
    h, w = img.shape[:2]

    if h < max_size or w < max_size:
        return img_cut

    for _ in range(num_rects):
        x = random.randint(0, w - max_size)
        y = random.randint(0, h - max_size)
        w_rect = random.randint(5, max_size)
        h_rect = random.randint(5, max_size)
        img_cut[y:y+h_rect, x:x+w_rect] = 255  # whiteout
    return img_cut


def grayscale_jitter(img, strength=30):
    jitter = np.random.randint(-strength, strength, size=img.shape, dtype=np.int16)
    jittered = np.clip(img.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    return jittered

def add_ink_speckles(img, num_dots=200, max_radius=5):
    noisy = img.copy()
    h, w = img.shape

    for _ in range(num_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, max_radius)
        intensity = random.randint(0, 60)  # Almost black
        cv2.circle(noisy, (x, y), radius, intensity, -1)
    return noisy


def horizontal_stretch(img, scale_x=1.2):
    """
    Stretches or compresses the image horizontally (width).
    Args:
        img (np.array): Input image.
        scale_x (float): Horizontal scaling factor.
    Returns:
        np.array: Horizontally stretched image.
    """
    h, w = img.shape[:2]
    new_w = int(w * scale_x)
    stretched = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
    return stretched

def vertical_stretch(img, scale_y=1.2):
    """
    Stretches or compresses the image vertically (height).
    Args:
        img (np.array): Input image.
        scale_y (float): Vertical scaling factor.
    Returns:
        np.array: Vertically stretched image.
    """
    h, w = img.shape[:2]
    new_h = int(h * scale_y)
    stretched = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
    return stretched

def blur_image(img, kernel_size=3):
    """
    Applies Gaussian blur to the entire image.
    Args:
        img (np.array): Input image.
        kernel_size (int): Size of the Gaussian blur kernel (must be odd).
    Returns:
        np.array: Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # make sure it's odd
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
    """
    Increases image contrast using CLAHE (adaptive histogram equalization).
    Works on grayscale images or converts color to grayscale first.

    Args:
        img (np.array): Input image.
        clip_limit (float): Contrast limit.
        tile_grid_size (tuple): Size of grid for histogram equalization.

    Returns:
        np.array: Image with increased contrast.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    return enhanced

img = cv2.imread('11_nbhv_textline_1.png', cv2.IMREAD_GRAYSCALE)

def increase_brightness(img, value=30):
    """
    Increases brightness by adding a constant value to all pixels.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # safer in HSV space
    h, s, v = cv2.split(hsv)
    
    # Clip to avoid overflow
    v = np.clip(v + value, 0, 255)
    
    brighter_hsv = cv2.merge((h, s, v))
    brighter_img = cv2.cvtColor(brighter_hsv, cv2.COLOR_HSV2BGR)
    return brighter_img

# combo 1: elastic distortion + blur
def elastic_blur(img): 
    distorted = elastic_distortion(img, alpha=10, sigma=2)
    blurred = blur_image(distorted, kernel_size=5)
    return blurred

# combo 2: brightness + fatten_letters
def brighten_fatten(img):
    bright = increase_brightness(img, value=30)
    fat = fatten_letters(bright)
    return fat

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

    # Enforce minimum shift by avoiding values too close to zero
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

def add_ink_speckles(img, num_dots=50, max_radius=2):
    noisy = img.copy()
    h, w = img.shape[:2]

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _ in range(num_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, max_radius)
        intensity = random.randint(0, 60)  # Almost black
        cv2.circle(noisy, (x, y), radius, intensity, -1)

    return noisy

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
        borderValue=(255, 255, 255)   # <-- white fill
    )

def random_gamma(img, gamma_range=(0.8, 1.2)):
    gamma = random.uniform(*gamma_range)
    inv = 1.0 / gamma
    table = np.array([((i/255.0)**inv)*255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def grid_distortion(img, num_steps=5, distort_limit=0.03):
    h, w = img.shape[:2]
    x_steps = np.linspace(0, w, num_steps+1, dtype=np.int32)
    y_steps = np.linspace(0, h, num_steps+1, dtype=np.int32)
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    for i in range(num_steps):
        for j in range(num_steps):
            dx = random.uniform(-distort_limit, distort_limit)*w/num_steps
            dy = random.uniform(-distort_limit, distort_limit)*h/num_steps
            x1, x2 = x_steps[j], x_steps[j+1]
            y1, y2 = y_steps[i], y_steps[i+1]
            map_x[y1:y2, x1:x2] = np.tile(np.linspace(x1+dx, x2+dx, x2-x1), (y2-y1, 1))
            map_y[y1:y2, x1:x2] = np.tile(np.linspace(y1+dy, y2+dy, y2-y1), (x2-x1, 1)).T
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def brightness_jitter(img, delta=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = np.clip(v + random.randint(-delta, delta), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def contrast_jitter(img, alpha_range=(0.9, 1.1)):
    alpha = random.uniform(*alpha_range)
    # blend img with its mean gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.clip(img.astype(np.float32)*alpha + gray_bgr*(1-alpha), 0,255).astype(np.uint8)

def black_blotch(img, num=3, max_size=10):
    out = img.copy()
    h,w = out.shape[:2]
    for _ in range(num):
        x,y = random.randint(0,w-1), random.randint(0,h-1)
        sz = random.randint(5, max_size)
        cv2.circle(out, (x,y), sz, (0,0,0), -1)
    return out

def motion_blur(img, degree=5, angle=None):
    if angle is None: angle = random.uniform(-45, 45)
    # build kernel
    M = cv2.getRotationMatrix2D((degree/2,degree/2), angle, 1)
    kernel = np.diag(np.ones(degree, dtype=np.float32))
    kernel = cv2.warpAffine(kernel, M, (degree,degree))
    kernel = kernel / degree
    return cv2.filter2D(img, -1, kernel)


def tiny_open(img, kernel_size=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened if img.ndim==2 else cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

def hsv_jitter(img, h_delta=5, s_delta=30, v_delta=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(int) + random.randint(-h_delta, h_delta)) % 180).astype(np.uint8)
    s = np.clip(s.astype(int) + random.randint(-s_delta, s_delta), 0, 255).astype(np.uint8)
    v = np.clip(v.astype(int) + random.randint(-v_delta, v_delta), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)





def channel_dropout(img):
    out = img.copy()
    ch = random.choice([0,1,2])
    out[:,:,ch] = 0
    return out

def tiny_median_blur(img, k=3):
    # k must be odd
    if k % 2 == 0: k += 1
    return cv2.medianBlur(img, k)


def res_jitter(img, scale_range=(0.9,1.1)):
    h,w = img.shape[:2]
    s = random.uniform(*scale_range)
    nh, nw = max(1,int(h*s)), max(1,int(w*s))
    small = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)

def sharpen(img):
    """
    Applies a simple sharpening filter.
    """
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def water_wave(img, strength_range=(1.0, 2.0), wavelength_range=(25, 40)):
    strength = random.uniform(*strength_range)
    wavelength = random.uniform(*wavelength_range)
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x + strength * np.sin(2 * np.pi * y / wavelength)
            map_y[y, x] = y
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if img.ndim == 3 else 255
    )


def fold_crinkle(img, num_lines=1, max_thickness=1, alpha_range=(10, 40)):
    img = img.copy()
    h, w = img.shape[:2]
    
    for _ in range(num_lines):
        x1 = random.randint(0, w - 1)
        y1 = 0
        x2 = random.randint(0, w - 1)
        y2 = h - 1
        thickness = random.randint(1, max_thickness)

        # Subtle white line by blending instead of direct overwrite
        overlay = img.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        alpha = random.uniform(*alpha_range) / 255.0  # blend factor

        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def low_contrast(img, factor_range=(0.7, 0.95)):
    factor = random.uniform(*factor_range)
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return (gray * factor + mean * (1 - factor)).astype(np.uint8)

def add_black_smudges(img, num_smudges=5, max_radius=8):
    img = img.copy()
    h, w = img.shape[:2]
    for _ in range(num_smudges):
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        radius = random.randint(2, max_radius)
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)
    return img


def random_smear(img, max_length=30):
    img = img.copy()
    h, w = img.shape[:2]
    for _ in range(5):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = min(w-1, x1 + random.randint(5, max_length))
        y2 = min(h-1, y1 + random.randint(-5, 5))
        thickness = random.randint(1, 3)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
    return img

def random_black_patches(img, num_patches=5, patch_size=10):
    img = img.copy()
    h, w = img.shape[:2]
    for _ in range(num_patches):
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img[y:y+patch_size, x:x+patch_size] = 0
    return img

import os
from glob import glob
#######
# === Mapping function names to callables ===
AUGMENTATIONS = {
    # 'randomlines': lambda img: add_random_white_lines(img, 2, 2, 4),
    # 'thin': thin_letters, # for nagari
    # 'fatten': fatten_letters, # for nagari
    'rotation': lambda img: apply_small_random_rotation(img, max_angle=3),
    'blurredpatches': lambda img: add_random_blur_marks(img, 5, 3),
    # 'graypatches': lambda img: add_random_gray_patches(img, 20, 20),
    'elastic': lambda img: elastic_distortion(img, alpha=10, sigma=3),
    # 'cutouts': lambda img: random_cutouts(img, 20, 20),
    'jitter': lambda img: grayscale_jitter(img, strength=20), # for nagari
    'hstretch': lambda img: horizontal_stretch(img, scale_x=1.2), # for nagari
    'vstretch': lambda img: vertical_stretch(img, scale_y=1.2), # for nagari
    'blur': lambda img: blur_image(img, kernel_size=5), # for nagari
    # 'multiplicative': lambda img: add_multiplicative_noise(img, (50, 100)),
    # 'contrast': increase_contrast,
    # 'brightness': lambda img: increase_brightness(img, value=50), 
    'elasticblur': elastic_blur, 
    # 'brightfatten': brighten_fatten,
    "perspective": perspective_warp, 
    'sine': lambda img: sine(img, amplitude=5, wavelength=100),
    'horizontal': lambda img: horizontal(img, strength=0.002),
    'jpeg': lambda img: add_jpeg_artifacts(img, quality=20), 
    'shift': lambda img: random_shift(img, min_dx=5, max_dx=20, min_dy=5, max_dy=20),
    'morph': lambda img: random_morph(img, min_kernel=2, max_kernel=4),
    'saltpepper': salt_and_pepper, # upto here for augs 8 
    "shear": lambda img: random_shear(img, shear_range=0.05), 
    'motion_blur': lambda img: motion_blur(img, degree=5), 
    'med_blur': lambda img: tiny_median_blur(img, k=3),
    'gaussian_noise': lambda img: add_gaussian_noise(img, mean=0, std=6),
    'sharpen':            sharpen,
    # 'black_smudges': lambda img: add_black_smudges(img, num_smudges=5, max_radius=10),
    # 'smear':         lambda img: random_smear(img, max_length=25),
    # # 'black_patches': lambda img: random_black_patches(img, num_patches=4, patch_size=12),
}


# === Main processing script ===
def apply_random_augmentations(input_folder, output_folder, num_augmentations=7):
    os.makedirs(output_folder, exist_ok=True)
    # only grab actual image extensions
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for e in exts:
        image_paths += glob(os.path.join(input_folder, e))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, ext   = os.path.splitext(filename)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Could not read {img_path}, skipping.")
            continue

        # save original
        original_out = os.path.join(output_folder, filename)
        cv2.imwrite(original_out, img)

        # random augs...
        chosen_augs = random.sample(list(AUGMENTATIONS.items()), k=num_augmentations)
        for suffix, aug_func in chosen_augs:
            aug_img = aug_func(img.copy())
            out_name = f"{name}_{suffix}.png"
            cv2.imwrite(os.path.join(output_folder, out_name), aug_img)

        print(f"✅ Processed {filename}: {', '.join(s for s,_ in chosen_augs)}")


import json
def augment_labels_json(
    labels_json_path,
    input_base_folder,       # e.g. "oldNepaliDataset/images"
    output_base_folder,      # e.g. "oldNepaliDataset3/images"
    output_json_path,
    num_augmentations=5
):
    os.makedirs(output_base_folder, exist_ok=True)

    # 1️⃣ Load original labels
    with open(labels_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    augmented_records = []
    for rec in records:
        orig_path = rec['image_path']
        # derive full path
        img_path = os.path.join(input_base_folder, os.path.relpath(orig_path, input_base_folder))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️  Couldn’t load {img_path}; skipping.")
            continue

        # 2️⃣ Save the original image into the output folder (optional)
        rel_save_dir = os.path.dirname(os.path.relpath(orig_path, input_base_folder))
        full_save_dir = os.path.join(output_base_folder, rel_save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        orig_filename = os.path.basename(orig_path)
        cv2.imwrite(os.path.join(full_save_dir, orig_filename), img)

        # Keep the original JSON entry, but fix its path to the new folder
        new_orig_rec = dict(rec)
        new_orig_rec['image_path'] = os.path.join(output_base_folder, os.path.relpath(orig_path, input_base_folder))

        
        augmented_records.append(new_orig_rec)

        # 3️⃣ Generate augmentations
        choices = random.sample(list(AUGMENTATIONS.items()), k=num_augmentations)
        for suffix, aug_fn in choices:
            aug_img = aug_fn(img.copy())
            name, ext = os.path.splitext(orig_filename)
            new_filename = f"{name}_{suffix}{ext}"
            out_path = os.path.join(full_save_dir, new_filename)
            cv2.imwrite(out_path, aug_img)

            # 4️⃣ Create a record for this augmented image
            aug_rec = {
                'text': rec['text'],
                # 'page': rec['page'],
                'image_path': os.path.join(output_base_folder, rel_save_dir, new_filename)
            }
            augmented_records.append(aug_rec)

        print(f"✅ {orig_filename} → {[s for s,_ in choices]}")

    # 5️⃣ Dump combined list
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_records, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Augmented JSON with {len(augmented_records)} entries written to {output_json_path}")

# === USAGE ===
if __name__ == '__main__':
    augment_labels_json(
        labels_json_path   = 'data/oldNepali_fullset/labels/labels_train.json',
        input_base_folder  = 'data/oldNepali_fullset/',
        output_base_folder = 'data/oldNepali_fullset_aug8/images',
        output_json_path   = 'data/oldNepali_fullset_aug8/labels.json',
        num_augmentations  = 8
    )
