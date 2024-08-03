import cv2
import numpy as np
import os
import glob



from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

def run(func, this_iter, desc="Processing"):
    with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
        results = list(
            tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
        )
    return results

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_overexposed(image_path, threshold=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return np.argmax(hist) > threshold


def is_underexposed(image_path, threshold=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return np.argmax(hist) < threshold


def is_low_resolution(image_path, min_width=640, min_height=480):
    image = cv2.imread(image_path)
    if image is None:
        return True
    height, width = image.shape[:2]
    return width < min_width or height < min_height


def has_invalid_annotation(label_path):
    if not os.path.exists(label_path):
        return True
    with open(label_path, 'r') as file:
        lines = file.readlines()
    return len(lines) == 0


def process_images(image_dir, label_dir, min_width=640, min_height=480, blur_threshold=100, exposure_threshold_high=200,
                   exposure_threshold_low=50):
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))

    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        label_path = os.path.join(label_dir, os.path.splitext(base_name)[0] + '.txt')

        if (is_blurry(image_path, blur_threshold) or
                is_overexposed(image_path, exposure_threshold_high) or
                is_underexposed(image_path, exposure_threshold_low) or
                is_low_resolution(image_path, min_width, min_height) or
                has_invalid_annotation(label_path)):
            print(f'Removing {image_path}')
            os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)


if __name__ == "__main__":
    IMAGE_DIR = r'C:\Users\19916\Desktop\1\datasets\trainB-SCI+Filter\images\val'
    LABEL_DIR = r'C:\Users\19916\Desktop\1\datasets\trainB-SCI+Filter\labels\val'
    process_images(IMAGE_DIR, LABEL_DIR)
