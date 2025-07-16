import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_plant_images(folder_path, image_size=(128, 128), max_images=1200):
    images = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if idx >= max_images:
            break
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.
            images.append(img)
    return np.array(images)

def load_plant_images_multiple(folder_paths, image_size=(128, 128), max_images=1200):
    """
    Load images from multiple folders and combine them into a single dataset.
    
    Args:
        folder_paths: List of folder paths or tuple of folder paths
        image_size: Target size for resizing images
        max_images: Maximum number of images to load from each folder
    
    Returns:
        numpy array of combined images
    """
    all_images = []
    images_per_folder = max_images // len(folder_paths)  # Distribute images evenly
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping...")
            continue
            
        folder_images = []
        for idx, filename in enumerate(os.listdir(folder_path)):
            if idx >= images_per_folder:
                break
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.
                folder_images.append(img)
        
        print(f"Loaded {len(folder_images)} images from {folder_path}")
        all_images.extend(folder_images)
    
    print(f"Total images loaded: {len(all_images)}")
    return np.array(all_images)

def add_salt_pepper_noise_rgb(image, amount=0.05):
    noisy = image.copy()
    h, w, c = noisy.shape
    num_salt = int(amount * h * w / 2)
    num_pepper = int(amount * h * w / 2)

    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    for i in range(c):
        noisy[coords[0], coords[1], i] = 1

    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    for i in range(c):
        noisy[coords[0], coords[1], i] = 0

    return noisy

def evaluate_denoising(clean, denoised):
    psnr_vals, ssim_vals = [], []
    for i in range(len(clean)):
        p = psnr(clean[i], denoised[i], data_range=1.0)
        s = ssim(clean[i], denoised[i], channel_axis=-1, data_range=1.0)
        psnr_vals.append(p)
        ssim_vals.append(s)
    return np.mean(psnr_vals), np.mean(ssim_vals)

def apply_median_filter_rgb(images, ksize=3):
    filtered_images = []
    for img in images:
        filtered = np.zeros_like(img)
        for c in range(3):  # channel R, G, B
            filtered[:, :, c] = cv2.medianBlur((img[:, :, c] * 255).astype(np.uint8), ksize)
        filtered_images.append(filtered.astype(np.float32) / 255.)
    return np.array(filtered_images)

def add_gaussian_noise(image, mean=0.0, var=0.01):
    """
    Menambahkan derau Gaussian ke gambar.
    """
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)