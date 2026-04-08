"""
Cleans and prepares scanned flower sketches for training.
Handles scanner noise, resizing, and normalization.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset
import random


# ── Configuration ────────────────────────────────────────────────────────────

IMAGE_SIZE = 64          # Training resolution (64×64). Increase to 128 if GPU allows.
DATA_DIR   = "data/raw"  # Put your scanned sketches here
OUT_DIR    = "data/processed"


# ── Noise removal ────────────────────────────────────────────────────────────

def clean_scan(img: Image.Image) -> Image.Image:
    """
    Remove scanner noise and normalize a sketch image.
    - Converts to grayscale
    - Applies a mild median filter to kill speckle noise
    - Inverts so lines are white on black (easier for the GAN)
    - Stretches contrast so the sketch is crisp
    """
    img = img.convert("L")                          # Grayscale
    img = img.filter(ImageFilter.MedianFilter(3))   # Denoise
    img = ImageOps.autocontrast(img, cutoff=2)      # Stretch contrast
    return img


def batch_clean(src_dir: str, dst_dir: str, size: int = IMAGE_SIZE):
    """Clean all images in src_dir and save to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

    print(f"Processing {len(files)} images...")
    for fname in files:
        img = Image.open(os.path.join(src_dir, fname))
        img = clean_scan(img)
        img = img.resize((size, size), Image.LANCZOS)
        img.save(os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png"))

    print(f"Saved {len(files)} cleaned images to '{dst_dir}'")


# ── Dataset with aggressive augmentation ─────────────────────────────────────
# With only ~100 images we need heavy augmentation so the GAN sees variety.

class FlowerSketchDataset(Dataset):
    """
    Loads cleaned sketch images and applies on-the-fly augmentation.
    Each epoch the model sees a different random transformation of every image,
    effectively multiplying your dataset many times over.
    """

    def __init__(self, img_dir: str, image_size: int = IMAGE_SIZE, augment: bool = True, color: bool = False):
        self.img_dir    = img_dir
        self.image_size = image_size
        self.augment    = augment
        self.color      = color      # False = grayscale, True = RGB
        self.paths      = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".png")
        ])

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No .png files found in '{img_dir}'.\n"
                f"Run preprocess.batch_clean() first."
            )

        mode = "color RGB" if color else "grayscale"
        print(f"Dataset: {len(self.paths)} images in '{img_dir}'  [{mode}]")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mode = "RGB" if self.color else "L"
        img  = Image.open(self.paths[idx]).convert(mode)

        if self.augment:
            img = self._augment(img)

        # Convert to tensor in [-1, 1]  (what the GAN expects)
        tensor = TF.to_tensor(img)          # [0, 1]
        tensor = tensor * 2.0 - 1.0        # [-1, 1]
        return tensor

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Applies random transformations. Each call returns a unique variant.
        The transforms are carefully chosen to preserve the sketch look.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)

        # Random rotation up to ±20°  (flowers look fine rotated)
        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, fill=255)   # fill=255 → white background

        # Random scale + crop
        scale = random.uniform(0.85, 1.15)
        new_size = int(self.image_size * scale)
        img = img.resize((new_size, new_size), Image.LANCZOS)
        img = TF.center_crop(img, self.image_size)

        # Slight brightness / contrast jitter (mimics scanner variation)
        img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        img = TF.adjust_contrast(img,   random.uniform(0.85, 1.15))

        # Tiny Gaussian blur (sometimes scans are slightly soft)
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        return img


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Clean your raw scans
    batch_clean(DATA_DIR, OUT_DIR, size=IMAGE_SIZE)

    # Step 2: Sanity-check the dataset
    ds = FlowerSketchDataset(OUT_DIR)
    sample = ds[0]
    print(f"Sample tensor shape : {sample.shape}")   # torch.Size([1, 64, 64])
    print(f"Value range         : [{sample.min():.2f}, {sample.max():.2f}]")  # [-1, 1]


def batch_clean_color(src_dir: str, dst_dir: str, size: int = 128):
    """
    Process color images (your digital drawings) for Phase 2 training.
    Keeps RGB, just resizes and lightly normalises contrast.
    No median filter or grayscale conversion — color must be preserved.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

    print(f"Processing {len(files)} color images...")
    for fname in files:
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png"))

    print(f"Saved {len(files)} color images to '{dst_dir}'")
