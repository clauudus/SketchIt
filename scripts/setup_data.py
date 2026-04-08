"""
This must be run ONCE before training the model to:
1. Create the expected folder structure
2. Clean your raw scanned sketches

Put your raw scanned images in  data/raw/  then run:
    python scripts/setup_data.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import batch_clean, IMAGE_SIZE

DIRS = [
    "data/raw",
    "data/processed",
    "output/samples",
    "output/checkpoints",
    "output/generated",
]

print("Creating project structure...")
for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"  OK {d}/")

raw_images = [f for f in os.listdir("data/raw") if f.lower().endswith(
    (".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

if len(raw_images) == 0:
    print("\n No images found in data/raw/")
    print("   Copy your scanned flower sketches there and run this script again.")
else:
    print(f"\nFound {len(raw_images)} raw images. Cleaning...")
    batch_clean("data/raw", "data/processed", size=IMAGE_SIZE)
    print("\nAll done! Now run:  python src/train.py")
