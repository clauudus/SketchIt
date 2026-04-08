"""
Prepares the flower photo dataset for Model 2 training.

Put your flower photos in data/photos_raw/ then run:
    python scripts/setup_photos.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import batch_clean_color

DIRS = [
    "data/photos_raw",
    "data/processed_photos",
    "output/photos/samples",
    "output/photos/checkpoints",
    "output/colorizer/samples",
    "output/colorizer/generated",
]

print("Setting up photo dataset structure...")
for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"  OK {d}/")

photos = [f for f in os.listdir("data/photos_raw") if f.lower().endswith(
    (".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

if len(photos) == 0:
    print("\n No photos found in data/photos_raw/")

else:
    print(f"\nFound {len(photos)} photos. Processing to 64×64 RGB...")
    batch_clean_color("data/photos_raw", "data/processed_photos", size=64)
    print("\nReady! Now run:")
    print("  python src/train_photos.py")
    print("\nThen after training, train the colorizer:")
    print("  python src/colorize.py --train")
    print("\nAnd finally, colorize a generated sketch:")
    print("  python src/colorize.py --sketch_seed 42")
