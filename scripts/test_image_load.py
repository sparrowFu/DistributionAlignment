"""
Simple test to check if images can be loaded.
Run this to identify problematic images.
"""

import time
from pathlib import Path
from PIL import Image
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

print("=" * 60)
print("Image Loading Test")
print("=" * 60)

# Load parquet file
print("\nLoading captions file...")
df = pd.read_parquet(config.CAPTIONS_PATH)
print(f"Loaded {len(df)} samples")

# Test first 10 images
print("\nTesting first 10 images:")

for idx in range(min(10, len(df))):
    image_file_name = df.loc[idx, "image_file_name"]
    image_path = config.IMAGES_DIR / image_file_name

    print(f"\n[{idx+1}/10] Testing: {image_file_name}")

    if not image_path.exists():
        print(f"  ✗ File not found: {image_path}")
        continue

    # Check file size
    file_size = image_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  Size: {file_size:.2f} MB")

    # Try to open image
    start = time.time()
    try:
        img = Image.open(image_path)
        elapsed = time.time() - start

        print(f"  ✓ Opened in {elapsed:.3f}s - {img.size} {img.mode}")

        # Try to convert to RGB
        if img.mode != "RGB":
            start = time.time()
            img = img.convert("RGB")
            elapsed = time.time() - start
            print(f"  ✓ Converted to RGB in {elapsed:.3f}s")

        img.close()

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)
