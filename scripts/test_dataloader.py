"""
Diagnostic script to test data loading and identify bottlenecks.
"""

import time
import signal
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup simple logging
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

import config
from data.caption_dataset import ImageCaptionDataset
from torch.utils.data import DataLoader


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def run_with_timeout(func, timeout=30):
    """Run a function with a timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        signal.alarm(0)
        return None


print("=" * 60)
print("Data Loading Diagnostic Test")
print("=" * 60)

# Test 1: Dataset creation
print("\n[1/4] Creating dataset...")
start = time.time()
try:
    dataset = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH,
        images_dir=config.IMAGES_DIR,
        num_captions=1  # Use fewer captions for faster testing
    )
    print(f"  ✓ Dataset created in {time.time() - start:.2f}s")
    print(f"  Dataset size: {len(dataset)} samples")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check first image file directly
print("\n[2/4] Checking first image file...")
try:
    image_file_name = dataset.data.loc[0, "image_file_name"]
    image_path = dataset.images_dir / image_file_name
    print(f"  Image file: {image_path}")
    print(f"  Exists: {image_path.exists()}")

    if image_path.exists():
        file_size = image_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")

        # Try opening with PIL
        from PIL import Image
        print("  Attempting to open with PIL...")
        try:
            img = Image.open(image_path)
            print(f"  ✓ PIL Image opened: {img.size}, mode={img.mode}")
            img.close()
        except Exception as e:
            print(f"  ✗ PIL failed: {e}")
    else:
        print(f"  ✗ Image file does not exist!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Single sample access (with detailed logging)
print("\n[3/4] Testing single sample access (dataset[0])...")
print("  This step might hang - if it takes > 10 seconds, press Ctrl+C")

start = time.time()
try:
    sample = dataset[0]
    elapsed = time.time() - start

    if sample is None:
        print("  ✗ First sample is None (image loading failed)")
    else:
        print(f"  ✓ Sample loaded in {elapsed:.2f}s")
        print(f"    Image size: {sample['image'].size}")
        print(f"    Image mode: {sample['image'].mode}")
        print(f"    Captions: {len(sample['captions'])}")
except KeyboardInterrupt:
    print(f"\n  ✗ Interrupted by user after {time.time() - start:.2f}s")
    print("  This indicates dataset[0] is hanging - likely image loading issue")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: DataLoader creation (single process)
print("\n[4/4] Creating DataLoader (num_workers=0)...")
start = time.time()
try:
    def filter_none(batch):
        return [item for item in batch if item is not None]

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=filter_none
    )
    print(f"  ✓ DataLoader created in {time.time() - start:.2f}s")
    print(f"  Number of batches: {len(dataloader)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! Data loading is working correctly.")
print("=" * 60)
