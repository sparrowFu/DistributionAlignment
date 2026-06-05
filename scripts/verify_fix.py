"""
Quick verification that the caption fix is loaded.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect
from data.caption_dataset import ImageCaptionDataset

# Check if the fixed code is loaded
source = inspect.getsource(ImageCaptionDataset._get_captions)

print("Checking if fixed code is loaded...")
print()

if "no caption available" in source:
    print("✓ FIXED CODE IS LOADED")
    print()
    print("The fix includes:")
    print("  - Handling numpy arrays correctly")
    print("  - Using placeholder for empty captions")
    print("  - Safe loop that won't infinite loop")
    print()
    print("You can now run:")
    print("  python scripts/test_dataloader.py")
else:
    print("✗ OLD CODE STILL LOADED")
    print()
    print("Please clear cache and restart Python:")
    print("  1. Delete __pycache__ folders")
    print("  2. Run in a new terminal/window")
