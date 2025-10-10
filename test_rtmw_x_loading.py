"""Quick test to verify RTMW-X model loading."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.pose_detector import RTMWCocktail14Detector
from config.settings import settings
import cv2
import numpy as np

print("Testing RTMW-X model loading...\n")

# Test RTMW-L (should work)
print("1. Loading RTMW-L...")
try:
    rtmw_l = RTMWCocktail14Detector(
        config_file=str(settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"),
        checkpoint_file=str(settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth")
    )
    print("✅ RTMW-L loaded successfully\n")
except Exception as e:
    print(f"❌ RTMW-L failed: {e}\n")

# Test RTMW-X (the key test)
print("2. Loading RTMW-X...")
try:
    rtmw_x = RTMWCocktail14Detector(
        config_file=str(settings.PROJECT_ROOT / "data/models/rtmw-x_8xb320-270e_cocktail14-384x288.py"),
        checkpoint_file=str(settings.PROJECT_ROOT / "data/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth")
    )
    print("✅ RTMW-X loaded successfully\n")

    # Try detection
    print("3. Testing detection on dummy image...")
    dummy_image = np.random.randint(0, 255, (384, 288, 3), dtype=np.uint8)
    poses = rtmw_x.detect(dummy_image)
    print(f"✅ Detection works: {len(poses)} poses detected\n")

    print("=" * 60)
    print("SUCCESS: RTMW-X is fully functional!")
    print("=" * 60)

except Exception as e:
    print(f"❌ RTMW-X failed: {e}\n")
    import traceback
    traceback.print_exc()
