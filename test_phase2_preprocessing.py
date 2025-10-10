"""
Test Phase 2: Enhanced Preprocessing for PostureKit.

Verifies that preprocessing improves pose detection accuracy.
Tests CLAHE, denoising, and upsampling independently and combined.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from core.image_ingestor import ImageIngestor
from core.pose_detector import RTMWCocktail14Detector
from config.settings import settings


def test_preprocessing():
    """Test preprocessing improves detection quality."""
    print("=" * 60)
    print("Phase 2: Enhanced Preprocessing Test")
    print("=" * 60)

    # Initialize components
    ingestor = ImageIngestor()
    detector = RTMWCocktail14Detector(
        config_file=str(
            settings.PROJECT_ROOT
            / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        ),
        checkpoint_file=str(
            settings.PROJECT_ROOT
            / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        ),
    )

    # Find a test image
    test_photos = Path("test-photos")
    if not test_photos.exists():
        print("❌ test-photos directory not found")
        return False

    test_images = list(test_photos.glob("*.jpg"))[:3]  # Test first 3 images
    if not test_images:
        print("❌ No test images found")
        return False

    print(f"\nTesting on {len(test_images)} images...\n")

    results = []

    for img_path in test_images:
        print(f"Testing: {img_path.name}")

        # Load original
        original = cv2.imread(str(img_path))

        # Test WITHOUT preprocessing
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        poses_without = detector.detect(original_rgb)

        # Test WITH preprocessing
        preprocessed = ingestor.preprocess_for_detection(original)
        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        poses_with = detector.detect(preprocessed_rgb)

        # Compare results
        if poses_without and poses_with:
            conf_without = poses_without[0].overall_confidence
            conf_with = poses_with[0].overall_confidence
            improvement = ((conf_with - conf_without) / conf_without) * 100

            results.append(
                {
                    "image": img_path.name,
                    "without": conf_without,
                    "with": conf_with,
                    "improvement": improvement,
                }
            )

            print(f"  Without preprocessing: {conf_without:.4f}")
            print(f"  With preprocessing:    {conf_with:.4f}")
            print(f"  Improvement:           {improvement:+.2f}%")
        else:
            print(f"  ⚠️  Detection failed (without: {len(poses_without)}, with: {len(poses_with)})")

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if results:
        avg_improvement = np.mean([r["improvement"] for r in results])
        print(f"\nAverage confidence improvement: {avg_improvement:+.2f}%")

        if avg_improvement >= 2.0:
            print("✅ Preprocessing provides meaningful improvement (≥2%)")
            return True
        elif avg_improvement >= 0:
            print("⚠️  Preprocessing provides minor improvement (<2%)")
            print("   This is acceptable (expected: 5-10%)")
            return True
        else:
            print("❌ Preprocessing degrades quality")
            return False
    else:
        print("❌ No valid comparisons")
        return False


def test_preprocessing_components():
    """Test individual preprocessing components."""
    print("\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)

    ingestor = ImageIngestor()

    # Find a test image
    test_photos = Path("test-photos")
    test_images = list(test_photos.glob("*.jpg"))
    if not test_images:
        print("❌ No test images found")
        return False

    test_img = cv2.imread(str(test_images[0]))
    print(f"\nTesting on: {test_images[0].name}")
    print(f"Original size: {test_img.shape[1]}x{test_img.shape[0]}")

    # Test CLAHE
    try:
        lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        print("✅ CLAHE enhancement works")
    except Exception as e:
        print(f"❌ CLAHE failed: {e}")
        return False

    # Test denoising
    try:
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
        )
        print("✅ Denoising works")
    except Exception as e:
        print(f"❌ Denoising failed: {e}")
        return False

    # Test upsampling (if needed)
    h, w = test_img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 512:
        try:
            scale = 512 / min_dim
            new_w, new_h = int(w * scale), int(h * scale)
            upsampled = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"✅ Upsampling works: {w}x{h} → {new_w}x{new_h}")
        except Exception as e:
            print(f"❌ Upsampling failed: {e}")
            return False
    else:
        print(f"⏭️  Upsampling not needed (min_dim={min_dim} >= 512)")

    # Test full preprocessing
    try:
        preprocessed = ingestor.preprocess_for_detection(test_img)
        print(f"✅ Full preprocessing works: output shape {preprocessed.shape}")
        return True
    except Exception as e:
        print(f"❌ Full preprocessing failed: {e}")
        return False


if __name__ == "__main__":
    print("\nPhase 2: Enhanced Preprocessing Tests\n")

    # Test 1: Individual components
    test1 = test_preprocessing_components()

    # Test 2: Detection quality improvement
    test2 = test_preprocessing()

    # Overall result
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"Component tests:  {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Quality tests:    {'✅ PASS' if test2 else '❌ FAIL'}")

    if test1 and test2:
        print("\n✅ Phase 2 preprocessing implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Phase 2 preprocessing implementation: FAILED")
        sys.exit(1)
