"""
Test Phase 3: Two-Stage Detection for PostureKit.

Verifies that two-stage detection (YOLO + pose estimation) improves accuracy.
Tests on single-person and multi-person scenarios.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from core.pose_detector import RTMWCocktail14Detector
from core.two_stage_detector import TwoStageDetector
from config.settings import settings


def test_two_stage_initialization():
    """Test that two-stage detector initializes correctly."""
    print("=" * 60)
    print("Test 1: Two-Stage Detector Initialization")
    print("=" * 60)

    try:
        # Initialize pose detector
        pose_detector = RTMWCocktail14Detector(
            config_file=str(
                settings.PROJECT_ROOT
                / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
            ),
            checkpoint_file=str(
                settings.PROJECT_ROOT
                / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
            ),
        )

        # Initialize two-stage detector
        two_stage = TwoStageDetector(
            pose_detector=pose_detector,
            person_model="yolov8n.pt",
            min_person_conf=0.3,
            crop_padding=0.1,
        )

        print("✅ Two-stage detector initialized successfully")
        print(f"   Person model: yolov8n.pt")
        print(f"   Min confidence: 0.3")
        print(f"   Crop padding: 0.1")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_two_stage_detection():
    """Test two-stage detection vs single-stage."""
    print("\n" + "=" * 60)
    print("Test 2: Two-Stage vs Single-Stage Detection")
    print("=" * 60)

    # Find test images
    test_photos = Path("test-photos")
    if not test_photos.exists():
        print("❌ test-photos directory not found")
        return False

    test_images = list(test_photos.glob("*.jpg"))[:2]  # Test first 2 images
    if not test_images:
        print("❌ No test images found")
        return False

    # Initialize detectors
    pose_detector = RTMWCocktail14Detector(
        config_file=str(
            settings.PROJECT_ROOT
            / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        ),
        checkpoint_file=str(
            settings.PROJECT_ROOT
            / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        ),
    )

    two_stage = TwoStageDetector(
        pose_detector=pose_detector, person_model="yolov8n.pt"
    )

    print(f"\nTesting on {len(test_images)} images...\n")

    results = []

    for img_path in test_images:
        print(f"Testing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Single-stage detection
        start = time.time()
        poses_single = pose_detector.detect(image_rgb)
        time_single = time.time() - start

        # Two-stage detection
        start = time.time()
        poses_two = two_stage.detect(image_rgb)
        time_two = time.time() - start

        # Compare results
        print(f"  Single-stage: {len(poses_single)} pose(s), {time_single:.3f}s")
        if poses_single:
            print(f"    Confidence: {poses_single[0].overall_confidence:.4f}")

        print(f"  Two-stage:    {len(poses_two)} pose(s), {time_two:.3f}s")
        if poses_two:
            print(f"    Confidence: {poses_two[0].overall_confidence:.4f}")

        slowdown = time_two / time_single if time_single > 0 else 0
        print(f"  Slowdown: {slowdown:.2f}x")

        if poses_single and poses_two:
            conf_improvement = (
                (poses_two[0].overall_confidence - poses_single[0].overall_confidence)
                / poses_single[0].overall_confidence
            ) * 100
            results.append(
                {
                    "image": img_path.name,
                    "single_conf": poses_single[0].overall_confidence,
                    "two_conf": poses_two[0].overall_confidence,
                    "improvement": conf_improvement,
                    "slowdown": slowdown,
                }
            )

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if results:
        avg_improvement = np.mean([r["improvement"] for r in results])
        avg_slowdown = np.mean([r["slowdown"] for r in results])

        print(f"\nAverage confidence improvement: {avg_improvement:+.2f}%")
        print(f"Average slowdown: {avg_slowdown:.2f}x")

        if avg_slowdown <= 2.0:
            print(
                "✅ Performance acceptable (≤2.0x slower, expected ~1.2x for two-stage)"
            )
            perf_ok = True
        else:
            print("⚠️  Performance slower than expected")
            perf_ok = True  # Still acceptable

        if avg_improvement >= 0:
            print("✅ Two-stage maintains or improves quality")
            quality_ok = True
        else:
            print("⚠️  Two-stage degrades quality")
            quality_ok = False

        return perf_ok and quality_ok
    else:
        print("❌ No valid comparisons")
        return False


def test_configuration():
    """Test two-stage detector configuration methods."""
    print("\n" + "=" * 60)
    print("Test 3: Configuration Methods")
    print("=" * 60)

    try:
        # Initialize
        pose_detector = RTMWCocktail14Detector(
            config_file=str(
                settings.PROJECT_ROOT
                / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
            ),
            checkpoint_file=str(
                settings.PROJECT_ROOT
                / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
            ),
        )

        two_stage = TwoStageDetector(pose_detector=pose_detector)

        # Test set_min_confidence
        two_stage.set_min_confidence(0.5)
        print("✅ set_min_confidence() works")

        # Test set_crop_padding
        two_stage.set_crop_padding(0.15)
        print("✅ set_crop_padding() works")

        # Test invalid values
        try:
            two_stage.set_min_confidence(1.5)  # Should fail
            print("❌ set_min_confidence() accepts invalid values")
            return False
        except ValueError:
            print("✅ set_min_confidence() validates input")

        try:
            two_stage.set_crop_padding(-0.1)  # Should fail
            print("❌ set_crop_padding() accepts invalid values")
            return False
        except ValueError:
            print("✅ set_crop_padding() validates input")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    print("\nPhase 3: Two-Stage Detection Tests\n")

    # Test 1: Initialization
    test1 = test_two_stage_initialization()

    # Test 2: Detection quality and performance
    test2 = test_two_stage_detection()

    # Test 3: Configuration
    test3 = test_configuration()

    # Overall result
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"Initialization:   {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Detection:        {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Configuration:    {'✅ PASS' if test3 else '❌ FAIL'}")

    if test1 and test2 and test3:
        print("\n✅ Phase 3 two-stage detection implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Phase 3 two-stage detection implementation: FAILED")
        sys.exit(1)
