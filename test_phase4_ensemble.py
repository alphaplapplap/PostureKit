"""
Test Phase 4: Ensemble Detection for PostureKit.

Verifies that ensemble detection (multiple models with fusion) improves accuracy.
Tests both weighted_average and confidence_weighted fusion methods.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from core.ensemble_detector import EnsembleDetector, EnsembleConfig
from core.pose_detector import RTMWCocktail14Detector
from config.settings import settings


def test_ensemble_initialization():
    """Test that ensemble detector initializes with 2 models."""
    print("=" * 60)
    print("Test 1: Ensemble Detector Initialization")
    print("=" * 60)

    try:
        # Model paths
        rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        rtmw_m_config = settings.PROJECT_ROOT / "data/models/rtmw-m_8xb1024-270e_cocktail14-256x192.py"
        rtmw_m_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-m_simcc-cocktail14_270e-256x192.pth"

        # Check files exist
        if not rtmw_l_config.exists():
            print(f"❌ RTMW-L config not found: {rtmw_l_config}")
            return False
        if not rtmw_l_checkpoint.exists():
            print(f"❌ RTMW-L checkpoint not found: {rtmw_l_checkpoint}")
            return False
        if not rtmw_m_config.exists():
            print(f"❌ RTMW-M config not found: {rtmw_m_config}")
            return False
        if not rtmw_m_checkpoint.exists():
            print(f"❌ RTMW-M checkpoint not found: {rtmw_m_checkpoint}")
            return False

        # Initialize ensemble
        config = EnsembleConfig(
            models=[
                {
                    "config": str(rtmw_l_config),
                    "checkpoint": str(rtmw_l_checkpoint),
                    "weight": 1.0,
                },
                {
                    "config": str(rtmw_m_config),
                    "checkpoint": str(rtmw_m_checkpoint),
                    "weight": 1.0,
                },
            ],
            fusion_method="confidence_weighted",
        )

        ensemble = EnsembleDetector(config)

        print(f"✅ Ensemble initialized successfully")
        print(f"   Model count: {ensemble.get_model_count()}")
        print(f"   Total weight: {ensemble.get_total_weight():.1f}")
        print(f"   Fusion method: {ensemble.config.fusion_method}")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ensemble_detection():
    """Test ensemble detection vs single-stage."""
    print("\n" + "=" * 60)
    print("Test 2: Ensemble vs Single-Stage Detection")
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

    # Model paths
    rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    rtmw_m_config = settings.PROJECT_ROOT / "data/models/rtmw-m_8xb1024-270e_cocktail14-256x192.py"
    rtmw_m_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-m_simcc-cocktail14_270e-256x192.pth"

    # Initialize single-stage detector
    single_stage = RTMWCocktail14Detector(
        config_file=str(rtmw_l_config), checkpoint_file=str(rtmw_l_checkpoint)
    )

    # Initialize ensemble
    config = EnsembleConfig(
        models=[
            {
                "config": str(rtmw_l_config),
                "checkpoint": str(rtmw_l_checkpoint),
                "weight": 1.0,
            },
            {
                "config": str(rtmw_m_config),
                "checkpoint": str(rtmw_m_checkpoint),
                "weight": 1.0,
            },
        ],
        fusion_method="confidence_weighted",
    )
    ensemble = EnsembleDetector(config)

    print(f"\nTesting on {len(test_images)} images...\n")

    results = []

    for img_path in test_images:
        print(f"Testing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Single-stage detection
        start = time.time()
        poses_single = single_stage.detect(image_rgb)
        time_single = time.time() - start

        # Ensemble detection
        start = time.time()
        poses_ensemble = ensemble.detect(image_rgb)
        time_ensemble = time.time() - start

        # Compare results
        print(f"  Single-stage: {len(poses_single)} pose(s), {time_single:.3f}s")
        if poses_single:
            print(f"    Confidence: {poses_single[0].overall_confidence:.4f}")

        print(f"  Ensemble:     {len(poses_ensemble)} pose(s), {time_ensemble:.3f}s")
        if poses_ensemble:
            print(f"    Confidence: {poses_ensemble[0].overall_confidence:.4f}")

        slowdown = time_ensemble / time_single if time_single > 0 else 0
        print(f"  Slowdown: {slowdown:.2f}x")

        if poses_single and poses_ensemble:
            conf_improvement = (
                (
                    poses_ensemble[0].overall_confidence
                    - poses_single[0].overall_confidence
                )
                / poses_single[0].overall_confidence
            ) * 100
            results.append(
                {
                    "image": img_path.name,
                    "single_conf": poses_single[0].overall_confidence,
                    "ensemble_conf": poses_ensemble[0].overall_confidence,
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

        if avg_slowdown <= 3.0:
            print(
                "✅ Performance acceptable (≤3.0x slower, expected ~2x for ensemble)"
            )
            perf_ok = True
        else:
            print("⚠️  Performance slower than expected")
            perf_ok = False

        if avg_improvement >= 0:
            print("✅ Ensemble maintains or improves quality")
            quality_ok = True
        else:
            print("⚠️  Ensemble degrades quality")
            quality_ok = False

        return perf_ok and quality_ok
    else:
        print("❌ No valid comparisons")
        return False


def test_fusion_methods():
    """Test both fusion methods."""
    print("\n" + "=" * 60)
    print("Test 3: Fusion Methods")
    print("=" * 60)

    try:
        # Model paths
        rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        rtmw_m_config = settings.PROJECT_ROOT / "data/models/rtmw-m_8xb1024-270e_cocktail14-256x192.py"
        rtmw_m_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-m_simcc-cocktail14_270e-256x192.pth"

        # Create ensemble
        config = EnsembleConfig(
            models=[
                {
                    "config": str(rtmw_l_config),
                    "checkpoint": str(rtmw_l_checkpoint),
                    "weight": 1.0,
                },
                {
                    "config": str(rtmw_m_config),
                    "checkpoint": str(rtmw_m_checkpoint),
                    "weight": 1.0,
                },
            ],
            fusion_method="weighted_average",
        )
        ensemble = EnsembleDetector(config)

        # Test changing fusion method
        print("✅ Initial method: weighted_average")

        ensemble.set_fusion_method("confidence_weighted")
        print("✅ Changed to: confidence_weighted")

        # Test invalid method
        try:
            ensemble.set_fusion_method("invalid_method")
            print("❌ set_fusion_method() accepts invalid methods")
            return False
        except ValueError:
            print("✅ set_fusion_method() validates input")

        return True

    except Exception as e:
        print(f"❌ Fusion method test failed: {e}")
        return False


if __name__ == "__main__":
    print("\nPhase 4: Ensemble Detection Tests\n")

    # Test 1: Initialization
    test1 = test_ensemble_initialization()

    # Test 2: Detection quality and performance
    test2 = test_ensemble_detection()

    # Test 3: Fusion methods
    test3 = test_fusion_methods()

    # Overall result
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"Initialization:   {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Detection:        {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Fusion methods:   {'✅ PASS' if test3 else '❌ FAIL'}")

    if test1 and test2 and test3:
        print("\n✅ Phase 4 ensemble detection implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Phase 4 ensemble detection implementation: FAILED")
        sys.exit(1)
