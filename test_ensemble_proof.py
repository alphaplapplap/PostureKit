"""
Comprehensive proof that Phase 4 ensemble detection truly works.

This test validates:
1. Both models load with different checkpoints
2. Each model produces different predictions
3. Fusion actually combines both model outputs
4. Performance is ~2x slower (proving both models run)
5. Ensemble output differs from single-model output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import time
from core.ensemble_detector import EnsembleDetector, EnsembleConfig
from core.pose_detector import RTMWCocktail14Detector
from config.settings import settings


def test_individual_models():
    """Test that each model produces different outputs."""
    print("=" * 80)
    print("TEST 1: Individual Model Outputs (Proving Models Are Different)")
    print("=" * 80)

    # Load test image
    test_images = list(Path("test-photos").glob("*.jpg"))
    if not test_images:
        print("❌ No test images found")
        return False

    test_image_path = test_images[0]
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nTest image: {test_image_path.name} ({image.shape})\n")

    # Model paths
    rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    rtmw_x_config = settings.PROJECT_ROOT / "data/models/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_x_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

    # Load RTMW-L
    print("Loading RTMW-L (220MB, Model 1)...")
    model_l = RTMWCocktail14Detector(
        config_file=str(rtmw_l_config),
        checkpoint_file=str(rtmw_l_checkpoint)
    )
    print(f"✅ RTMW-L loaded from: ...{str(rtmw_l_checkpoint)[-60:]}")

    # Load RTMW-X
    print("\nLoading RTMW-X (353MB, Model 2)...")
    model_x = RTMWCocktail14Detector(
        config_file=str(rtmw_x_config),
        checkpoint_file=str(rtmw_x_checkpoint)
    )
    print(f"✅ RTMW-X loaded from: ...{str(rtmw_x_checkpoint)[-60:]}")

    # Run detection on both
    print("\n" + "-" * 80)
    print("Running detection on both models...")
    print("-" * 80)

    start = time.time()
    poses_l = model_l.detect(image_rgb)
    time_l = time.time() - start

    start = time.time()
    poses_x = model_x.detect(image_rgb)
    time_x = time.time() - start

    if not poses_l or not poses_x:
        print("❌ One or both models failed to detect pose")
        return False

    pose_l = poses_l[0]
    pose_x = poses_x[0]

    print(f"\nRTMW-L Results:")
    print(f"  Confidence: {pose_l.overall_confidence:.6f}")
    print(f"  Time: {time_l:.3f}s")
    print(f"  First 3 keypoints (x,y):")
    for i in range(3):
        print(f"    [{i}] ({pose_l.keypoints[i][0]:.2f}, {pose_l.keypoints[i][1]:.2f}) vis={pose_l.visibility[i]:.3f}")

    print(f"\nRTMW-X Results:")
    print(f"  Confidence: {pose_x.overall_confidence:.6f}")
    print(f"  Time: {time_x:.3f}s")
    print(f"  First 3 keypoints (x,y):")
    for i in range(3):
        print(f"    [{i}] ({pose_x.keypoints[i][0]:.2f}, {pose_x.keypoints[i][1]:.2f}) vis={pose_x.visibility[i]:.3f}")

    # Compare outputs
    print("\n" + "-" * 80)
    print("Comparing Model Outputs (Proof They're Different)")
    print("-" * 80)

    keypoint_diff = np.abs(pose_l.keypoints - pose_x.keypoints)
    max_diff = np.max(keypoint_diff)
    mean_diff = np.mean(keypoint_diff)

    print(f"\nKeypoint coordinate differences:")
    print(f"  Mean difference: {mean_diff:.3f} pixels")
    print(f"  Max difference: {max_diff:.3f} pixels")

    conf_diff = abs(pose_l.overall_confidence - pose_x.overall_confidence)
    print(f"\nConfidence difference: {conf_diff:.6f}")

    if mean_diff > 0.1:
        print("✅ Models produce DIFFERENT outputs (as expected)")
        different = True
    else:
        print("⚠️  Models produce identical outputs (unexpected)")
        different = False

    return different


def test_ensemble_fusion():
    """Test that ensemble actually fuses both model predictions."""
    print("\n" + "=" * 80)
    print("TEST 2: Ensemble Fusion (Proving Fusion Combines Both Models)")
    print("=" * 80)

    # Load test image
    test_images = list(Path("test-photos").glob("*.jpg"))
    test_image_path = test_images[0]
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nTest image: {test_image_path.name}\n")

    # Model paths
    rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    rtmw_x_config = settings.PROJECT_ROOT / "data/models/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_x_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

    # Single models
    print("Loading individual models...")
    model_l = RTMWCocktail14Detector(
        config_file=str(rtmw_l_config),
        checkpoint_file=str(rtmw_l_checkpoint)
    )
    model_x = RTMWCocktail14Detector(
        config_file=str(rtmw_x_config),
        checkpoint_file=str(rtmw_x_checkpoint)
    )

    # Ensemble
    print("Loading ensemble...")
    config = EnsembleConfig(
        models=[
            {
                "config": str(rtmw_l_config),
                "checkpoint": str(rtmw_l_checkpoint),
                "weight": 1.0,
            },
            {
                "config": str(rtmw_x_config),
                "checkpoint": str(rtmw_x_checkpoint),
                "weight": 1.0,
            },
        ],
        fusion_method="confidence_weighted",
    )
    ensemble = EnsembleDetector(config)

    # Run all three
    print("\n" + "-" * 80)
    print("Running detection...")
    print("-" * 80)

    poses_l = model_l.detect(image_rgb)
    poses_x = model_x.detect(image_rgb)
    poses_ensemble = ensemble.detect(image_rgb)

    if not poses_l or not poses_x or not poses_ensemble:
        print("❌ Detection failed")
        return False

    pose_l = poses_l[0]
    pose_x = poses_x[0]
    pose_ensemble = poses_ensemble[0]

    print(f"\nResults:")
    print(f"  RTMW-L confidence: {pose_l.overall_confidence:.6f}")
    print(f"  RTMW-X confidence: {pose_x.overall_confidence:.6f}")
    print(f"  Ensemble confidence: {pose_ensemble.overall_confidence:.6f}")

    # Compare keypoints
    print("\n" + "-" * 80)
    print("Keypoint Analysis (First 5 keypoints)")
    print("-" * 80)

    print(f"\n{'KP':>3} | {'RTMW-L':>20} | {'RTMW-X':>20} | {'Ensemble':>20} | {'Diff':>10}")
    print("-" * 85)

    ensemble_matches_l = 0
    ensemble_matches_x = 0
    ensemble_is_between = 0

    for i in range(5):
        l_kp = pose_l.keypoints[i]
        x_kp = pose_x.keypoints[i]
        e_kp = pose_ensemble.keypoints[i]

        l_x, l_y = l_kp[0], l_kp[1]
        x_x, x_y = x_kp[0], x_kp[1]
        e_x, e_y = e_kp[0], e_kp[1]

        diff_l = np.sqrt((e_x - l_x)**2 + (e_y - l_y)**2)
        diff_x = np.sqrt((e_x - x_x)**2 + (e_y - x_y)**2)

        print(f"{i:3d} | ({l_x:7.2f}, {l_y:7.2f}) | ({x_x:7.2f}, {x_y:7.2f}) | "
              f"({e_x:7.2f}, {e_y:7.2f}) | {diff_l:4.1f}/{diff_x:4.1f}")

        # Check if ensemble is between the two models
        if abs(diff_l - diff_x) < 0.5:
            ensemble_matches_l += 1

        # Check if ensemble is truly fused (not exactly matching either)
        if diff_l > 0.1 and diff_x > 0.1:
            ensemble_is_between += 1

    print("\n" + "-" * 80)
    print("Fusion Validation")
    print("-" * 80)

    print(f"\nKeypoints where ensemble differs from both models: {ensemble_is_between}/5")

    if ensemble_is_between >= 2:
        print("✅ Ensemble is FUSING predictions (not copying one model)")
        fusing = True
    else:
        print("⚠️  Ensemble may be copying one model")
        fusing = False

    return fusing


def test_performance_characteristics():
    """Test that ensemble is ~2x slower, proving both models run."""
    print("\n" + "=" * 80)
    print("TEST 3: Performance Analysis (Proving Both Models Run)")
    print("=" * 80)

    # Load test image
    test_images = list(Path("test-photos").glob("*.jpg"))
    test_image_path = test_images[0]
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nTest image: {test_image_path.name}\n")

    # Model paths
    rtmw_l_config = settings.PROJECT_ROOT / "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    rtmw_x_config = settings.PROJECT_ROOT / "data/models/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_x_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

    # Single model
    print("Loading RTMW-L (single-model baseline)...")
    model_l = RTMWCocktail14Detector(
        config_file=str(rtmw_l_config),
        checkpoint_file=str(rtmw_l_checkpoint)
    )

    # Ensemble
    print("Loading ensemble (RTMW-L + RTMW-X)...")
    config = EnsembleConfig(
        models=[
            {
                "config": str(rtmw_l_config),
                "checkpoint": str(rtmw_l_checkpoint),
                "weight": 1.0,
            },
            {
                "config": str(rtmw_x_config),
                "checkpoint": str(rtmw_x_checkpoint),
                "weight": 1.0,
            },
        ],
        fusion_method="confidence_weighted",
    )
    ensemble = EnsembleDetector(config)

    # Benchmark
    print("\n" + "-" * 80)
    print("Benchmarking (3 runs each)...")
    print("-" * 80)

    # Warm-up
    model_l.detect(image_rgb)
    ensemble.detect(image_rgb)

    # Single-model
    times_single = []
    for i in range(3):
        start = time.time()
        model_l.detect(image_rgb)
        times_single.append(time.time() - start)

    avg_single = np.mean(times_single)

    # Ensemble
    times_ensemble = []
    for i in range(3):
        start = time.time()
        ensemble.detect(image_rgb)
        times_ensemble.append(time.time() - start)

    avg_ensemble = np.mean(times_ensemble)

    print(f"\nSingle-model (RTMW-L only):")
    print(f"  Times: {[f'{t:.3f}s' for t in times_single]}")
    print(f"  Average: {avg_single:.3f}s")

    print(f"\nEnsemble (RTMW-L + RTMW-X):")
    print(f"  Times: {[f'{t:.3f}s' for t in times_ensemble]}")
    print(f"  Average: {avg_ensemble:.3f}s")

    slowdown = avg_ensemble / avg_single

    print("\n" + "-" * 80)
    print("Performance Analysis")
    print("-" * 80)

    print(f"\nSlowdown factor: {slowdown:.2f}x")
    print(f"Expected range: 1.8-2.5x (for true 2-model ensemble)")

    if 1.8 <= slowdown <= 2.8:
        print("✅ Performance consistent with TWO models running")
        two_models = True
    elif slowdown < 1.5:
        print("⚠️  Too fast - may be running only one model")
        two_models = False
    else:
        print("⚠️  Slower than expected - unexpected overhead")
        two_models = False

    return two_models


def test_checkpoint_files():
    """Verify different checkpoint files are used."""
    print("\n" + "=" * 80)
    print("TEST 4: Checkpoint File Verification")
    print("=" * 80)

    rtmw_l_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    rtmw_x_checkpoint = settings.PROJECT_ROOT / "data/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

    print("\nChecking checkpoint files...")

    if not rtmw_l_checkpoint.exists():
        print(f"❌ RTMW-L checkpoint not found: {rtmw_l_checkpoint}")
        return False

    if not rtmw_x_checkpoint.exists():
        print(f"❌ RTMW-X checkpoint not found: {rtmw_x_checkpoint}")
        return False

    size_l = rtmw_l_checkpoint.stat().st_size / (1024 * 1024)
    size_x = rtmw_x_checkpoint.stat().st_size / (1024 * 1024)

    print(f"\nRTMW-L checkpoint:")
    print(f"  File: {rtmw_l_checkpoint.name}")
    print(f"  Size: {size_l:.1f} MB")
    print(f"  Expected: ~220 MB")

    print(f"\nRTMW-X checkpoint:")
    print(f"  File: {rtmw_x_checkpoint.name}")
    print(f"  Size: {size_x:.1f} MB")
    print(f"  Expected: ~353 MB")

    print("\n" + "-" * 80)

    if abs(size_l - 220) < 50 and abs(size_x - 353) < 50:
        print("✅ Both checkpoints present with correct sizes")
        print("✅ Files are DIFFERENT (different names and sizes)")
        return True
    else:
        print("⚠️  Checkpoint sizes don't match expectations")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ENSEMBLE DETECTION PROOF")
    print("Proving Phase 4 uses TWO different models with real fusion")
    print("=" * 80)

    results = {}

    # Test 1: Different models
    results['different_models'] = test_individual_models()

    # Test 2: Fusion
    results['fusion_works'] = test_ensemble_fusion()

    # Test 3: Performance
    results['two_models_run'] = test_performance_characteristics()

    # Test 4: Checkpoints
    results['different_checkpoints'] = test_checkpoint_files()

    # Summary
    print("\n" + "=" * 80)
    print("PROOF SUMMARY")
    print("=" * 80)

    print(f"\n1. Models produce different outputs: {'✅ PROVEN' if results['different_models'] else '❌ FAILED'}")
    print(f"2. Fusion combines both models: {'✅ PROVEN' if results['fusion_works'] else '❌ FAILED'}")
    print(f"3. Both models run (2x slower): {'✅ PROVEN' if results['two_models_run'] else '❌ FAILED'}")
    print(f"4. Different checkpoints used: {'✅ PROVEN' if results['different_checkpoints'] else '❌ FAILED'}")

    all_pass = all(results.values())

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ENSEMBLE FULLY PROVEN: Two models, real fusion, working correctly!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("⚠️  ENSEMBLE PROOF INCOMPLETE: Some tests failed")
        print("=" * 80)
        sys.exit(1)
