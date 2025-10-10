"""
RIGOROUS PROOF that flexible configuration actually works.

Tests:
1. Different models produce different predictions
2. Ensemble actually uses both models
3. Configuration parameters actually change behavior
4. No silent failures or fallbacks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from swift_bridge import PostureKitBridge


def test_models_produce_different_outputs():
    """PROOF: RTMW-L and RTMW-X produce different predictions."""
    print("="*80)
    print("TEST 1: Models produce DIFFERENT outputs (proving they're different)")
    print("="*80)

    # Load image
    test_images = list(Path("test-photos").glob("*.jpg"))
    image = cv2.imread(str(test_images[0]))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test RTMW-L
    print("\n1. Testing RTMW-L...")
    bridge_l = PostureKitBridge(pose_models='rtmw-l')
    result_l = bridge_l.detect_pose(image_rgb)
    kp_l = np.array(result_l['keypoints'])
    print(f"   RTMW-L first keypoint: ({kp_l[0][0]:.2f}, {kp_l[0][1]:.2f})")

    # Test RTMW-X
    print("\n2. Testing RTMW-X...")
    bridge_x = PostureKitBridge(pose_models='rtmw-x')
    result_x = bridge_x.detect_pose(image_rgb)
    kp_x = np.array(result_x['keypoints'])
    print(f"   RTMW-X first keypoint: ({kp_x[0][0]:.2f}, {kp_x[0][1]:.2f})")

    # Compare
    print("\n3. Comparing outputs...")
    diff = np.abs(kp_l[:, :2] - kp_x[:, :2])
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    print(f"   Mean difference: {mean_diff:.2f} pixels")
    print(f"   Max difference: {max_diff:.2f} pixels")

    if mean_diff > 1.0:
        print("   ✅ PROVEN: Models produce DIFFERENT outputs")
        return True
    else:
        print("   ❌ FAILED: Models produce IDENTICAL outputs (same model loaded?)")
        return False


def test_ensemble_uses_both_models():
    """PROOF: Ensemble actually uses both models and fuses predictions."""
    print("\n" + "="*80)
    print("TEST 2: Ensemble FUSES both models (not just using one)")
    print("="*80)

    # Load image
    test_images = list(Path("test-photos").glob("*.jpg"))
    image = cv2.imread(str(test_images[0]))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get individual model predictions
    print("\n1. Getting RTMW-L prediction...")
    bridge_l = PostureKitBridge(pose_models='rtmw-l')
    result_l = bridge_l.detect_pose(image_rgb)
    kp_l = np.array(result_l['keypoints'])

    print("2. Getting RTMW-X prediction...")
    bridge_x = PostureKitBridge(pose_models='rtmw-x')
    result_x = bridge_x.detect_pose(image_rgb)
    kp_x = np.array(result_x['keypoints'])

    # Get ensemble prediction
    print("3. Getting ensemble prediction...")
    bridge_ens = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
    result_ens = bridge_ens.detect_pose(image_rgb)
    kp_ens = np.array(result_ens['keypoints'])

    # Check if ensemble is between the two models (proof of fusion)
    print("\n4. Checking fusion (is ensemble between both models?)...")
    print(f"\n   Keypoint 0:")
    print(f"     RTMW-L:   ({kp_l[0][0]:.2f}, {kp_l[0][1]:.2f})")
    print(f"     RTMW-X:   ({kp_x[0][0]:.2f}, {kp_x[0][1]:.2f})")
    print(f"     Ensemble: ({kp_ens[0][0]:.2f}, {kp_ens[0][1]:.2f})")
    print(f"     Expected: ({(kp_l[0][0] + kp_x[0][0])/2:.2f}, {(kp_l[0][1] + kp_x[0][1])/2:.2f})")

    # Calculate expected fusion (average)
    expected = (kp_l[:, :2] + kp_x[:, :2]) / 2

    # Check how close ensemble is to expected average
    fusion_error = np.abs(kp_ens[:, :2] - expected)
    mean_error = np.mean(fusion_error)

    print(f"\n   Fusion accuracy: {mean_error:.3f} pixels from expected average")

    if mean_error < 5.0:  # Allow small deviation due to confidence weighting
        print("   ✅ PROVEN: Ensemble is FUSING both models (close to average)")
        return True
    else:
        # Check if it's copying one model
        dist_to_l = np.mean(np.abs(kp_ens[:, :2] - kp_l[:, :2]))
        dist_to_x = np.mean(np.abs(kp_ens[:, :2] - kp_x[:, :2]))

        print(f"   Distance to RTMW-L: {dist_to_l:.2f} pixels")
        print(f"   Distance to RTMW-X: {dist_to_x:.2f} pixels")

        if dist_to_l < 0.1:
            print("   ❌ FAILED: Ensemble is just copying RTMW-L")
        elif dist_to_x < 0.1:
            print("   ❌ FAILED: Ensemble is just copying RTMW-X")
        else:
            print("   ❌ FAILED: Ensemble output doesn't match expected fusion")
        return False


def test_fusion_methods_differ():
    """PROOF: Different fusion methods produce different results."""
    print("\n" + "="*80)
    print("TEST 3: Fusion methods produce DIFFERENT results")
    print("="*80)

    # Load image
    test_images = list(Path("test-photos").glob("*.jpg"))
    image = cv2.imread(str(test_images[0]))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test confidence_weighted
    print("\n1. Testing confidence_weighted fusion...")
    bridge_conf = PostureKitBridge(
        pose_models=['rtmw-l', 'rtmw-x'],
        fusion_method='confidence_weighted'
    )
    result_conf = bridge_conf.detect_pose(image_rgb)
    kp_conf = np.array(result_conf['keypoints'])

    # Test weighted_average
    print("2. Testing weighted_average fusion...")
    bridge_avg = PostureKitBridge(
        pose_models=['rtmw-l', 'rtmw-x'],
        fusion_method='weighted_average'
    )
    result_avg = bridge_avg.detect_pose(image_rgb)
    kp_avg = np.array(result_avg['keypoints'])

    # Compare
    print("\n3. Comparing fusion methods...")
    diff = np.abs(kp_conf[:, :2] - kp_avg[:, :2])
    mean_diff = np.mean(diff)

    print(f"   First keypoint (confidence_weighted): ({kp_conf[0][0]:.2f}, {kp_conf[0][1]:.2f})")
    print(f"   First keypoint (weighted_average):    ({kp_avg[0][0]:.2f}, {kp_avg[0][1]:.2f})")
    print(f"   Mean difference: {mean_diff:.3f} pixels")

    if mean_diff > 0.01:
        print("   ✅ PROVEN: Fusion methods produce DIFFERENT results")
        return True
    else:
        print("   ❌ FAILED: Fusion methods produce IDENTICAL results")
        return False


def test_model_validation():
    """PROOF: Invalid models are rejected."""
    print("\n" + "="*80)
    print("TEST 4: Invalid models are REJECTED")
    print("="*80)

    print("\n1. Testing invalid model name...")
    try:
        bridge = PostureKitBridge(pose_models='invalid-model')
        print("   ❌ FAILED: Invalid model was accepted!")
        return False
    except ValueError as e:
        print(f"   ✅ PROVEN: Invalid model rejected: {e}")

    print("\n2. Testing empty list...")
    try:
        bridge = PostureKitBridge(pose_models=[])
        print("   ⚠️  Empty list accepted (might cause issues)")
    except (ValueError, IndexError) as e:
        print(f"   ✅ Empty list handled: {e}")

    return True


def test_performance_differences():
    """PROOF: Different configs have different performance."""
    print("\n" + "="*80)
    print("TEST 5: Performance differences (proving different code paths)")
    print("="*80)

    import time

    test_images = list(Path("test-photos").glob("*.jpg"))
    image = cv2.imread(str(test_images[0]))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Time RTMW-L
    print("\n1. Benchmarking RTMW-L...")
    bridge_l = PostureKitBridge(pose_models='rtmw-l')
    bridge_l.detect_pose(image_rgb)  # Warm-up

    times_l = []
    for _ in range(3):
        start = time.time()
        bridge_l.detect_pose(image_rgb)
        times_l.append(time.time() - start)
    avg_l = np.mean(times_l)
    print(f"   Average: {avg_l:.3f}s")

    # Time RTMW-X
    print("\n2. Benchmarking RTMW-X...")
    bridge_x = PostureKitBridge(pose_models='rtmw-x')
    bridge_x.detect_pose(image_rgb)  # Warm-up

    times_x = []
    for _ in range(3):
        start = time.time()
        bridge_x.detect_pose(image_rgb)
        times_x.append(time.time() - start)
    avg_x = np.mean(times_x)
    print(f"   Average: {avg_x:.3f}s")

    # Time Ensemble
    print("\n3. Benchmarking Ensemble...")
    bridge_ens = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
    bridge_ens.detect_pose(image_rgb)  # Warm-up

    times_ens = []
    for _ in range(3):
        start = time.time()
        bridge_ens.detect_pose(image_rgb)
        times_ens.append(time.time() - start)
    avg_ens = np.mean(times_ens)
    print(f"   Average: {avg_ens:.3f}s")

    # Compare
    print("\n4. Performance analysis...")
    print(f"   RTMW-L:   {avg_l:.3f}s (1.0x)")
    print(f"   RTMW-X:   {avg_x:.3f}s ({avg_x/avg_l:.2f}x)")
    print(f"   Ensemble: {avg_ens:.3f}s ({avg_ens/avg_l:.2f}x)")

    # RTMW-X should be slower than L
    if avg_x > avg_l * 1.2:
        print("   ✅ RTMW-X is slower (proves it's a different model)")
        x_slower = True
    else:
        print("   ❌ RTMW-X not slower (might be using same model)")
        x_slower = False

    # Ensemble should be ~2x slower
    if avg_ens > avg_l * 1.8:
        print("   ✅ Ensemble is ~2x slower (proves both models run)")
        ens_slower = True
    else:
        print(f"   ❌ Ensemble only {avg_ens/avg_l:.2f}x slower (might be using one model)")
        ens_slower = False

    return x_slower and ens_slower


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RIGOROUS PROOF OF FLEXIBLE CONFIGURATION")
    print("No bullshit - actual evidence")
    print("="*80)

    results = {}

    # Run all tests
    results['different_outputs'] = test_models_produce_different_outputs()
    results['ensemble_fusion'] = test_ensemble_uses_both_models()
    results['fusion_methods'] = test_fusion_methods_differ()
    results['validation'] = test_model_validation()
    results['performance'] = test_performance_differences()

    # Summary
    print("\n" + "="*80)
    print("PROOF SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PROVEN" if passed else "❌ FAILED"
        print(f"{status} | {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PROVEN - Flexible configuration ACTUALLY WORKS")
        print("   - Different models load different checkpoints")
        print("   - Ensemble actually fuses both models")
        print("   - Fusion methods actually differ")
        print("   - Validation works")
        print("   - Performance matches expectations")
    else:
        print("❌ SOME TESTS FAILED - I was bullshitting you")
        failed = [name for name, passed in results.items() if not passed]
        print(f"   Failed: {', '.join(failed)}")
    print("="*80)

    sys.exit(0 if all_passed else 1)
