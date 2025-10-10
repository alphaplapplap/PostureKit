"""
Test flexible model configuration for PostureKit.

Demonstrates all configuration options:
- Single models: rtmw-l, rtmw-x
- Ensemble: any combination
- Two-stage with any model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import time
from swift_bridge import PostureKitBridge


def test_model_config(config_name, **kwargs):
    """Test a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"Config: {kwargs}")
    print('='*80)

    try:
        # Initialize bridge
        start = time.time()
        bridge = PostureKitBridge(**kwargs)
        init_time = time.time() - start

        print(f"✅ Initialized in {init_time:.2f}s")

        # Load test image
        test_images = list(Path("test-photos").glob("*.jpg"))
        if not test_images:
            print("⚠️  No test images found")
            return

        image = cv2.imread(str(test_images[0]))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        start = time.time()
        result = bridge.detect_pose(image_rgb)
        detect_time = time.time() - start

        if result:
            print(f"✅ Detection successful in {detect_time:.3f}s")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Keypoints: {len(result.get('keypoints', []))} detected")
        else:
            print("⚠️  No pose detected")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("FLEXIBLE MODEL CONFIGURATION TEST")
    print("="*80)

    tests = [
        # Single models
        ("RTMW-L (default)", {}),
        ("RTMW-L (explicit)", {'pose_models': 'rtmw-l'}),
        ("RTMW-X (explicit)", {'pose_models': 'rtmw-x'}),

        # Ensemble configurations
        ("Ensemble: RTMW-L + RTMW-X", {'pose_models': ['rtmw-l', 'rtmw-x']}),
        ("Ensemble: RTMW-X only (via list)", {'pose_models': ['rtmw-x']}),
        ("Ensemble with weighted average", {
            'pose_models': ['rtmw-l', 'rtmw-x'],
            'fusion_method': 'weighted_average'
        }),

        # Backward compatibility
        ("Backward compat: use_ensemble=True", {'use_ensemble': True}),

        # Two-stage with different models
        ("Two-stage with RTMW-L", {
            'pose_models': 'rtmw-l',
            'use_two_stage': True
        }),
        ("Two-stage with RTMW-X", {
            'pose_models': 'rtmw-x',
            'use_two_stage': True
        }),
    ]

    results = {}

    for test_name, config in tests:
        results[test_name] = test_model_config(test_name, **config)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} | {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Flexible configuration working!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
