"""
Debug utilities for model configuration.

When things break (and they will), use these to figure out what's actually happening.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def print_config_info(bridge):
    """
    Print detailed configuration info for debugging.

    Usage:
        from utils.debug_config import print_config_info
        bridge = PostureKitBridge(...)
        print_config_info(bridge)
    """
    print("\n" + "="*80)
    print("POSTUREKIT CONFIGURATION DEBUG INFO")
    print("="*80)

    # Detector type
    detector_type = type(bridge.detector).__name__
    print(f"\nDetector type: {detector_type}")

    # Check what's actually loaded
    if hasattr(bridge, 'pose_detector'):
        print(f"Single pose detector: {type(bridge.pose_detector).__name__}")
        if hasattr(bridge.pose_detector, 'checkpoint_file'):
            print(f"  Checkpoint: {Path(bridge.pose_detector.checkpoint_file).name}")

    # Ensemble info
    if detector_type == 'EnsembleDetector':
        print(f"\nEnsemble configuration:")
        print(f"  Number of models: {len(bridge.detector.detectors)}")
        print(f"  Fusion method: {bridge.detector.config.fusion_method}")

        for i, detector in enumerate(bridge.detector.detectors):
            model_config = bridge.detector.config.models[i]
            checkpoint_path = Path(model_config['checkpoint'])
            print(f"\n  Model {i+1}:")
            print(f"    Checkpoint: {checkpoint_path.name}")
            print(f"    Size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
            print(f"    Weight: {model_config['weight']}")

    # Two-stage info
    if detector_type == 'TwoStageDetector':
        print(f"\nTwo-stage configuration:")
        print(f"  Person detector: {bridge.detector.person_model}")
        print(f"  Min confidence: {bridge.detector.min_person_conf}")
        print(f"  Crop padding: {bridge.detector.crop_padding}")

    print("\n" + "="*80)


def validate_model_files(models_dir: Path = None) -> Dict[str, Any]:
    """
    Validate that all required model files exist and have correct sizes.

    Returns dict with validation results.
    """
    from config.settings import settings

    if models_dir is None:
        models_dir = settings.PROJECT_ROOT / "data" / "models"

    expected_files = {
        'rtmw-l': {
            'config': 'rtmw-l_8xb320-270e_cocktail14-384x288.py',
            'checkpoint': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
            'expected_size_mb': 220,
            'tolerance_mb': 5,
        },
        'rtmw-x': {
            'config': 'rtmw-x_8xb320-270e_cocktail14-384x288.py',
            'checkpoint': 'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
            'expected_size_mb': 353,
            'tolerance_mb': 5,
        },
    }

    results = {}

    print("\n" + "="*80)
    print("MODEL FILE VALIDATION")
    print("="*80)

    for model_name, files in expected_files.items():
        print(f"\n{model_name.upper()}:")
        results[model_name] = {}

        # Check config
        config_path = models_dir / files['config']
        if config_path.exists():
            print(f"  ✅ Config: {files['config']}")
            results[model_name]['config'] = True
        else:
            print(f"  ❌ Config: {files['config']} NOT FOUND")
            results[model_name]['config'] = False

        # Check checkpoint
        checkpoint_path = models_dir / files['checkpoint']
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            expected = files['expected_size_mb']
            tolerance = files['tolerance_mb']

            if abs(size_mb - expected) < tolerance:
                print(f"  ✅ Checkpoint: {files['checkpoint']}")
                print(f"     Size: {size_mb:.1f} MB (expected ~{expected} MB)")
                results[model_name]['checkpoint'] = True
            else:
                print(f"  ⚠️  Checkpoint: {files['checkpoint']}")
                print(f"     Size: {size_mb:.1f} MB (expected ~{expected} MB)")
                print(f"     WARNING: Size mismatch (might be corrupted)")
                results[model_name]['checkpoint'] = 'size_mismatch'
        else:
            print(f"  ❌ Checkpoint: {files['checkpoint']} NOT FOUND")
            results[model_name]['checkpoint'] = False

    print("\n" + "="*80)

    return results


def quick_test(bridge, test_image_path: str = None):
    """
    Quick sanity test of a configured bridge.

    Returns detection result or raises error.
    """
    import cv2
    import numpy as np
    import time

    print("\n" + "="*80)
    print("QUICK CONFIGURATION TEST")
    print("="*80)

    # Get test image
    if test_image_path is None:
        from pathlib import Path
        test_photos = list(Path("test-photos").glob("*.jpg"))
        if not test_photos:
            raise FileNotFoundError("No test images in test-photos/")
        test_image_path = str(test_photos[0])

    print(f"\nTest image: {Path(test_image_path).name}")

    # Load image
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {test_image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")

    # Run detection
    print("\nRunning detection...")
    start = time.time()
    result = bridge.detect_pose(image_rgb)
    elapsed = time.time() - start

    if result is None:
        print("❌ Detection failed (returned None)")
        return None

    print(f"✅ Detection successful in {elapsed:.3f}s")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")
    print(f"   Keypoints: {len(result.get('keypoints', []))}")

    # Check keypoint values for sanity
    if 'keypoints' in result:
        kps = np.array(result['keypoints'])
        print(f"   First keypoint: ({kps[0][0]:.2f}, {kps[0][1]:.2f})")

        # Sanity checks
        if np.any(np.isnan(kps)):
            print("   ⚠️  WARNING: NaN values in keypoints")
        if np.all(kps[:, :2] == 0):
            print("   ⚠️  WARNING: All keypoints are (0, 0)")

    print("="*80 + "\n")

    return result


if __name__ == "__main__":
    # Can be run standalone for debugging
    print("Model file validation:\n")
    results = validate_model_files()

    all_ok = all(
        info.get('config') and info.get('checkpoint') == True
        for info in results.values()
    )

    if all_ok:
        print("\n✅ All model files validated")
    else:
        print("\n⚠️  Some model files missing or incorrect")
