# PostureKit Troubleshooting Guide

When things break (and they will), start here.

## Quick Debug Commands

```python
from src.swift_bridge import PostureKitBridge
from src.utils.debug_config import print_config_info, validate_model_files, quick_test

# 1. Validate model files exist and have correct sizes
validate_model_files()

# 2. Test your configuration
bridge = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
print_config_info(bridge)
quick_test(bridge)
```

## Common Issues

### 1. "Unknown model 'X'" Error

**Error:**
```
ValueError: Unknown model 'rtmw-m'. Available: rtmw-l, rtmw-x
```

**Cause:** Typo in model name or model not in registry.

**Fix:**
```python
# Check available models
from src.swift_bridge import PostureKitBridge
print(PostureKitBridge.AVAILABLE_MODELS.keys())
# dict_keys(['rtmw-l', 'rtmw-x'])

# Use correct name
bridge = PostureKitBridge(pose_models='rtmw-l')  # Not 'rtmw-m'
```

### 2. "Model checkpoint not found" Error

**Error:**
```
FileNotFoundError: Model checkpoint not found: /path/to/model.pth
```

**Cause:** Model file missing or in wrong location.

**Fix:**
```bash
# Check what files exist
ls -lh data/models/*.pth

# Validate all files
python3 -c "from src.utils.debug_config import validate_model_files; validate_model_files()"
```

**Expected files:**
- `rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth` (220MB)
- `rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth` (353MB)

### 3. Ensemble Returns Same Result as Single Model

**Symptom:** Ensemble output identical to RTMW-L, no fusion happening.

**Debug:**
```python
from src.utils.debug_config import print_config_info

bridge = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
print_config_info(bridge)

# Should show:
# Detector type: EnsembleDetector
# Number of models: 2
# Model 1: rtmw-dw-x-l_...pth (220MB)
# Model 2: rtmw-x_...pth (353MB)
```

**If it shows 1 model or wrong checkpoints:**
- Check logs for "Fusion failed" or "Using fallback"
- Run `test_config_proof.py` to verify fusion works
- Check that both checkpoint files exist and aren't corrupted

### 4. Performance Too Fast (Ensemble Not Running Both Models)

**Symptom:** Ensemble is same speed as single model.

**Debug:**
```python
import time
import cv2
from pathlib import Path

# Benchmark
bridge_single = PostureKitBridge(pose_models='rtmw-l')
bridge_ensemble = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])

image = cv2.imread(str(list(Path("test-photos").glob("*.jpg"))[0]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Time single
start = time.time()
bridge_single.detect_pose(image)
time_single = time.time() - start

# Time ensemble
start = time.time()
bridge_ensemble.detect_pose(image)
time_ensemble = time.time() - start

print(f"Single: {time_single:.3f}s")
print(f"Ensemble: {time_ensemble:.3f}s")
print(f"Slowdown: {time_ensemble/time_single:.2f}x")

# Should be ~2x slower
if time_ensemble < time_single * 1.5:
    print("⚠️  Ensemble too fast - might not be running both models")
```

### 5. Fusion Returns NaN or Zero Keypoints

**Symptom:** Detection works but keypoints are invalid.

**Debug:**
```python
import numpy as np

result = bridge.detect_pose(image)
kps = np.array(result['keypoints'])

print(f"Shape: {kps.shape}")  # Should be (133, 3)
print(f"NaN values: {np.any(np.isnan(kps))}")  # Should be False
print(f"All zeros: {np.all(kps[:, :2] == 0)}")  # Should be False
print(f"First 3: {kps[:3]}")
```

**Common causes:**
- Visibility validation issue (check `src/core/pose_detector.py` line 64)
- Confidence normalization issue (check `src/core/ensemble_detector.py` fusion methods)

### 6. Different Results Each Run (Non-Deterministic)

**Expected behavior:** Slight variations are normal due to:
- Image preprocessing randomness
- Floating-point precision
- Model internal state

**If results vary wildly (>10 pixels):**
```python
# Run detection multiple times
results = []
for _ in range(5):
    result = bridge.detect_pose(image)
    results.append(np.array(result['keypoints']))

# Check variance
variance = np.var([r[0] for r in results], axis=0)
print(f"Variance in first keypoint: {variance}")

# Should be < 1.0 pixel
if np.any(variance > 1.0):
    print("⚠️  High variance - check for randomness in pipeline")
```

### 7. Model Confusion (Wrong Model Loading)

**Symptom:** RTMW-X performs same as RTMW-L.

**Verify different models:**
```python
# Test that models are actually different
bridge_l = PostureKitBridge(pose_models='rtmw-l')
bridge_x = PostureKitBridge(pose_models='rtmw-x')

result_l = bridge_l.detect_pose(image)
result_x = bridge_x.detect_pose(image)

kp_l = np.array(result_l['keypoints'])
kp_x = np.array(result_x['keypoints'])

diff = np.abs(kp_l - kp_x)
print(f"Mean difference: {np.mean(diff):.2f} pixels")

# Should be > 1 pixel
if np.mean(diff) < 1.0:
    print("⚠️  Models producing identical output - might be loading same checkpoint")
    print(f"Check: ls -l data/models/*.pth")
```

### 8. Import Errors

**Error:**
```
ImportError: cannot import name 'PostureKitBridge'
```

**Fix:**
```python
import sys
from pathlib import Path

# Make sure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from swift_bridge import PostureKitBridge  # Not src.swift_bridge
```

### 9. Memory Issues with Ensemble

**Symptom:** Out of memory errors with ensemble.

**Memory usage:**
- RTMW-L: ~220MB
- RTMW-X: ~353MB
- Ensemble: ~573MB (both loaded)

**Fix:**
```python
# Use single larger model instead
bridge = PostureKitBridge(pose_models='rtmw-x')  # 353MB instead of 573MB

# Or use two-stage with smaller model
bridge = PostureKitBridge(
    pose_models='rtmw-l',
    use_two_stage=True  # Better quality without double memory
)
```

### 10. Confidence Values Look Wrong

**Symptom:** Confidence > 1.0 or < 0.0

**Debug:**
```python
result = bridge.detect_pose(image)
conf = result.get('confidence')

print(f"Confidence: {conf}")
print(f"Valid range: {0.0 <= conf <= 1.0}")

if conf > 1.0:
    print("⚠️  Confidence > 1.0 - check normalization in fusion code")
if conf < 0.0:
    print("⚠️  Confidence < 0.0 - check calculation")
```

**Known issue:** Ensemble fusion divides visibility (0-2 range) by 2 to get confidence (0-1 range). Check `src/core/ensemble_detector.py` line 293.

## Nuclear Options

### Reset Everything

```bash
# Remove all Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Redownload model files
# (if you have the download script)

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Verify Installation

```bash
# Run all validation tests
python3 test_config_proof.py        # Proves flexible config works
python3 test_ensemble_proof.py      # Proves ensemble fusion works
python3 test_phase4_ensemble.py     # Proves Phase 4 works
python3 test_flexible_config.py     # Tests all 9 configurations
```

All tests should pass ✅.

## Getting Help

When reporting issues, include:

```python
from src.utils.debug_config import print_config_info, validate_model_files

# 1. Model files
validate_model_files()

# 2. Configuration
bridge = PostureKitBridge(...)  # Your config
print_config_info(bridge)

# 3. Python/system info
import sys, torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Platform: {sys.platform}")

# 4. Error traceback (full output)
```

## Known Limitations

1. **Both models must be downloaded** - No automatic fallback to single model
2. **No model auto-download** - Must manually download checkpoints
3. **Ensemble is slow** - 2-3x slower than single model (expected)
4. **Empty list fails** - `pose_models=[]` will crash (not handled gracefully)
5. **Fusion requires compatible keypoints** - All models must output same keypoint format

## Prevention (Before Things Break)

```python
# Always validate before deploying
from src.utils.debug_config import validate_model_files, quick_test

# 1. Check files exist
results = validate_model_files()
assert all(r['checkpoint'] == True for r in results.values()), "Missing model files"

# 2. Test configuration
bridge = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
result = quick_test(bridge)
assert result is not None, "Detection failed"

# 3. Verify performance
import time
start = time.time()
bridge.detect_pose(test_image)
elapsed = time.time() - start
assert elapsed > 0.5, "Too fast - models might not be running"

print("✅ Configuration validated")
```

Good luck debugging. You'll need it.
