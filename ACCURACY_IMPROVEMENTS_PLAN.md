# PostureKit Accuracy Improvements Implementation Plan

**Date:** 2025-10-09
**Branch:** `feature/accuracy-improvements`
**Status:** Implementation in progress

## Overview

This document outlines the plan to implement 4 accuracy improvements for PostureKit's pose similarity search system. Combined, these improvements should deliver **30-50% overall accuracy gains** with acceptable performance trade-offs.

## Selected Improvements

| Feature | Accuracy Gain | Complexity | Performance Impact | Files to Modify |
|---------|--------------|------------|-------------------|-----------------|
| Visual Features | 15-25% (search) | ⭐⭐ Easy-Med | -30% slower | Enable existing code |
| Preprocessing | 5-10% | ⭐ Easy | -5% slower | 1 file (+30 lines) |
| Two-Stage Detection | 5-10% | ⭐⭐⭐ Medium | -20% slower | 2 new files |
| Ensemble Detection | 10-15% | ⭐⭐⭐ Medium | -60% slower | 3 new files |

**Total Expected Gain:** 35-60% combined accuracy improvement
**Total Performance Impact:** ~2-3x slower (acceptable for quality)

---

## Phase 1: Visual Features Integration (HIGHEST PRIORITY)

### Why First?
- Easiest implementation (code already exists)
- Highest ROI for search quality (15-25% improvement)
- No new dependencies needed
- Quick win to validate approach

### Current Status
- ✅ `src/core/visual_feature_extractor.py` fully implemented (325 lines)
- ✅ Database tables (`visual_features`, `fused_features`) already exist
- ✅ Storage manager supports visual/fused features
- ❌ Not currently being called during indexing
- ❌ Similarity search uses only geometric features (52-dim)

### Implementation Steps

#### 1.1 Modify `src/swift_bridge.py`

**Add imports (after line 32):**
```python
from core.visual_feature_extractor import VisualFeatureExtractor, VisualFeatures
from core.multimodal_fusion import FusedFeatures, fuse_features
```

**Initialize in `__init__` (after line 93):**
```python
self.visual_extractor = VisualFeatureExtractor()  # Lazy loads model
```

**Modify `index_directory()` loop (around line 380):**
```python
# After geometric feature extraction
features = self.feature_extractor.extract(pose)

# ADD: Extract visual features
visual_features = self.visual_extractor.extract(image_rgb)

# ADD: Fuse geometric + visual features
fused_features = fuse_features(features, visual_features)

# Modified storage call
pose_id = self.storage_manager.store_detection(
    image_path=image_path,
    image_metadata=image_metadata,
    pose_result=pose,
    features=features,
    visual_features=visual_features,    # NEW
    fused_features=fused_features        # NEW
)
```

#### 1.2 Modify `src/intelligence/similarity_engine.py`

**Change index dimension (line 106):**
```python
# OLD: dimension = 52  (geometric only)
dimension = vectors.shape[1]  # Now 628 (fused features)
```

**Update `build_index()` to use fused features (line 66-78):**
```python
# OLD: Query GeometricFeatures table
query = session.query(
    FusedFeatures.pose_id,           # CHANGED
    FusedFeatures.feature_vector,    # CHANGED
    PoseDetection.overall_confidence,
    Image.file_path
).join(
    PoseDetection, FusedFeatures.pose_id == PoseDetection.id  # CHANGED
).join(
    Image, PoseDetection.image_id == Image.id
).order_by(PoseDetection.created_at)
```

**Update docstrings:**
- Line 161: Change "52-dim geometric" → "628-dim fused"
- Line 364: Update scale factor for 628-dim vectors (empirical tuning needed)

#### 1.3 Test Visual Features

```bash
# Clear existing index
rm -rf data/indices/*

# Index test directory with visual features
python3 -c "
import sys
sys.path.insert(0, 'venv/lib/python3.9/site-packages')
sys.path.insert(0, 'src')
from swift_bridge import PostureKitBridge

bridge = PostureKitBridge()
result = bridge.index_directory('test-photos', recursive=False)
print(result)
"

# Verify database
psql $DATABASE_URL -c "
SELECT COUNT(*) as geometric FROM geometric_features;
SELECT COUNT(*) as visual FROM visual_features;
SELECT COUNT(*) as fused FROM fused_features;
"

# Check FAISS index dimension
python3 -c "
import faiss
index = faiss.read_index('data/indices/pose_features.index')
print(f'Index dimension: {index.d}')  # Should be 628
print(f'Total vectors: {index.ntotal}')
"
```

**Success Criteria:**
- ✅ Visual features table populated
- ✅ Fused features table populated (628-dim vectors)
- ✅ FAISS index dimension = 628
- ✅ Search returns results with similarity scores

---

## Phase 2: Enhanced Preprocessing

### Why Second?
- Easy implementation (~30 lines)
- Immediate 5-10% accuracy boost
- Minimal performance impact
- Improves quality for all subsequent processing

### Implementation Steps

#### 2.1 Modify `src/core/image_ingestor.py`

**Add preprocessing method (after line 224):**
```python
def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better pose detection.

    Applies:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Non-local means denoising
    - Resolution upsampling for small images

    Args:
        image: Input image in BGR format

    Returns:
        Enhanced image in BGR format
    """
    # 1. Apply CLAHE to each channel (improves keypoint visibility)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply to L channel only
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Denoise (reduces false keypoint detections)
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, None, h=3, hColor=3,
        templateWindowSize=7, searchWindowSize=21
    )

    # 3. Upsample small images (improves detection on small subjects)
    h, w = denoised.shape[:2]
    min_dim = min(h, w)
    if min_dim < 512:
        scale = 512 / min_dim
        new_w, new_h = int(w * scale), int(h * scale)
        upsampled = cv2.resize(denoised, (new_w, new_h),
                              interpolation=cv2.INTER_CUBIC)
        return upsampled

    return denoised
```

**Update `process_image()` to use preprocessing (line 288):**
```python
# Load original image
original_image = self.load_image(file_path)

# ADD: Preprocess for better detection
preprocessed = self.preprocess_for_detection(original_image)

# Extract metadata from original (not preprocessed)
metadata = self.extract_metadata(file_path, original_image)

# Normalize preprocessed image for pose detection
normalized_image = self.normalize_image(
    preprocessed,  # CHANGED: was original_image
    target_size=target_size,
    convert_to_rgb=True
)
```

#### 2.2 Test Preprocessing

```bash
# Test on challenging images (low contrast, small subjects, noisy)
python3 -c "
from pathlib import Path
import cv2
import numpy as np
from src.core.image_ingestor import ImageIngestor
from src.core.pose_detector import RTMWCocktail14Detector
from config.settings import settings

ingestor = ImageIngestor()
detector = RTMWCocktail14Detector(
    config_file=str(settings.PROJECT_ROOT / 'data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py'),
    checkpoint_file=str(settings.PROJECT_ROOT / 'data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth')
)

# Test image
test_img_path = Path('test-photos/test1.jpg')
original = cv2.imread(str(test_img_path))
preprocessed = ingestor.preprocess_for_detection(original)

# Detect on both
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

poses_orig = detector.detect(original_rgb)
poses_prep = detector.detect(preprocessed_rgb)

print(f'Original: {len(poses_orig)} poses, conf: {poses_orig[0].overall_confidence if poses_orig else 0:.3f}')
print(f'Preprocessed: {len(poses_prep)} poses, conf: {poses_prep[0].overall_confidence if poses_prep else 0:.3f}')
"
```

**Success Criteria:**
- ✅ Preprocessing improves detection confidence by 5-10%
- ✅ No visual artifacts in preprocessed images
- ✅ Performance impact <10%

---

## Phase 3: Two-Stage Detection

### Why Third?
- Medium complexity
- Good accuracy gain (5-10%)
- Acceptable performance cost (-20%)
- Improves multi-person scenarios

### Implementation Steps

#### 3.1 Install YOLOv8

```bash
pip install ultralytics
```

#### 3.2 Create `src/core/two_stage_detector.py`

```python
"""
Two-stage pose detection: Person detection → Pose estimation.
Improves accuracy by detecting persons first, then running pose on crops.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from ultralytics import YOLO

from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TwoStageDetector:
    """
    Two-stage detection pipeline:
    1. YOLO person detection (bounding boxes)
    2. Pose estimation on person crops (higher effective resolution)
    """

    def __init__(
        self,
        pose_detector: RTMWCocktail14Detector,
        person_model: str = "yolov8n.pt",  # Nano model (6MB, fast)
        min_person_conf: float = 0.3,
        crop_padding: float = 0.1  # 10% padding around bbox
    ):
        """
        Initialize two-stage detector.

        Args:
            pose_detector: Existing RTMWCocktail14Detector instance
            person_model: YOLO model for person detection
            min_person_conf: Minimum confidence for person detections
            crop_padding: Padding ratio around person bbox
        """
        self.pose_detector = pose_detector
        self.yolo = YOLO(person_model)
        self.min_person_conf = min_person_conf
        self.crop_padding = crop_padding

        logger.info(f"TwoStageDetector initialized with {person_model}")

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect poses using two-stage pipeline.

        Args:
            image: Input image in RGB format

        Returns:
            List of PoseResult objects
        """
        # Stage 1: Detect persons
        results = self.yolo(image, classes=[0], verbose=False)  # class 0 = person

        if not results or len(results[0].boxes) == 0:
            logger.debug("No persons detected by YOLO")
            return []

        all_poses = []

        # Stage 2: Pose estimation on each person crop
        for i, box in enumerate(results[0].boxes):
            if box.conf[0] < self.min_person_conf:
                continue

            # Get bbox with padding
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            crop, offset = self._crop_with_padding(image, x1, y1, x2, y2)

            # Detect pose on crop
            poses = self.pose_detector.detect(crop)

            # Map keypoints back to original image coordinates
            for pose in poses:
                pose.keypoints[:, 0] += offset[0]  # x offset
                pose.keypoints[:, 1] += offset[1]  # y offset
                pose.bbox[0] += offset[0]  # bbox x
                pose.bbox[1] += offset[1]  # bbox y
                pose.person_id = i  # Use YOLO detection order as person_id
                all_poses.append(pose)

        return all_poses

    def _crop_with_padding(
        self,
        image: np.ndarray,
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop image with padding around bbox.

        Returns:
            (cropped_image, (x_offset, y_offset))
        """
        h, w = image.shape[:2]

        # Add padding
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_x = bbox_w * self.crop_padding
        pad_y = bbox_h * self.crop_padding

        # Clamp to image bounds
        x1_pad = int(max(0, x1 - pad_x))
        y1_pad = int(max(0, y1 - pad_y))
        x2_pad = int(min(w, x2 + pad_x))
        y2_pad = int(min(h, y2 + pad_y))

        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        return crop, (x1_pad, y1_pad)
```

#### 3.3 Modify `src/swift_bridge.py`

**Add import and init option:**
```python
from core.two_stage_detector import TwoStageDetector

def __init__(self, db_url: Optional[str] = None, use_two_stage: bool = False):
    # ... existing init ...

    if use_two_stage:
        self.detector = TwoStageDetector(self.pose_detector)
    else:
        self.detector = self.pose_detector
```

**Use unified detector interface:**
```python
# In index_directory() and detect_pose()
poses = self.detector.detect(image_rgb)  # Works for both single and two-stage
```

#### 3.4 Test Two-Stage

```bash
# Test on multi-person image
python3 -c "
from src.swift_bridge import PostureKitBridge

# Single-stage
bridge_single = PostureKitBridge(use_two_stage=False)
result_single = bridge_single.detect_pose_from_file('test-photos/multi_person.jpg')

# Two-stage
bridge_two = PostureKitBridge(use_two_stage=True)
result_two = bridge_two.detect_pose_from_file('test-photos/multi_person.jpg')

print(f'Single-stage: conf={result_single[\"confidence\"]:.3f}')
print(f'Two-stage: conf={result_two[\"confidence\"]:.3f}')
"
```

**Success Criteria:**
- ✅ Two-stage improves keypoint accuracy by 5-10%
- ✅ Better handling of multi-person images
- ✅ Performance within 20% slowdown

---

## Phase 4: Ensemble Detection

### Why Last?
- Most complex implementation
- Requires downloading additional model (~400MB)
- Highest performance cost (-60%)
- But delivers 10-15% accuracy gain

### Implementation Steps

#### 4.1 Download Additional Model

Options:
1. **RTMW-X** (larger RTMW): ~400MB, similar architecture
2. **ViTPose-H** (different architecture): ~600MB, better diversity

**Recommendation:** RTMW-X (easier integration, proven compatibility)

```bash
# Download RTMW-X
cd data/models
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmw-x_simcc-cocktail14_270e-256x192-13a2546d_20231208.pth
wget https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb320-270e_cocktail14-256x192.py
```

#### 4.2 Create `src/core/ensemble_detector.py`

```python
"""
Ensemble pose detection using multiple models with weighted voting.
Improves robustness by combining predictions from different models.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detection."""
    models: List[Dict[str, any]]  # List of {config, checkpoint, weight}
    fusion_method: str = "weighted_average"  # or "confidence_weighted"
    min_agreement: float = 0.5  # Minimum overlap to consider valid


class EnsembleDetector:
    """
    Ensemble of multiple pose detection models.
    Fuses predictions using weighted averaging.
    """

    def __init__(self, config: EnsembleConfig):
        """
        Initialize ensemble with multiple models.

        Args:
            config: EnsembleConfig with model specifications
        """
        self.config = config
        self.detectors = []

        # Load all models
        for model_spec in config.models:
            detector = RTMWCocktail14Detector(
                config_file=model_spec['config'],
                checkpoint_file=model_spec['checkpoint']
            )
            self.detectors.append({
                'detector': detector,
                'weight': model_spec.get('weight', 1.0)
            })

        logger.info(f"Ensemble initialized with {len(self.detectors)} models")

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect poses using ensemble of models.

        Args:
            image: Input image in RGB format

        Returns:
            List of fused PoseResult objects
        """
        # Get predictions from all models
        all_predictions = []
        for det_spec in self.detectors:
            poses = det_spec['detector'].detect(image)
            if poses:
                all_predictions.append({
                    'poses': poses,
                    'weight': det_spec['weight']
                })

        if not all_predictions:
            return []

        # Fuse predictions
        if self.config.fusion_method == "weighted_average":
            return self._fuse_weighted_average(all_predictions)
        elif self.config.fusion_method == "confidence_weighted":
            return self._fuse_confidence_weighted(all_predictions)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

    def _fuse_weighted_average(self, predictions: List[Dict]) -> List[PoseResult]:
        """
        Fuse predictions using weighted average of keypoints.
        """
        # For simplicity, take first detection from each model
        # (More sophisticated: match poses by IoU before fusing)

        total_weight = sum(p['weight'] for p in predictions)

        # Initialize with first prediction
        base_pose = predictions[0]['poses'][0]
        fused_keypoints = np.zeros_like(base_pose.keypoints)
        fused_visibility = np.zeros_like(base_pose.visibility)

        # Weighted average of keypoints
        for pred in predictions:
            if not pred['poses']:
                continue
            pose = pred['poses'][0]
            weight = pred['weight'] / total_weight
            fused_keypoints += pose.keypoints * weight
            fused_visibility += pose.visibility * weight

        # Create fused result
        fused = PoseResult(
            keypoints=fused_keypoints,
            visibility=fused_visibility,
            bbox=base_pose.bbox,
            overall_confidence=float(np.mean([p['poses'][0].overall_confidence
                                              for p in predictions if p['poses']])),
            person_id=base_pose.person_id
        )

        return [fused]

    def _fuse_confidence_weighted(self, predictions: List[Dict]) -> List[PoseResult]:
        """
        Fuse using confidence-weighted averaging.
        Higher confidence predictions get more weight.
        """
        # Similar to weighted_average, but use per-keypoint confidence
        # Implementation: weight each keypoint by its visibility score

        base_pose = predictions[0]['poses'][0]
        fused_keypoints = np.zeros_like(base_pose.keypoints)
        total_confidence = np.zeros((len(base_pose.keypoints),))

        for pred in predictions:
            if not pred['poses']:
                continue
            pose = pred['poses'][0]
            # Weight by both model weight and per-keypoint visibility
            weights = pose.visibility * pred['weight']
            fused_keypoints += pose.keypoints * weights[:, np.newaxis]
            total_confidence += weights

        # Normalize
        fused_keypoints /= (total_confidence[:, np.newaxis] + 1e-6)
        fused_visibility = total_confidence / sum(p['weight'] for p in predictions)

        fused = PoseResult(
            keypoints=fused_keypoints,
            visibility=fused_visibility,
            bbox=base_pose.bbox,
            overall_confidence=float(np.mean(fused_visibility)),
            person_id=base_pose.person_id
        )

        return [fused]
```

#### 4.3 Modify `src/swift_bridge.py`

```python
from core.ensemble_detector import EnsembleDetector, EnsembleConfig

def __init__(self, db_url: Optional[str] = None, use_ensemble: bool = False):
    # ... existing code ...

    if use_ensemble:
        config = EnsembleConfig(
            models=[
                {
                    'config': str(settings.PROJECT_ROOT / 'data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py'),
                    'checkpoint': str(settings.PROJECT_ROOT / 'data/models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'),
                    'weight': 1.0
                },
                {
                    'config': str(settings.PROJECT_ROOT / 'data/models/rtmw-x_8xb320-270e_cocktail14-256x192.py'),
                    'checkpoint': str(settings.PROJECT_ROOT / 'data/models/rtmw-x_simcc-cocktail14_270e-256x192-13a2546d_20231208.pth'),
                    'weight': 1.2  # Slightly higher weight for larger model
                }
            ],
            fusion_method="confidence_weighted"
        )
        self.detector = EnsembleDetector(config)
    else:
        self.detector = self.pose_detector
```

#### 4.4 Test Ensemble

```bash
# Compare single vs ensemble
python3 -c "
from src.swift_bridge import PostureKitBridge
import time

# Single model
bridge_single = PostureKitBridge(use_ensemble=False)
start = time.time()
result_single = bridge_single.detect_pose_from_file('test-photos/test1.jpg')
time_single = time.time() - start

# Ensemble
bridge_ensemble = PostureKitBridge(use_ensemble=True)
start = time.time()
result_ensemble = bridge_ensemble.detect_pose_from_file('test-photos/test1.jpg')
time_ensemble = time.time() - start

print(f'Single: conf={result_single[\"confidence\"]:.3f}, time={time_single:.2f}s')
print(f'Ensemble: conf={result_ensemble[\"confidence\"]:.3f}, time={time_ensemble:.2f}s')
print(f'Slowdown: {time_ensemble/time_single:.1f}x')
"
```

**Success Criteria:**
- ✅ Ensemble improves confidence by 10-15%
- ✅ More robust keypoint localization
- ✅ Performance within 60% slowdown (1.6x slower)

---

## Testing Strategy

### Unit Tests
Each component should be tested independently:

1. **Visual Features:**
   - Extract features from test image
   - Verify 576-dim output
   - Check L2 normalization

2. **Preprocessing:**
   - Test CLAHE, denoising, upsampling
   - Verify no artifacts
   - Measure confidence improvement

3. **Two-Stage:**
   - Test person detection
   - Verify coordinate mapping
   - Compare accuracy vs single-stage

4. **Ensemble:**
   - Test weighted fusion
   - Verify prediction agreement
   - Measure accuracy improvement

### Integration Tests

```bash
# Full pipeline test
python3 -c "
from src.swift_bridge import PostureKitBridge

# Initialize with all features
bridge = PostureKitBridge(use_ensemble=True)  # Ensemble includes preprocessing

# Index test directory
result = bridge.index_directory('test-photos', recursive=False)
print(f'Indexed: {result[\"poses_indexed\"]} poses')

# Search test
import cv2
test_img = cv2.imread('test-photos/test1.jpg')
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
pose_data = bridge.detect_pose(test_img_rgb)
features = bridge.extract_features(pose_data)
similar = bridge.search_similar(features['feature_vector'], k=5)

print(f'Found {len(similar)} similar poses')
for i, match in enumerate(similar, 1):
    print(f'{i}. {match[\"image_path\"]}: {match[\"similarity_score\"]:.3f}')
"
```

### Performance Benchmarks

```bash
# Benchmark each improvement
python3 benchmark_accuracy.py --baseline  # Current system
python3 benchmark_accuracy.py --visual    # +Visual features
python3 benchmark_accuracy.py --preprocess  # +Preprocessing
python3 benchmark_accuracy.py --two-stage   # +Two-stage
python3 benchmark_accuracy.py --ensemble    # All features
```

Create `benchmark_accuracy.py`:
```python
"""Benchmark accuracy improvements."""
import argparse
import time
from pathlib import Path
from src.swift_bridge import PostureKitBridge

def benchmark(use_visual=False, use_preprocess=False, use_two_stage=False, use_ensemble=False):
    bridge = PostureKitBridge(
        use_two_stage=use_two_stage,
        use_ensemble=use_ensemble
    )

    test_images = list(Path('test-photos').glob('*.jpg'))

    total_time = 0
    total_confidence = 0
    detected = 0

    for img_path in test_images:
        start = time.time()
        result = bridge.detect_pose_from_file(str(img_path))
        elapsed = time.time() - start

        if result:
            total_time += elapsed
            total_confidence += result['confidence']
            detected += 1

    avg_time = total_time / len(test_images)
    avg_conf = total_confidence / detected if detected > 0 else 0

    print(f"Images: {len(test_images)}")
    print(f"Detected: {detected}")
    print(f"Avg confidence: {avg_conf:.3f}")
    print(f"Avg time: {avg_time:.3f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--two-stage', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()

    if args.baseline:
        print("=== BASELINE ===")
        benchmark()
    elif args.visual:
        print("=== VISUAL FEATURES ===")
        benchmark(use_visual=True)
    elif args.preprocess:
        print("=== PREPROCESSING ===")
        benchmark(use_preprocess=True)
    elif args.two-stage:
        print("=== TWO-STAGE ===")
        benchmark(use_two_stage=True)
    elif args.ensemble:
        print("=== ENSEMBLE ===")
        benchmark(use_ensemble=True)
```

---

## Rollout Strategy

### Development
1. ✅ Document plan (this file)
2. Initialize git repo
3. Create feature branch
4. Implement Phase 1 (Visual Features)
5. Test and commit
6. Implement Phase 2 (Preprocessing)
7. Test and commit
8. Implement Phase 3 (Two-Stage)
9. Test and commit
10. Implement Phase 4 (Ensemble)
11. Test and commit

### Deployment
- Features will be opt-in via flags
- Default: Visual features enabled, others optional
- Users can enable ensemble for quality-critical use cases

### Backward Compatibility
- Existing geometric-only index still works
- Migration script to rebuild index with visual features
- No database schema changes needed (tables already exist)

---

## Git Workflow

```bash
# Initialize repo (if not already)
git init
git add .
git commit -m "Initial commit before accuracy improvements"

# Create feature branch
git checkout -b feature/accuracy-improvements

# Commit after each phase
git add src/swift_bridge.py src/intelligence/similarity_engine.py
git commit -m "Phase 1: Enable visual features integration"

git add src/core/image_ingestor.py
git commit -m "Phase 2: Add enhanced preprocessing (CLAHE, denoising, upsampling)"

git add src/core/two_stage_detector.py
git commit -m "Phase 3: Implement two-stage detection with YOLO"

git add src/core/ensemble_detector.py
git commit -m "Phase 4: Add ensemble detection with weighted fusion"

# Push to remote
git push -u origin feature/accuracy-improvements
```

---

## Rollback Plan

If any phase causes issues:

```bash
# Revert to previous commit
git revert HEAD

# Or reset to before feature branch
git checkout main
git branch -D feature/accuracy-improvements
```

Database rollback:
```sql
-- Clear visual/fused features if needed
TRUNCATE TABLE visual_features CASCADE;
TRUNCATE TABLE fused_features CASCADE;

-- Rebuild geometric-only index
python3 -c "
from src.intelligence.similarity_engine import SimilarityEngine
from src.storage.storage_manager import StorageManager
storage = StorageManager()
engine = SimilarityEngine(storage)
engine.build_index(force_rebuild=True)
"
```

---

## Success Metrics

### Accuracy
- Pose detection confidence improves by 5-15% (per phase)
- Combined improvement: 30-50%
- Search results more relevant (user feedback)

### Performance
- Indexing: 2-3x slower (acceptable)
- Search: No change (uses pre-computed features)
- Memory: +200MB for visual extractor

### Quality
- Better handling of:
  - Low-contrast images (preprocessing)
  - Multi-person scenes (two-stage)
  - Challenging poses (ensemble)
  - Visual similarity (visual features)

---

## Next Steps

1. ✅ Complete this documentation
2. Initialize git repository
3. Create feature branch
4. Implement Phase 1 (Visual Features) - **START HERE**
5. Progressive implementation of remaining phases

**Current Status:** Documentation complete, ready to begin Phase 1 implementation.
