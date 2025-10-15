"""
Pose Detector for PostureKit - Cocktail14 RTMW-L Implementation.
Wraps MMPose 1.3.2 with Cocktail14 RTMW-L checkpoint for 133-keypoint detection.

Cocktail14 specifics:
- Trained on 14 combined datasets for robust generalization
- Uses COCO-WholeBody format (133 keypoints)
- Optimized for diverse poses and viewpoints
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
import weakref
import threading
import gc

# PyTorch patch is handled by core._torch_patch module (imported in swift_bridge.py)
# No need to patch here - the global patch is already applied

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples, PoseDataSample

from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def _has_mps_module() -> bool:
    """Check if torch.mps module is available (PyTorch 2.1+)."""
    return hasattr(torch, 'mps')


# Person detection (optional, for two-stage approach)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.info("ultralytics not available - person detection disabled")


@dataclass
class PoseResult:
    """
    Result from pose detection on a single person.

    Attributes:
        keypoints: Array of shape (133, 3) where each row is [x, y, confidence]
        visibility: Array of shape (133,) with COCO visibility flags:
                   0 = not labeled (not visible in image)
                   1 = labeled but occluded (person present but keypoint hidden)
                   2 = labeled and visible (keypoint clearly visible)
        bbox: Bounding box [x, y, width, height]
        overall_confidence: Mean confidence across all keypoints
        person_id: Index of person in image (0-based)
    """
    keypoints: np.ndarray  # (133, 3)
    visibility: np.ndarray  # (133,)
    bbox: np.ndarray       # (4,) [x, y, w, h]
    overall_confidence: float
    person_id: int

    def __post_init__(self):
        """Validate data after initialization."""
        assert self.keypoints.shape == (133, 3), \
            f"Expected keypoints shape (133, 3), got {self.keypoints.shape}"
        assert self.visibility.shape == (133,), \
            f"Expected visibility shape (133,), got {self.visibility.shape}"
        # Allow continuous visibility values [0, 2] for ensemble fusion
        assert np.all((self.visibility >= 0) & (self.visibility <= 2)), \
            f"Visibility must be in [0, 2], got range [{np.min(self.visibility)}, {np.max(self.visibility)}]"
        assert self.bbox.shape == (4,), \
            f"Expected bbox shape (4,), got {self.bbox.shape}"
        assert 0.0 <= self.overall_confidence <= 1.0, \
            f"Confidence must be in [0, 1], got {self.overall_confidence}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'keypoints': self.keypoints.tolist(),
            'visibility': self.visibility.tolist(),
            'bbox': self.bbox.tolist(),
            'overall_confidence': float(self.overall_confidence),
            'person_id': self.person_id,
        }

    def get_visible_keypoints(self) -> np.ndarray:
        """Get only visible keypoints (visibility == 2)."""
        return self.keypoints[self.visibility == 2]

    def count_visible(self) -> int:
        """Count visible keypoints."""
        return int((self.visibility == 2).sum())

    def count_occluded(self) -> int:
        """Count occluded keypoints."""
        return int((self.visibility == 1).sum())


class PoseDetectorError(Exception):
    """Base exception for Pose Detector errors."""
    pass


class ModelInitializationError(PoseDetectorError):
    """Raised when model fails to initialize."""
    pass


class DetectionError(PoseDetectorError):
    """Raised when pose detection fails."""
    pass


class RTMWCocktail14Detector:
    """
    Wrapper for MMPose Cocktail14 RTMW-L model.

    Cocktail14 Model Details:
    - Architecture: RTMW-L (large variant)
    - Training: 14 combined datasets (COCO, MPII, AIC, etc.)
    - Keypoints: 133 (COCO-WholeBody format)
    - Input size: 288x384 (height x width)

    Features:
    - Lazy loading (model loaded on first use)
    - Apple Silicon MPS optimization
    - Multi-person detection with NMS
    - Confidence-based filtering

    Attributes:
        config_file: Path to Cocktail14 config
        checkpoint_file: Path to Cocktail14 weights
        device: Torch device (mps, cuda, or cpu)
        detection_threshold: Minimum confidence for valid detection
    """

    # Cocktail14 default paths (can be overridden)
    DEFAULT_CONFIG = 'rtmpose-l_8xb32-270e_cocktail14-384x288.py'
    DEFAULT_CHECKPOINT = 'rtmpose-l_8xb32-270e_cocktail14-384x288.pth'

    # Class-level model cache with weak references to prevent memory leaks
    # Shared across all instances to avoid redundant model loading (each model ~500MB)
    _model_cache = weakref.WeakValueDictionary()
    _detector_cache = weakref.WeakValueDictionary()
    _cache_lock = threading.Lock()  # Protect cache access from concurrent loads

    def __init__(
        self,
        config_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        device: Optional[str] = None,
        detection_threshold: float = 0.3,
        use_person_detector: bool = True,
        storage_manager: Optional['StorageManager'] = None
    ):
        """
        Initialize Cocktail14 RTMW-L Detector.

        Args:
            config_file: Path to config file (uses default if None)
            checkpoint_file: Path to checkpoint (uses default if None)
            device: Device to use ('mps', 'cuda', 'cpu')
            detection_threshold: Minimum confidence for valid keypoint
            use_person_detector: Use YOLOv8 person detection (two-stage)
            storage_manager: StorageManager for bias correction learning (optional)
        """
        # Use provided paths or fall back to settings, then defaults
        self.config_file = config_file or getattr(
            settings, 'MMPOSE_CONFIG_PATH', self.DEFAULT_CONFIG
        )
        self.checkpoint_file = checkpoint_file or getattr(
            settings, 'MMPOSE_CHECKPOINT_PATH', self.DEFAULT_CHECKPOINT
        )
        self.device = device or getattr(settings, 'DEVICE', 'mps')
        self.detection_threshold = detection_threshold
        self.use_person_detector = use_person_detector and YOLO_AVAILABLE

        # Model is loaded lazily on first use
        self._model = None
        self._device_obj = None
        self._person_detector = None

        # Bias correction (if storage_manager provided)
        self.bias_corrector = None
        self.feature_extractor = None
        if storage_manager is not None:
            try:
                from src.learning.correction_learner import BiasCorrector
                from src.core.geometric_feature_extractor import GeometricFeatureExtractor

                self.bias_corrector = BiasCorrector(storage_manager)
                self.feature_extractor = GeometricFeatureExtractor()
                logger.info(f"BiasCorrector initialized with {len(self.bias_corrector.bias_table)} learned biases")
            except Exception as e:
                logger.warning(f"Failed to initialize BiasCorrector: {e}", exc_info=True)
                self.bias_corrector = None
                self.feature_extractor = None

        logger.info(
            f"RTMWCocktail14Detector initialized",
            extra={'extra_data': {
                'device': self.device,
                'threshold': self.detection_threshold,
                'config': str(self.config_file),
                'checkpoint': str(self.checkpoint_file),
                'person_detector': self.use_person_detector,
                'bias_correction': self.bias_corrector is not None
            }}
        )

    def _initialize_device(self) -> torch.device:
        """
        Initialize and validate torch device.

        Returns:
            Torch device object

        Raises:
            ModelInitializationError: If device is unavailable
        """
        try:
            if self.device == 'mps':
                if not torch.backends.mps.is_available():
                    logger.warning("MPS not available, falling back to CPU")
                    self.device = 'cpu'
                    return torch.device('cpu')
                device_obj = torch.device('mps')
                logger.info("Using Apple Silicon MPS (Metal Performance Shaders)")

            elif self.device == 'cuda':
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device = 'cpu'  # Update device string for status reporting
                    return torch.device('cpu')
                device_obj = torch.device('cuda')
                logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")

            else:
                device_obj = torch.device('cpu')
                logger.info("Using CPU")

            return device_obj

        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize device: {e}") from e

    def _load_model(self) -> None:
        """
        Load Cocktail14 RTMW-L model (lazy loading with caching).

        Raises:
            ModelInitializationError: If model loading fails
        """
        if self._model is not None:
            return  # Already loaded

        # Thread-safe cache check with double-check locking
        cache_key = (str(self.config_file), str(self.checkpoint_file), self.device)

        # Fast path: check cache without lock
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            self._model = cached_model
            self._device_obj = self._initialize_device()
            logger.info("Reusing cached Cocktail14 model (memory efficient)")
            return

        # Slow path: load model with lock to prevent duplicate loads
        with self._cache_lock:
            # Double-check inside lock
            cached_model = self._model_cache.get(cache_key)
            if cached_model is not None:
                self._model = cached_model
                self._device_obj = self._initialize_device()
                logger.info("Reusing cached Cocktail14 model (memory efficient, race avoided)")
                return

            logger.info("Loading Cocktail14 RTMW-L model...")

            try:
                # Validate paths
                config_path = Path(self.config_file)
                checkpoint_path = Path(self.checkpoint_file)

                if not checkpoint_path.exists():
                    raise ModelInitializationError(
                        f"Cocktail14 checkpoint not found at: {checkpoint_path}\n"
                        f"Download from: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                        f"rtmpose-l_simcc-cocktail14_pt-ucoco_270e-384x288-8d0b6c28_20230728.pth"
                    )

                if not config_path.exists():
                    raise ModelInitializationError(
                        f"Config file not found at: {config_path}\n"
                        f"Expected Cocktail14 config from MMPose configs"
                    )

                # Initialize device
                self._device_obj = self._initialize_device()
                device_str = str(self._device_obj)

                logger.info(f"Loading model on device: {device_str}")

                # Load model with MMPose 1.3.2 API
                # Suppress mmengine checkpoint loading messages to prevent stdout pollution
                import sys
                import os
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                devnull_stdout = open(os.devnull, 'w')
                devnull_stderr = open(os.devnull, 'w')
                try:
                    # Redirect to devnull during model loading
                    sys.stdout = devnull_stdout
                    sys.stderr = devnull_stderr

                    # torch.load is monkey-patched at module level to use weights_only=False
                    self._model = init_model(
                        str(config_path),
                        str(checkpoint_path),
                        device=device_str
                    )
                finally:
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    devnull_stdout.close()
                    devnull_stderr.close()

                # Explicitly set to eval mode to ensure inference-only behavior
                self._model.eval()

                # Store in cache for reuse by other detector instances (still inside lock)
                self._model_cache[cache_key] = self._model

                logger.info(
                    "Cocktail14 model loaded successfully",
                    extra={'extra_data': {
                        'config': str(config_path),
                        'weights': str(checkpoint_path),
                        'device': str(self._device_obj)
                    }}
                )

            except Exception as e:
                if isinstance(e, ModelInitializationError):
                    raise
                raise ModelInitializationError(f"Failed to load Cocktail14 model: {e}") from e

    def _extract_pose_results(
        self,
        data_samples: List[PoseDataSample],
        image_shape: Tuple[int, int]
    ) -> List[PoseResult]:
        """
        Extract PoseResult objects from MMPose 1.3.2 data samples.

        Args:
            data_samples: MMPose PoseDataSample objects
            image_shape: (height, width) of input image

        Returns:
            List of PoseResult objects, one per detected person
        """
        results = []

        if not data_samples:
            logger.debug("No poses detected in image")
            return results

        for person_idx, sample in enumerate(data_samples):
            try:
                pred_instances = sample.pred_instances

                # Debug: Log available fields in pred_instances
                logger.debug(f"pred_instances fields: {dir(pred_instances)}")
                if hasattr(pred_instances, 'keypoints_visible'):
                    logger.debug("pred_instances has keypoints_visible field!")

                # Extract keypoints (133, 2) and scores (133,)
                keypoints = pred_instances.keypoints  # (N, 133, 2) where N=1 usually
                keypoint_scores = pred_instances.keypoint_scores  # (N, 133)

                # Try to extract explicit visibility if available
                keypoints_visible = None
                if hasattr(pred_instances, 'keypoints_visible'):
                    keypoints_visible = pred_instances.keypoints_visible
                    logger.debug(f"Found keypoints_visible: shape={keypoints_visible.shape if hasattr(keypoints_visible, 'shape') else 'N/A'}")
                    if hasattr(keypoints_visible, 'shape'):
                        logger.debug(f"Raw keypoints_visible values (first 10): {keypoints_visible.flatten()[:10]}")

                # Handle batch dimension
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]  # Take first (and usually only) detection
                    keypoint_scores = keypoint_scores[0]
                    if keypoints_visible is not None and len(keypoints_visible.shape) == 2:
                        keypoints_visible = keypoints_visible[0]

                # Convert to numpy if tensor
                if torch.is_tensor(keypoints):
                    keypoints = keypoints.detach().cpu().numpy()
                if torch.is_tensor(keypoint_scores):
                    keypoint_scores = keypoint_scores.detach().cpu().numpy()
                if keypoints_visible is not None and torch.is_tensor(keypoints_visible):
                    keypoints_visible = keypoints_visible.detach().cpu().numpy()

                # Comprehensive array validation (prevent buffer overflow/corruption)
                if keypoints.shape != (133, 2):
                    logger.warning(
                        f"Invalid keypoint shape: {keypoints.shape}, expected (133, 2). Skipping detection."
                    )
                    continue

                if keypoint_scores.shape != (133,):
                    logger.warning(
                        f"Invalid keypoint_scores shape: {keypoint_scores.shape}, expected (133,). Skipping detection."
                    )
                    continue

                # Validate all values are finite (no inf/nan)
                if not np.all(np.isfinite(keypoints)):
                    logger.warning("Keypoints contain inf/nan values. Skipping detection.")
                    continue

                if not np.all(np.isfinite(keypoint_scores)):
                    logger.warning("Keypoint scores contain inf/nan values. Skipping detection.")
                    continue

                # Reasonable bounds check (prevent buffer overflow from corrupt data)
                if np.max(np.abs(keypoints)) > 100000:
                    logger.warning(
                        f"Keypoints outside reasonable bounds: max={np.max(np.abs(keypoints))}. Skipping detection."
                    )
                    continue

                # Debug: Log raw keypoint scores to diagnose confidence issue
                logger.debug(f"Raw keypoint_scores shape: {keypoint_scores.shape}")
                logger.debug(f"Keypoint scores - min: {keypoint_scores.min():.3f}, max: {keypoint_scores.max():.3f}, mean: {keypoint_scores.mean():.3f}")
                logger.debug(f"Sample keypoint scores (indices 15-22, ankles/feet): {keypoint_scores[15:23]}")

                # Fix coordinate space if MMPose didn't transform back to original
                # Keypoints should be in original image coordinates, but sometimes aren't
                x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
                img_h, img_w = image_shape

                # Log raw coordinates before any transformation
                logger.debug(f"Raw MMPose keypoints: x:[{x_coords.min():.1f}, {x_coords.max():.1f}], "
                           f"y:[{y_coords.min():.1f}, {y_coords.max():.1f}], image={img_w}x{img_h}")

                # Mark out-of-bounds keypoints as missing BEFORE clipping
                # Use 5px margin to account for rounding errors
                margin = 5
                out_of_bounds = (
                    (x_coords < -margin) | (x_coords > img_w + margin) |
                    (y_coords < -margin) | (y_coords > img_h + margin)
                )

                if out_of_bounds.any():
                    num_oob = out_of_bounds.sum()
                    # Mark as missing in visibility array
                    if keypoints_visible is not None:
                        keypoints_visible[out_of_bounds] = 0.0  # v=0 (missing)
                        logger.debug(f"Marked {num_oob} out-of-bounds keypoints as v=0 (missing)")
                    else:
                        # Fallback: set confidence to 0 if no visibility data
                        keypoint_scores[out_of_bounds] = 0.0
                        logger.debug(f"Marked {num_oob} out-of-bounds keypoints with 0 confidence")

                if (x_coords.min() < -10 or x_coords.max() > img_w + 10 or
                    y_coords.min() < -10 or y_coords.max() > img_h + 10):
                    # Clip to bounds for display purposes
                    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, img_w - 1)
                    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, img_h - 1)

                    logger.debug(f"Clipped coordinates to ({img_w}x{img_h})")

                # Combine into (133, 3) array [x, y, confidence] with pre-allocation
                keypoints_with_conf = np.empty((133, 3), dtype=np.float32)
                keypoints_with_conf[:, :2] = keypoints
                keypoints_with_conf[:, 2] = np.clip(keypoint_scores, 0.0, 1.0)

                # Extract bounding box
                if hasattr(pred_instances, 'bboxes') and len(pred_instances.bboxes) > 0:
                    bbox = pred_instances.bboxes[0]  # (4,) or (5,) [x1, y1, x2, y2, (score)]
                    if torch.is_tensor(bbox):
                        bbox = bbox.detach().cpu().numpy()

                    # Convert [x1, y1, x2, y2] to [x, y, w, h]
                    bbox = np.array([
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ], dtype=np.float32)
                else:
                    # Fallback: compute bbox from keypoints
                    valid_keypoints = keypoints_with_conf[
                        keypoints_with_conf[:, 2] > self.detection_threshold
                    ]
                    if len(valid_keypoints) > 0:
                        x_min, y_min = valid_keypoints[:, :2].min(axis=0)
                        x_max, y_max = valid_keypoints[:, :2].max(axis=0)
                        bbox = np.array([
                            x_min, y_min, x_max - x_min, y_max - y_min
                        ], dtype=np.float32)
                    else:
                        # Fallback: compute bbox from visible keypoints only
                        # Note: visibility computed below, so use threshold here
                        visible_keypoints = keypoints_with_conf[
                            keypoints_with_conf[:, 2] > self.detection_threshold
                        ]
                        if len(visible_keypoints) > 0:
                            x_min, y_min = visible_keypoints[:, :2].min(axis=0)
                            x_max, y_max = visible_keypoints[:, :2].max(axis=0)
                            bbox = np.array([
                                x_min, y_min, x_max - x_min, y_max - y_min
                            ], dtype=np.float32)
                        else:
                            bbox = np.array([0, 0, image_shape[1], image_shape[0]], dtype=np.float32)

                # Convert MMPose visibility confidence scores to COCO format
                # MMPose returns confidence scores (0.0-1.0), not COCO flags
                visibility_array = np.zeros(133, dtype=np.int8)
                if keypoints_visible is not None:
                    # Convert confidence to COCO visibility: 0=missing, 1=occluded, 2=visible
                    visibility_array[keypoints_visible >= 0.5] = 2  # Visible
                    visibility_array[(keypoints_visible >= 0.1) & (keypoints_visible < 0.5)] = 1  # Occluded
                    visibility_array[keypoints_visible < 0.1] = 0  # Missing

                    visible_count = (visibility_array == 2).sum()
                    occluded_count = (visibility_array == 1).sum()
                    missing_count = (visibility_array == 0).sum()
                    logger.debug(f"MMPose visibility: {visible_count} visible, {occluded_count} occluded, {missing_count} missing")
                else:
                    # Fallback: compute visibility from confidence scores
                    visibility_array[keypoint_scores >= 0.5] = 2  # Visible
                    visibility_array[(keypoint_scores >= 0.1) & (keypoint_scores < 0.5)] = 1  # Occluded
                    visibility_array[keypoint_scores < 0.1] = 0  # Missing

                    visible_count = (visibility_array == 2).sum()
                    occluded_count = (visibility_array == 1).sum()
                    missing_count = (visibility_array == 0).sum()
                    logger.debug(f"Computed visibility from confidence: {visible_count} visible, {occluded_count} occluded, {missing_count} missing")

                # Calculate overall confidence from only visible keypoints
                visible_scores = keypoint_scores[visibility_array == 2]
                if len(visible_scores) > 0:
                    overall_confidence = float(np.clip(visible_scores.mean(), 0.0, 1.0))
                else:
                    # No visible keypoints, fallback to all scores
                    overall_confidence = float(np.clip(keypoint_scores.mean(), 0.0, 1.0))

                result = PoseResult(
                    keypoints=keypoints_with_conf,
                    visibility=visibility_array,
                    bbox=bbox,
                    overall_confidence=overall_confidence,
                    person_id=person_idx
                )

                # Validate detection before adding
                if not self._validate_detection(result, image_shape):
                    logger.debug(f"Skipping invalid detection for person {person_idx}")
                    continue

                # Apply learned bias corrections if available
                if self.bias_corrector is not None and self.feature_extractor is not None:
                    try:
                        # Extract geometric features
                        geometric_features = self.feature_extractor.extract(result)

                        # Apply bias correction
                        corrected_keypoints = self.bias_corrector.apply_correction(
                            result.keypoints,
                            geometric_features.feature_vector
                        )

                        # Update result with corrected keypoints
                        result.keypoints = corrected_keypoints

                        logger.debug(f"Applied bias corrections to person {person_idx}")
                    except Exception as e:
                        logger.warning(f"Failed to apply bias correction: {e}")
                        # Continue with uncorrected keypoints

                results.append(result)

                logger.debug(
                    f"Extracted Cocktail14 pose for person {person_idx}",
                    extra={'extra_data': {
                        'confidence': overall_confidence,
                        'valid_keypoints': int((keypoint_scores > self.detection_threshold).sum())
                    }}
                )

            except Exception as e:
                logger.warning(
                    f"Failed to extract pose for person {person_idx}: {e}",
                    exc_info=True
                )
                continue

        return results

    def _validate_detection(self, result: PoseResult, image_shape: Tuple[int, int]) -> bool:
        """
        Validate if detection is a plausible human body.

        Filters out:
        - Hands/feet detected as bodies (too small bbox)
        - Partial detections with insufficient keypoints
        - Detections mostly out of frame

        Args:
            result: PoseResult to validate
            image_shape: (height, width) of image

        Returns:
            True if valid, False if should be rejected
        """
        keypoints = result.keypoints
        bbox = result.bbox
        img_h, img_w = image_shape

        # 1. Reject tiny bounding boxes (hands/feet mistaken as bodies)
        MIN_BBOX_SIZE = 50  # pixels
        if bbox[2] < MIN_BBOX_SIZE or bbox[3] < MIN_BBOX_SIZE:
            logger.debug(f"Rejected detection: bbox too small ({bbox[2]}x{bbox[3]})")
            return False

        # 2. Reject if mostly out of frame
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        if bbox_x + bbox_w < 0 or bbox_x > img_w or bbox_y + bbox_h < 0 or bbox_y > img_h:
            logger.debug("Rejected detection: bbox out of frame")
            return False

        # 3. Require minimum visible keypoints
        MIN_VISIBLE_KEYPOINTS = 8  # At least torso visible
        visible_count = (keypoints[:, 2] > self.detection_threshold).sum()
        if visible_count < MIN_VISIBLE_KEYPOINTS:
            logger.debug(f"Rejected detection: only {visible_count} visible keypoints")
            return False

        # 4. Check for plausible body structure (shoulders exist)
        l_shoulder_conf = keypoints[5, 2]
        r_shoulder_conf = keypoints[6, 2]
        has_shoulders = (l_shoulder_conf > self.detection_threshold or
                         r_shoulder_conf > self.detection_threshold)

        if not has_shoulders:
            logger.debug("Rejected detection: no shoulders detected")
            return False

        # 5. Check shoulder width is reasonable (not a hand)
        if l_shoulder_conf > self.detection_threshold and r_shoulder_conf > self.detection_threshold:
            shoulder_width = np.linalg.norm(keypoints[5, :2] - keypoints[6, :2])
            MIN_SHOULDER_WIDTH = 5  # pixels (relaxed from 20px for distant people)
            if shoulder_width < MIN_SHOULDER_WIDTH:
                logger.debug(f"Rejected detection: shoulder width too narrow ({shoulder_width:.1f}px)")
                return False

        return True

    def _load_person_detector(self) -> None:
        """Load YOLOv8 person detector (lazy loading with caching)."""
        if self._person_detector is not None:
            return  # Already loaded

        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, cannot use person detector")
            self.use_person_detector = False
            return

        # Check cache first
        detector_key = 'yolov8n.pt'
        if detector_key in self._detector_cache:
            self._person_detector = self._detector_cache[detector_key]
            logger.info("Reusing cached YOLOv8n detector (memory efficient)")
            return

        logger.info("Loading YOLOv8n person detector...")
        try:
            self._person_detector = YOLO('yolov8n.pt')  # Lightweight nano model
            self._detector_cache[detector_key] = self._person_detector
            logger.info("YOLOv8n person detector loaded")
        except Exception as e:
            logger.error(f"Failed to load person detector: {e}")
            self.use_person_detector = False
            self._person_detector = None

    def _detect_people(self, image: np.ndarray, min_conf: float = 0.5) -> List[np.ndarray]:
        """
        Detect people in image using YOLO.

        Args:
            image: Input image (H, W, 3) in RGB
            min_conf: Minimum confidence for person detection

        Returns:
            List of bboxes in [x1, y1, x2, y2] format
        """
        if self._person_detector is None:
            self._load_person_detector()

        if self._person_detector is None:
            return []  # Fallback to whole-image detection

        try:
            # Run YOLO detection (class 0 = person)
            results = self._person_detector(image, classes=[0], verbose=False)

            bboxes = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    if conf >= min_conf:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        bboxes.append(bbox)

            logger.debug(f"YOLO detected {len(bboxes)} people with conf >= {min_conf}")
            return bboxes

        except Exception as e:
            logger.warning(f"Person detection failed: {e}, falling back to whole-image")
            return []

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect poses in image using Cocktail14 RTMW-L.

        Uses two-stage detection if use_person_detector=True:
        1. YOLO detects person bboxes
        2. Pose estimation on each bbox

        Otherwise uses single-stage whole-image detection.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            List of PoseResult objects, one per detected person

        Raises:
            DetectionError: If detection fails
        """
        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        logger.debug(f"Running Cocktail14 detection on image shape: {image.shape}")

        try:
            # Two-stage: Person detection → Pose estimation
            if self.use_person_detector:
                person_bboxes = self._detect_people(image, min_conf=0.5)

                if person_bboxes:
                    logger.debug(f"Running pose estimation on {len(person_bboxes)} detected people")

                    # Run pose estimation on each person bbox
                    data_samples = inference_topdown(
                        self._model,
                        image,
                        bboxes=person_bboxes  # Pass YOLO bboxes to MMPose
                    )
                else:
                    # Fall back to single-stage if YOLO finds nothing
                    logger.info("No people detected by YOLO, falling back to single-stage detection")
                    data_samples = inference_topdown(
                        self._model,
                        image
                    )

            # Single-stage: Whole-image detection
            else:
                # Run inference using MMPose 1.3.2 API
                # Note: Cocktail14 expects specific input size (384x288)
                data_samples = inference_topdown(
                    self._model,
                    image
                )

            # Extract results
            results = self._extract_pose_results(data_samples, image.shape[:2])

            # CRITICAL: Delete data_samples to free GPU memory
            del data_samples
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                if _has_mps_module():
                    try:
                        torch.mps.synchronize()  # Wait for GPU operations to complete
                        torch.mps.empty_cache()
                    except Exception:
                        # Silently ignore if MPS methods fail
                        pass
                # Else: PyTorch < 2.1, skip MPS cache cleanup
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(
                f"Cocktail14 detected {len(results)} pose(s)",
                extra={'extra_data': {
                    'num_poses': len(results),
                    'confidences': [r.overall_confidence for r in results],
                    'method': 'two-stage' if self.use_person_detector else 'single-stage'
                }}
            )

            return results

        except Exception as e:
            if isinstance(e, (ModelInitializationError, DetectionError)):
                raise
            raise DetectionError(f"Cocktail14 detection failed: {e}") from e

    def detect_batch(self, images: List[np.ndarray], progress_callback=None) -> List[List[PoseResult]]:
        """
        Detect poses in multiple images using batch processing (GPU-optimized).

        Processes images sequentially but keeps GPU memory loaded for efficiency.
        Significantly faster than cold-starting for each image when using MPS/CUDA.

        Args:
            images: List of input images as numpy arrays (H, W, 3) in RGB format
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of Lists: For each image, a list of PoseResult objects

        Raises:
            DetectionError: If batch detection fails
        """
        if not images:
            return []

        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        logger.info(f"Running batch detection on {len(images)} images (device={self.device})")

        try:
            all_results = []

            # Process images sequentially with GPU memory staying hot
            for idx, image in enumerate(images):
                if progress_callback:
                    progress_callback(idx + 1, len(images))

                # Use standard detect - GPU stays loaded and warm
                results = self.detect(image)
                all_results.append(results)

                # Periodic GPU cache cleanup to prevent accumulation
                if idx % 10 == 0 and idx > 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        if _has_mps_module():
                            try:
                                torch.mps.empty_cache()
                            except Exception:
                                pass
                        # Else: PyTorch < 2.1, skip MPS cache cleanup

            logger.info(f"Batch detection complete: {len(images)} images processed")
            return all_results

        except Exception as e:
            raise DetectionError(f"Batch detection failed: {e}") from e

    def detect_single(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Detect single pose (returns highest confidence detection).

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            PoseResult for highest confidence detection, or None if no poses
        """
        results = self.detect(image)

        if not results:
            return None

        return max(results, key=lambda r: r.overall_confidence)

    def detect_from_crop(
        self,
        crop: np.ndarray,
        bbox_in_original: np.ndarray
    ) -> Optional[PoseResult]:
        """
        Detect pose from cropped person image.

        Args:
            crop: Cropped image of person (H, W, 3) in RGB
            bbox_in_original: Bounding box in original image [x, y, w, h]

        Returns:
            PoseResult with keypoints mapped back to original image coordinates
        """
        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        logger.debug(f"Detecting pose in crop shape: {crop.shape}")

        try:
            # Run inference on crop
            data_samples = inference_topdown(
                self._model,
                crop,
                bboxes=None
            )

            if not data_samples or len(data_samples) == 0:
                logger.debug("No pose detected in crop")
                return None

            # Extract first (should only be one person in crop)
            sample = data_samples[0]
            pred_instances = sample.pred_instances

            # Extract keypoints (133, 2) and scores (133,)
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores

            # Handle batch dimension
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]
                keypoint_scores = keypoint_scores[0]

            # Convert to numpy if tensor
            if torch.is_tensor(keypoints):
                keypoints = keypoints.detach().cpu().numpy()
            if torch.is_tensor(keypoint_scores):
                keypoint_scores = keypoint_scores.detach().cpu().numpy()

            # Map keypoints back to original image coordinates
            x_offset, y_offset = bbox_in_original[0], bbox_in_original[1]
            keypoints_original = keypoints.copy()
            keypoints_original[:, 0] += x_offset
            keypoints_original[:, 1] += y_offset

            # Combine into (133, 3) with pre-allocation
            keypoints_with_conf = np.empty((133, 3), dtype=np.float32)
            keypoints_with_conf[:, :2] = keypoints_original
            keypoints_with_conf[:, 2] = np.clip(keypoint_scores, 0.0, 1.0)

            # Overall confidence
            overall_confidence = float(np.clip(keypoint_scores.mean(), 0.0, 1.0))

            # Compute visibility from confidence scores
            visibility_array = np.zeros(133, dtype=np.int8)
            visibility_array[keypoint_scores >= 0.5] = 2  # Visible
            visibility_array[(keypoint_scores >= 0.1) & (keypoint_scores < 0.5)] = 1  # Occluded
            visibility_array[keypoint_scores < 0.1] = 0  # Missing

            # Create result
            result = PoseResult(
                keypoints=keypoints_with_conf,
                visibility=visibility_array,
                bbox=bbox_in_original.copy(),
                overall_confidence=overall_confidence,
                person_id=0  # Will be set by caller
            )

            logger.debug(f"Pose detected with confidence: {overall_confidence:.3f}")

            return result

        except Exception as e:
            logger.warning(f"Failed to detect pose in crop: {e}", exc_info=True)
            return None

    def is_model_loaded(self) -> bool:
        """Check if Cocktail14 model is loaded in memory."""
        return self._model is not None

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self._model is not None:
            # Don't delete if model is in cache (shared by other instances)
            self._model = None
            self._device_obj = None

        if self._person_detector is not None:
            self._person_detector = None

        # Force garbage collection to release memory
        gc.collect()

        # Clear GPU cache with retry logic (synchronize BEFORE empty_cache)
        for attempt in range(3):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for operations to finish first
                    torch.cuda.empty_cache()  # Then clear cache
                elif torch.backends.mps.is_available():
                    if _has_mps_module():
                        try:
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    # Else: PyTorch < 2.1, skip MPS cache cleanup
                break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Failed to clear GPU cache after {attempt + 1} attempts: {e}")
                else:
                    import time
                    time.sleep(0.1)

        logger.info("Models unloaded from memory")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except Exception:
            pass  # Suppress errors during cleanup


# Alias for backward compatibility
PoseDetector = RTMWCocktail14Detector
