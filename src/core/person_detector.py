"""
Person Detector using YOLOv8.
Detects individual people in images before pose estimation.
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import logging
import weakref

from ultralytics import YOLO
import torch

from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def _has_mps_module() -> bool:
    """
    Check if torch.mps module is available (PyTorch 2.1+).

    PyTorch 2.0.x has torch.backends.mps but not torch.mps module.
    PyTorch 2.1+ has both torch.backends.mps and torch.mps.

    Returns:
        True if torch.mps module exists, False otherwise
    """
    return hasattr(torch, 'mps')


@dataclass
class PersonDetection:
    """
    Detected person bounding box.

    Attributes:
        bbox: Bounding box [x, y, width, height]
        confidence: Detection confidence (0-1)
        person_id: Index of person in image
        crop: Cropped image of person (for pose detection)
    """
    bbox: np.ndarray  # (4,) [x, y, w, h]
    confidence: float
    person_id: int
    crop: Optional[np.ndarray] = None  # (H, W, 3)

    def __post_init__(self):
        """Validate detection."""
        assert self.bbox.shape == (4,), f"Expected bbox (4,), got {self.bbox.shape}"
        assert 0.0 <= self.confidence <= 1.0, f"Confidence must be [0,1], got {self.confidence}"


class PersonDetectorError(Exception):
    """Base exception for person detector errors."""
    pass


class YOLOPersonDetector:
    """
    Detects individual people in images using YOLOv8.

    Features:
    - Detects all people in image
    - Returns bounding boxes and cropped images
    - Filters by confidence threshold
    - Handles overlapping detections with NMS

    Attributes:
        model: YOLOv8 model
        confidence_threshold: Minimum confidence for detection
        device: Torch device (mps, cuda, cpu)
    """

    # Class-level model cache with weak references to prevent memory leaks
    _model_cache = weakref.WeakValueDictionary()

    def __init__(
        self,
        model_name: str = 'yolov8n.pt',  # nano model (fastest)
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize person detector.

        Args:
            model_name: YOLOv8 model ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
            confidence_threshold: Minimum confidence for person detection
            device: Device to use ('mps', 'cuda', 'cpu'). Uses settings default if None.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or settings.DEVICE
        self.model = None

        # Model path
        self.model_name = model_name

        logger.info(
            f"YOLOPersonDetector initialized",
            extra={'extra_data': {
                'model': model_name,
                'threshold': confidence_threshold,
                'device': self.device
            }}
        )

    def _load_model(self):
        """Load YOLO model (lazy loading with caching)."""
        if self.model is not None:
            return

        # Check cache first (TOCTOU-safe: use get() to avoid race)
        cache_key = (self.model_name, self.device)
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            self.model = cached_model
            logger.info(f"Reusing cached YOLO model: {self.model_name}")
            return

        logger.info(f"Loading YOLOv8 model: {self.model_name}")

        try:
            # Load model
            self.model = YOLO(self.model_name)

            # Set device
            if self.device == 'mps' and torch.backends.mps.is_available():
                # YOLOv8 automatically handles MPS
                logger.info("Using Apple Silicon MPS")
            elif self.device == 'cuda' and torch.cuda.is_available():
                logger.info("Using CUDA GPU")
            else:
                logger.info("Using CPU")

            # Cache the model
            self._model_cache[cache_key] = self.model

            logger.info("YOLOv8 model loaded successfully")

        except Exception as e:
            raise PersonDetectorError(f"Failed to load YOLO model: {e}") from e

    def detect_people(
        self,
        image: np.ndarray,
        return_crops: bool = True
    ) -> List[PersonDetection]:
        """
        Detect all people in image.

        Args:
            image: Input image (H, W, 3) in RGB format
            return_crops: Whether to return cropped person images

        Returns:
            List of PersonDetection objects, one per detected person
        """
        # Load model if needed
        if self.model is None:
            self._load_model()

        logger.debug(f"Detecting people in image shape: {image.shape}")

        try:
            # Run detection
            # classes=0 means only detect 'person' class
            results = self.model.predict(
                image,
                classes=[0],  # 0 = person in COCO dataset
                conf=self.confidence_threshold,
                verbose=False
            )

            detections = []

            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i, box in enumerate(boxes):
                    # Get bounding box in xyxy format
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Convert to xywh format
                    bbox = np.array([
                        x1,
                        y1,
                        x2 - x1,  # width
                        y2 - y1   # height
                    ], dtype=np.float32)

                    # Get confidence
                    confidence = float(box.conf[0])

                    # Crop person if requested
                    crop = None
                    if return_crops:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Add small padding (5%)
                        h, w = image.shape[:2]
                        pad_x = int((x2 - x1) * 0.05)
                        pad_y = int((y2 - y1) * 0.05)

                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(w, x2 + pad_x)
                        y2 = min(h, y2 + pad_y)

                        crop = image[y1:y2, x1:x2].copy()

                    detection = PersonDetection(
                        bbox=bbox,
                        confidence=confidence,
                        person_id=i,
                        crop=crop
                    )

                    detections.append(detection)

            # CRITICAL: Delete YOLO results to free GPU memory
            del results
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                if _has_mps_module():
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception:
                        # Ignore cache cleanup errors
                        pass
                # Else: PyTorch < 2.1, skip MPS cache cleanup
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(
                f"Detected {len(detections)} people",
                extra={'extra_data': {
                    'count': len(detections),
                    'confidences': [d.confidence for d in detections]
                }}
            )

            return detections

        except Exception as e:
            raise PersonDetectorError(f"Person detection failed: {e}") from e

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def unload_model(self) -> None:
        """Unload YOLO model from memory to free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache
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

            logger.info("YOLO model unloaded from memory")
