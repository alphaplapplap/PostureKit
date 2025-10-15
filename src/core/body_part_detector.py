"""
Body Part Detector for PostureKit - NudeNet Integration.

Detects semantic body parts using NudeNet (YOLOv8-based) to complement
COCO-WholeBody keypoints with region-level bounding boxes.

18 Detection Classes:
- FACE_MALE, FACE_FEMALE
- BELLY_EXPOSED, BELLY_COVERED
- FEET_EXPOSED, FEET_COVERED
- ARMPITS_EXPOSED, ARMPITS_COVERED
- FEMALE_BREAST_EXPOSED, FEMALE_BREAST_COVERED, MALE_BREAST_EXPOSED
- BUTTOCKS_EXPOSED, BUTTOCKS_COVERED
- FEMALE_GENITALIA_EXPOSED, FEMALE_GENITALIA_COVERED, MALE_GENITALIA_EXPOSED
- ANUS_EXPOSED, ANUS_COVERED
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy import NudeNet to avoid loading model at module import time
_NUDENET_DETECTOR = None


def _get_detector(model_size: str = "320n"):
    """Lazy-load NudeNet detector (singleton pattern)."""
    global _NUDENET_DETECTOR
    if _NUDENET_DETECTOR is None:
        try:
            from nudenet import NudeDetector
            # NudeNet 3.4.2+ uses default model automatically
            _NUDENET_DETECTOR = NudeDetector()
            logger.info(f"NudeNet detector loaded (model_size parameter ignored in v3.4.2+)")
        except ImportError as e:
            logger.error(f"Failed to import NudeNet: {e}")
            logger.error("Install with: pip install nudenet")
            raise ImportError("nudenet is not installed. Run: pip install nudenet>=3.4.2")
    return _NUDENET_DETECTOR


@dataclass
class BodyPartDetection:
    """
    Single body part detection result.

    Attributes:
        part_name: NudeNet class label (e.g., "FACE_MALE", "FEET_EXPOSED")
        confidence: Detection confidence score (0-1)
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        is_exposed: True if part has "_EXPOSED" suffix (vs "_COVERED")
        canonical_region: Simplified region name (face, torso, feet, etc.)
    """
    part_name: str
    confidence: float
    bbox: np.ndarray  # (4,) array: [x1, y1, x2, y2]
    is_exposed: bool
    canonical_region: str

    def __post_init__(self):
        """Validate detection data."""
        assert 0.0 <= self.confidence <= 1.0, \
            f"Confidence must be in [0, 1], got {self.confidence}"
        assert self.bbox.shape == (4,), \
            f"BBox must be (4,), got {self.bbox.shape}"
        assert self.bbox[0] < self.bbox[2], "Invalid bbox: x1 >= x2"
        assert self.bbox[1] < self.bbox[3], "Invalid bbox: y1 >= y2"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'part_name': self.part_name,
            'confidence': float(self.confidence),
            'bbox': self.bbox.tolist(),
            'is_exposed': self.is_exposed,
            'canonical_region': self.canonical_region
        }

    def get_area(self) -> float:
        """Calculate bounding box area in pixels."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def get_center(self) -> Tuple[float, float]:
        """Get bounding box center point."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)


class BodyPartDetector:
    """
    Detects semantic body parts using NudeNet (YOLOv8-based).

    Provides region-level bounding boxes that complement COCO-WholeBody keypoints:
    - Keypoints: Precise joint locations (133 points)
    - Body Parts: Semantic regions with spatial extent (18 classes)

    Usage:
        detector = BodyPartDetector(model_size="320n")
        detections = detector.detect(image_rgb)
        canonical = detector.get_canonical_regions(detections)
    """

    # Mapping from NudeNet labels to canonical region names
    REGION_MAPPING = {
        'face': ['FACE_MALE', 'FACE_FEMALE'],
        'torso': ['BELLY_EXPOSED', 'BELLY_COVERED',
                  'FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED',
                  'MALE_BREAST_EXPOSED'],
        'feet': ['FEET_EXPOSED', 'FEET_COVERED'],
        'armpits': ['ARMPITS_EXPOSED', 'ARMPITS_COVERED'],
        'buttocks': ['BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED'],
        'genitalia': ['FEMALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_COVERED',
                     'MALE_GENITALIA_EXPOSED'],
        'anus': ['ANUS_EXPOSED', 'ANUS_COVERED']
    }

    # Reverse mapping: label -> canonical region
    LABEL_TO_REGION = {}
    for region, labels in REGION_MAPPING.items():
        for label in labels:
            LABEL_TO_REGION[label] = region

    # Default category-specific confidence thresholds (optimized for clothed subjects)
    # These values are tuned based on empirical testing to maximize coverage while
    # maintaining detection quality. Categories with lower defaults (0.15-0.20) are
    # harder to detect or commonly occluded by clothing.
    DEFAULT_THRESHOLDS = {
        'FACE_MALE': 0.20,
        'FACE_FEMALE': 0.20,
        'BELLY_EXPOSED': 0.15,
        'BELLY_COVERED': 0.15,
        'FEET_EXPOSED': 0.25,
        'FEET_COVERED': 0.25,
        'ARMPITS_EXPOSED': 0.15,
        'ARMPITS_COVERED': 0.15,
        'FEMALE_BREAST_EXPOSED': 0.35,
        'FEMALE_BREAST_COVERED': 0.35,
        'MALE_BREAST_EXPOSED': 0.25,
        'BUTTOCKS_EXPOSED': 0.25,
        'BUTTOCKS_COVERED': 0.25,
        'FEMALE_GENITALIA_EXPOSED': 0.35,
        'FEMALE_GENITALIA_COVERED': 0.35,
        'MALE_GENITALIA_EXPOSED': 0.35,
        'ANUS_EXPOSED': 0.35,
        'ANUS_COVERED': 0.35,
    }

    def __init__(
        self,
        model_size: str = "320n",
        min_confidence: float = 0.3,
        category_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Body Part Detector.

        Args:
            model_size: "320n" (fast, ~15 FPS on CPU) or "640m" (accurate, ~8 FPS)
            min_confidence: Minimum confidence threshold for detections (fallback)
            category_thresholds: Per-category confidence thresholds (18 classes).
                                If None, uses DEFAULT_THRESHOLDS. Pass empty dict
                                to disable adaptive thresholds.
        """
        self.model_size = model_size
        self.min_confidence = min_confidence
        self.detector = None  # Lazy-loaded

        # Use custom thresholds or defaults
        self.category_thresholds = (
            category_thresholds if category_thresholds is not None
            else self.DEFAULT_THRESHOLDS.copy()
        )

        logger.info(
            f"BodyPartDetector initialized",
            extra={'extra_data': {
                'model_size': model_size,
                'min_confidence': min_confidence,
                'using_adaptive_thresholds': bool(self.category_thresholds)
            }}
        )

    def detect(
        self,
        image: np.ndarray,
        min_confidence: Optional[float] = None,
        use_adaptive: bool = True
    ) -> List[BodyPartDetection]:
        """
        Detect body parts in image with optional adaptive thresholds.

        Args:
            image: RGB image as numpy array (H, W, 3)
            min_confidence: Override default confidence threshold (fallback for unknown categories)
            use_adaptive: If True, use per-category thresholds from self.category_thresholds

        Returns:
            List of body part detections with bounding boxes
        """
        if min_confidence is None:
            min_confidence = self.min_confidence

        # Lazy-load detector
        if self.detector is None:
            self.detector = _get_detector(self.model_size)

        try:
            # NudeNet expects RGB image, returns list of dicts
            raw_detections = self.detector.detect(image)

            # Convert to BodyPartDetection objects with adaptive filtering
            detections = []
            for d in raw_detections:
                conf = d['score']
                part_name = d['class']

                # Determine threshold for this detection
                if use_adaptive and self.category_thresholds:
                    # Use category-specific threshold (fallback to global if not found)
                    threshold = self.category_thresholds.get(part_name, min_confidence)
                else:
                    # Use global threshold
                    threshold = min_confidence

                # Filter by threshold
                if conf < threshold:
                    continue

                # NudeNet returns [x, y, width, height] format (XYWH)
                # Convert to [x1, y1, x2, y2] format (XYXY) for consistency
                box_xywh = d['box']
                x, y, w, h = box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)  # [x1, y1, x2, y2]

                is_exposed = '_EXPOSED' in part_name
                canonical_region = self.LABEL_TO_REGION.get(part_name, 'unknown')

                detection = BodyPartDetection(
                    part_name=part_name,
                    confidence=conf,
                    bbox=bbox,
                    is_exposed=is_exposed,
                    canonical_region=canonical_region
                )
                detections.append(detection)

            logger.debug(
                f"Detected {len(detections)}/{len(raw_detections)} body parts",
                extra={'extra_data': {
                    'num_detections': len(detections),
                    'num_raw': len(raw_detections),
                    'min_confidence': min_confidence,
                    'adaptive': use_adaptive
                }}
            )

            return detections

        except Exception as e:
            logger.error(f"Body part detection failed: {e}", exc_info=True)
            return []

    def get_canonical_regions(
        self,
        detections: List[BodyPartDetection]
    ) -> Dict[str, BodyPartDetection]:
        """
        Map detections to canonical body regions.

        If multiple detections map to same region (e.g., BELLY_EXPOSED and BELLY_COVERED),
        returns the highest-confidence detection.

        Args:
            detections: List of body part detections

        Returns:
            Dict mapping canonical region name to best detection
            Example: {'face': <BodyPartDetection>, 'feet': <BodyPartDetection>, ...}
        """
        canonical = {}

        for detection in detections:
            region = detection.canonical_region
            if region == 'unknown':
                continue

            # Keep highest confidence detection per region
            if region not in canonical or detection.confidence > canonical[region].confidence:
                canonical[region] = detection

        logger.debug(
            f"Canonical regions: {list(canonical.keys())}",
            extra={'extra_data': {
                'regions': list(canonical.keys()),
                'num_regions': len(canonical)
            }}
        )

        return canonical

    def get_visible_regions(
        self,
        detections: List[BodyPartDetection],
        min_confidence: float = 0.5
    ) -> List[str]:
        """
        Get list of visible canonical regions above confidence threshold.

        Args:
            detections: List of body part detections
            min_confidence: Minimum confidence for "visible" classification

        Returns:
            List of canonical region names (e.g., ['face', 'feet', 'torso'])
        """
        canonical = self.get_canonical_regions(detections)
        return [
            region for region, det in canonical.items()
            if det.confidence >= min_confidence
        ]

    def filter_by_region(
        self,
        detections: List[BodyPartDetection],
        region: str
    ) -> List[BodyPartDetection]:
        """
        Filter detections to specific canonical region.

        Args:
            detections: List of body part detections
            region: Canonical region name ('face', 'torso', 'feet', etc.)

        Returns:
            List of detections belonging to specified region
        """
        return [d for d in detections if d.canonical_region == region]

    def get_exposure_summary(
        self,
        detections: List[BodyPartDetection]
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Get exposure status for each region.

        Returns:
            Dict mapping region to (is_exposed, confidence) tuple
            Example: {'torso': (True, 0.95), 'feet': (False, 0.82), ...}
        """
        canonical = self.get_canonical_regions(detections)
        return {
            region: (det.is_exposed, det.confidence)
            for region, det in canonical.items()
        }


class BodyPartDetectorError(Exception):
    """Base exception for Body Part Detector errors."""
    pass


# Convenience function for simple use cases
def detect_body_parts(
    image: np.ndarray,
    model_size: str = "320n",
    min_confidence: float = 0.3
) -> List[BodyPartDetection]:
    """
    Convenience function to detect body parts in single image.

    Args:
        image: RGB image as numpy array
        model_size: "320n" (fast) or "640m" (accurate)
        min_confidence: Minimum detection confidence

    Returns:
        List of body part detections
    """
    detector = BodyPartDetector(model_size=model_size, min_confidence=min_confidence)
    return detector.detect(image)
