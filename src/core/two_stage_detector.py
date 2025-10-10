"""
Two-Stage Pose Detection for PostureKit.

Improves accuracy by detecting persons first, then running pose estimation on crops.
Expected accuracy improvement: 5-10%
Performance impact: ~20% slower

Architecture:
1. YOLO person detection (bounding boxes)
2. Pose estimation on person crops (higher effective resolution)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics package required for two-stage detection. "
        "Install with: pip install ultralytics"
    )

from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TwoStageDetector:
    """
    Two-stage detection pipeline for improved pose accuracy.

    Stage 1: YOLO person detection (fast, robust bounding boxes)
    Stage 2: Pose estimation on person crops (higher effective resolution)

    Benefits:
    - Better handling of multi-person scenarios
    - Improved keypoint accuracy via higher effective resolution
    - More robust to background clutter

    Attributes:
        pose_detector: RTMWCocktail14Detector instance for pose estimation
        yolo: YOLO model for person detection
        min_person_conf: Minimum confidence for person detections
        crop_padding: Padding ratio around person bbox
    """

    def __init__(
        self,
        pose_detector: RTMWCocktail14Detector,
        person_model: str = "yolov8n.pt",  # Nano model (6MB, fast)
        min_person_conf: float = 0.3,
        crop_padding: float = 0.1,  # 10% padding around bbox
    ):
        """
        Initialize two-stage detector.

        Args:
            pose_detector: Existing RTMWCocktail14Detector instance
            person_model: YOLO model for person detection
                         Options: yolov8n.pt (6MB), yolov8s.pt (22MB), yolov8m.pt (52MB)
            min_person_conf: Minimum confidence for person detections (0.0-1.0)
            crop_padding: Padding ratio around person bbox (0.0-1.0)
                         0.1 = 10% padding around detected person
        """
        self.pose_detector = pose_detector
        self.min_person_conf = min_person_conf
        self.crop_padding = crop_padding

        # Load YOLO model (downloads automatically if not present)
        try:
            self.yolo = YOLO(person_model)
            logger.info(
                f"TwoStageDetector initialized: model={person_model}, "
                f"min_conf={min_person_conf}, padding={crop_padding}"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model {person_model}: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect poses using two-stage pipeline.

        Process:
        1. Detect all persons in image using YOLO
        2. For each person:
           - Crop region with padding
           - Run pose estimation on crop
           - Map keypoints back to original coordinates

        Args:
            image: Input image in RGB format (H, W, 3)

        Returns:
            List of PoseResult objects, one per detected person
        """
        # Stage 1: Detect persons
        try:
            results = self.yolo(image, classes=[0], verbose=False)  # class 0 = person
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

        if not results or len(results[0].boxes) == 0:
            logger.debug("No persons detected by YOLO")
            return []

        all_poses = []
        num_persons = len(results[0].boxes)
        logger.debug(f"YOLO detected {num_persons} person(s)")

        # Stage 2: Pose estimation on each person crop
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])

            if conf < self.min_person_conf:
                logger.debug(
                    f"Person {i+1}/{num_persons}: confidence {conf:.3f} < threshold {self.min_person_conf}"
                )
                continue

            # Get bbox with padding
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            crop, offset = self._crop_with_padding(image, x1, y1, x2, y2)

            logger.debug(
                f"Person {i+1}/{num_persons}: bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), "
                f"conf={conf:.3f}, crop_size={crop.shape[1]}x{crop.shape[0]}"
            )

            # Detect pose on crop
            try:
                poses = self.pose_detector.detect(crop)
            except Exception as e:
                logger.warning(f"Pose detection failed for person {i+1}: {e}")
                continue

            if not poses:
                logger.debug(f"No pose detected for person {i+1}")
                continue

            # Map keypoints back to original image coordinates
            for pose in poses:
                # Translate keypoints by crop offset
                pose.keypoints[:, 0] += offset[0]  # x coordinates
                pose.keypoints[:, 1] += offset[1]  # y coordinates

                # Translate bbox
                pose.bbox[0] += offset[0]  # bbox x
                pose.bbox[1] += offset[1]  # bbox y

                # Assign person_id based on YOLO detection order
                pose.person_id = i

                all_poses.append(pose)

        logger.info(
            f"Two-stage detection complete: {len(all_poses)} pose(s) from {num_persons} person(s)"
        )

        return all_poses

    def _crop_with_padding(
        self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop image region with padding around bbox.

        Padding ensures we don't miss keypoints near bbox edges.

        Args:
            image: Input image (H, W, 3)
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            Tuple of (cropped_image, (x_offset, y_offset))
            - cropped_image: Cropped region with padding
            - x_offset, y_offset: Offset to map coordinates back to original
        """
        h, w = image.shape[:2]

        # Calculate padding
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_x = bbox_w * self.crop_padding
        pad_y = bbox_h * self.crop_padding

        # Apply padding and clamp to image bounds
        x1_pad = int(max(0, x1 - pad_x))
        y1_pad = int(max(0, y1 - pad_y))
        x2_pad = int(min(w, x2 + pad_x))
        y2_pad = int(min(h, y2 + pad_y))

        # Crop image
        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        return crop, (x1_pad, y1_pad)

    def set_min_confidence(self, min_conf: float) -> None:
        """
        Update minimum person detection confidence threshold.

        Args:
            min_conf: New threshold (0.0-1.0)
                     Lower values: More detections but more false positives
                     Higher values: Fewer detections but higher precision
        """
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError(f"min_conf must be in [0.0, 1.0], got {min_conf}")

        self.min_person_conf = min_conf
        logger.info(f"Updated min_person_conf to {min_conf}")

    def set_crop_padding(self, padding: float) -> None:
        """
        Update crop padding ratio.

        Args:
            padding: New padding ratio (0.0-1.0)
                    0.0 = No padding (tight crop)
                    0.1 = 10% padding (recommended)
                    0.2 = 20% padding (more context)
        """
        if not 0.0 <= padding <= 1.0:
            raise ValueError(f"padding must be in [0.0, 1.0], got {padding}")

        self.crop_padding = padding
        logger.info(f"Updated crop_padding to {padding}")
