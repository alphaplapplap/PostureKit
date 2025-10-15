"""
Multi-person pose detection pipeline.
Combines person detection (YOLO) with pose estimation (MMPose).
"""
import numpy as np
from typing import List, Tuple
import logging

from src.core.person_detector import YOLOPersonDetector, PersonDetection
from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.core.geometric_feature_extractor import GeometricFeatureExtractor, GeometricFeatures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MultiPersonPipeline:
    """
    Complete pipeline for multi-person pose detection.

    Pipeline:
    1. Detect all people in image (YOLO)
    2. For each person, detect pose (MMPose)
    3. Estimate viewpoint for each pose
    4. Extract features for each pose

    Attributes:
        person_detector: YOLOv8 person detector
        pose_detector: MMPose pose detector
        feature_extractor: Geometric feature extractor
    """

    def __init__(
        self,
        person_confidence: float = 0.5,
        pose_confidence: float = 0.3
    ):
        """
        Initialize multi-person pipeline.

        Args:
            person_confidence: Minimum confidence for person detection
            pose_confidence: Minimum confidence for pose detection
        """
        self.person_detector = YOLOPersonDetector(
            confidence_threshold=person_confidence
        )
        self.pose_detector = RTMWCocktail14Detector(
            detection_threshold=pose_confidence
        )
        self.feature_extractor = GeometricFeatureExtractor()

        logger.info("MultiPersonPipeline initialized")

    def process_image(
        self,
        image: np.ndarray
    ) -> List[Tuple[PersonDetection, PoseResult, None, GeometricFeatures]]:
        """
        Process image through complete multi-person pipeline.

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            List of tuples (person_detection, pose_result, None, features)
            One tuple per detected person with successful pose estimation
        """
        logger.info("Processing image through multi-person pipeline")

        # Step 1: Detect all people
        logger.debug("Step 1: Detecting people")
        person_detections = self.person_detector.detect_people(image, return_crops=True)

        if not person_detections:
            logger.info("No people detected in image")
            return []

        logger.info(f"Detected {len(person_detections)} people")

        # Step 2: Process each person
        results = []

        for person in person_detections:
            logger.debug(f"Processing person {person.person_id}")

            try:
                # Detect pose from crop
                pose_result = self.pose_detector.detect_from_crop(
                    person.crop,
                    person.bbox
                )

                if pose_result is None:
                    logger.warning(f"No pose detected for person {person.person_id}")
                    continue

                # Set person ID
                pose_result.person_id = person.person_id

                # Extract features (no viewpoint estimation)
                features = self.feature_extractor.extract(pose_result)

                # Add to results (viewpoint is None)
                results.append((person, pose_result, None, features))

                logger.debug(f"Successfully processed person {person.person_id}")

            except Exception as e:
                logger.error(f"Failed to process person {person.person_id}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully processed {len(results)}/{len(person_detections)} people")

        return results
