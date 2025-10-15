"""
Classification Manager for PostureKit.
Handles batch classification workflows and persistence of predicted labels.
"""
from typing import List, Dict, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass
import logging
from datetime import datetime

from sqlalchemy import select, update

from src.intelligence.pose_classifier import PoseClassifier, CategoryPrediction
from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, TrainingLabel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """
    Result of a classification operation with metadata for user review.

    This bundles the prediction with the pose metadata users need to decide
    whether to approve or reject the automatic label.
    """
    pose_id: UUID
    prediction: CategoryPrediction
    image_path: str
    current_category: Optional[str]  # Existing category if any
    timestamp: datetime


class ClassificationManager:
    """
    Manages the complete classification workflow from prediction to persistence.

    This orchestrates the classifier and storage manager to provide high-level
    operations like batch classification, prediction review, and label saving.
    Users interact with this manager rather than the lower-level components.

    The key workflow this enables:
    1. User labels a seed set of poses manually (minimum 10-50 examples)
    2. System predicts categories for remaining unlabeled poses
    3. User reviews predictions, approving high-confidence ones
    4. Approved predictions become training labels
    5. Repeat: more labels → better classifier → better predictions

    This creates an active learning loop where human effort focuses on edge
    cases while the system handles obvious classifications automatically.
    """

    def __init__(
        self,
        classifier: PoseClassifier,
        storage_manager: StorageManager
    ):
        """
        Initialize Classification Manager.

        Args:
            classifier: PoseClassifier instance for predictions
            storage_manager: StorageManager for database operations
        """
        self.classifier = classifier
        self.storage_manager = storage_manager

        logger.info("ClassificationManager initialized")

    def classify_unlabeled_batch(
        self,
        min_confidence: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[ClassificationResult]:
        """
        Classify all unlabeled poses and return results for user review.

        This generates predictions for poses without categories, filtering to
        only those above the confidence threshold. The results include all
        metadata needed for users to review and approve predictions in the UI.

        Args:
            min_confidence: Minimum confidence threshold (0-1)
            max_results: Maximum number of results to return (None = unlimited)

        Returns:
            List of ClassificationResult objects ordered by confidence (highest first)
        """
        logger.info(
            f"Classifying unlabeled poses (min_confidence={min_confidence})"
        )

        # Get predictions from classifier
        predictions = self.classifier.batch_predict(
            unlabeled_only=True,
            min_confidence=min_confidence
        )

        # Enrich with metadata for review
        results = []

        with self.storage_manager.session_scope() as session:
            for pose_id, prediction in predictions:
                # Fetch image path and current category
                pose = session.execute(
                    select(PoseDetection)
                    .where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()

                if pose is None:
                    continue

                image = session.execute(
                    select(PoseDetection)
                    .join(PoseDetection.image)
                    .where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()

                result = ClassificationResult(
                    pose_id=pose_id,
                    prediction=prediction,
                    image_path=image.image.file_path if image else "unknown",
                    current_category=pose.category,
                    timestamp=datetime.now()
                )

                results.append(result)

        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.prediction.confidence, reverse=True)

        # Limit results if requested
        if max_results is not None:
            results = results[:max_results]

        logger.info(
            f"Generated {len(results)} classification results",
            extra={'extra_data': {
                'total_results': len(results),
                'min_confidence': min_confidence
            }}
        )

        return results

    def save_predictions(
        self,
        predictions: List[Tuple[UUID, str]],
        mark_as_corrected: bool = False
    ) -> int:
        """
        Save approved predictions to the database as TrainingLabel records.

        For each pose, this creates a new TrainingLabel if one doesn't exist,
        or updates the existing one if it does. The category gets set to the
        predicted value.

        Args:
            predictions: List of (pose_id, category) tuples to save
            mark_as_corrected: Whether to mark as corrected (not used for predictions)

        Returns:
            Number of labels successfully saved
        """
        if not predictions:
            logger.warning("No predictions to save")
            return 0

        logger.info(f"Saving {len(predictions)} predictions to database")

        saved_count = 0

        with self.storage_manager.session_scope() as session:
            for pose_id, category in predictions:
                try:
                    # Check if TrainingLabel already exists for this pose
                    existing_label = session.execute(
                        select(TrainingLabel)
                        .where(TrainingLabel.pose_id == pose_id)
                    ).scalar_one_or_none()

                    if existing_label:
                        # Update existing label
                        existing_label.category = category
                        logger.debug(f"Updated category '{category}' for pose {pose_id}")
                    else:
                        # Create new TrainingLabel
                        new_label = TrainingLabel(
                            pose_id=pose_id,
                            category=category,
                            difficulty=None,
                            tags=None,
                            notes=None
                        )
                        session.add(new_label)
                        logger.debug(f"Created category '{category}' for pose {pose_id}")

                    saved_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save prediction for pose {pose_id}: {e}",
                        exc_info=True
                    )
                    continue

            # Commit all changes atomically
            session.commit()

        logger.info(
            f"Successfully saved {saved_count}/{len(predictions)} predictions",
            extra={'extra_data': {
                'saved': saved_count,
                'total': len(predictions)
            }}
        )

        return saved_count

    def auto_label_high_confidence(
        self,
        confidence_threshold: float = 0.9,
        max_auto_labels: int = 100
    ) -> int:
        """
        Automatically label poses with very high confidence predictions.

        This is for power users who want to accelerate labeling by automatically
        accepting predictions above a very high confidence threshold. Only use
        this when you trust the classifier—typically after you've manually
        validated its accuracy on a test set.

        Args:
            confidence_threshold: Minimum confidence for auto-labeling (0.9-1.0)
            max_auto_labels: Maximum number of poses to auto-label in one batch

        Returns:
            Number of poses automatically labeled
        """
        logger.info(
            f"Auto-labeling poses with confidence >= {confidence_threshold}"
        )

        # Get high-confidence predictions
        predictions = self.classifier.batch_predict(
            unlabeled_only=True,
            min_confidence=confidence_threshold
        )

        # Limit to max_auto_labels
        predictions = predictions[:max_auto_labels]

        if not predictions:
            logger.info("No high-confidence predictions found")
            return 0

        # Convert to save format
        labels_to_save = [
            (pose_id, pred.category)
            for pose_id, pred in predictions
        ]

        # Save with corrected flag since these are high-confidence
        saved = self.save_predictions(labels_to_save, mark_as_corrected=True)

        logger.info(
            f"Auto-labeled {saved} poses",
            extra={'extra_data': {
                'saved': saved,
                'threshold': confidence_threshold
            }}
        )

        return saved

    def get_classification_stats(self) -> Dict[str, any]:
        """
        Get statistics about classification readiness and performance.

        Returns:
            Dictionary with comprehensive classification statistics
        """
        stats = self.classifier.get_classifier_stats()

        # Add manager-specific stats
        with self.storage_manager.session_scope() as session:
            # Count unlabeled poses
            unlabeled_count = session.execute(
                select(PoseDetection)
                .where(PoseDetection.category.is_(None))
            ).all()

            stats['unlabeled_poses'] = len(unlabeled_count)

        stats['can_classify'] = stats['ready'] and stats['unlabeled_poses'] > 0

        return stats
