"""
Pose Classifier for PostureKit.
Predicts pose categories using K-nearest neighbors classification with similarity search.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from uuid import UUID
from collections import Counter
import logging

from src.intelligence.similarity_engine import SimilarityEngine, SimilarPose
from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures, TrainingLabel
from src.utils.logging_config import get_logger
from sqlalchemy import select, func

logger = get_logger(__name__)


@dataclass
class CategoryPrediction:
    """
    A predicted category with confidence and explanation.

    This provides full transparency into how the classification decision was made,
    showing users which similar poses influenced the prediction and how strongly.
    """
    category: str
    confidence: float  # 0-1, higher means more certain
    vote_distribution: Dict[str, float]  # Category -> weighted vote score
    supporting_poses: List[SimilarPose]  # K nearest neighbors used for voting
    k: int  # Number of neighbors used

    def explain(self) -> str:
        """
        Generate human-readable explanation of the prediction.

        This helps users understand and trust the classification by showing
        the reasoning process: which poses voted, how similar they were,
        and what the vote distribution looked like.
        """
        explanation = f"Predicted category: {self.category} (confidence: {self.confidence:.2%})\n\n"
        explanation += f"Based on {self.k} similar poses:\n"

        # Show vote distribution
        explanation += "\nVote distribution:\n"
        sorted_votes = sorted(
            self.vote_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for cat, score in sorted_votes:
            percentage = (score / sum(self.vote_distribution.values())) * 100
            explanation += f"  {cat}: {percentage:.1f}%\n"

        # Show top supporting poses
        explanation += f"\nTop supporting poses:\n"
        for i, pose in enumerate(self.supporting_poses[:5], 1):
            explanation += f"  {i}. Category: {pose.category or 'unlabeled'}, "
            explanation += f"Similarity: {pose.similarity_score:.3f}, "
            explanation += f"Distance: {pose.distance:.3f}\n"

        return explanation


class PoseClassifierError(Exception):
    """Base exception for pose classifier errors."""
    pass


class InsufficientTrainingDataError(PoseClassifierError):
    """Raised when not enough labeled poses exist for classification."""
    pass


class PoseClassifier:
    """
    K-nearest neighbors classifier for pose categorization.

    This classifier leverages the similarity engine to find poses geometrically
    similar to the query, then uses weighted voting among those neighbors to
    predict the category. The voting is weighted by similarity—closer neighbors
    have more influence than distant ones.

    The key advantage of KNN for pose classification is its interpretability.
    Unlike neural networks that are black boxes, you can always see exactly
    which training examples influenced each prediction and why. This transparency
    is critical for users who need to understand and trust the system.

    Attributes:
        similarity_engine: SimilarityEngine for finding nearest neighbors
        storage_manager: StorageManager for database access
        k: Number of nearest neighbors to consider (default 5)
        min_labeled_poses: Minimum labeled poses required for classification
    """

    DEFAULT_K = 5
    MIN_LABELED_POSES = 10

    def __init__(
        self,
        similarity_engine: SimilarityEngine,
        storage_manager: StorageManager,
        k: int = 5,
        min_labeled_poses: int = 10
    ):
        """
        Initialize Pose Classifier.

        Args:
            similarity_engine: SimilarityEngine instance for neighbor search
            storage_manager: StorageManager for database access
            k: Number of nearest neighbors for voting (typically 3-10)
            min_labeled_poses: Minimum labeled training examples required
        """
        self.similarity_engine = similarity_engine
        self.storage_manager = storage_manager
        self.k = k
        self.min_labeled_poses = min_labeled_poses

        logger.info(
            f"PoseClassifier initialized with k={k}",
            extra={'extra_data': {'k': k, 'min_labeled': min_labeled_poses}}
        )

    def predict(
        self,
        query_vector: np.ndarray,
        k: Optional[int] = None
    ) -> CategoryPrediction:
        """
        Predict category for a pose given its feature vector.

        The prediction process works as follows:
        1. Find K nearest labeled poses using similarity search
        2. Weight each neighbor's vote by its similarity (inverse of distance)
        3. Sum weighted votes per category
        4. Return category with highest vote and confidence score

        Args:
            query_vector: 52-dimensional geometric feature vector
            k: Number of neighbors to use (overrides default if provided)

        Returns:
            CategoryPrediction with predicted category and explanation

        Raises:
            InsufficientTrainingDataError: If too few labeled poses exist
            PoseClassifierError: If prediction fails
        """
        k = k or self.k

        # Validate we have enough labeled training data
        labeled_count = self._count_labeled_poses()
        if labeled_count < self.min_labeled_poses:
            raise InsufficientTrainingDataError(
                f"Need at least {self.min_labeled_poses} labeled poses, "
                f"found {labeled_count}. Label more poses before classification."
            )

        try:
            # Find K nearest neighbors, but search more to ensure we get K labeled ones
            # Some neighbors might be unlabeled, so we search 2*K and filter
            search_k = k * 2
            similar_poses = self.similarity_engine.search(query_vector, k=search_k)

            # Filter to only labeled poses
            labeled_poses = [p for p in similar_poses if p.category is not None]

            if len(labeled_poses) < k:
                logger.warning(
                    f"Found only {len(labeled_poses)} labeled neighbors, wanted {k}"
                )
                # Use what we have, but adjust k
                k = len(labeled_poses)
            else:
                # Take top K labeled poses
                labeled_poses = labeled_poses[:k]

            if not labeled_poses:
                raise InsufficientTrainingDataError(
                    "No labeled neighbors found. Cannot classify."
                )

            # Weighted voting based on similarity scores
            vote_distribution = self._weighted_voting(labeled_poses)

            # Predict category with highest vote
            predicted_category = max(vote_distribution.items(), key=lambda x: x[1])[0]

            # Calculate confidence based on vote concentration
            total_votes = sum(vote_distribution.values())
            confidence = vote_distribution[predicted_category] / total_votes

            # Adjust confidence based on vote spread
            # If votes are very split, reduce confidence even if one category leads
            num_categories_voted = len(vote_distribution)
            if num_categories_voted > 1:
                # Penalize confidence if votes are spread across many categories
                spread_penalty = 1.0 - (num_categories_voted - 1) * 0.15
                confidence *= max(spread_penalty, 0.5)

            prediction = CategoryPrediction(
                category=predicted_category,
                confidence=confidence,
                vote_distribution=vote_distribution,
                supporting_poses=labeled_poses,
                k=k
            )

            logger.info(
                f"Predicted category: {predicted_category} (confidence: {confidence:.2%})",
                extra={'extra_data': {
                    'category': predicted_category,
                    'confidence': confidence,
                    'k': k,
                    'votes': vote_distribution
                }}
            )

            return prediction

        except InsufficientTrainingDataError:
            raise
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise PoseClassifierError(f"Classification failed: {e}") from e

    def predict_by_pose_id(
        self,
        pose_id: UUID,
        k: Optional[int] = None
    ) -> CategoryPrediction:
        """
        Predict category for an existing pose in the database.

        This is a convenience method that fetches the feature vector for the
        given pose and then performs classification. Useful for batch
        classification of unlabeled poses.

        Args:
            pose_id: ID of pose to classify
            k: Number of neighbors to use

        Returns:
            CategoryPrediction with predicted category
        """
        # Fetch feature vector from database
        with self.storage_manager.session_scope() as session:
            features = session.execute(
                select(GeometricFeatures)
                .where(GeometricFeatures.pose_id == pose_id)
            ).scalar_one_or_none()

            if features is None:
                raise ValueError(f"No features found for pose {pose_id}")

            feature_vector = np.array(features.feature_vector, dtype=np.float32)

        return self.predict(feature_vector, k=k)

    def batch_predict(
        self,
        unlabeled_only: bool = True,
        min_confidence: float = 0.7,
        k: Optional[int] = None
    ) -> List[Tuple[UUID, CategoryPrediction]]:
        """
        Predict categories for multiple poses in the database.

        This is useful for automatically labeling large batches of poses after
        you've labeled a seed set. Only predictions above the minimum confidence
        threshold are returned to avoid propagating low-quality labels.

        Args:
            unlabeled_only: Only predict for poses without categories
            min_confidence: Minimum confidence threshold (0-1)
            k: Number of neighbors to use

        Returns:
            List of (pose_id, prediction) tuples for poses meeting criteria
        """
        predictions = []

        with self.storage_manager.session_scope() as session:
            # Build query for target poses
            query = select(PoseDetection.id, GeometricFeatures.feature_vector)
            query = query.join(
                GeometricFeatures,
                PoseDetection.id == GeometricFeatures.pose_id
            )

            if unlabeled_only:
                # Left join with TrainingLabel and filter for null categories
                query = query.outerjoin(
                    TrainingLabel,
                    PoseDetection.id == TrainingLabel.pose_id
                ).where(TrainingLabel.category.is_(None))

            results = session.execute(query).all()

            logger.info(f"Batch predicting for {len(results)} poses")

            for pose_id, feature_vector in results:
                try:
                    feature_vec = np.array(feature_vector, dtype=np.float32)
                    prediction = self.predict(feature_vec, k=k)

                    # Only include predictions above confidence threshold
                    if prediction.confidence >= min_confidence:
                        predictions.append((pose_id, prediction))
                        logger.debug(
                            f"Predicted {prediction.category} for pose {pose_id} "
                            f"(confidence: {prediction.confidence:.2%})"
                        )
                    else:
                        logger.debug(
                            f"Skipped pose {pose_id} - low confidence "
                            f"({prediction.confidence:.2%})"
                        )

                except Exception as e:
                    logger.warning(f"Failed to predict for pose {pose_id}: {e}")
                    continue

            logger.info(
                f"Batch prediction complete: {len(predictions)} poses classified",
                extra={'extra_data': {
                    'total_processed': len(results),
                    'predictions_made': len(predictions),
                    'min_confidence': min_confidence
                }}
            )

        return predictions

    def _weighted_voting(self, neighbors: List[SimilarPose]) -> Dict[str, float]:
        """
        Perform weighted voting among neighbors.

        Each neighbor's vote is weighted by its similarity score. This ensures
        that very similar poses have more influence than marginally similar ones.
        The weighting function is: vote_weight = similarity_score

        We use similarity scores (higher = more similar) rather than distances
        because they're already normalized to [0,1] and are more intuitive for
        weighting—a similarity of 0.9 means the pose should have strong influence.

        Args:
            neighbors: List of SimilarPose objects with categories

        Returns:
            Dictionary mapping categories to weighted vote scores
        """
        votes: Dict[str, float] = {}

        for neighbor in neighbors:
            if neighbor.category is None:
                continue

            # Weight by similarity score (higher similarity = more weight)
            weight = neighbor.similarity_score

            # Accumulate weighted votes per category
            votes[neighbor.category] = votes.get(neighbor.category, 0.0) + weight

        return votes

    def _count_labeled_poses(self) -> int:
        """Count number of labeled poses in database."""
        with self.storage_manager.session_scope() as session:
            count = session.execute(
                select(func.count())
                .select_from(TrainingLabel)
                .where(TrainingLabel.category.isnot(None))
            ).scalar()

            return count or 0

    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of categories in labeled training data.

        This helps users understand their training set composition and identify
        categories that need more examples for balanced classification.

        Returns:
            Dictionary mapping categories to counts
        """
        with self.storage_manager.session_scope() as session:
            results = session.execute(
                select(TrainingLabel.category, func.count())
                .where(TrainingLabel.category.isnot(None))
                .group_by(TrainingLabel.category)
            ).all()

            distribution = {category: count for category, count in results}

            logger.debug(f"Category distribution: {distribution}")
            return distribution

    def evaluate_accuracy(
        self,
        test_poses: Optional[List[UUID]] = None,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate classifier accuracy on labeled poses using cross-validation.

        This queries poses that have TrainingLabels with non-null categories,
        then performs leave-one-out prediction to measure accuracy.
        """
        k = k or self.k

        with self.storage_manager.session_scope() as session:
            # Get labeled poses by joining with TrainingLabel
            query = (
                select(
                    PoseDetection.id,
                    TrainingLabel.category,
                    GeometricFeatures.feature_vector
                )
                .join(TrainingLabel, PoseDetection.id == TrainingLabel.pose_id)
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
                .where(TrainingLabel.category.isnot(None))
            )

            if test_poses:
                query = query.where(PoseDetection.id.in_(test_poses))

            results = session.execute(query).all()

            if len(results) < 2:
                raise InsufficientTrainingDataError(
                    "Need at least 2 labeled poses for evaluation"
                )

            correct = 0
            total = len(results)
            confusion_matrix = {}

            for pose_id, true_category, feature_vector in results:
                try:
                    # Predict using similar poses (excluding this one)
                    feature_vec = np.array(feature_vector, dtype=np.float32)
                    similar_poses = self.similarity_engine.search(
                        feature_vec,
                        k=k + 1,
                        exclude_pose_id=pose_id
                    )

                    # Filter to labeled poses and take top K
                    labeled_poses = [p for p in similar_poses if p.category is not None][:k]

                    if not labeled_poses:
                        continue

                    # Vote and predict
                    votes = self._weighted_voting(labeled_poses)
                    predicted_category = max(votes.items(), key=lambda x: x[1])[0]

                    # Check if correct
                    if predicted_category == true_category:
                        correct += 1

                    # Update confusion matrix
                    key = (true_category, predicted_category)
                    confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

                except Exception as e:
                    logger.warning(f"Evaluation failed for pose {pose_id}: {e}")
                    total -= 1
                    continue

            accuracy = correct / total if total > 0 else 0.0

            metrics = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'k': k,
                'confusion_matrix': confusion_matrix
            }

            logger.info(
                f"Evaluation complete: {accuracy:.2%} accuracy ({correct}/{total})",
                extra={'extra_data': metrics}
            )

            return metrics

    def get_classifier_stats(self) -> Dict[str, any]:
        """
        Get statistics about the classifier state.

        Returns:
            Dictionary with classifier metadata and statistics
        """
        labeled_count = self._count_labeled_poses()
        category_dist = self.get_category_distribution()

        return {
            'k': self.k,
            'min_labeled_poses': self.min_labeled_poses,
            'labeled_poses': labeled_count,
            'ready': labeled_count >= self.min_labeled_poses,
            'categories': list(category_dist.keys()),
            'category_distribution': category_dist
        }
