"""
Training Intelligence for PostureKit.
Identifies most valuable poses for annotation to maximize dataset quality.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from sqlalchemy import select, func, and_, or_
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures, TrainingLabel
from src.intelligence.similarity_engine import SimilarityEngine
from src.intelligence.pose_classifier import PoseClassifier
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SamplingStrategy(Enum):
    """Strategies for selecting valuable training samples."""
    DIVERSITY = "diversity"  # Maximize feature space coverage
    UNCERTAINTY = "uncertainty"  # Focus on hard-to-classify poses
    HYBRID = "hybrid"  # Combination of diversity and uncertainty
    CLUSTER_CENTERS = "cluster_centers"  # Representative of each cluster
    BOUNDARY = "boundary"  # Near decision boundaries


@dataclass
class TrainingCandidate:
    """
    A pose candidate for annotation with priority score.

    Attributes:
        pose_id: Database ID of pose detection
        priority_score: Higher = more valuable for training (0-1)
        reasoning: Why this pose was selected
        feature_vector: 52-dim geometric features
        metadata: Additional context (image path, confidence, etc.)
    """
    pose_id: str
    priority_score: float
    reasoning: str
    feature_vector: np.ndarray
    metadata: Dict

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'pose_id': self.pose_id,
            'priority_score': float(self.priority_score),
            'reasoning': self.reasoning,
            'feature_vector': self.feature_vector.tolist(),
            'metadata': self.metadata
        }


class TrainingIntelligenceError(Exception):
    """Base exception for training intelligence errors."""
    pass


class TrainingIntelligence:
    """
    Analyzes dataset to identify most valuable poses for annotation.

    This module implements active learning strategies to maximize training
    efficiency by selecting diverse, informative, and representative samples.

    Key capabilities:
    - Diversity sampling: Cover entire feature space
    - Uncertainty sampling: Focus on hard examples
    - Cluster analysis: Find representative samples
    - Coverage gap detection: Identify underrepresented regions
    - Training value scoring: Rank poses by annotation value

    Attributes:
        storage_manager: Database access
        similarity_engine: For feature similarity queries
        classifier: For uncertainty estimation
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        similarity_engine: SimilarityEngine,
        classifier: Optional[PoseClassifier] = None
    ):
        """
        Initialize Training Intelligence.

        Args:
            storage_manager: StorageManager instance
            similarity_engine: SimilarityEngine for similarity queries
            classifier: Optional PoseClassifier for uncertainty sampling
        """
        self.storage_manager = storage_manager
        self.similarity_engine = similarity_engine
        self.classifier = classifier

        # KMeans clustering cache to avoid refitting on every call
        self._kmeans_cache: Dict[Tuple[int, int], Tuple] = {}  # (n_clusters, n_samples) -> (model, centers, timestamp)
        self._kmeans_cache_ttl: int = 3600  # 1 hour TTL in seconds

        logger.info("TrainingIntelligence initialized")

    def identify_training_candidates(
        self,
        n_candidates: int = 20,
        strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        exclude_labeled: bool = True
    ) -> List[TrainingCandidate]:
        """
        Identify top N poses most valuable for annotation.

        Args:
            n_candidates: Number of candidates to return
            strategy: Sampling strategy to use
            exclude_labeled: Skip poses that already have training labels

        Returns:
            List of TrainingCandidate objects sorted by priority

        Raises:
            TrainingIntelligenceError: If identification fails
        """
        logger.info(
            f"Identifying {n_candidates} training candidates using {strategy.value} strategy"
        )

        # Get unlabeled poses with features
        unlabeled_poses = self._get_unlabeled_poses() if exclude_labeled else self._get_all_poses()

        if len(unlabeled_poses) == 0:
            logger.warning("No unlabeled poses available")
            return []

        # Apply strategy-specific selection
        if strategy == SamplingStrategy.DIVERSITY:
            candidates = self._diversity_sampling(unlabeled_poses, n_candidates)
        elif strategy == SamplingStrategy.UNCERTAINTY:
            candidates = self._uncertainty_sampling(unlabeled_poses, n_candidates)
        elif strategy == SamplingStrategy.CLUSTER_CENTERS:
            candidates = self._cluster_center_sampling(unlabeled_poses, n_candidates)
        elif strategy == SamplingStrategy.BOUNDARY:
            candidates = self._boundary_sampling(unlabeled_poses, n_candidates)
        else:  # HYBRID
            candidates = self._hybrid_sampling(unlabeled_poses, n_candidates)

        logger.info(
            f"Identified {len(candidates)} training candidates",
            extra={'extra_data': {'strategy': strategy.value, 'count': len(candidates)}}
        )

        return candidates

    def _get_unlabeled_poses(self) -> List[Dict]:
        """Get all poses without training labels."""
        with self.storage_manager.session_scope() as session:
            # Query poses that don't have training labels
            query = (
                select(
                    PoseDetection.id,
                    PoseDetection.overall_confidence,
                    PoseDetection.image_id,
                    GeometricFeatures.feature_vector
                )
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
                .outerjoin(TrainingLabel, PoseDetection.id == TrainingLabel.pose_id)
                .where(TrainingLabel.id.is_(None))  # No training label exists
            )

            results = session.execute(query).all()

            poses = []
            for row in results:
                poses.append({
                    'pose_id': str(row.id),
                    'confidence': row.overall_confidence,
                    'image_id': str(row.image_id),
                    'feature_vector': np.array(row.feature_vector, dtype=np.float32)
                })

            logger.debug(f"Found {len(poses)} unlabeled poses")
            return poses

    def _get_all_poses(self) -> List[Dict]:
        """Get all poses with features."""
        with self.storage_manager.session_scope() as session:
            query = (
                select(
                    PoseDetection.id,
                    PoseDetection.overall_confidence,
                    PoseDetection.image_id,
                    GeometricFeatures.feature_vector
                )
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
            )

            results = session.execute(query).all()

            poses = []
            for row in results:
                poses.append({
                    'pose_id': str(row.id),
                    'confidence': row.overall_confidence,
                    'image_id': str(row.image_id),
                    'feature_vector': np.array(row.feature_vector, dtype=np.float32)
                })

            return poses

    def _diversity_sampling(
        self,
        poses: List[Dict],
        n_candidates: int
    ) -> List[TrainingCandidate]:
        """
        Select poses that maximize diversity in feature space.

        Uses farthest-first traversal to ensure selected poses are
        maximally distant from each other.
        """
        if len(poses) <= n_candidates:
            return [self._create_candidate(p, 1.0, "High diversity") for p in poses]

        # Extract feature vectors
        features = np.vstack([p['feature_vector'] for p in poses])

        # Farthest-first traversal
        selected_indices = []

        # Start with random pose
        selected_indices.append(np.random.randint(len(poses)))

        # Iteratively select farthest pose from selected set
        for _ in range(n_candidates - 1):
            # Calculate minimum distance to any selected pose
            selected_features = features[selected_indices]
            distances = cdist(features, selected_features, metric='euclidean')
            min_distances = distances.min(axis=1)

            # Set already selected to -1
            min_distances[selected_indices] = -1

            # Select pose with maximum minimum distance
            farthest_idx = min_distances.argmax()
            selected_indices.append(farthest_idx)

        # Create candidates with diversity scores
        candidates = []
        for idx in selected_indices:
            score = 1.0  # All equally valuable for diversity
            candidates.append(
                self._create_candidate(
                    poses[idx],
                    score,
                    "Maximizes feature space coverage"
                )
            )

        return candidates

    def _uncertainty_sampling(
        self,
        poses: List[Dict],
        n_candidates: int
    ) -> List[TrainingCandidate]:
        """
        Select poses with highest prediction uncertainty.

        Requires trained classifier to estimate uncertainty.
        """
        if self.classifier is None:
            logger.warning("No classifier available for uncertainty sampling, using diversity")
            return self._diversity_sampling(poses, n_candidates)

        # For now, fall back to diversity since classifier.predict_batch isn't implemented
        logger.warning("Uncertainty sampling not fully implemented, using diversity")
        return self._diversity_sampling(poses, n_candidates)

    def _cluster_center_sampling(
        self,
        poses: List[Dict],
        n_candidates: int
    ) -> List[TrainingCandidate]:
        """
        Select representative poses from each cluster with caching.

        Uses K-means clustering to find natural groups, then selects
        the pose closest to each cluster center. Caches fitted models
        to avoid expensive refitting.
        """
        if len(poses) <= n_candidates:
            return [self._create_candidate(p, 1.0, "Cluster representative") for p in poses]

        # Extract features
        features = np.vstack([p['feature_vector'] for p in poses])

        # Perform K-means clustering with caching
        n_clusters = min(n_candidates, len(poses))
        cache_key = (n_clusters, len(poses))

        import time
        current_time = time.time()

        # Check cache and TTL
        if cache_key in self._kmeans_cache:
            cached_model, cached_centers, cached_time = self._kmeans_cache[cache_key]
            cache_age = current_time - cached_time

            if cache_age < self._kmeans_cache_ttl:
                # Use cached model
                logger.debug(f"Using cached KMeans model (age: {cache_age:.1f}s)")
                cluster_centers = cached_centers
            else:
                # Cache expired, refit
                logger.debug(f"KMeans cache expired ({cache_age:.1f}s > {self._kmeans_cache_ttl}s), refitting")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(features)
                cluster_centers = kmeans.cluster_centers_
                self._kmeans_cache[cache_key] = (kmeans, cluster_centers.copy(), current_time)
        else:
            # Not in cache, fit new model
            logger.debug(f"Fitting new KMeans model (n_clusters={n_clusters}, n_samples={len(poses)})")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(features)
            cluster_centers = kmeans.cluster_centers_
            self._kmeans_cache[cache_key] = (kmeans, cluster_centers.copy(), current_time)

            # Limit cache size to prevent memory growth
            if len(self._kmeans_cache) > 10:
                # Remove oldest entry
                oldest_key = min(self._kmeans_cache.keys(), key=lambda k: self._kmeans_cache[k][2])
                del self._kmeans_cache[oldest_key]
                logger.debug(f"Evicted oldest KMeans cache entry")

        # Find pose closest to each cluster center (pre-allocate for performance)
        n_centers = len(cluster_centers)
        selected_indices = np.empty(n_centers, dtype=np.intp)  # intp for array indices
        for i, center in enumerate(cluster_centers):
            distances = np.linalg.norm(features - center, axis=1)
            selected_indices[i] = distances.argmin()

        # Create candidates
        candidates = []
        for idx in selected_indices:
            candidates.append(
                self._create_candidate(
                    poses[idx],
                    1.0,
                    "Representative of cluster"
                )
            )

        return candidates

    def _boundary_sampling(
        self,
        poses: List[Dict],
        n_candidates: int
    ) -> List[TrainingCandidate]:
        """
        Select poses near decision boundaries.

        Useful for refining class boundaries. Requires classifier.
        """
        if self.classifier is None:
            logger.warning("No classifier for boundary sampling, using diversity")
            return self._diversity_sampling(poses, n_candidates)

        # Fall back to diversity for now
        logger.warning("Boundary sampling not fully implemented, using diversity")
        return self._diversity_sampling(poses, n_candidates)

    def _hybrid_sampling(
        self,
        poses: List[Dict],
        n_candidates: int
    ) -> List[TrainingCandidate]:
        """
        Combine diversity and uncertainty sampling.

        Selects poses that are both diverse and uncertain.
        """
        # Get diversity candidates (2x more than needed)
        diversity_candidates = self._diversity_sampling(poses, min(n_candidates * 2, len(poses)))

        # If no classifier, just return diversity sample
        if self.classifier is None:
            return diversity_candidates[:n_candidates]

        # For now, return diversity candidates
        return diversity_candidates[:n_candidates]

    def _create_candidate(
        self,
        pose: Dict,
        score: float,
        reasoning: str
    ) -> TrainingCandidate:
        """Create TrainingCandidate from pose dict."""
        return TrainingCandidate(
            pose_id=pose['pose_id'],
            priority_score=float(score),
            reasoning=reasoning,
            feature_vector=pose['feature_vector'],
            metadata={
                'confidence': pose['confidence'],
                'image_id': pose['image_id']
            }
        )

    def analyze_dataset_coverage(self) -> Dict[str, any]:
        """
        Analyze current dataset coverage and identify gaps.

        Returns:
            Dictionary with coverage statistics and recommendations
        """
        logger.info("Analyzing dataset coverage")

        with self.storage_manager.session_scope() as session:
            # Get label distribution
            label_query = (
                select(TrainingLabel.category, func.count())
                .where(TrainingLabel.category.isnot(None))
                .group_by(TrainingLabel.category)
            )
            label_results = session.execute(label_query).all()
            label_dist = dict(label_results) if label_results else {}

            # Get total poses
            total_poses = session.execute(
                select(func.count()).select_from(PoseDetection)
            ).scalar() or 0

            # Get labeled poses
            labeled_poses = session.execute(
                select(func.count()).select_from(TrainingLabel)
                .where(TrainingLabel.category.isnot(None))
            ).scalar() or 0

            # Calculate coverage percentage
            coverage_pct = (labeled_poses / total_poses * 100) if total_poses > 0 else 0

            # Identify underrepresented categories (< 10% of labeled data)
            if labeled_poses > 0:
                min_count = labeled_poses * 0.1
                underrepresented = [
                    cat for cat, count in label_dist.items()
                    if count < min_count
                ]
            else:
                underrepresented = []

            analysis = {
                'total_poses': total_poses,
                'labeled_poses': labeled_poses,
                'unlabeled_poses': total_poses - labeled_poses,
                'coverage_percentage': coverage_pct,
                'category_distribution': label_dist,
                'underrepresented_categories': underrepresented,
                'recommendations': self._generate_recommendations(
                    total_poses, labeled_poses, label_dist
                )
            }

            logger.info(
                f"Coverage analysis complete: {coverage_pct:.1f}% labeled",
                extra={'extra_data': analysis}
            )

            return analysis

    def _generate_recommendations(
        self,
        total_poses: int,
        labeled_poses: int,
        label_dist: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations based on dataset state."""
        recommendations = []

        if labeled_poses == 0:
            recommendations.append("Start by labeling 50-100 diverse poses to establish baseline")
        elif labeled_poses < 100:
            recommendations.append(f"Need {100 - labeled_poses} more labels for reliable classification")

        if labeled_poses > 0 and len(label_dist) < 3:
            recommendations.append("Consider adding more category diversity to dataset")

        if label_dist:
            max_count = max(label_dist.values())
            min_count = min(label_dist.values())
            if max_count > min_count * 3:
                recommendations.append("Class imbalance detected - focus on underrepresented categories")

        coverage_pct = (labeled_poses / total_poses * 100) if total_poses > 0 else 0
        if coverage_pct < 10:
            recommendations.append("Low coverage - use diversity sampling to explore feature space")
        elif coverage_pct > 50:
            recommendations.append("Good coverage - use uncertainty sampling to refine boundaries")

        return recommendations
