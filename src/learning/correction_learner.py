"""
Bias Correction Learner for PostureKit.
Learns systematic biases in pose detection and applies them to future detections.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pickle
import logging
from datetime import datetime
from collections import defaultdict

from sqlalchemy import select, func

from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Keypoint indices for major body parts
BODY_PARTS = {
    'nose': [0],
    'left_eye': [1],
    'right_eye': [2],
    'left_ear': [3],
    'right_ear': [4],
    'left_shoulder': [5],
    'right_shoulder': [6],
    'left_elbow': [7],
    'right_elbow': [8],
    'left_wrist': [9],
    'right_wrist': [10],
    'left_hip': [11],
    'right_hip': [12],
    'left_knee': [13],
    'right_knee': [14],
    'left_ankle': [15],
    'right_ankle': [16],
}


@dataclass
class BiasEstimate:
    """
    Estimated systematic bias for a keypoint in a specific pose category.

    Attributes:
        keypoint_index: Which keypoint this applies to
        pose_category: Pose type identifier (e.g., 'frontal_arms_down')
        offset_x: Mean X-axis correction in pixels
        offset_y: Mean Y-axis correction in pixels
        std_x: Standard deviation of X corrections
        std_y: Standard deviation of Y corrections
        sample_count: Number of corrections used to estimate this bias
        confidence: Reliability score (0-1) based on consistency
    """
    keypoint_index: int
    pose_category: str
    offset_x: float
    offset_y: float
    std_x: float
    std_y: float
    sample_count: int
    confidence: float


class CorrectionLearnerError(Exception):
    """Base exception for correction learner errors."""
    pass


class InsufficientDataError(CorrectionLearnerError):
    """Raised when insufficient training data available."""
    pass


class BiasCorrector:
    """
    Learns and applies systematic bias corrections to pose detections.

    This learner doesn't try to predict exact keypoint positions. Instead, it
    identifies systematic biases in the pose detector (e.g., "left elbows are
    consistently 8 pixels too high") and applies simple offset corrections.

    The approach works by:
    1. Extracting correction deltas from manually corrected poses
    2. Grouping poses by type (frontal/profile, arms up/down, etc.)
    3. Computing average offsets for each keypoint in each pose category
    4. Storing these as lookup tables
    5. Applying relevant offsets to new detections based on their category

    This requires far less training data than full position prediction because
    we're only estimating ~20-50 bias parameters instead of 266 position targets.

    Attributes:
        storage_manager: Database access
        bias_table: Dict mapping (keypoint_index, pose_category) to BiasEstimate
        model_dir: Directory for saving trained biases
        min_training_samples: Minimum corrections needed before training
        min_samples_per_bias: Minimum samples to trust a specific bias estimate
    """

    DEFAULT_MODEL_DIR = Path('data/models/corrections')
    MIN_TRAINING_SAMPLES = 10  # Lowered from 50 for faster incremental learning
    MIN_SAMPLES_PER_BIAS = 3   # Lowered from 5 for faster incremental learning

    def __init__(
        self,
        storage_manager: StorageManager,
        model_dir: Optional[Path] = None,
        min_training_samples: int = 50,
        min_samples_per_bias: int = 5
    ):
        """
        Initialize Bias Corrector.

        Args:
            storage_manager: StorageManager for database access
            model_dir: Directory to save/load bias tables
            min_training_samples: Minimum total corrections before training
            min_samples_per_bias: Minimum samples to trust a specific bias
        """
        self.storage_manager = storage_manager
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.min_training_samples = min_training_samples
        self.min_samples_per_bias = min_samples_per_bias

        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Bias lookup table: (keypoint_idx, pose_category) -> BiasEstimate
        self.bias_table: Dict[Tuple[int, str], BiasEstimate] = {}

        # Try to load existing biases
        self._load_biases()

        logger.info(
            f"BiasCorrector initialized",
            extra={'extra_data': {
                'model_dir': str(self.model_dir),
                'min_samples': self.min_training_samples,
                'biases_loaded': len(self.bias_table)
            }}
        )

    def _categorize_pose(self, features: np.ndarray) -> str:
        """
        Categorize pose based on geometric features.

        Uses the 52-dim feature vector to determine pose type. We create
        simple categories based on arm position and body orientation since
        these affect detection biases most.

        Args:
            features: 52-dim normalized feature vector

        Returns:
            Category string like 'frontal_arms_down' or 'profile_arms_raised'
        """
        # Features are normalized to [0, 1], but angles were divided by 180
        # So we need to multiply back to get rough angle ranges

        # Left/right armpit angles (features 8-9) indicate arm position
        # Close to 90 degrees (0.5 normalized) = arms at sides
        # Close to 0 or 180 (0 or 1 normalized) = arms raised/extended
        left_armpit = features[8] * 180
        right_armpit = features[9] * 180
        avg_armpit = (left_armpit + right_armpit) / 2

        if avg_armpit < 60:
            arm_position = 'arms_raised'
        elif avg_armpit > 120:
            arm_position = 'arms_wide'
        else:
            arm_position = 'arms_down'

        # Shoulder width ratio (feature 18 in limb ratios section) indicates viewing angle
        # High ratio = frontal view, low ratio = profile view
        # In normalized space, compare to typical frontal threshold
        shoulder_width_ratio = features[18]

        if shoulder_width_ratio > 0.7:  # Wide shoulders visible
            orientation = 'frontal'
        elif shoulder_width_ratio < 0.3:  # Narrow profile
            orientation = 'profile'
        else:
            orientation = 'oblique'

        return f"{orientation}_{arm_position}"

    def _extract_correction_data(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Extract correction deltas from database.

        Queries all corrected poses and computes the difference between
        original detection and user-corrected positions.

        Returns:
            List of (keypoint_index, pose_category, delta_xy) tuples
        """
        logger.info("Extracting correction data from database")

        with self.storage_manager.session_scope() as session:
            # Query corrected poses with features
            query = (
                select(PoseDetection, GeometricFeatures)
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
                .where(
                    PoseDetection.is_corrected == True,
                    PoseDetection.original_keypoints.isnot(None)
                )
            )

            results = session.execute(query).all()

            if len(results) < self.min_training_samples:
                raise InsufficientDataError(
                    f"Need at least {self.min_training_samples} corrections, "
                    f"found {len(results)}"
                )

            # Extract deltas for each keypoint in each pose
            correction_data = []

            for pose, features in results:
                # Reshape keypoints from flat arrays to (133, 3)
                original_kp = np.array(pose.original_keypoints).reshape(133, 3)
                corrected_kp = np.array(pose.keypoints).reshape(133, 3)

                # Get pose category from features
                feature_vec = np.array(features.feature_vector, dtype=np.float32)
                category = self._categorize_pose(feature_vec)

                # For each keypoint, compute delta
                for kp_idx in range(133):
                    # Only consider keypoints with reasonable confidence in both versions
                    if original_kp[kp_idx, 2] < 0.3 or corrected_kp[kp_idx, 2] < 0.3:
                        continue

                    # Compute XY delta (ignore confidence dimension)
                    delta = corrected_kp[kp_idx, :2] - original_kp[kp_idx, :2]

                    # Only include if user actually moved it (>1 pixel)
                    if np.linalg.norm(delta) > 1.0:
                        correction_data.append((kp_idx, category, delta))

            logger.info(
                f"Extracted {len(correction_data)} individual keypoint corrections "
                f"from {len(results)} poses"
            )

            return correction_data

    def train(self) -> Dict[str, int]:
        """
        Train bias correction model.

        Extracts correction deltas, groups by keypoint and pose category,
        computes average offsets, and stores as bias table.

        Returns:
            Dictionary with training statistics
        """
        logger.info("Training bias correction model")

        # Extract correction data
        correction_data = self._extract_correction_data()

        # Group corrections by (keypoint_index, pose_category)
        grouped = defaultdict(list)
        for kp_idx, category, delta in correction_data:
            grouped[(kp_idx, category)].append(delta)

        # Compute bias estimates for each group
        self.bias_table = {}
        total_biases = 0
        skipped_small_samples = 0

        for (kp_idx, category), deltas in grouped.items():
            deltas_array = np.array(deltas)
            n_samples = len(deltas_array)

            # Skip if too few samples for reliable estimate
            if n_samples < self.min_samples_per_bias:
                skipped_small_samples += 1
                continue

            # Compute statistics
            mean_delta = deltas_array.mean(axis=0)
            std_delta = deltas_array.std(axis=0)

            # Confidence based on consistency (low std = high confidence)
            # Use coefficient of variation: std/mean, clamped to reasonable range
            cv_x = std_delta[0] / (abs(mean_delta[0]) + 1e-6)
            cv_y = std_delta[1] / (abs(mean_delta[1]) + 1e-6)
            cv_avg = (cv_x + cv_y) / 2
            confidence = np.clip(1.0 / (1.0 + cv_avg), 0.0, 1.0)

            # Store bias estimate
            bias = BiasEstimate(
                keypoint_index=kp_idx,
                pose_category=category,
                offset_x=float(mean_delta[0]),
                offset_y=float(mean_delta[1]),
                std_x=float(std_delta[0]),
                std_y=float(std_delta[1]),
                sample_count=n_samples,
                confidence=float(confidence)
            )

            self.bias_table[(kp_idx, category)] = bias
            total_biases += 1

        # Save to disk
        self._save_biases()

        stats = {
            'total_corrections': len(correction_data),
            'unique_keypoint_category_pairs': len(grouped),
            'biases_learned': total_biases,
            'skipped_insufficient_data': skipped_small_samples
        }

        logger.info(
            f"Training complete: learned {total_biases} biases",
            extra={'extra_data': stats}
        )

        return stats

    def apply_correction(
        self,
        keypoints: np.ndarray,
        geometric_features: np.ndarray
    ) -> np.ndarray:
        """
        Apply learned bias corrections to detected keypoints.

        Args:
            keypoints: Detected keypoints (133, 3) with [x, y, confidence]
            geometric_features: 52-dim feature vector for this pose

        Returns:
            Corrected keypoints (133, 3)
        """
        if not self.bias_table:
            logger.debug("No bias table loaded, returning uncorrected keypoints")
            return keypoints

        # Determine pose category
        category = self._categorize_pose(geometric_features)

        # Copy keypoints so we don't modify original
        corrected = keypoints.copy()
        corrections_applied = 0

        # Apply bias correction to each keypoint that has a learned bias
        for kp_idx in range(133):
            key = (kp_idx, category)

            if key not in self.bias_table:
                continue

            bias = self.bias_table[key]

            # Only apply if confidence is reasonable and keypoint is detected
            if bias.confidence < 0.3 or keypoints[kp_idx, 2] < 0.3:
                continue

            # Apply offset
            corrected[kp_idx, 0] += bias.offset_x
            corrected[kp_idx, 1] += bias.offset_y
            corrections_applied += 1

        if corrections_applied > 0:
            logger.debug(
                f"Applied {corrections_applied} bias corrections to {category} pose"
            )

        return corrected

    def _save_biases(self):
        """Save bias table to disk."""
        try:
            bias_path = self.model_dir / 'bias_table.pkl'
            with open(bias_path, 'wb') as f:
                pickle.dump(self.bias_table, f)

            # Save metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'num_biases': len(self.bias_table),
                'min_samples': self.min_training_samples,
                'min_samples_per_bias': self.min_samples_per_bias
            }
            metadata_path = self.model_dir / 'metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved {len(self.bias_table)} biases to {bias_path}")

        except Exception as e:
            logger.error(f"Failed to save biases: {e}", exc_info=True)

    def _load_biases(self):
        """Load bias table from disk."""
        try:
            bias_path = self.model_dir / 'bias_table.pkl'

            if not bias_path.exists():
                logger.debug("No saved biases found")
                return

            with open(bias_path, 'rb') as f:
                self.bias_table = pickle.load(f)

            logger.info(f"Loaded {len(self.bias_table)} biases from {bias_path}")

        except Exception as e:
            logger.warning(f"Failed to load biases: {e}")
            self.bias_table = {}

    def get_bias_summary(self) -> Dict:
        """
        Get summary of learned biases.

        Returns:
            Dictionary with bias statistics
        """
        if not self.bias_table:
            return {
                'total_biases': 0,
                'categories': [],
                'keypoints_with_biases': []
            }

        # Group by category
        categories = set()
        keypoints_with_biases = set()
        total_samples = 0
        avg_confidence = []

        for (kp_idx, category), bias in self.bias_table.items():
            categories.add(category)
            keypoints_with_biases.add(kp_idx)
            total_samples += bias.sample_count
            avg_confidence.append(bias.confidence)

        return {
            'total_biases': len(self.bias_table),
            'categories': sorted(list(categories)),
            'keypoints_with_biases': sorted(list(keypoints_with_biases)),
            'total_training_samples': total_samples,
            'avg_confidence': float(np.mean(avg_confidence)) if avg_confidence else 0.0
        }

    def clear_biases(self):
        """Clear all learned biases from memory and disk."""
        logger.info("Clearing bias correction model")

        self.bias_table = {}

        try:
            bias_path = self.model_dir / 'bias_table.pkl'
            if bias_path.exists():
                bias_path.unlink()

            metadata_path = self.model_dir / 'metadata.pkl'
            if metadata_path.exists():
                metadata_path.unlink()

            logger.info("Biases cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear biases: {e}", exc_info=True)
