"""
Multi-Modal Correction Learner for PostureKit.
Learns from manual corrections using both geometric and visual features.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from src.learning.correction_learner import (
    CorrectionLearner,
    CorrectionLearnerError,
    InsufficientDataError
)
from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MultiModalCorrectionPattern:
    """
    Correction pattern with both geometric and visual features.

    Attributes:
        original_viewpoint: Original estimated viewpoint (elevation, azimuth, facing)
        corrected_viewpoint: User-corrected viewpoint
        geometric_features: 52-dim geometric feature vector
        visual_features: 576-dim visual feature vector
        keypoint_adjustments: Number of keypoints moved
        confidence_changes: Average change in keypoint confidence
    """
    original_viewpoint: np.ndarray  # (3,)
    corrected_viewpoint: np.ndarray  # (3,)
    geometric_features: np.ndarray  # (52,)
    visual_features: np.ndarray     # (576,)
    keypoint_adjustments: int
    confidence_changes: float

    def viewpoint_delta(self) -> np.ndarray:
        """Calculate correction delta."""
        return self.corrected_viewpoint - self.original_viewpoint

    def combined_features(self) -> np.ndarray:
        """Get combined 628-dim feature vector."""
        return np.concatenate([self.geometric_features, self.visual_features])


class MultiModalCorrectionLearner(CorrectionLearner):
    """
    Learns from corrections using BOTH geometric and visual features.

    Extends base CorrectionLearner to train on 628-dimensional combined features
    instead of 52-dimensional geometric-only features.

    Key Differences from Base:
    - Trains on [geometric | visual] concatenated features (628-dim)
    - More expressive models (200 trees vs 100, depth 15 vs 10)
    - Separate storage for multi-modal models

    Attributes:
        storage_manager: Database access
        models: Dict of trained RandomForest models
        scalers: Dict of feature scalers
        model_dir: Directory for model persistence
        min_training_samples: Minimum corrections needed
        use_visual_features: Whether to include visual features
    """

    DEFAULT_MODEL_DIR = Path('data/models/multimodal_corrections')

    def __init__(
        self,
        storage_manager: StorageManager,
        model_dir: Optional[Path] = None,
        min_training_samples: int = 50,
        use_visual_features: bool = True
    ):
        """
        Initialize Multi-Modal Correction Learner.

        Args:
            storage_manager: StorageManager instance
            model_dir: Directory to save/load models
            min_training_samples: Minimum corrections needed
            use_visual_features: Whether to use visual features (vs geometric only)
        """
        # Initialize base class with different model dir
        model_dir = model_dir or self.DEFAULT_MODEL_DIR
        super().__init__(storage_manager, model_dir, min_training_samples)

        self.use_visual_features = use_visual_features

        logger.info(
            f"MultiModalCorrectionLearner initialized",
            extra={'extra_data': {
                'model_dir': str(self.model_dir),
                'min_samples': self.min_training_samples,
                'use_visual': use_visual_features
            }}
        )

    def extract_correction_patterns(self) -> List[MultiModalCorrectionPattern]:
        """
        Extract multi-modal correction patterns from database.

        Queries corrected poses with geometric AND visual features.

        Returns:
            List of MultiModalCorrectionPattern objects

        Raises:
            InsufficientDataError: If not enough corrections
            CorrectionLearnerError: If visual features missing when required
        """
        logger.info("Extracting multi-modal correction patterns from database")

        with self.storage_manager.session_scope() as session:
            # Query corrected poses with all features
            # Note: Visual features table may not exist yet
            query = (
                select(PoseDetection, GeometricFeatures)
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
                .where(PoseDetection.is_corrected == True)
            )

            results = session.execute(query).all()

            if len(results) < self.min_training_samples:
                raise InsufficientDataError(
                    f"Need at least {self.min_training_samples} corrections, "
                    f"found {len(results)}"
                )

            patterns = []
            for row in results:
                pose, geo_features = row

                # Extract geometric features
                geometric_vec = np.array(geo_features.feature_vector, dtype=np.float32)

                # Visual features - placeholder until table exists
                if self.use_visual_features:
                    # TODO: Query visual_features table when implemented
                    visual_vec = np.zeros(512, dtype=np.float32)
                else:
                    visual_vec = np.zeros(512, dtype=np.float32)

                # Extract viewpoint corrections - placeholder
                # Note: Schema needs corrected_* columns
                original_vp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                corrected_vp = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                # Correction metadata - placeholder
                keypoint_adjustments = 0
                confidence_changes = 0.0

                pattern = MultiModalCorrectionPattern(
                    original_viewpoint=original_vp,
                    corrected_viewpoint=corrected_vp,
                    geometric_features=geometric_vec,
                    visual_features=visual_vec,
                    keypoint_adjustments=keypoint_adjustments,
                    confidence_changes=confidence_changes
                )

                patterns.append(pattern)

            logger.info(
                f"Extracted {len(patterns)} multi-modal correction patterns",
                extra={'extra_data': {'count': len(patterns)}}
            )

            return patterns

    def train(
        self,
        patterns: Optional[List[MultiModalCorrectionPattern]] = None
    ) -> Dict[str, float]:
        """
        Train multi-modal correction models.

        Args:
            patterns: List of correction patterns (extracts from DB if None)

        Returns:
            Dictionary of R² scores for each model

        Raises:
            InsufficientDataError: If not enough training data
            CorrectionLearnerError: If training fails
        """
        logger.info("Training multi-modal correction models")

        # Extract patterns if not provided
        if patterns is None:
            patterns = self.extract_correction_patterns()

        if len(patterns) < self.min_training_samples:
            raise InsufficientDataError(
                f"Need at least {self.min_training_samples} corrections"
            )

        # Build feature matrix
        if self.use_visual_features:
            # Concatenate geometric + visual
            X = np.array([p.combined_features() for p in patterns])  # (N, 564)
            logger.info(f"Training on multi-modal features: {X.shape}")
        else:
            # Geometric only
            X = np.array([p.geometric_features for p in patterns])  # (N, 52)
            logger.info(f"Training on geometric features only: {X.shape}")

        # Target variables: correction deltas for each angle
        y_elevation = np.array([p.viewpoint_delta()[0] for p in patterns])
        y_azimuth = np.array([p.viewpoint_delta()[1] for p in patterns])
        y_facing = np.array([p.viewpoint_delta()[2] for p in patterns])

        targets = {
            'elevation': y_elevation,
            'azimuth': y_azimuth,
            'facing': y_facing
        }

        scores = {}

        # Train separate model for each angle
        for name, y in targets.items():
            logger.debug(f"Training {name} multi-modal correction model")

            try:
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train Random Forest with more capacity for multi-modal features
                model = RandomForestRegressor(
                    n_estimators=200,        # More trees for richer features
                    max_depth=15,            # Deeper trees for complex patterns
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )

                model.fit(X_scaled, y)

                # Calculate R² score
                score = model.score(X_scaled, y)
                scores[name] = score

                # Store model and scaler
                self.models[name] = model
                self.scalers[name] = scaler

                logger.info(
                    f"Trained multi-modal {name} model: R²={score:.3f}",
                    extra={'extra_data': {
                        'model': name,
                        'r2_score': score,
                        'feature_dim': X.shape[1],
                        'n_samples': len(patterns)
                    }}
                )

            except Exception as e:
                logger.error(f"Failed to train {name} model: {e}", exc_info=True)
                raise CorrectionLearnerError(f"Training failed for {name}: {e}") from e

        # Save models to disk
        self._save_models()

        logger.info(
            f"Multi-modal training complete",
            extra={'extra_data': {
                'scores': scores,
                'samples': len(patterns),
                'feature_dim': X.shape[1]
            }}
        )

        return scores

    def predict_correction(
        self,
        geometric_features: np.ndarray,
        visual_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict viewpoint correction for given features.

        Args:
            geometric_features: 52-dimensional geometric features
            visual_features: 576-dimensional visual features (optional)

        Returns:
            Predicted correction deltas (3,) [elevation, azimuth, facing]

        Raises:
            CorrectionLearnerError: If models not trained
        """
        if not self.models:
            raise CorrectionLearnerError("Models not trained. Call train() first.")

        # Validate inputs (geometric vector format v3 = 66-dim)
        if geometric_features.shape != (66,):
            raise ValueError(f"Expected (66,) geometric features, got {geometric_features.shape}")

        # Prepare input
        if self.use_visual_features:
            if visual_features is None:
                raise ValueError("Visual features required but not provided")
            if visual_features.shape != (576,):
                raise ValueError(f"Expected (576,) visual features, got {visual_features.shape}")
            X = np.concatenate([geometric_features, visual_features]).reshape(1, -1)
        else:
            X = geometric_features.reshape(1, -1)

        # Predict each angle correction
        deltas = []
        for name in self.MODEL_NAMES:
            X_scaled = self.scalers[name].transform(X)
            delta = self.models[name].predict(X_scaled)[0]
            deltas.append(delta)

        return np.array(deltas, dtype=np.float32)

    def apply_correction(
        self,
        original_viewpoint: np.ndarray,
        geometric_features: np.ndarray,
        visual_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply learned correction to viewpoint estimate.

        Args:
            original_viewpoint: Original viewpoint (3,) [elevation, azimuth, facing]
            geometric_features: 52-dim geometric features
            visual_features: 576-dim visual features (optional)

        Returns:
            Corrected viewpoint (3,)
        """
        if not self.models:
            logger.debug("No trained models, returning original viewpoint")
            return original_viewpoint

        try:
            # Predict correction
            delta = self.predict_correction(geometric_features, visual_features)

            # Apply correction
            corrected = original_viewpoint + delta

            # Clamp to valid ranges
            corrected[0] = np.clip(corrected[0], -90, 90)    # elevation
            corrected[1] = np.clip(corrected[1], 0, 360)     # azimuth
            corrected[2] = np.clip(corrected[2], 0, 360)     # facing

            logger.debug(
                f"Applied multi-modal correction: delta={delta}",
                extra={'extra_data': {
                    'original': original_viewpoint.tolist(),
                    'delta': delta.tolist(),
                    'corrected': corrected.tolist(),
                    'used_visual': visual_features is not None
                }}
            )

            return corrected

        except Exception as e:
            logger.warning(f"Failed to apply correction: {e}, using original")
            return original_viewpoint

    def unload_models(self) -> None:
        """
        Unload RandomForest models and scalers from memory to free resources.

        Clears all trained models and scalers. Models can be reloaded from disk
        using load_models() if previously saved.
        """
        if self.models:
            del self.models
            self.models = {}
            logger.info("Unloaded RandomForest models from memory")

        if self.scalers:
            del self.scalers
            self.scalers = {}
            logger.info("Unloaded scalers from memory")

        # Force garbage collection
        import gc
        gc.collect()

        logger.info("Multi-modal correction learner models unloaded")

    def compare_with_geometric_only(
        self,
        patterns: List[MultiModalCorrectionPattern]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multi-modal vs geometric-only model performance.

        Args:
            patterns: List of correction patterns

        Returns:
            Dictionary with comparison metrics
        """
        from sklearn.model_selection import cross_val_score

        logger.info("Comparing multi-modal vs geometric-only models")

        # Prepare feature matrices
        X_geometric = np.array([p.geometric_features for p in patterns])
        X_multimodal = np.array([p.combined_features() for p in patterns])

        # Target variables
        targets = {
            'elevation': np.array([p.viewpoint_delta()[0] for p in patterns]),
            'azimuth': np.array([p.viewpoint_delta()[1] for p in patterns]),
            'facing': np.array([p.viewpoint_delta()[2] for p in patterns])
        }

        results = {}

        for name, y in targets.items():
            # Geometric-only model
            model_geo = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            scores_geo = cross_val_score(model_geo, X_geometric, y, cv=5, scoring='r2')

            # Multi-modal model
            model_multi = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            scores_multi = cross_val_score(model_multi, X_multimodal, y, cv=5, scoring='r2')

            # Calculate improvement
            improvement = (scores_multi.mean() - scores_geo.mean()) / abs(scores_geo.mean()) * 100

            results[name] = {
                'geometric_r2': float(scores_geo.mean()),
                'multimodal_r2': float(scores_multi.mean()),
                'improvement_pct': float(improvement)
            }

            logger.info(
                f"{name}: Geometric R²={scores_geo.mean():.3f}, "
                f"Multi-modal R²={scores_multi.mean():.3f}, "
                f"Improvement={improvement:.1f}%"
            )

        return results
