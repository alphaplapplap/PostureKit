"""
Multi-Modal Fusion for PostureKit.
Combines geometric and visual features into unified representation.
"""
import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass
import logging

from src.core.geometric_feature_extractor import GeometricFeatures
from src.core.visual_feature_extractor import VisualFeatures
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusedFeatures:
    """
    Fused multi-modal features combining geometric and visual.

    Attributes:
        fused_vector: Combined feature vector (628-dim)
        geometric_vector: Original geometric features (52-dim)
        visual_vector: Original visual features (576-dim)
        fusion_method: Method used for fusion
    """
    fused_vector: np.ndarray       # (628,) float32
    geometric_vector: np.ndarray   # (52,) float32
    visual_vector: np.ndarray      # (576,) float32
    fusion_method: str

    def __post_init__(self):
        """Validate feature dimensions."""
        assert self.fused_vector.shape == (628,), \
            f"Fused vector must be (628,), got {self.fused_vector.shape}"
        assert self.geometric_vector.shape == (52,), \
            f"Geometric vector must be (52,), got {self.geometric_vector.shape}"
        assert self.visual_vector.shape == (576,), \
            f"Visual vector must be (576,), got {self.visual_vector.shape}"
        assert self.fused_vector.dtype == np.float32, \
            f"Fused vector must be float32, got {self.fused_vector.dtype}"

    @property
    def feature_vector(self):
        """Alias for fused_vector for backwards compatibility."""
        return self.fused_vector

    @property
    def geometric_dim(self):
        """Dimension of geometric features."""
        return 52

    @property
    def visual_dim(self):
        """Dimension of visual features."""
        return 576

    @property
    def shape(self):
        """Convenience property to access fused_vector.shape."""
        return self.fused_vector.shape

    @property
    def dtype(self):
        """Convenience property to access fused_vector.dtype."""
        return self.fused_vector.dtype

    def __array__(self, dtype=None):
        """Support numpy operations on this object."""
        if dtype is not None:
            return self.fused_vector.astype(dtype)
        return self.fused_vector

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'fused_vector': self.fused_vector.tolist(),
            'geometric_vector': self.geometric_vector.tolist(),
            'visual_vector': self.visual_vector.tolist(),
            'fusion_method': self.fusion_method
        }


class MultiModalFusionError(Exception):
    """Base exception for Multi-Modal Fusion errors."""
    pass


class IncompatibleFeaturesError(MultiModalFusionError):
    """Raised when feature dimensions don't match expected."""
    pass


class MultiModalFusion:
    """
    Fuses geometric and visual features into unified representation.

    Fusion Strategies:
    1. 'concatenate': Simple concatenation [geometric | visual]
    2. 'weighted': Weighted concatenation with learned weights
    3. 'normalized': Concatenate with per-modality normalization

    Attributes:
        fusion_method: Fusion strategy to use
        geometric_weight: Weight for geometric features (weighted method)
        visual_weight: Weight for visual features (weighted method)
    """

    VALID_METHODS = ['concatenate', 'weighted', 'normalized']

    def __init__(
        self,
        fusion_method: Optional[Literal['concatenate', 'weighted', 'normalized']] = None,
        fusion_type: Optional[Literal['concatenate', 'weighted', 'normalized']] = None,
        geometric_weight: float = 1.0,
        visual_weight: float = 1.0
    ):
        """
        Initialize Multi-Modal Fusion.

        Args:
            fusion_method: Fusion strategy to use (preferred parameter name)
            fusion_type: Fusion strategy to use (alias for fusion_method, for backwards compatibility)
            geometric_weight: Weight for geometric features (only for 'weighted')
            visual_weight: Weight for visual features (only for 'weighted')
        """
        # Handle both parameter names for backwards compatibility
        method = fusion_method or fusion_type or 'concatenate'

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid fusion method '{method}'. "
                f"Must be one of: {self.VALID_METHODS}"
            )

        self.fusion_method = method
        self.geometric_weight = geometric_weight
        self.visual_weight = visual_weight

        logger.info(
            f"MultiModalFusion initialized: method={method}, "
            f"geometric_weight={geometric_weight}, visual_weight={visual_weight}"
        )

    def _validate_inputs(
        self,
        geometric_features,
        visual_features
    ) -> tuple:
        """
        Validate and normalize input features.

        Accepts either feature objects or raw numpy arrays.

        Returns:
            Tuple of (geometric_array, visual_array) as numpy arrays
        """
        # Handle geometric features (object or array)
        if isinstance(geometric_features, np.ndarray):
            geometric_vec = geometric_features
        elif hasattr(geometric_features, 'feature_vector'):
            geometric_vec = geometric_features.feature_vector
        else:
            raise IncompatibleFeaturesError(
                f"Invalid geometric features type: {type(geometric_features)}"
            )

        # Handle visual features (object or array)
        if isinstance(visual_features, np.ndarray):
            visual_vec = visual_features
        elif hasattr(visual_features, 'feature_vector'):
            visual_vec = visual_features.feature_vector
        else:
            raise IncompatibleFeaturesError(
                f"Invalid visual features type: {type(visual_features)}"
            )

        # Validate dimensions
        if geometric_vec.shape != (52,):
            raise IncompatibleFeaturesError(
                f"Expected 52-dim geometric features, got {geometric_vec.shape}"
            )

        if visual_vec.shape != (576,):
            raise IncompatibleFeaturesError(
                f"Expected 576-dim visual features, got {visual_vec.shape}"
            )

        return geometric_vec, visual_vec

    def _concatenate_fusion(
        self,
        geometric_vec: np.ndarray,
        visual_vec: np.ndarray
    ) -> np.ndarray:
        """
        Simple concatenation: [geometric | visual]

        Args:
            geometric_vec: 52-dim geometric features
            visual_vec: 576-dim visual features

        Returns:
            628-dim fused vector
        """
        fused = np.concatenate([geometric_vec, visual_vec])
        return fused.astype(np.float32)

    def _weighted_fusion(
        self,
        geometric_vec: np.ndarray,
        visual_vec: np.ndarray
    ) -> np.ndarray:
        """
        Weighted concatenation: [w_g * geometric | w_v * visual]

        Allows adjusting relative importance of each modality.

        Args:
            geometric_vec: 52-dim geometric features
            visual_vec: 576-dim visual features

        Returns:
            628-dim fused vector
        """
        weighted_geometric = geometric_vec * self.geometric_weight
        weighted_visual = visual_vec * self.visual_weight

        fused = np.concatenate([weighted_geometric, weighted_visual])
        return fused.astype(np.float32)

    def _normalized_fusion(
        self,
        geometric_vec: np.ndarray,
        visual_vec: np.ndarray
    ) -> np.ndarray:
        """
        Normalized concatenation: [norm(geometric) | norm(visual)]

        L2 normalizes each modality separately before concatenation.
        Ensures both modalities contribute equally regardless of scale.

        Args:
            geometric_vec: 52-dim geometric features
            visual_vec: 576-dim visual features

        Returns:
            628-dim fused vector
        """
        # L2 normalize each modality
        geometric_norm = np.linalg.norm(geometric_vec)
        visual_norm = np.linalg.norm(visual_vec)

        if geometric_norm > 0:
            geometric_normalized = geometric_vec / geometric_norm
        else:
            geometric_normalized = geometric_vec

        if visual_norm > 0:
            visual_normalized = visual_vec / visual_norm
        else:
            visual_normalized = visual_vec

        fused = np.concatenate([geometric_normalized, visual_normalized])
        return fused.astype(np.float32)

    def fuse(
        self,
        geometric_features,
        visual_features
    ) -> FusedFeatures:
        """
        Fuse geometric and visual features into unified representation.

        Accepts either feature objects or raw numpy arrays.

        Args:
            geometric_features: 52-dim geometric features (GeometricFeatures object or numpy array)
            visual_features: 576-dim visual features (VisualFeatures object or numpy array)

        Returns:
            FusedFeatures with 628-dim combined vector

        Raises:
            IncompatibleFeaturesError: If feature dimensions don't match
            MultiModalFusionError: If fusion fails
        """
        # Validate and normalize inputs
        geometric_vec, visual_vec = self._validate_inputs(geometric_features, visual_features)

        logger.debug(
            f"Fusing features using {self.fusion_method} method: "
            f"geometric_shape={geometric_vec.shape}, "
            f"visual_shape={visual_vec.shape}"
        )

        try:
            # Apply fusion strategy
            if self.fusion_method == 'concatenate':
                fused_vec = self._concatenate_fusion(geometric_vec, visual_vec)
            elif self.fusion_method == 'weighted':
                fused_vec = self._weighted_fusion(geometric_vec, visual_vec)
            elif self.fusion_method == 'normalized':
                fused_vec = self._normalized_fusion(geometric_vec, visual_vec)
            else:
                raise MultiModalFusionError(f"Unknown fusion method: {self.fusion_method}")

            # Create result
            result = FusedFeatures(
                fused_vector=fused_vec,
                geometric_vector=geometric_vec,
                visual_vector=visual_vec,
                fusion_method=self.fusion_method
            )

            logger.debug(
                f"Fusion complete: fused_shape={fused_vec.shape}, "
                f"fused_norm={float(np.linalg.norm(fused_vec)):.4f}"
            )

            return result

        except Exception as e:
            if isinstance(e, (IncompatibleFeaturesError, MultiModalFusionError)):
                raise
            raise MultiModalFusionError(f"Fusion failed: {e}") from e

    def fuse_from_arrays(
        self,
        geometric_array: np.ndarray,
        visual_array: np.ndarray
    ) -> np.ndarray:
        """
        Fuse geometric and visual feature arrays directly.

        This is a convenience method that works with raw numpy arrays
        instead of feature objects.

        Args:
            geometric_array: 52-dim geometric features as numpy array
            visual_array: 576-dim visual features as numpy array

        Returns:
            628-dim fused vector as numpy array

        Raises:
            IncompatibleFeaturesError: If feature dimensions don't match
        """
        # Validate inputs
        if geometric_array.shape != (52,):
            raise IncompatibleFeaturesError(
                f"Expected (52,) geometric features, got {geometric_array.shape}"
            )
        if visual_array.shape != (576,):
            raise IncompatibleFeaturesError(
                f"Expected (576,) visual features, got {visual_array.shape}"
            )

        # Apply fusion strategy
        if self.fusion_method == 'concatenate':
            fused = self._concatenate_fusion(geometric_array, visual_array)
        elif self.fusion_method == 'weighted':
            fused = self._weighted_fusion(geometric_array, visual_array)
        elif self.fusion_method == 'normalized':
            fused = self._normalized_fusion(geometric_array, visual_array)
        else:
            raise MultiModalFusionError(f"Unknown fusion method: {self.fusion_method}")

        return fused

    def split_fused(self, fused_vector: np.ndarray) -> tuple:
        """
        Split fused vector back into geometric and visual components.

        Args:
            fused_vector: 628-dim fused feature vector

        Returns:
            Tuple of (geometric_vector, visual_vector)

        Raises:
            IncompatibleFeaturesError: If fused vector has wrong dimensions
        """
        if fused_vector.shape != (628,):
            raise IncompatibleFeaturesError(
                f"Expected (628,) fused vector, got {fused_vector.shape}"
            )

        geometric_vec = fused_vector[:52]
        visual_vec = fused_vector[52:]

        return geometric_vec, visual_vec

    def split_features(self, fused_vector: np.ndarray) -> tuple:
        """
        Alias for split_fused() for backwards compatibility.

        Args:
            fused_vector: 628-dim fused feature vector

        Returns:
            Tuple of (geometric_vector, visual_vector)
        """
        return self.split_fused(fused_vector)

    def set_weights(self, geometric_weight: float, visual_weight: float) -> None:
        """
        Update fusion weights (only affects 'weighted' method).

        Args:
            geometric_weight: New weight for geometric features
            visual_weight: New weight for visual features
        """
        self.geometric_weight = geometric_weight
        self.visual_weight = visual_weight

        logger.info(
            f"Updated fusion weights: geometric_weight={geometric_weight}, "
            f"visual_weight={visual_weight}"
        )


def create_fusion(
    method: Literal['concatenate', 'weighted', 'normalized'] = 'concatenate',
    **kwargs
) -> MultiModalFusion:
    """
    Factory function to create fusion instance.

    Args:
        method: Fusion method to use
        **kwargs: Additional arguments for fusion (weights, etc.)

    Returns:
        Configured MultiModalFusion instance
    """
    return MultiModalFusion(fusion_method=method, **kwargs)
