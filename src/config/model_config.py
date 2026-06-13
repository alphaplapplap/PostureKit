"""
Model Configuration for PostureKit.
Defines all 4 detection model layers and their configurable parameters.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Layer 1: Pose Detection Models (RTMPose)
# ============================================================================

class PoseModel(Enum):
    """Available pose detection models."""
    RTMW_L = "rtmw-l"     # 8xb1024-270e, 256×192 - Balanced
    RTMW_M = "rtmw-m"     # 8xb1024-270e, 256×192 - Fast
    RTMW_X = "rtmw-x"     # 8xb320-270e, 384×288 - Accurate
    ENSEMBLE = "ensemble"  # Multi-model fusion

    def get_config_file(self) -> str:
        """Get model configuration file path."""
        config_map = {
            PoseModel.RTMW_L: "data/models/rtmw-l_8xb320-270e_cocktail14-384x288.py",
            PoseModel.RTMW_M: "data/models/rtmw-m_8xb1024-270e_cocktail14-256x192.py",
            PoseModel.RTMW_X: "data/models/rtmw-x_8xb320-270e_cocktail14-384x288.py",
        }
        return config_map.get(self, "")


class EnsembleFusionMethod(Enum):
    """Ensemble fusion methods for multi-model pose detection."""
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by detection confidence
    AVERAGE = "average"                          # Simple average
    MAX_CONFIDENCE = "max"                       # Take maximum confidence


# ============================================================================
# Layer 2: Body Part Detection Models (NudeNet)
# ============================================================================

class BodyPartModel(Enum):
    """Available body part detection models (NudeNet)."""
    NUDENET_320N = "320n"   # Fast (~15 FPS on CPU)
    NUDENET_640M = "640m"   # Accurate (~8 FPS on CPU)
    DISABLED = "disabled"   # Skip body part detection

    def get_fps_estimate(self) -> str:
        """Get FPS estimate for this model."""
        fps_map = {
            BodyPartModel.NUDENET_320N: "~15 FPS (CPU)",
            BodyPartModel.NUDENET_640M: "~8 FPS (CPU)",
            BodyPartModel.DISABLED: "N/A",
        }
        return fps_map.get(self, "Unknown")


# ============================================================================
# Layer 3: Visual Feature Extraction Models (Appearance)
# ============================================================================

class VisualModel(Enum):
    """Available visual feature extraction models."""
    MOBILENET_V3_SMALL = "mobilenet_v3_small"  # 576-dim embeddings
    DISABLED = "disabled"                       # Skip visual features

    def get_feature_dim(self) -> int:
        """Get feature dimensionality."""
        dim_map = {
            VisualModel.MOBILENET_V3_SMALL: 576,
            VisualModel.DISABLED: 0,
        }
        return dim_map.get(self, 0)


# ============================================================================
# Layer 4: Search Feature Mode
# ============================================================================

class SearchFeatureMode(Enum):
    """Feature types used for similarity search."""
    GEOMETRIC_ONLY = "geometric"   # 66-dim keypoint-based features (v3)
    FUSED_MULTIMODAL = "fused"     # 642-dim geometric+visual features (66 + 576)

    def get_feature_dim(self) -> int:
        """Get feature dimensionality for this mode (geometric vector format v3)."""
        dim_map = {
            SearchFeatureMode.GEOMETRIC_ONLY: 66,
            SearchFeatureMode.FUSED_MULTIMODAL: 642,
        }
        return dim_map.get(self, 66)

    def requires_visual_features(self) -> bool:
        """Check if this mode requires visual features."""
        return self == SearchFeatureMode.FUSED_MULTIMODAL


# ============================================================================
# Complete Model Configuration
# ============================================================================

@dataclass
class ModelConfiguration:
    """
    Complete configuration for all 4 detection model layers.

    Layer 1: Pose Detection (RTMPose) - Detects 133 keypoints
    Layer 2: Body Part Detection (NudeNet) - Detects 18 body part regions
    Layer 3: Visual Features (MobileNet) - Extracts 576-dim appearance features
    Layer 4: Search Features - Selects which features to use for similarity
    """

    # ========================================================================
    # Layer 1: Pose Detection
    # ========================================================================
    pose_model: PoseModel = PoseModel.ENSEMBLE
    pose_models_ensemble: List[str] = field(default_factory=lambda: ["rtmw-l", "rtmw-x"])
    ensemble_fusion_method: EnsembleFusionMethod = EnsembleFusionMethod.CONFIDENCE_WEIGHTED

    # ========================================================================
    # Layer 2: Body Part Detection
    # ========================================================================
    body_part_model: BodyPartModel = BodyPartModel.NUDENET_320N
    body_part_confidence: float = 0.3  # Detection confidence threshold

    # ========================================================================
    # Layer 3: Visual Features
    # ========================================================================
    visual_model: VisualModel = VisualModel.DISABLED  # User doesn't care about clothing

    # ========================================================================
    # Layer 4: Search Features
    # ========================================================================
    search_feature_mode: SearchFeatureMode = SearchFeatureMode.GEOMETRIC_ONLY

    # ========================================================================
    # Hardware
    # ========================================================================
    device: str = "mps"  # mps, cuda, cpu
    num_threads: int = 12

    def validate(self) -> List[str]:
        """
        Validate configuration consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check: Fused mode requires visual features
        if self.search_feature_mode == SearchFeatureMode.FUSED_MULTIMODAL:
            if self.visual_model == VisualModel.DISABLED:
                errors.append(
                    "SearchFeatureMode.FUSED_MULTIMODAL requires visual features. "
                    "Enable visual_model or switch to GEOMETRIC_ONLY mode."
                )

        # Check: Ensemble requires models list
        if self.pose_model == PoseModel.ENSEMBLE:
            if not self.pose_models_ensemble or len(self.pose_models_ensemble) < 2:
                errors.append(
                    "PoseModel.ENSEMBLE requires at least 2 models in pose_models_ensemble list."
                )

        # Check: Confidence threshold in valid range
        if not (0.0 <= self.body_part_confidence <= 1.0):
            errors.append(
                f"body_part_confidence must be in [0.0, 1.0], got {self.body_part_confidence}"
            )

        # Check: Valid device
        valid_devices = ["mps", "cuda", "cpu"]
        if self.device not in valid_devices:
            errors.append(
                f"device must be one of {valid_devices}, got '{self.device}'"
            )

        return errors

    def get_search_dimension(self) -> int:
        """Get the expected feature dimension for similarity search."""
        return self.search_feature_mode.get_feature_dim()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'pose_model': self.pose_model.value,
            'pose_models_ensemble': self.pose_models_ensemble,
            'ensemble_fusion_method': self.ensemble_fusion_method.value,
            'body_part_model': self.body_part_model.value,
            'body_part_confidence': self.body_part_confidence,
            'visual_model': self.visual_model.value,
            'search_feature_mode': self.search_feature_mode.value,
            'device': self.device,
            'num_threads': self.num_threads,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfiguration':
        """Create from dictionary."""
        return cls(
            pose_model=PoseModel(config_dict.get('pose_model', 'ensemble')),
            pose_models_ensemble=config_dict.get('pose_models_ensemble', ["rtmw-l", "rtmw-x"]),
            ensemble_fusion_method=EnsembleFusionMethod(
                config_dict.get('ensemble_fusion_method', 'confidence_weighted')
            ),
            body_part_model=BodyPartModel(config_dict.get('body_part_model', '320n')),
            body_part_confidence=config_dict.get('body_part_confidence', 0.3),
            visual_model=VisualModel(config_dict.get('visual_model', 'disabled')),
            search_feature_mode=SearchFeatureMode(
                config_dict.get('search_feature_mode', 'geometric')
            ),
            device=config_dict.get('device', 'mps'),
            num_threads=config_dict.get('num_threads', 12),
        )

    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "PostureKit Model Configuration",
            "=" * 60,
            f"[1] Pose Detection:   {self.pose_model.value}",
        ]

        if self.pose_model == PoseModel.ENSEMBLE:
            lines.append(f"    Models:           {', '.join(self.pose_models_ensemble)}")
            lines.append(f"    Fusion:           {self.ensemble_fusion_method.value}")

        lines.extend([
            f"[2] Body Parts:       {self.body_part_model.value} "
            f"({self.body_part_model.get_fps_estimate()})",
            f"    Confidence:       {self.body_part_confidence}",
            f"[3] Visual Features:  {self.visual_model.value}",
            f"[4] Search Mode:      {self.search_feature_mode.value} "
            f"({self.get_search_dimension()}-dim)",
            f"[*] Device:           {self.device}",
            f"[*] Threads:          {self.num_threads}",
            "=" * 60,
        ])

        return "\n".join(lines)


# ============================================================================
# Default Configurations
# ============================================================================

# Recommended configuration for most users
DEFAULT_CONFIG = ModelConfiguration(
    pose_model=PoseModel.ENSEMBLE,
    pose_models_ensemble=["rtmw-l", "rtmw-x"],
    ensemble_fusion_method=EnsembleFusionMethod.CONFIDENCE_WEIGHTED,
    body_part_model=BodyPartModel.NUDENET_320N,
    body_part_confidence=0.3,
    visual_model=VisualModel.DISABLED,  # User doesn't care about clothing
    search_feature_mode=SearchFeatureMode.GEOMETRIC_ONLY,
    device="mps",
    num_threads=12,
)

# Fast configuration (prioritize speed)
FAST_CONFIG = ModelConfiguration(
    pose_model=PoseModel.RTMW_M,
    body_part_model=BodyPartModel.NUDENET_320N,
    visual_model=VisualModel.DISABLED,
    search_feature_mode=SearchFeatureMode.GEOMETRIC_ONLY,
    device="mps",
    num_threads=12,
)

# Accurate configuration (prioritize accuracy)
ACCURATE_CONFIG = ModelConfiguration(
    pose_model=PoseModel.ENSEMBLE,
    pose_models_ensemble=["rtmw-l", "rtmw-x"],
    ensemble_fusion_method=EnsembleFusionMethod.CONFIDENCE_WEIGHTED,
    body_part_model=BodyPartModel.NUDENET_640M,
    visual_model=VisualModel.DISABLED,
    search_feature_mode=SearchFeatureMode.GEOMETRIC_ONLY,
    device="mps",
    num_threads=12,
)

# Multimodal configuration (appearance-aware search)
MULTIMODAL_CONFIG = ModelConfiguration(
    pose_model=PoseModel.ENSEMBLE,
    pose_models_ensemble=["rtmw-l", "rtmw-x"],
    ensemble_fusion_method=EnsembleFusionMethod.CONFIDENCE_WEIGHTED,
    body_part_model=BodyPartModel.NUDENET_320N,
    visual_model=VisualModel.MOBILENET_V3_SMALL,  # Enable visual features
    search_feature_mode=SearchFeatureMode.FUSED_MULTIMODAL,  # Use fused features
    device="mps",
    num_threads=12,
)
