"""
Configuration management for PostureKit.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""
    
    # Project root directory
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # PostgreSQL Database Configuration
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'posturekit')  # Base name, suffix added by profile
    DB_USER: str = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')

    # Database Profile (irl/2d/3d) - determines which isolated database to use
    DB_PROFILE: str = os.getenv('DB_PROFILE', 'irl')

    def get_database_name(self, profile: Optional[str] = None) -> str:
        """
        Get database name for a specific profile.

        Args:
            profile: Database profile (irl/2d/3d). Uses DB_PROFILE if None.

        Returns:
            Full database name (e.g., 'posturekit_irl')
        """
        active_profile = profile or self.DB_PROFILE
        if active_profile not in ['irl', '2d', '3d']:
            raise ValueError(f"Invalid database profile: {active_profile}. Must be 'irl', '2d', or '3d'")
        return f"{self.DB_NAME}_{active_profile}"

    def get_index_dir(self, profile: Optional[str] = None) -> Path:
        """
        Get FAISS index directory for a specific profile.

        Args:
            profile: Database profile (irl/2d/3d). Uses DB_PROFILE if None.

        Returns:
            Path to profile-specific index directory (e.g., 'data/indices/irl/')
        """
        active_profile = profile or self.DB_PROFILE
        if active_profile not in ['irl', '2d', '3d']:
            raise ValueError(f"Invalid database profile: {active_profile}. Must be 'irl', '2d', or '3d'")
        return self.PROJECT_ROOT / "data" / "indices" / active_profile

    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL connection string using active profile."""
        db_name = self.get_database_name()
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{db_name}"
        )

    def get_database_url(self, profile: Optional[str] = None) -> str:
        """
        Get database URL for a specific profile.

        Args:
            profile: Database profile (irl/2d/3d). Uses DB_PROFILE if None.

        Returns:
            Full PostgreSQL connection string
        """
        db_name = self.get_database_name(profile)
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{db_name}"
        )
    
    # Image Processing Configuration
    IMAGE_TARGET_WIDTH: int = int(os.getenv('IMAGE_TARGET_WIDTH', '1024'))
    IMAGE_TARGET_HEIGHT: int = int(os.getenv('IMAGE_TARGET_HEIGHT', '1024'))
    
    @property
    def IMAGE_TARGET_SIZE(self) -> tuple[int, int]:
        """Target image size for processing."""
        return (self.IMAGE_TARGET_WIDTH, self.IMAGE_TARGET_HEIGHT)
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Path = PROJECT_ROOT / os.getenv('LOG_FILE', 'logs/posturekit.log')
    
    # Model Configuration
    # Modern MMPoseInferencer approach - use full model name for MMPose 1.3.2
    MMPOSE_MODEL_NAME: str = os.getenv('MMPOSE_MODEL_NAME', 'rtmw-l_8xb320-270e_cocktail14-384x288')
    MMPOSE_DET_MODEL: str = os.getenv('MMPOSE_DET_MODEL', 'rtmdet-m')

    # Optional: Custom model weights path (if not using auto-download)
    MMPOSE_MODEL_PATH: Optional[Path] = None
    MMPOSE_CONFIG_PATH: Optional[Path] = None

    @property
    def MMPOSE_CHECKPOINT_PATH(self) -> Optional[Path]:
        """Alias for MMPOSE_MODEL_PATH for backward compatibility."""
        return self.MMPOSE_MODEL_PATH

    def __init__(self) -> None:
        """Initialize settings and resolve paths."""
        # Support legacy custom model paths for backward compatibility
        model_path = os.getenv('MMPOSE_MODEL_PATH')
        if model_path:
            self.MMPOSE_MODEL_PATH = self.PROJECT_ROOT / model_path

        config_path = os.getenv('MMPOSE_CONFIG_PATH')
        if config_path:
            self.MMPOSE_CONFIG_PATH = self.PROJECT_ROOT / config_path

        # Auto-detect downloaded model config and checkpoint
        if not self.MMPOSE_CONFIG_PATH or not self.MMPOSE_MODEL_PATH:
            # Check for downloaded RTMW-L model
            models_dir = self.PROJECT_ROOT / "data" / "models"
            rtmw_config = models_dir / "rtmw-l_8xb320-270e_cocktail14-384x288.py"
            rtmw_checkpoint = models_dir / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

            if rtmw_config.exists() and rtmw_checkpoint.exists():
                if not self.MMPOSE_CONFIG_PATH:
                    self.MMPOSE_CONFIG_PATH = rtmw_config
                    self.MMPOSE_MODEL_NAME = str(rtmw_config)  # Use explicit path instead of alias
                if not self.MMPOSE_MODEL_PATH:
                    self.MMPOSE_MODEL_PATH = rtmw_checkpoint
    
    # Device Configuration
    DEVICE: str = os.getenv('DEVICE', 'mps')

    # Threading Configuration
    NUM_THREADS: int = int(os.getenv('NUM_THREADS', '4'))

    # Pose Model Constants
    NUM_KEYPOINTS: int = 133  # RTMW-L COCO-WholeBody format
    KEYPOINT_COORDINATES_DIM: int = 3  # (x, y, confidence)

    @property
    def KEYPOINTS_FLAT_DIM(self) -> int:
        """Total flattened keypoint dimension."""
        return self.NUM_KEYPOINTS * self.KEYPOINT_COORDINATES_DIM

    # Feature Vector Dimensions
    GEOMETRIC_FEATURE_DIM: int = 52  # 12 angles + 10 ratios + 15 body angles + 8 symmetry + 7 occlusion
    VISUAL_FEATURE_DIM: int = 576  # MobileNetV3-Small output dimension

    @property
    def FUSED_FEATURE_DIM(self) -> int:
        """Total fused feature dimension (geometric + visual)."""
        return self.GEOMETRIC_FEATURE_DIM + self.VISUAL_FEATURE_DIM

    # ========================================================================
    # 4-Layer Model Configuration
    # ========================================================================

    # Layer 1: Pose Detection (RTMPose)
    POSE_MODEL: str = os.getenv('POSE_MODEL', 'ensemble')  # rtmw-l, rtmw-m, rtmw-x, ensemble
    POSE_MODELS_ENSEMBLE: str = os.getenv('POSE_MODELS_ENSEMBLE', 'rtmw-l,rtmw-x')  # Comma-separated
    ENSEMBLE_FUSION_METHOD: str = os.getenv('ENSEMBLE_FUSION_METHOD', 'confidence_weighted')

    @property
    def POSE_MODELS_ENSEMBLE_LIST(self) -> list[str]:
        """Parse ensemble models list from comma-separated string."""
        return [m.strip() for m in self.POSE_MODELS_ENSEMBLE.split(',') if m.strip()]

    # Layer 2: Body Part Detection (NudeNet)
    BODY_PART_MODEL: str = os.getenv('BODY_PART_MODEL', '320n')  # 320n, 640m, disabled
    BODY_PART_CONFIDENCE: float = float(os.getenv('BODY_PART_CONFIDENCE', '0.3'))

    # Layer 3: Visual Features (Appearance)
    VISUAL_MODEL: str = os.getenv('VISUAL_MODEL', 'mobilenet_v3_small')  # mobilenet_v3_small, disabled

    # Layer 4: Search Features
    SEARCH_FEATURE_MODE: str = os.getenv('SEARCH_FEATURE_MODE', 'geometric')  # geometric, fused

    # ========================================================================
    # Pose-Based Search Configuration (Appearance-Free)
    # ========================================================================

    # Browse Quality Scoring Weights
    # Formula: w1 * body_part_conf + w2 * pose_conf + w3 * completeness
    BROWSE_QUALITY_WEIGHT_BODY_PARTS: float = float(os.getenv('BROWSE_QUALITY_WEIGHT_BODY_PARTS', '0.50'))
    BROWSE_QUALITY_WEIGHT_POSE: float = float(os.getenv('BROWSE_QUALITY_WEIGHT_POSE', '0.30'))
    BROWSE_QUALITY_WEIGHT_COMPLETENESS: float = float(os.getenv('BROWSE_QUALITY_WEIGHT_COMPLETENESS', '0.20'))

    # Joint Importance Weighting (for similarity search)
    ENABLE_JOINT_IMPORTANCE_WEIGHTING: bool = os.getenv('ENABLE_JOINT_IMPORTANCE_WEIGHTING', 'false').lower() == 'true'
    JOINT_WEIGHT_CRITICAL: float = float(os.getenv('JOINT_WEIGHT_CRITICAL', '3.0'))  # Spine, hips, shoulders
    JOINT_WEIGHT_IMPORTANT: float = float(os.getenv('JOINT_WEIGHT_IMPORTANT', '2.0'))  # Elbows, knees
    JOINT_WEIGHT_MODERATE: float = float(os.getenv('JOINT_WEIGHT_MODERATE', '1.0'))  # Wrists, ankles
    JOINT_WEIGHT_MINOR: float = float(os.getenv('JOINT_WEIGHT_MINOR', '0.5'))  # Hands, feet details

    # OKS (Object Keypoint Similarity) Metric
    ENABLE_OKS_METRIC: bool = os.getenv('ENABLE_OKS_METRIC', 'false').lower() == 'true'
    OKS_SIGMAS: dict = {
        # Standard COCO keypoint sigmas (κ values for uncertainty)
        # Smaller sigmas = stricter matching for that keypoint
        'head': 0.026,
        'shoulders': 0.079,
        'elbows': 0.072,
        'wrists': 0.062,
        'hips': 0.107,
        'knees': 0.087,
        'ankles': 0.089,
    }

    # Mirror/Flip Search
    ENABLE_FLIP_SEARCH: bool = os.getenv('ENABLE_FLIP_SEARCH', 'false').lower() == 'true'
    FLIP_SEARCH_MERGE_TOP_K: int = int(os.getenv('FLIP_SEARCH_MERGE_TOP_K', '10'))  # Merge top N from each orientation

    # Pose Plausibility Scoring
    ENABLE_PLAUSIBILITY_SCORING: bool = os.getenv('ENABLE_PLAUSIBILITY_SCORING', 'false').lower() == 'true'
    PLAUSIBILITY_WEIGHT: float = float(os.getenv('PLAUSIBILITY_WEIGHT', '0.2'))  # Boost factor (0-0.5)
    PLAUSIBILITY_MIN_LIMB_SYMMETRY: float = float(os.getenv('PLAUSIBILITY_MIN_LIMB_SYMMETRY', '0.8'))  # Left ≈ Right

    def validate(self) -> list[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check database credentials
        if not self.DB_PASSWORD:
            errors.append("DB_PASSWORD not set in environment")
        
        # Check directories exist
        if not self.LOG_FILE.parent.exists():
            try:
                self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {e}")
        
        # Validate device selection
        valid_devices = ['mps', 'cuda', 'cpu']
        if self.DEVICE not in valid_devices:
            errors.append(
                f"Invalid DEVICE '{self.DEVICE}'. Must be one of: {valid_devices}"
            )
        
        return errors


# Global settings instance
settings = Settings()


def validate_settings() -> None:
    """
    Validate settings and raise exception if invalid.
    
    Raises:
        ValueError: If any settings are invalid
    """
    errors = settings.validate()
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )