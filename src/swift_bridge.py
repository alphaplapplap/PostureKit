"""
Swift Bridge for PostureKit - Direct Python Integration
Provides a clean interface between Swift and Python modules.
"""
import os
import sys
import json
import logging
import hashlib
import warnings
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import numpy as np

# CRITICAL: Enable MPS fallback for unsupported operations
# Must be set BEFORE any PyTorch imports or model loading
# PyTorch MPS (Metal Performance Shaders) on Apple Silicon doesn't support all operations
# This enables automatic CPU fallback for unsupported ops (e.g., hardsigmoid)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Suppress FutureWarnings from mmpose/PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

# Add project root to path for imports (parent of src directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL: Import torch patch BEFORE any mmpose/mmengine imports
import src.core._torch_patch as core_torch_patch

logger = logging.getLogger(__name__)


# Worker function for parallel thumbnail generation (must be at module level for pickling)
def _process_thumbnail_worker(args):
    """
    Worker function for parallel thumbnail generation.

    Args:
        args: Tuple of (image_id, file_path)

    Returns:
        Dict with keys: id, thumbnail_bytes, success, error
    """
    import cv2
    from pathlib import Path
    from src.utils.thumbnail_generator import ThumbnailGenerator

    image_id, file_path = args

    try:
        # Check if file exists
        if not Path(file_path).exists():
            return {
                'id': image_id,
                'thumbnail_bytes': None,
                'success': False,
                'error': 'File not found',
                'file_path': file_path
            }

        # Load image
        image_bgr = cv2.imread(str(file_path))
        if image_bgr is None:
            return {
                'id': image_id,
                'thumbnail_bytes': None,
                'success': False,
                'error': 'Failed to load image',
                'file_path': file_path
            }

        # Generate thumbnail (using optimized cv2.imencode path)
        generator = ThumbnailGenerator(size=200, quality=80)
        thumbnail_bytes = generator.generate_from_array(image_bgr, format_bgr=True)

        if thumbnail_bytes:
            return {
                'id': image_id,
                'thumbnail_bytes': thumbnail_bytes,
                'success': True,
                'error': None,
                'file_path': file_path
            }
        else:
            return {
                'id': image_id,
                'thumbnail_bytes': None,
                'success': False,
                'error': 'Thumbnail generation returned None',
                'file_path': file_path
            }

    except Exception as e:
        return {
            'id': image_id,
            'thumbnail_bytes': None,
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


# Import core modules (required)
try:
    from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
    from src.core.geometric_feature_extractor import GeometricFeatureExtractor, GeometricFeatures
    from src.core.visual_feature_extractor import VisualFeatureExtractor
    from src.core.multimodal_fusion import MultiModalFusion
    from src.core.ensemble_detector import EnsembleDetector, EnsembleConfig
    from src.intelligence.similarity_engine import SimilarityEngine
    from src.storage.storage_manager import StorageManager
    from src.storage.models import Image, PoseDetection, GeometricFeatures as GeometricFeaturesModel
    from src.core.image_ingestor import ImageMetadata

    # Apply mmengine patch after imports (now mmengine is in sys.modules)
    core_torch_patch._apply_mmengine_patch()

    CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Core modules not available: {e}")
    CORE_AVAILABLE = False
    raise  # Re-raise - core modules are required

# Import optional modules (two-stage detection)
try:
    from src.core.two_stage_detector import TwoStageDetector
    TWO_STAGE_AVAILABLE = True
except ImportError as e:
    logger.info(f"Two-stage detection not available (optional): {e}")
    TwoStageDetector = None
    TWO_STAGE_AVAILABLE = False


class PostureKitBridge:
    """
    Bridge class for Swift integration.
    Provides simplified interface to PostureKit Python modules.
    """

    # Model registry with configurations
    AVAILABLE_MODELS = {
        'rtmw-l': {
            'name': 'RTMW-L',
            'config': 'rtmw-l_8xb320-270e_cocktail14-384x288.py',
            'checkpoint': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
            'size_mb': 220,
            'resolution': (384, 288),
            'description': 'Baseline model, good general performance'
        },
        'rtmw-x': {
            'name': 'RTMW-X',
            'config': 'rtmw-x_8xb320-270e_cocktail14-384x288.py',
            'checkpoint': 'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
            'size_mb': 353,
            'resolution': (384, 288),
            'description': 'Larger model, better for difficult cases'
        },
    }

    def __init__(
        self,
        db_url: Optional[str] = None,
        use_two_stage: bool = None,
        use_ensemble: bool = False,
        pose_models: Union[str, List[str]] = None,
        fusion_method: str = "confidence_weighted",
        num_threads: int = 4,
        device: Optional[str] = None,
        body_part_model: Optional[str] = None,
        body_part_confidence: float = 0.3,
        body_part_thresholds: Optional[Dict[str, float]] = None,
        visual_model: Optional[str] = None,
        search_feature_mode: Optional[str] = None,
        skip_models: bool = False,
    ):
        """
        Initialize PostureKit bridge.

        Args:
            db_url: Database URL (defaults to PostgreSQL from settings)
            use_two_stage: Enable two-stage detection (YOLO + pose estimation)
                          Default False. Set True for 5-10% accuracy improvement.
            use_ensemble: DEPRECATED - Use pose_models instead.
                         Enable ensemble detection (multiple models with fusion)
                         Default False. Set True for 10-15% accuracy improvement.
            pose_models: Which pose model(s) to use. Options:
                        - 'rtmw-l' (default): RTMW-L only (220MB, fast, good quality)
                        - 'rtmw-x': RTMW-X only (353MB, slower, best quality)
                        - ['rtmw-l', 'rtmw-x']: Ensemble with both (2.5x slower, 10-15% better)
                        - ['rtmw-l']: RTMW-L via ensemble (same as 'rtmw-l')
                        - ['rtmw-x']: RTMW-X via ensemble (same as 'rtmw-x')
            fusion_method: Fusion method for ensemble (when pose_models is a list)
                          Options: 'confidence_weighted' (default) or 'weighted_average'
            num_threads: Number of threads for CPU inference (PyTorch, OpenCV, OpenMP)
                        Default 4. Higher values (8, 16) improve performance on multi-core systems.
            device: Device to use for inference ('mps', 'cuda', 'cpu')
                   Default None (uses settings.DEVICE). 'mps' for Apple Silicon GPU acceleration.
            body_part_model: NudeNet model size ('320n', '640m', 'disabled')
                            Default None (uses settings.BODY_PART_MODEL). '320n' is fast, '640m' is accurate.
            body_part_confidence: Minimum confidence for body part detection
                                 Default 0.3. Range: 0.0-1.0
            visual_model: Visual feature model ('mobilenet_v3_small', 'disabled')
                         Default None (uses settings.VISUAL_MODEL). Used for appearance-based matching.
            search_feature_mode: Feature mode for similarity search ('geometric', 'fused')
                                Default None (uses settings.SEARCH_FEATURE_MODE). 'geometric' = pose structure only.
            skip_models: Skip loading pose detection models (for search-only operations)
                        Default False. Set True to avoid 400MB+ model load when only searching.
        """
        # Import settings and model config
        from src.config.settings import settings
        from src.config.model_config import SearchFeatureMode, BodyPartModel, VisualModel
        import contextlib
        import sys

        if db_url is None:
            db_url = settings.DATABASE_URL

        # Load model configuration from settings if not provided
        self.body_part_model = body_part_model or settings.BODY_PART_MODEL
        self.body_part_confidence = body_part_confidence
        self.body_part_thresholds = body_part_thresholds  # Per-category thresholds for adaptive detection
        self.visual_model = visual_model or settings.VISUAL_MODEL
        self.search_feature_mode_str = search_feature_mode or settings.SEARCH_FEATURE_MODE

        # Convert string to enum
        if self.search_feature_mode_str == 'geometric':
            self.search_feature_mode = SearchFeatureMode.GEOMETRIC_ONLY
        elif self.search_feature_mode_str == 'fused':
            self.search_feature_mode = SearchFeatureMode.FUSED_MULTIMODAL
        else:
            self.search_feature_mode = SearchFeatureMode.GEOMETRIC_ONLY

        # Read use_two_stage from environment variable if not provided
        # Swift sets USE_TWO_STAGE_DETECTION env var from user settings
        if use_two_stage is None:
            use_two_stage = os.environ.get('USE_TWO_STAGE_DETECTION', 'false').lower() == 'true'

        print(f"[BRIDGE INIT DEBUG] DB Profile: {settings.DB_PROFILE}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Database URL: {db_url}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Index directory: {settings.get_index_dir()}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Body Part Model: {self.body_part_model}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Visual Model: {self.visual_model}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Search Feature Mode: {self.search_feature_mode.value}", file=sys.stderr, flush=True)
        print(f"[BRIDGE INIT DEBUG] Use Two-Stage Detection: {use_two_stage}", file=sys.stderr, flush=True)

        # NOTE: Threading must be configured via environment variables (set by Swift)
        # before PyTorch is imported. torch.set_num_*_threads() cannot be called
        # after parallel work has started.

        # Skip model loading if only search is needed (saves 30-60s initialization)
        if skip_models:
            logger.info("Skipping pose detection model loading (search-only mode)")
            self.detector = None
            self.pose_detector = None
        else:
            # Determine which models to use
            # Handle backward compatibility with use_ensemble
            if pose_models is None:
                if use_ensemble:
                    pose_models = ['rtmw-l', 'rtmw-x']
                else:
                    pose_models = 'rtmw-l'  # Default single model

            # Normalize to list
            if isinstance(pose_models, str):
                model_list = [pose_models]
                use_ensemble_mode = False
            else:
                model_list = pose_models
                use_ensemble_mode = len(model_list) > 1

            # Validate models
            for model_name in model_list:
                if model_name not in self.AVAILABLE_MODELS:
                    available = ', '.join(self.AVAILABLE_MODELS.keys())
                    raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

            # Build model specifications
            model_specs = []
            for model_name in model_list:
                model_info = self.AVAILABLE_MODELS[model_name]
                config_path = settings.PROJECT_ROOT / "data" / "models" / model_info['config']
                checkpoint_path = settings.PROJECT_ROOT / "data" / "models" / model_info['checkpoint']

                if not config_path.exists():
                    raise FileNotFoundError(f"Model config not found: {config_path}")
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

                model_specs.append({
                    'name': model_name,
                    'config': str(config_path),
                    'checkpoint': str(checkpoint_path),
                    'weight': 1.0,
                    'info': model_info
                })

            # Suppress mmengine logging during model loading
            import logging as py_logging
            mmengine_logger = py_logging.getLogger('mmengine')
            old_level = mmengine_logger.level
            mmengine_logger.setLevel(py_logging.ERROR)

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            # Use context managers to guarantee file handle cleanup
            with open(os.devnull, 'w') as devnull_out, \
                 open(os.devnull, 'w') as devnull_err:

                try:
                    # Redirect stdout/stderr to suppress mmpose output
                    sys.stdout = devnull_out
                    sys.stderr = devnull_err

                    # Initialize detector based on mode
                    if use_ensemble_mode:
                        # Ensemble mode: Multiple models with fusion
                        ensemble_config = EnsembleConfig(
                            models=[{
                                "config": spec['config'],
                                "checkpoint": spec['checkpoint'],
                                "weight": spec['weight'],
                            } for spec in model_specs],
                            fusion_method=fusion_method,
                        )
                        self.detector = EnsembleDetector(ensemble_config, device=device)
                        model_names = ' + '.join([spec['name'].upper() for spec in model_specs])
                        logger.info(f"Ensemble detection enabled: {model_names} ({len(model_specs)} models, {fusion_method})")

                    else:
                        # Single model mode
                        spec = model_specs[0]
                        self.pose_detector = RTMWCocktail14Detector(
                            config_file=spec['config'],
                            checkpoint_file=spec['checkpoint'],
                            device=device
                        )

                        if use_two_stage:
                            # Two-stage mode: YOLO + pose estimation
                            if TWO_STAGE_AVAILABLE:
                                self.detector = TwoStageDetector(
                                    pose_detector=self.pose_detector,
                                    person_model="yolov8x.pt",
                                    min_person_conf=0.05,
                                    crop_padding=0.1,
                                )
                                logger.info(f"Two-stage detection enabled: {spec['name'].upper()}")
                            else:
                                # Fall back to single-stage if ultralytics not available
                                logger.warning(
                                    "Two-stage detection requested but ultralytics not installed. "
                                    "Falling back to single-stage. Install with: pip install ultralytics"
                                )
                                self.detector = self.pose_detector
                                logger.info(f"Single-stage detection (fallback): {spec['name'].upper()} ({spec['info']['size_mb']}MB)")
                        else:
                            # Single-stage mode (default)
                            self.detector = self.pose_detector
                            logger.info(f"Single-stage detection: {spec['name'].upper()} ({spec['info']['size_mb']}MB)")

                finally:
                    # Restore stdout/stderr (file handles auto-closed by context manager)
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    mmengine_logger.setLevel(old_level)

        self.feature_extractor = GeometricFeatureExtractor()

        # Initialize visual extractor only if enabled
        if self.visual_model != 'disabled':
            self.visual_extractor = VisualFeatureExtractor(device=device)  # Lazy-loads model on first use
        else:
            self.visual_extractor = None

        self.fusion_engine = MultiModalFusion(fusion_method='concatenate')

        # Ensure PostgreSQL is running before attempting database operations
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', 5432))
                if result != 0:
                    raise RuntimeError(
                        "PostgreSQL is not running on localhost:5432.\n"
                        "Please start PostgreSQL: brew services start postgresql@16"
                    )
        except socket.error as e:
            raise RuntimeError(
                f"Cannot connect to PostgreSQL: {e}\n"
                "Please start PostgreSQL: brew services start postgresql@16"
            )

        self.storage_manager = StorageManager(db_url)

        # Create database tables if they don't exist
        from src.storage.models import Base
        Base.metadata.create_all(self.storage_manager.engine)

        # Initialize similarity engine with configured feature mode
        # Pass database_profile to ensure correct index path (data/indices/irl/)
        self.similarity_engine = SimilarityEngine(
            self.storage_manager,
            database_profile=settings.DB_PROFILE,
            feature_mode=self.search_feature_mode
        )

        logger.info("PostureKit bridge initialized with existing model files")


    def detect_pose(self, image_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect pose in image.

        NOTE: This method returns only the first detected person for backward compatibility
        with Swift code that expects a single pose result. For multi-person detection,
        use detect_multi_person_poses() instead.

        Args:
            image_array: Image as numpy array (H, W, 3) in RGB format

        Returns:
            Dictionary with pose data or None if no pose detected
        """
        try:
            # Use detect_multi_person() for ensemble detectors to get all people
            # (then return first one for backward compatibility)
            if hasattr(self.detector, 'detect_multi_person'):
                logger.debug("Using ensemble multi-person detection (returning first person)")
                poses = self.detector.detect_multi_person(image_array)
            else:
                logger.debug("Using standard detection")
                poses = self.detector.detect(image_array)

            if not poses:
                return None

            # Return first pose (highest confidence) - for backward compatibility
            pose = poses[0]

            return {
                'keypoints': pose.keypoints.tolist(),
                'visibility': pose.visibility.tolist(),
                'bbox': pose.bbox.tolist(),
                'confidence': float(pose.overall_confidence),
                'person_id': int(pose.person_id)
            }

        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            return None

    def detect_pose_from_file(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect pose in image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with pose data or None if no pose detected
        """
        try:
            import cv2

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect pose
            return self.detect_pose(image_rgb)

        except Exception as e:
            logger.error(f"Pose detection from file failed: {e}")
            return None

    def detect_multi_person_poses(self, image_array: np.ndarray, refine_bbox: bool = True) -> List[Dict[str, Any]]:
        """
        Detect multiple people and their poses in image.

        Args:
            image_array: Image as numpy array (H, W, 3) in RGB format
            refine_bbox: Whether to refine bounding boxes using keypoints (default: True)

        Returns:
            List of dictionaries with pose data, one per detected person
            Each dictionary contains: keypoints, visibility, bbox, confidence, person_id
        """
        try:
            # Use detect_multi_person() for ensemble detectors to properly handle multiple people
            # This fixes the bug where ensemble mode only returned the first person
            if hasattr(self.detector, 'detect_multi_person'):
                logger.debug("Using ensemble multi-person detection")
                poses = self.detector.detect_multi_person(image_array)
            else:
                logger.debug("Using standard detection")
                poses = self.detector.detect(image_array)

            if not poses:
                return []

            # Convert all poses to dictionaries
            results = []
            for pose in poses:
                bbox = pose.bbox

                # Refine bbox using keypoints if requested
                if refine_bbox:
                    bbox = self._refine_bbox_from_keypoints(pose)

                results.append({
                    'keypoints': pose.keypoints.tolist(),
                    'visibility': pose.visibility.tolist(),
                    'bbox': bbox.tolist(),
                    'confidence': float(pose.overall_confidence),
                    'person_id': int(pose.person_id)
                })

            logger.info(f"Detected {len(results)} person(s) in image")
            return results

        except Exception as e:
            logger.error(f"Multi-person pose detection failed: {e}")
            return []

    def detect_multi_person_poses_from_file(self, image_path: str, refine_bbox: bool = True) -> List[Dict[str, Any]]:
        """
        Detect multiple people and their poses in image file.

        Args:
            image_path: Path to image file
            refine_bbox: Whether to refine bounding boxes using keypoints (default: True)

        Returns:
            List of dictionaries with pose data, one per detected person
        """
        try:
            import cv2

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect all poses
            return self.detect_multi_person_poses(image_rgb, refine_bbox=refine_bbox)

        except Exception as e:
            logger.error(f"Multi-person pose detection from file failed: {e}")
            return []

    def _refine_bbox_from_keypoints(self, pose_result: PoseResult) -> np.ndarray:
        """
        Refine bounding box using visible keypoints for tighter fit.

        Strategy:
        1. Start with keypoints that have high confidence (> 0.5)
        2. Compute tight bbox from visible keypoints
        3. Add confidence-weighted padding for occluded areas
        4. Ensure minimum bbox size to prevent over-tight crops

        Args:
            pose_result: PoseResult object

        Returns:
            Refined bbox as [x, y, w, h] numpy array
        """
        keypoints = pose_result.keypoints  # (133, 3) [x, y, confidence]
        visibility = pose_result.visibility  # (133,) [0, 1, 2]

        # Get visible keypoints (visibility == 2, confidence > 0.5)
        visible_mask = (visibility == 2) & (keypoints[:, 2] > 0.5)
        visible_kps = keypoints[visible_mask, :2]

        if len(visible_kps) < 8:
            # Not enough visible keypoints, use medium-confidence keypoints
            medium_mask = keypoints[:, 2] > 0.3
            visible_kps = keypoints[medium_mask, :2]

        if len(visible_kps) < 3:
            # Still not enough, fall back to original bbox
            return pose_result.bbox

        # Compute tight bbox from visible keypoints
        x_min, y_min = visible_kps.min(axis=0)
        x_max, y_max = visible_kps.max(axis=0)

        # Calculate bbox dimensions
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Add adaptive padding based on occlusion level
        occlusion_ratio = (visibility < 2).sum() / len(visibility)
        # More occlusion → more padding (5-15%)
        padding_ratio = 0.05 + (occlusion_ratio * 0.10)

        pad_x = bbox_w * padding_ratio
        pad_y = bbox_h * padding_ratio

        # Apply padding
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = x_max + pad_x
        y_max = y_max + pad_y

        # Ensure minimum bbox size (prevent over-tight crops on distant people)
        MIN_BBOX_SIZE = 50  # pixels
        if (x_max - x_min) < MIN_BBOX_SIZE or (y_max - y_min) < MIN_BBOX_SIZE:
            # Fall back to original bbox
            return pose_result.bbox

        # Create refined bbox
        refined_bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)

        logger.debug(
            f"Bbox refinement: original={pose_result.bbox}, "
            f"refined={refined_bbox}, occlusion={occlusion_ratio:.2f}"
        )

        return refined_bbox
    
    def extract_features(self, pose_data: Dict[str, Any], image_array: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Extract and fuse geometric and visual features from pose data and image.

        Args:
            pose_data: Dictionary with pose data from detect_pose
            image_array: Image as numpy array (H, W, 3) in RGB format (optional for backwards compatibility)

        Returns:
            Dictionary with feature data including fused 628-dim vector, or None if extraction failed
        """
        try:
            # Reconstruct PoseResult object
            pose_result = PoseResult(
                keypoints=np.array(pose_data['keypoints']),
                visibility=np.array(pose_data['visibility']),
                bbox=np.array(pose_data['bbox']),
                overall_confidence=pose_data['confidence'],
                person_id=pose_data['person_id']
            )

            # Extract geometric features
            geometric_features = self.feature_extractor.extract(pose_result)

            # Extract visual features if image provided and enabled
            visual_features = None
            fused_vector = None
            if image_array is not None and self.visual_extractor is not None:
                try:
                    visual_features = self.visual_extractor.extract(image_array)
                    # Fuse geometric + visual features
                    fused_features = self.fusion_engine.fuse(geometric_features, visual_features)
                    fused_vector = fused_features.fused_vector.tolist()
                except Exception as e:
                    logger.warning(f"Visual feature extraction failed, using geometric only: {e}")

            return {
                'feature_vector': geometric_features.feature_vector.tolist(),  # 52-dim geometric (backwards compat)
                'feature_confidence': geometric_features.feature_confidence.tolist(),  # 52-dim confidence scores
                'fused_vector': fused_vector,  # 628-dim fused (for search)
                'joint_angles': geometric_features.joint_angles,
                'limb_ratios': geometric_features.limb_ratios,
                'body_angles': geometric_features.body_angles,
                'symmetry_scores': geometric_features.symmetry_scores,
                'occlusion_pattern': geometric_features.occlusion_pattern.tolist()
            }

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def search_similar(self, feature_vector: List[float], k: int = 20,
                      min_confidence: float = 0.5,
                      query_confidence: Optional[List[float]] = None,
                      min_feature_confidence: float = 0.35,
                      min_valid_overlap: int = 12,
                      required_regions: Optional[List[str]] = None,
                      min_region_confidence: float = 0.3,
                      min_similarity: float = 0.0,
                      deduplicate_images: bool = True,
                      query_keypoints: Optional[List[float]] = None,
                      query_bbox: Optional[List[float]] = None,
                      enable_flip_search: bool = False) -> List[Dict[str, Any]]:
        """
        Search for similar poses with optional body part filtering and confidence-aware matching.

        Args:
            feature_vector: 52-dimensional feature vector
            k: Number of results to return
            min_confidence: Minimum detection confidence
            query_confidence: Optional 52-dim confidence scores for query features
            min_feature_confidence: Minimum confidence for feature dimensions (default: 0.5)
            min_valid_overlap: Minimum valid dimensions required for matching (default: 20)
            required_regions: Optional list of required body parts (e.g., ['face', 'feet', 'torso'])
            min_region_confidence: Minimum confidence for body part detection (default: 0.3)
            deduplicate_images: If True, return only one pose per image file (default: True)
            query_keypoints: Optional 399-dim query keypoints for OKS/flip search (133 × 3)
            query_bbox: Optional query bbox [x, y, w, h] for OKS search
            enable_flip_search: If True, also search for horizontally flipped matches

        Returns:
            List of similar pose results with visible_regions and valid_dimensions fields
            If flip search enabled, includes 'is_flipped_match' field
            If OKS enabled, includes 'oks_similarity' field
        """
        try:
            import sys
            # Debug: Check index status
            print(f'[SEARCH DEBUG] Index path: {self.similarity_engine.index_path}', file=sys.stderr, flush=True)
            print(f'[SEARCH DEBUG] Index exists: {self.similarity_engine.index_path.exists()}', file=sys.stderr, flush=True)
            if self.similarity_engine.index is None:
                print(f'[SEARCH DEBUG] Index is None, checking if we need to build...', file=sys.stderr, flush=True)
                # Try to load existing index first
                if not self.similarity_engine.load_index():
                    print('[SEARCH DEBUG] No existing index found, need to build from scratch', file=sys.stderr, flush=True)
                    print('SEARCH: Index not loaded, building now (may take 60-90s for large datasets)...', file=sys.stderr, flush=True)
                    self.similarity_engine.build_index()
                    print('SEARCH: Index build complete, proceeding with search...', file=sys.stderr, flush=True)
                else:
                    print(f'[SEARCH DEBUG] Loaded existing index with {self.similarity_engine.index.ntotal} poses', file=sys.stderr, flush=True)
            else:
                print(f'[SEARCH DEBUG] Index already loaded with {self.similarity_engine.index.ntotal} poses', file=sys.stderr, flush=True)

            # Convert to numpy arrays
            query_vector = np.array(feature_vector, dtype=np.float32)
            query_conf_array = np.array(query_confidence, dtype=np.float32) if query_confidence else None
            query_kp_array = np.array(query_keypoints, dtype=np.float32).reshape(133, 3) if query_keypoints else None
            query_bbox_array = np.array(query_bbox, dtype=np.float32) if query_bbox else None

            # Prepare search kwargs
            search_kwargs = {
                'k': k,
                'min_confidence': min_confidence,
                'min_feature_confidence': min_feature_confidence,
                'min_valid_overlap': min_valid_overlap,
                'deduplicate_images': deduplicate_images,
                'required_regions': required_regions,
                'min_region_confidence': min_region_confidence,
                'min_similarity': min_similarity,
                'query_keypoints': query_kp_array,
                'query_bbox': query_bbox_array
            }

            # Choose search method based on flip search setting
            if enable_flip_search and query_kp_array is not None:
                # Use flip search (searches both normal and mirrored orientations)
                results = self.similarity_engine.search_with_flip(
                    feature_vector=query_vector,
                    keypoints=query_kp_array,
                    query_confidence=query_conf_array,
                    **search_kwargs
                )
            else:
                # Standard search (normal orientation only)
                results = self.similarity_engine.search_by_feature(
                    feature_vector=query_vector,
                    query_confidence=query_conf_array,
                    **search_kwargs
                )

            # Convert to Swift-friendly format.
            #
            # The base64 thumbnail (~17 KB), keypoints (399 floats) and detailed regions are by
            # far the heaviest per-result fields, and the response scales with result count. A low
            # similarity threshold can return tens of thousands of results — at full fidelity that
            # is a ~0.5 GB single-line JSON blob, which overruns the Swift stdin/stdout reader and
            # comes back as ZERO results. Past a cap we omit those fields; Swift then loads each
            # visible cell's thumbnail from disk (it already falls back to that) and simply skips
            # the skeleton overlay / detailed-region breakdown for very large sets.
            import base64
            HEAVY_PAYLOAD_CAP = 1000
            include_heavy = len(results) <= HEAVY_PAYLOAD_CAP
            swift_results = []
            for result in results:
                result_dict = {
                    # Core identifiers
                    'pose_id': result['pose_id'],
                    'image_id': result.get('image_id'),
                    'image_path': result['image_path'],
                    'person_id': result.get('person_id'),
                    'rank': result['rank'],

                    # Similarity scores
                    'similarity_score': result['similarity_score'],
                    'distance': result.get('distance'),  # Raw distance (L2 or OKS)
                    'base_similarity': result.get('base_similarity'),  # Before plausibility boost
                    'detection_confidence': result['detection_confidence'],

                    # Quality metrics (new features from pose search enhancements)
                    'plausibility_score': result.get('plausibility_score'),  # Anatomical validity (0-1)
                    'oks_similarity': result.get('oks_similarity'),  # COCO-standard pose similarity
                    'is_flipped_match': result.get('is_flipped_match', False),  # Mirrored pose flag

                    # Bounding box & dimensions
                    'bbox': result.get('bbox', []),  # FIX: Was missing! [x, y, w, h]
                    'image_width': result.get('image_width'),
                    'image_height': result.get('image_height'),
                    'file_size': result.get('file_size'),

                    # Pose data (lightweight)
                    'visible_regions': result.get('visible_regions', []),

                    # Metadata
                    'is_corrected': result.get('is_corrected', False),
                    'category': result.get('category'),
                    'valid_dimensions': result.get('valid_dimensions')  # Confidence-aware search
                }

                # Heavy fields only for reasonably-sized result sets; otherwise omit to keep the
                # JSON response transferable (Swift loads thumbnails from disk on demand).
                if include_heavy:
                    result_dict['keypoints'] = result.get('keypoints', [])  # 133 × 3 [x, y, confidence]
                    result_dict['visible_regions_detailed'] = result.get('visible_regions_detailed', [])
                    if result.get('thumbnail'):
                        result_dict['thumbnail_base64'] = base64.b64encode(result['thumbnail']).decode('utf-8')

                swift_results.append(result_dict)

            if not include_heavy:
                logger.info(
                    f"Large result set ({len(results)} > {HEAVY_PAYLOAD_CAP}): omitted thumbnails/keypoints/"
                    f"detailed-regions to keep the response transferable; Swift will load thumbnails from disk"
                )

            return swift_results

        except Exception as e:
            import sys
            import traceback
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            print(f'SEARCH ERROR: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return []
    
    def index_directory(self, directory_path: str, recursive: bool = True,
                       min_confidence: float = 0.3, skip_indexed: bool = True,
                       delete_missing: bool = False) -> Dict[str, Any]:
        """
        Index all images in a directory.

        Args:
            directory_path: Path to directory containing images
            recursive: Whether to search subdirectories
            min_confidence: Minimum pose confidence to index
            skip_indexed: Whether to skip already indexed images
            delete_missing: Whether to delete database entries for missing files

        Returns:
            Dictionary with indexing statistics
        """
        try:
            from pathlib import Path
            import cv2

            # Cleanup missing files first if requested
            deleted_images = 0
            deleted_poses = 0
            deleted_from_index = 0
            if delete_missing:
                logger.info("Cleaning up missing files before indexing...")
                from cleanup_missing_files import cleanup_missing_files
                deleted_images, deleted_poses, deleted_from_index = cleanup_missing_files(
                    self.storage_manager, self.similarity_engine
                )
                logger.info(f"Cleanup complete: {deleted_images} images, {deleted_poses} poses, {deleted_from_index} from index")

            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")

            # Find image files (exclude macOS resource forks and hidden files)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if recursive:
                all_files = [f for f in directory.rglob('*') if f.suffix.lower() in image_extensions]
                image_files = [f for f in all_files
                              if not f.name.startswith('._')
                              and not f.name.startswith('.')]
            else:
                all_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
                image_files = [f for f in all_files
                              if not f.name.startswith('._')
                              and not f.name.startswith('.')]

            # Log filtering results
            filtered_count = len(all_files) - len(image_files)
            if filtered_count > 0:
                import sys
                print(f"DEBUG: Filtered out {filtered_count} hidden/resource fork files", file=sys.stderr, flush=True)

            total_images = len(image_files)
            processed_images = 0
            poses_indexed = 0
            failed_images = 0
            skipped_images = 0
            failed_index_additions = 0  # Poses stored but not added to FAISS index

            for image_path in image_files:
                # Compute content hash for deduplication
                try:
                    with open(image_path, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception as e:
                    print(f"Failed to hash {image_path}: {e}", file=sys.stderr, flush=True)
                    content_hash = None

                # Check if already indexed (by path or content hash)
                if skip_indexed:
                    with self.storage_manager.session_scope() as session:
                        # Check by file path first (fastest)
                        existing = session.query(Image).filter(
                            Image.file_path == str(image_path)
                        ).first()

                        # If not found by path, check by content hash (duplicate detection)
                        if not existing and content_hash:
                            existing = session.query(Image).filter(
                                Image.content_hash == content_hash
                            ).first()

                        if existing:
                            skipped_images += 1
                            processed_images += 1

                            # Send progress update
                            progress = {
                                'type': 'progress',
                                'current_file': str(image_path.name),
                                'images_processed': processed_images,
                                'total_images': total_images,
                                'poses_indexed': poses_indexed,
                                'failed_images': failed_images,
                                'skipped_images': skipped_images,
                                'failed_index_additions': failed_index_additions,
                                'progress': processed_images / total_images if total_images > 0 else 0
                            }
                            # Debug: Log progress to stderr so we can see what's being sent
                            import sys
                            print(f"DEBUG PROGRESS: poses={poses_indexed}, failed={failed_images}, processed={processed_images}", file=sys.stderr, flush=True)
                            print(f"PROGRESS:{json.dumps(progress)}", flush=True)
                            continue


                try:
                    # Send progress update - currently processing
                    progress = {
                        'type': 'progress',
                        'current_file': str(image_path.name),
                        'images_processed': processed_images,
                        'total_images': total_images,
                        'poses_indexed': poses_indexed,
                        'failed_images': failed_images,
                        'skipped_images': skipped_images,
                        'failed_index_additions': failed_index_additions,
                        'progress': processed_images / total_images if total_images > 0 else 0
                    }
                    # Debug: Log progress to stderr so we can see what's being sent
                    import sys
                    print(f"DEBUG PROGRESS: poses={poses_indexed}, failed_imgs={failed_images}, failed_idx={failed_index_additions}, processed={processed_images}", file=sys.stderr, flush=True)
                    print(f"PROGRESS:{json.dumps(progress)}", flush=True)

                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"ERROR [Image Load Failed]: {image_path.name}", file=sys.stderr, flush=True)
                        failed_images += 1
                        processed_images += 1
                        continue

                    # Handle different image formats
                    if len(image.shape) == 2:
                        # Grayscale image (H, W) - convert to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif len(image.shape) == 3:
                        if image.shape[2] == 1:
                            # Grayscale with channel dimension (H, W, 1)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        elif image.shape[2] == 3:
                            # Standard BGR image
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        elif image.shape[2] == 4:
                            # RGBA image - convert to RGB
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                        else:
                            # Unsupported channel count
                            print(f"ERROR [Unsupported Format]: {image_path.name} shape={image.shape}", file=sys.stderr, flush=True)
                            failed_images += 1
                            processed_images += 1
                            continue
                    else:
                        # Unexpected image shape
                        print(f"ERROR [Unexpected Shape]: {image_path.name} shape={image.shape}", file=sys.stderr, flush=True)
                        failed_images += 1
                        processed_images += 1
                        continue

                    # Detect poses
                    try:
                        # Use detect_multi_person() for ensemble detectors to properly handle multiple people
                        if hasattr(self.detector, 'detect_multi_person'):
                            poses = self.detector.detect_multi_person(image_rgb)
                        else:
                            poses = self.detector.detect(image_rgb)
                        logger.debug(f"Detected {len(poses)} pose(s) in {image_path.name}")
                    except Exception as e:
                        import sys
                        logger.error(f"Pose detection failed for {image_path.name}: {e}", exc_info=True)
                        print(f"ERROR [Pose Detection]: {image_path.name}: {e}", file=sys.stderr, flush=True)
                        failed_images += 1
                        processed_images += 1
                        continue

                    if not poses:
                        logger.debug(f"No poses detected in {image_path.name}")
                        processed_images += 1
                        continue

                    # Filter poses by confidence BEFORE processing
                    valid_poses = [p for p in poses if p.overall_confidence >= min_confidence]
                    if not valid_poses:
                        logger.debug(f"No poses above confidence threshold ({min_confidence}) in {image_path.name}")
                        processed_images += 1
                        continue

                    logger.info(f"Processing {len(valid_poses)} person(s) in {image_path.name}")

                    # Generate thumbnail ONCE per image (not per person)
                    thumbnail_bytes = None
                    try:
                        from src.utils.thumbnail_generator import ThumbnailGenerator
                        generator = ThumbnailGenerator(size=200, quality=80)
                        thumbnail_bytes = generator.generate_from_array(image, format_bgr=True)
                        if thumbnail_bytes:
                            logger.debug(f"Generated thumbnail for {image_path.name}: {len(thumbnail_bytes)} bytes")
                        else:
                            logger.warning(f"Thumbnail generation returned None for {image_path.name}")
                    except Exception as e:
                        logger.error(f"Thumbnail generation failed for {image_path.name}: {e}", exc_info=True)
                        print(f"ERROR [Thumbnail Generation]: {image_path.name}: {e}", file=sys.stderr, flush=True)

                    # Create image metadata ONCE
                    channels = image.shape[2] if len(image.shape) == 3 else 1
                    image_metadata = ImageMetadata(
                        file_path=image_path,
                        original_width=image.shape[1],
                        original_height=image.shape[0],
                        file_size_bytes=image_path.stat().st_size,
                        channels=channels,
                        dtype=str(image.dtype),
                        content_hash=content_hash
                    )

                    # Process each person
                    for pose in valid_poses:
                        person_id = pose.person_id
                        logger.debug(f"Processing person {person_id} in {image_path.name}")

                        # Extract geometric features
                        try:
                            features = self.feature_extractor.extract(pose)
                            logger.debug(f"Extracted geometric features for person {person_id}")
                        except Exception as e:
                            logger.error(f"Geometric feature extraction failed for person {person_id}: {e}", exc_info=True)
                            print(f"ERROR [Geometric Features]: {image_path.name} person {person_id}: {e}", file=sys.stderr, flush=True)
                            continue  # Skip this person

                        # Extract visual features from person's bbox crop (ACCURACY IMPROVEMENT)
                        visual_features = None
                        if self.visual_extractor is not None:
                            try:
                                # Crop to person's bbox for better visual features
                                x, y, w, h = [int(v) for v in pose.bbox]
                                # Ensure bbox is within image bounds
                                x = max(0, min(x, image_rgb.shape[1] - 1))
                                y = max(0, min(y, image_rgb.shape[0] - 1))
                                w = min(w, image_rgb.shape[1] - x)
                                h = min(h, image_rgb.shape[0] - y)

                                if w > 10 and h > 10:  # Ensure valid crop size
                                    person_crop = image_rgb[y:y+h, x:x+w]
                                    visual_features = self.visual_extractor.extract(person_crop)
                                    logger.debug(f"Extracted visual features for person {person_id} from bbox crop")
                                else:
                                    logger.warning(f"Person {person_id} bbox too small for visual feature extraction")
                            except Exception as e:
                                logger.error(f"Visual feature extraction failed for person {person_id}: {e}", exc_info=True)
                                print(f"ERROR [Visual Features]: {image_path.name} person {person_id}: {e}", file=sys.stderr, flush=True)
                                visual_features = None
                        else:
                            logger.debug(f"Visual features disabled")

                        # Fuse geometric and visual features
                        fused_features = None
                        if visual_features is not None:
                            try:
                                fused_features = self.fusion_engine.fuse(features, visual_features)
                                logger.debug(f"Fused features for person {person_id}")
                            except Exception as e:
                                logger.error(f"Feature fusion failed for person {person_id}: {e}", exc_info=True)
                                print(f"ERROR [Feature Fusion]: {image_path.name} person {person_id}: {e}", file=sys.stderr, flush=True)
                                fused_features = None

                        # Detect body parts in person's bbox crop (ACCURACY IMPROVEMENT)
                        # OPTIMIZED: Lower bbox threshold (30x30), dual-pass detection, adaptive thresholds
                        body_part_detections = []
                        if self.body_part_model != 'disabled':
                            try:
                                from src.core.body_part_detector import BodyPartDetector
                                if not hasattr(self, '_body_detector'):
                                    self._body_detector = BodyPartDetector(
                                        model_size=self.body_part_model,
                                        min_confidence=self.body_part_confidence,
                                        category_thresholds=self.body_part_thresholds
                                    )
                                    logger.info(f"Initialized BodyPartDetector: model={self.body_part_model}, adaptive_thresholds={bool(self.body_part_thresholds)}")

                                # Crop to person's bbox
                                x, y, w, h = [int(v) for v in pose.bbox]
                                x = max(0, min(x, image_rgb.shape[1] - 1))
                                y = max(0, min(y, image_rgb.shape[0] - 1))
                                w = min(w, image_rgb.shape[1] - x)
                                h = min(h, image_rgb.shape[0] - y)

                                # OPTIMIZATION 1: Lowered bbox threshold from 50x50 to 30x30
                                if w > 30 and h > 30:
                                    # Pass 1: Detect on person crop
                                    person_crop = image_rgb[y:y+h, x:x+w]
                                    body_parts_crop = self._body_detector.detect(
                                        person_crop,
                                        min_confidence=self.body_part_confidence,
                                        use_adaptive=True
                                    )

                                    # Adjust bbox coordinates back to original image space
                                    for part in body_parts_crop:
                                        part.bbox[0] += x  # x1
                                        part.bbox[1] += y  # y1
                                        part.bbox[2] += x  # x2
                                        part.bbox[3] += y  # y2

                                    body_part_detections.extend(body_parts_crop)

                                    # OPTIMIZATION 2: Dual-pass detection if few parts found
                                    # Run full-image detection to catch faces/distant parts
                                    if len(body_part_detections) < 2:
                                        body_parts_full = self._body_detector.detect(
                                            image_rgb,
                                            min_confidence=self.body_part_confidence * 0.8,  # Slightly lower threshold
                                            use_adaptive=True
                                        )

                                        # Filter to parts within person's bbox region (with 50px margin)
                                        for part in body_parts_full:
                                            part_center_x = (part.bbox[0] + part.bbox[2]) / 2
                                            part_center_y = (part.bbox[1] + part.bbox[3]) / 2

                                            if (x - 50 <= part_center_x <= x + w + 50 and
                                                y - 50 <= part_center_y <= y + h + 50):
                                                # Check if not already detected (avoid duplicates)
                                                is_duplicate = False
                                                for existing in body_part_detections:
                                                    # Simple overlap check
                                                    overlap_x = min(part.bbox[2], existing.bbox[2]) - max(part.bbox[0], existing.bbox[0])
                                                    overlap_y = min(part.bbox[3], existing.bbox[3]) - max(part.bbox[1], existing.bbox[1])
                                                    if overlap_x > 0 and overlap_y > 0:
                                                        overlap_area = overlap_x * overlap_y
                                                        part_area = (part.bbox[2] - part.bbox[0]) * (part.bbox[3] - part.bbox[1])
                                                        if overlap_area / part_area > 0.5:  # 50% overlap = duplicate
                                                            is_duplicate = True
                                                            break

                                                if not is_duplicate:
                                                    body_part_detections.append(part)

                                    logger.debug(f"Detected {len(body_part_detections)} body parts for person {person_id} (dual-pass, adaptive thresholds)")
                                else:
                                    logger.warning(f"Person {person_id} bbox too small for body part detection ({w}x{h} < 30x30)")
                            except ImportError as e:
                                logger.warning(f"NudeNet not available: {e}")
                            except Exception as e:
                                logger.error(f"Body part detection failed for person {person_id}: {e}", exc_info=True)
                                print(f"WARNING [Body Part Detection]: {image_path.name} person {person_id}: {e}", file=sys.stderr, flush=True)
                        else:
                            logger.debug(f"Body part detection disabled")

                        # Store in database
                        try:
                            pose_id = self.storage_manager.store_detection(
                                image_path=image_path,
                                image_metadata=image_metadata,
                                pose_result=pose,
                                features=features,
                                visual_features=visual_features,
                                fused_features=fused_features,
                                thumbnail_bytes=thumbnail_bytes,
                                body_part_detections=body_part_detections
                            )
                            logger.debug(f"Stored person {person_id} (pose {pose_id}) in database")
                        except Exception as e:
                            logger.error(f"Database storage failed for person {person_id}: {e}", exc_info=True)
                            print(f"ERROR [Database Storage]: {image_path.name} person {person_id}: {e}", file=sys.stderr, flush=True)
                            continue  # Skip this person

                        # Add to FAISS index
                        try:
                            index_added = self.similarity_engine.add_pose(pose_id)
                        except Exception as e:
                            logger.error(f"FAISS index addition failed for pose {pose_id}: {e}", exc_info=True)
                            print(f"ERROR [FAISS Index]: pose {pose_id}: {e}", file=sys.stderr, flush=True)
                            index_added = False

                        if index_added:
                            poses_indexed += 1
                            logger.debug(f"Added person {person_id} to FAISS index")
                        else:
                            failed_index_additions += 1
                            print(f"WARNING [Index Add Failed]: person {person_id} - likely missing FusedFeatures", file=sys.stderr, flush=True)
                            logger.warning(f"Person {person_id} stored but failed to add to FAISS index")

                    processed_images += 1

                    # Send progress update after processing
                    progress = {
                        'type': 'progress',
                        'current_file': str(image_path.name),
                        'images_processed': processed_images,
                        'total_images': total_images,
                        'poses_indexed': poses_indexed,
                        'failed_images': failed_images,
                        'skipped_images': skipped_images,
                        'failed_index_additions': failed_index_additions,
                        'progress': processed_images / total_images if total_images > 0 else 0
                    }
                    # Debug: Log progress to stderr so we can see what's being sent
                    import sys
                    print(f"DEBUG PROGRESS: poses={poses_indexed}, failed_imgs={failed_images}, failed_idx={failed_index_additions}, processed={processed_images}", file=sys.stderr, flush=True)
                    print(f"PROGRESS:{json.dumps(progress)}", flush=True)

                except Exception as e:
                    import sys
                    import traceback
                    error_msg = f"Unexpected error processing {image_path}: {type(e).__name__}: {e}"
                    print(f"ERROR [Unexpected]: {error_msg}", file=sys.stderr, flush=True)
                    logger.error(error_msg, exc_info=True)
                    # Print stack trace to stderr for debugging
                    traceback.print_exc(file=sys.stderr)
                    failed_images += 1
                    processed_images += 1
                    continue
            
            # Build final index
            self.similarity_engine.build_index(force_rebuild=True)

            return {
                'total_images': total_images,
                'processed_images': processed_images,
                'poses_indexed': poses_indexed,
                'failed_images': failed_images,
                'skipped_images': skipped_images,
                'failed_index_additions': failed_index_additions,
                'deleted_images': deleted_images,
                'deleted_poses': deleted_poses,
                'deleted_from_index': deleted_from_index,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Directory indexing failed: {e}")
            return {
                'total_images': 0,
                'processed_images': 0,
                'poses_indexed': 0,
                'failed_images': 0,
                'skipped_images': 0,
                'failed_index_additions': 0,
                'success': False,
                'error': str(e)
            }

    def browse_by_body_parts(
        self,
        required_regions: Optional[List[str]] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        k: int = 50,
        sort_by: str = 'confidence',
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Browse images by body parts without uploading a query image.

        Allows users to filter database by NudeNet body part detections using
        custom per-category confidence thresholds.

        Args:
            required_regions: List of body part class names to filter by
                             (e.g., ['FEET_EXPOSED', 'FACE_FEMALE'])
                             If None or empty, returns random sample.
            category_thresholds: Per-category confidence thresholds (18 classes).
                                Key: part_name (e.g., 'FEET_EXPOSED')
                                Value: threshold (0.0-1.0)
            k: Number of results to return
            sort_by: Sort method - 'confidence', 'quality', 'diversity', 'random', or 'recent'
            seed: Random seed for reproducible shuffling (when sort_by='random')

        Returns:
            List of result dicts (same format as similarity search results)
            Each dict contains: pose_id, image_path, thumbnail, visible_regions, quality_score, etc.
        """
        from src.storage.models import Image, BodyPart, PoseDetection
        from sqlalchemy import func, and_, or_

        try:
            with self.storage_manager.session_scope() as session:
                # Build base query
                query = session.query(
                    Image,
                    PoseDetection,
                    func.array_agg(BodyPart.part_name).label('parts'),
                    func.avg(BodyPart.confidence).label('avg_conf')
                ).join(
                    PoseDetection, Image.id == PoseDetection.image_id
                ).join(
                    BodyPart, and_(
                        BodyPart.image_id == Image.id,
                        BodyPart.person_index == PoseDetection.person_id
                    )
                )

                # Apply filters if required_regions specified
                if required_regions:
                    if category_thresholds:
                        # Build per-category confidence filters
                        filters = []
                        for region in required_regions:
                            threshold = category_thresholds.get(region, 0.3)
                            filters.append(and_(
                                BodyPart.part_name == region,
                                BodyPart.confidence >= threshold
                            ))

                        if filters:
                            query = query.filter(or_(*filters))
                    else:
                        # No custom thresholds, just filter by part name
                        query = query.filter(BodyPart.part_name.in_(required_regions))

                # Group by image + pose
                query = query.group_by(Image.id, PoseDetection.id)

                # Ensure ALL required parts are present (using HAVING clause)
                if required_regions and len(required_regions) > 1:
                    query = query.having(
                        func.count(func.distinct(BodyPart.part_name)) >= len(required_regions)
                    )

                # Apply sorting
                if sort_by == 'confidence':
                    query = query.order_by(func.avg(BodyPart.confidence).desc())
                elif sort_by == 'quality':
                    # Sort by composite quality score (SQL-level computation)
                    # Formula: 0.5 * body_part_conf + 0.3 * pose_conf + 0.2 * completeness
                    # Completeness: num_parts / max(5, num_required)
                    min_parts = max(5.0, float(len(required_regions)) if required_regions else 5.0)
                    quality_expr = (
                        0.5 * func.avg(BodyPart.confidence) +
                        0.3 * PoseDetection.overall_confidence +
                        0.2 * (func.count(BodyPart.part_name) / min_parts)
                    )
                    query = query.order_by(quality_expr.desc())
                elif sort_by == 'diversity':
                    # Random sampling weighted by quality (for variety)
                    # Uses random() * quality_score for probabilistic sampling
                    quality_weight = (func.avg(BodyPart.confidence) * PoseDetection.overall_confidence)
                    query = query.order_by((func.random() * quality_weight).desc())
                elif sort_by == 'recent':
                    query = query.order_by(Image.created_at.desc())
                elif sort_by == 'random':
                    # Pure random (uniform distribution)
                    query = query.order_by(func.random())

                # Limit results
                results = query.limit(k).all()

                # Format results (match similarity search format for UI consistency)
                formatted = []
                from src.core.body_part_detector import BodyPartDetector
                import base64

                for idx, (image, pose, parts, avg_conf) in enumerate(results):
                    # Get canonical regions from part names
                    canonical_regions = list(set(
                        BodyPartDetector.LABEL_TO_REGION.get(p, 'unknown')
                        for p in parts
                    ))

                    # Compute composite quality score (Python-side for result dict)
                    # Matches SQL-side computation for consistency
                    # Formula: 0.5 * body_part_conf + 0.3 * pose_conf + 0.2 * completeness
                    min_parts = max(5, len(required_regions) if required_regions else 5)
                    completeness = min(1.0, len(parts) / min_parts)
                    quality_score = float(np.clip(
                        0.5 * avg_conf +
                        0.3 * pose.overall_confidence +
                        0.2 * completeness,
                        0.0, 1.0
                    ))

                    formatted.append({
                        'pose_id': str(pose.id),
                        'image_path': image.file_path,
                        'image_id': str(image.id),
                        'detection_confidence': pose.overall_confidence,
                        'thumbnail_base64': base64.b64encode(image.thumbnail).decode('utf-8') if image.thumbnail else None,
                        'visible_regions': canonical_regions,
                        'visible_regions_detailed': [
                            {'part_name': p, 'confidence': float(avg_conf)} for p in parts
                        ],
                        'keypoints': pose.keypoints,
                        'bbox': pose.bbox if pose.bbox else [],  # FIX: Was missing!
                        'rank': idx + 1,
                        'quality_score': quality_score,  # Composite quality (0-1, body parts + pose + completeness)
                        'body_part_confidence': float(avg_conf),  # Average body part confidence
                        'image_width': image.width,
                        'image_height': image.height,
                        'file_size': image.file_size_bytes,
                        'person_id': pose.person_id,
                        'is_corrected': pose.is_corrected,
                        'distance': 0.0  # Not applicable for browse mode
                    })

                logger.info(f"Browse by body parts: found {len(formatted)} results")
                return formatted

        except Exception as e:
            logger.error(f"Browse by body parts failed: {e}", exc_info=True)
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database and index statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            db_stats = self.storage_manager.get_statistics()
            index_stats = self.similarity_engine.get_statistics()
            
            return {
                'database': db_stats,
                'index': index_stats,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'database': {},
                'index': {},
                'success': False,
                'error': str(e)
            }
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build or rebuild the FAISS index.

        Args:
            force_rebuild: Whether to force rebuild even if index exists

        Returns:
            True if successful
        """
        try:
            self.similarity_engine.build_index(force_rebuild=force_rebuild)
            return True
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False

    def backfill_thumbnails(self) -> Dict[str, Any]:
        """
        Generate thumbnails for images that don't have them (OPTIMIZED VERSION).

        This is a one-time operation to backfill thumbnails for already-indexed images.
        New images will get thumbnails automatically during indexing.

        Optimizations:
        - Parallel processing with 8 workers (ProcessPoolExecutor)
        - Progress updates every 50 images (not every 1)
        - Direct cv2.imencode() for faster encoding
        - Keyset pagination (ID-based) for constant-time queries
        - Bulk database updates (bulk_update_mappings)
        - Batch size increased to 500

        Returns:
            Dict with keys:
                - images_processed: Total images checked
                - images_updated: Images that got new thumbnails
                - images_failed: Images that failed thumbnail generation
                - images_skipped: Images that already had thumbnails
        """
        from sqlalchemy import select
        from src.storage.models import Image
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import sys
        import uuid

        logger.info("Starting OPTIMIZED thumbnail backfill operation...")
        print("THUMBNAIL_BACKFILL_INIT", file=sys.stderr, flush=True)

        images_processed = 0
        images_updated = 0
        images_failed = 0
        images_skipped = 0

        try:
            # First, get total count
            with self.storage_manager.session_scope() as session:
                from sqlalchemy import func
                total_images = session.query(func.count(Image.id)).filter(Image.thumbnail == None).scalar()
                logger.info(f"Found {total_images} images without thumbnails")
                print(f"THUMBNAIL_COUNT: {total_images}", file=sys.stderr, flush=True)

            if total_images == 0:
                return {
                    'images_processed': 0,
                    'images_updated': 0,
                    'images_failed': 0,
                    'images_skipped': 0
                }

            # OPTIMIZATION: Larger batch size for better throughput
            batch_size = 500
            last_id = None  # For keyset pagination

            # M2 Pro has 8 performance cores, use all of them
            num_workers = 8

            while images_processed < total_images:
                # OPTIMIZATION: Keyset pagination (ID-based) instead of OFFSET
                # This maintains constant query time regardless of position
                with self.storage_manager.session_scope() as session:
                    if last_id is None:
                        # First batch
                        query = (
                            select(Image.id, Image.file_path)
                            .where(Image.thumbnail == None)
                            .order_by(Image.id)
                            .limit(batch_size)
                        )
                    else:
                        # Subsequent batches: use keyset pagination
                        query = (
                            select(Image.id, Image.file_path)
                            .where(Image.thumbnail == None)
                            .where(Image.id > last_id)
                            .order_by(Image.id)
                            .limit(batch_size)
                        )

                    batch_rows = session.execute(query).all()

                    if not batch_rows:
                        break

                    # Prepare work items for parallel processing
                    work_items = [(str(row.id), row.file_path) for row in batch_rows]
                    last_id = batch_rows[-1].id  # Update cursor for next batch

                # OPTIMIZATION: Process batch in parallel with 8 workers
                results = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all work items
                    future_to_item = {
                        executor.submit(_process_thumbnail_worker, item): item
                        for item in work_items
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_item):
                        result = future.result()
                        results.append(result)
                        images_processed += 1

                        # OPTIMIZATION: Print progress every 50 images instead of every 1
                        if images_processed % 50 == 0 or images_processed == total_images:
                            current_file = Path(result['file_path']).name if result.get('file_path') else ""
                            progress = {
                                'images_processed': images_processed,
                                'images_total': total_images,
                                'images_updated': images_updated + sum(1 for r in results if r['success']),
                                'images_failed': images_failed + sum(1 for r in results if not r['success']),
                                'current_file': current_file
                            }
                            print(json.dumps(progress), flush=True)

                # OPTIMIZATION: Bulk database update instead of individual ORM updates
                # This is MUCH faster for large batches
                successful_updates = []
                for result in results:
                    if result['success']:
                        successful_updates.append({
                            'id': uuid.UUID(result['id']),
                            'thumbnail': result['thumbnail_bytes']
                        })
                        images_updated += 1
                    else:
                        images_failed += 1
                        logger.warning(f"Failed to process {result['file_path']}: {result.get('error', 'Unknown error')}")

                # Bulk update in database
                if successful_updates:
                    with self.storage_manager.session_scope() as session:
                        session.bulk_update_mappings(Image, successful_updates)
                        session.commit()
                        logger.info(f"Bulk updated {len(successful_updates)} thumbnails (total: {images_updated}/{total_images})")

            logger.info(f"Thumbnail backfill complete: {images_updated} updated, {images_failed} failed")

            # Final progress update
            progress = {
                'images_processed': images_processed,
                'images_total': total_images,
                'images_updated': images_updated,
                'images_failed': images_failed,
                'current_file': ''
            }
            print(json.dumps(progress), flush=True)

        except Exception as e:
            logger.error(f"Thumbnail backfill failed: {e}", exc_info=True)
            return {
                'images_processed': images_processed,
                'images_updated': images_updated,
                'images_failed': images_failed,
                'images_skipped': images_skipped,
                'error': str(e)
            }

        return {
            'images_processed': images_processed,
            'images_updated': images_updated,
            'images_failed': images_failed,
            'images_skipped': images_skipped
        }

    def get_current_profile(self) -> str:
        """
        Get the current database profile.

        Returns:
            Current profile name ('irl', '2d', or '3d')
        """
        from src.config.settings import settings
        return settings.DB_PROFILE

    def get_profile_stats(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific profile (or current profile).

        Args:
            profile: Profile name ('irl', '2d', '3d'). If None, uses current profile.

        Returns:
            Dictionary with profile statistics:
                - profile: Profile name
                - total_images: Number of indexed images
                - total_poses: Number of detected poses
                - index_size: Number of poses in FAISS index
                - database_name: Database name
                - index_path: Path to FAISS index directory
        """
        from src.config.settings import settings
        import sys

        if profile is None:
            profile = settings.DB_PROFILE

        # Validate profile
        valid_profiles = ['irl', '2d', '3d']
        if profile not in valid_profiles:
            logger.error(f"Invalid profile: {profile}")
            return {
                'profile': profile,
                'error': f"Invalid profile. Must be one of: {', '.join(valid_profiles)}",
                'success': False
            }

        try:
            # Create temporary storage manager for this profile
            profile_db_url = settings.get_database_url(profile)
            temp_storage = StorageManager(connection_string=profile_db_url)

            try:
                # Get database stats
                db_stats = temp_storage.get_statistics()

                # Get index path
                index_dir = settings.get_index_dir(profile)
                index_path = index_dir / "index.faiss"
                index_exists = index_path.exists()

                # If this is the current profile, use loaded index
                if profile == settings.DB_PROFILE:
                    index_size = self.similarity_engine.index.ntotal if self.similarity_engine.index else 0
                else:
                    # For other profiles, check if index file exists
                    if index_exists:
                        # Could load it to get size, but that's expensive
                        # Instead, just mark as existing
                        index_size = -1  # Unknown size
                    else:
                        index_size = 0

                return {
                    'profile': profile,
                    'total_images': db_stats.get('total_images', 0),
                    'total_poses': db_stats.get('total_poses', 0),
                    'index_size': index_size,
                    'database_name': settings.get_database_name(profile),
                    'index_path': str(index_dir),
                    'index_exists': index_exists,
                    'success': True
                }
            finally:
                # Always close temporary storage
                temp_storage.close()

        except Exception as e:
            logger.error(f"Failed to get profile stats for {profile}: {e}", exc_info=True)
            return {
                'profile': profile,
                'error': str(e),
                'success': False
            }

    def switch_profile(self, new_profile: str) -> Dict[str, Any]:
        """
        Switch to a different database profile at runtime.

        This reinitializes the storage manager and similarity engine with the new profile's
        database and FAISS index, without requiring an app restart.

        Args:
            new_profile: Profile name ('irl', '2d', or '3d')

        Returns:
            Dictionary with switch result:
                - success: True if successful
                - old_profile: Previous profile name
                - new_profile: New profile name
                - error: Error message if failed
        """
        from src.config.settings import settings
        import sys
        import os

        # Validate profile
        valid_profiles = ['irl', '2d', '3d']
        if new_profile not in valid_profiles:
            error_msg = f"Invalid profile '{new_profile}'. Must be one of: {', '.join(valid_profiles)}"
            logger.error(error_msg)
            return {
                'success': False,
                'old_profile': settings.DB_PROFILE,
                'new_profile': new_profile,
                'error': error_msg
            }

        old_profile = settings.DB_PROFILE

        # No-op if already on this profile
        if new_profile == old_profile:
            logger.info(f"Already on profile '{new_profile}', no switch needed")
            return {
                'success': True,
                'old_profile': old_profile,
                'new_profile': new_profile,
                'message': 'Already on requested profile'
            }

        try:
            logger.info(f"Switching database profile: {old_profile} → {new_profile}")
            print(f"[PROFILE SWITCH] {old_profile} → {new_profile}", file=sys.stderr, flush=True)

            # Update settings DB_PROFILE (this affects all subsequent calls)
            settings.DB_PROFILE = new_profile

            # Also update environment variable for consistency
            os.environ['DB_PROFILE'] = new_profile

            # Get new database URL and index directory
            new_db_url = settings.get_database_url(new_profile)
            new_index_dir = settings.get_index_dir(new_profile)

            print(f"[PROFILE SWITCH] New DB URL: {new_db_url}", file=sys.stderr, flush=True)
            print(f"[PROFILE SWITCH] New Index Dir: {new_index_dir}", file=sys.stderr, flush=True)

            # Close old resources before reinitializing
            print(f"[PROFILE SWITCH] Closing old similarity engine...", file=sys.stderr, flush=True)
            if hasattr(self, 'similarity_engine') and self.similarity_engine:
                self.similarity_engine.close()

            print(f"[PROFILE SWITCH] Closing old storage manager...", file=sys.stderr, flush=True)
            if hasattr(self, 'storage_manager') and self.storage_manager:
                self.storage_manager.close()

            # Reinitialize storage manager with new profile
            print(f"[PROFILE SWITCH] Creating new storage manager...", file=sys.stderr, flush=True)
            self.storage_manager = StorageManager(connection_string=new_db_url)

            # Create database tables if they don't exist
            from src.storage.models import Base
            Base.metadata.create_all(self.storage_manager.engine)

            # Reinitialize similarity engine with new profile
            print(f"[PROFILE SWITCH] Creating new similarity engine...", file=sys.stderr, flush=True)
            self.similarity_engine = SimilarityEngine(
                self.storage_manager,
                database_profile=new_profile,
                feature_mode=self.search_feature_mode
            )

            logger.info(f"Successfully switched to profile '{new_profile}'")
            print(f"[PROFILE SWITCH] Success!", file=sys.stderr, flush=True)

            return {
                'success': True,
                'old_profile': old_profile,
                'new_profile': new_profile,
                'database_name': settings.get_database_name(new_profile),
                'index_path': str(new_index_dir)
            }

        except Exception as e:
            error_msg = f"Failed to switch profile: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"[PROFILE SWITCH ERROR] {e}", file=sys.stderr, flush=True)

            # Try to restore old profile
            try:
                settings.DB_PROFILE = old_profile
                os.environ['DB_PROFILE'] = old_profile
                logger.warning(f"Restored old profile '{old_profile}' after failed switch")
            except Exception as restore_error:
                logger.error(f"Failed to restore old profile: {restore_error}")

            return {
                'success': False,
                'old_profile': old_profile,
                'new_profile': new_profile,
                'error': error_msg
            }


# Global bridge instance with thread-safe initialization
_bridge_instance = None
_bridge_lock = threading.Lock()

def get_bridge() -> PostureKitBridge:
    """Get or create global bridge instance (thread-safe)."""
    global _bridge_instance
    if _bridge_instance is None:  # Fast path without lock
        with _bridge_lock:
            if _bridge_instance is None:  # Double-check inside lock
                _bridge_instance = PostureKitBridge()
    return _bridge_instance

def detect_pose(image_array: np.ndarray) -> Optional[Dict[str, Any]]:
    """Convenience function for pose detection."""
    return get_bridge().detect_pose(image_array)

def extract_features(pose_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for feature extraction."""
    return get_bridge().extract_features(pose_data)

def search_similar(feature_vector: List[float], k: int = 20,
                  min_confidence: float = 0.5,
                  required_regions: Optional[List[str]] = None,
                  min_region_confidence: float = 0.3) -> List[Dict[str, Any]]:
    """Convenience function for similarity search with optional body part filtering."""
    return get_bridge().search_similar(feature_vector, k, min_confidence, required_regions, min_region_confidence)

def index_directory(directory_path: str, recursive: bool = True,
                   min_confidence: float = 0.3, skip_indexed: bool = True,
                   delete_missing: bool = False) -> Dict[str, Any]:
    """Convenience function for directory indexing."""
    return get_bridge().index_directory(directory_path, recursive, min_confidence, skip_indexed, delete_missing)

def get_statistics() -> Dict[str, Any]:
    """Convenience function for getting statistics."""
    return get_bridge().get_statistics()

def build_index(force_rebuild: bool = False) -> bool:
    """Convenience function for building index."""
    return get_bridge().build_index(force_rebuild)
