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
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

# Suppress FutureWarnings from mmpose/PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# CRITICAL: Import torch patch BEFORE any mmpose/mmengine imports
import core._torch_patch

logger = logging.getLogger(__name__)

# Import with fallbacks for missing dependencies
try:
    from core.pose_detector import RTMWCocktail14Detector, PoseResult
    from core.geometric_feature_extractor import GeometricFeatureExtractor, GeometricFeatures
    from core.visual_feature_extractor import VisualFeatureExtractor
    from core.multimodal_fusion import MultiModalFusion
    from core.two_stage_detector import TwoStageDetector
    from core.ensemble_detector import EnsembleDetector, EnsembleConfig
    from intelligence.similarity_engine import SimilarityEngine
    from storage.storage_manager import StorageManager
    from storage.models import Image, PoseDetection, GeometricFeatures as GeometricFeaturesModel
    from core.image_ingestor import ImageMetadata

    # Apply mmengine patch after imports (now mmengine is in sys.modules)
    core._torch_patch._apply_mmengine_patch()

    FULL_STACK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Full stack not available: {e}")
    FULL_STACK_AVAILABLE = False


class PostureKitBridge:
    """
    Bridge class for Swift integration.
    Provides simplified interface to PostureKit Python modules.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        use_two_stage: bool = False,
        use_ensemble: bool = False,
    ):
        """
        Initialize PostureKit bridge.

        Args:
            db_url: Database URL (defaults to PostgreSQL from settings)
            use_two_stage: Enable two-stage detection (YOLO + pose estimation)
                          Default False. Set True for 5-10% accuracy improvement.
            use_ensemble: Enable ensemble detection (multiple models with fusion)
                         Default False. Set True for 10-15% accuracy improvement.
                         Note: Overrides use_two_stage if both are True.
        """
        # Import settings to get DATABASE_URL
        from config.settings import settings
        import contextlib

        if db_url is None:
            db_url = settings.DATABASE_URL

        # Initialize components with model paths
        # Use settings.PROJECT_ROOT instead of __file__ (which may not be set when running via -c)
        config_path = settings.PROJECT_ROOT / "data" / "models" / "rtmw-l_8xb320-270e_cocktail14-384x288.py"
        checkpoint_path = settings.PROJECT_ROOT / "data" / "models" / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

        # Suppress mmengine logging and redirect stdout to prevent checkpoint messages
        import logging as py_logging
        mmengine_logger = py_logging.getLogger('mmengine')
        old_level = mmengine_logger.level
        mmengine_logger.setLevel(py_logging.ERROR)  # Only show errors

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            # Redirect both stdout and stderr to suppress all mmpose output during init
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            self.pose_detector = RTMWCocktail14Detector(
                config_file=str(config_path),
                checkpoint_file=str(checkpoint_path)
            )
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            mmengine_logger.setLevel(old_level)

        # Initialize detector based on mode
        if use_ensemble:
            # Ensemble mode: Use multiple models with fusion
            rtmw_m_config = settings.PROJECT_ROOT / "data" / "models" / "rtmw-m_8xb1024-270e_cocktail14-256x192.py"
            rtmw_m_checkpoint = settings.PROJECT_ROOT / "data" / "models" / "rtmw-m_simcc-cocktail14_270e-256x192.pth"

            ensemble_config = EnsembleConfig(
                models=[
                    {
                        "config": str(config_path),
                        "checkpoint": str(checkpoint_path),
                        "weight": 1.0,  # RTMW-L (384x288)
                    },
                    {
                        "config": str(rtmw_m_config),
                        "checkpoint": str(rtmw_m_checkpoint),
                        "weight": 1.0,  # RTMW-M (256x192)
                    },
                ],
                fusion_method="confidence_weighted",  # Weight by per-keypoint confidence
            )
            self.detector = EnsembleDetector(ensemble_config)
            logger.info("Ensemble detection enabled (2 models)")

        elif use_two_stage:
            # Two-stage mode: YOLO + pose estimation
            self.detector = TwoStageDetector(
                pose_detector=self.pose_detector,
                person_model="yolov8n.pt",  # Nano model (6MB, fast)
                min_person_conf=0.3,
                crop_padding=0.1,
            )
            logger.info("Two-stage detection enabled")

        else:
            # Single-stage mode (default)
            self.detector = self.pose_detector
            logger.info("Single-stage detection (default)")

        self.feature_extractor = GeometricFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()  # Lazy-loads model on first use
        self.fusion_engine = MultiModalFusion(fusion_method='concatenate')
        self.storage_manager = StorageManager(db_url)

        # Create database tables if they don't exist
        from src.storage.models import Base
        Base.metadata.create_all(self.storage_manager.engine)

        self.similarity_engine = SimilarityEngine(self.storage_manager)

        logger.info("PostureKit bridge initialized with existing model files")
    
    def detect_pose(self, image_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect pose in image.

        Args:
            image_array: Image as numpy array (H, W, 3) in RGB format

        Returns:
            Dictionary with pose data or None if no pose detected
        """
        try:
            poses = self.detector.detect(image_array)

            if not poses:
                return None

            # Return first pose (highest confidence)
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
    
    def extract_features(self, pose_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract geometric features from pose data.
        
        Args:
            pose_data: Dictionary with pose data from detect_pose
            
        Returns:
            Dictionary with feature data or None if extraction failed
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
            
            # Extract features
            features = self.feature_extractor.extract(pose_result)
            
            return {
                'feature_vector': features.feature_vector.tolist(),
                'joint_angles': features.joint_angles,
                'limb_ratios': features.limb_ratios,
                'body_angles': features.body_angles,
                'symmetry_scores': features.symmetry_scores,
                'occlusion_pattern': features.occlusion_pattern.tolist()
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def search_similar(self, feature_vector: List[float], k: int = 20, 
                      min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar poses.
        
        Args:
            feature_vector: 52-dimensional feature vector
            k: Number of results to return
            min_confidence: Minimum detection confidence
            
        Returns:
            List of similar pose results
        """
        try:
            # Ensure index is built
            if not self.similarity_engine.index:
                self.similarity_engine.build_index()
            
            # Convert to numpy array
            query_vector = np.array(feature_vector, dtype=np.float32)
            
            # Search
            results = self.similarity_engine.search_by_feature(
                feature_vector=query_vector,
                k=k,
                min_confidence=min_confidence
            )
            
            # Convert to Swift-friendly format
            swift_results = []
            for result in results:
                swift_results.append({
                    'pose_id': result['pose_id'],
                    'similarity_score': result['similarity_score'],
                    'image_path': result['image_path'],
                    'detection_confidence': result['detection_confidence'],
                    'rank': result['rank']
                })
            
            return swift_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def index_directory(self, directory_path: str, recursive: bool = True,
                       min_confidence: float = 0.3, skip_indexed: bool = True) -> Dict[str, Any]:
        """
        Index all images in a directory.

        Args:
            directory_path: Path to directory containing images
            recursive: Whether to search subdirectories
            min_confidence: Minimum pose confidence to index
            skip_indexed: Whether to skip already indexed images

        Returns:
            Dictionary with indexing statistics
        """
        try:
            from pathlib import Path
            import cv2

            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")

            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if recursive:
                image_files = [f for f in directory.rglob('*')
                              if f.suffix.lower() in image_extensions]
            else:
                image_files = [f for f in directory.iterdir()
                              if f.suffix.lower() in image_extensions]

            total_images = len(image_files)
            processed_images = 0
            poses_indexed = 0
            failed_images = 0
            skipped_images = 0

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
                                'progress': processed_images / total_images if total_images > 0 else 0
                            }
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
                        'progress': processed_images / total_images if total_images > 0 else 0
                    }
                    print(f"PROGRESS:{json.dumps(progress)}", flush=True)

                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
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
                            print(f"Unsupported image format: {image.shape}", file=sys.stderr, flush=True)
                            failed_images += 1
                            processed_images += 1
                            continue
                    else:
                        # Unexpected image shape
                        print(f"Unexpected image shape: {image.shape}", file=sys.stderr, flush=True)
                        failed_images += 1
                        processed_images += 1
                        continue

                    # Detect poses
                    poses = self.detector.detect(image_rgb)

                    for pose in poses:
                        if pose.overall_confidence < min_confidence:
                            continue

                        # Extract geometric features
                        features = self.feature_extractor.extract(pose)

                        # Extract visual features
                        visual_features = self.visual_extractor.extract(image_rgb)

                        # Fuse geometric and visual features
                        fused_features = self.fusion_engine.fuse(features, visual_features)

                        # Create image metadata
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

                        # Store in database
                        pose_id = self.storage_manager.store_detection(
                            image_path=image_path,
                            image_metadata=image_metadata,
                            pose_result=pose,
                            features=features,
                            visual_features=visual_features,
                            fused_features=fused_features
                        )

                        # Add to FAISS index
                        self.similarity_engine.add_pose(pose_id)

                        poses_indexed += 1

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
                        'progress': processed_images / total_images if total_images > 0 else 0
                    }
                    print(f"PROGRESS:{json.dumps(progress)}", flush=True)

                except Exception as e:
                    import sys
                    print(f"Failed to process {image_path}: {e}", file=sys.stderr, flush=True)
                    logger.warning(f"Failed to process {image_path}: {e}")
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
                'success': False,
                'error': str(e)
            }
    
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


# Global bridge instance
_bridge_instance = None

def get_bridge() -> PostureKitBridge:
    """Get or create global bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = PostureKitBridge()
    return _bridge_instance

def detect_pose(image_array: np.ndarray) -> Optional[Dict[str, Any]]:
    """Convenience function for pose detection."""
    return get_bridge().detect_pose(image_array)

def extract_features(pose_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for feature extraction."""
    return get_bridge().extract_features(pose_data)

def search_similar(feature_vector: List[float], k: int = 20, 
                  min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """Convenience function for similarity search."""
    return get_bridge().search_similar(feature_vector, k, min_confidence)

def index_directory(directory_path: str, recursive: bool = True,
                   min_confidence: float = 0.3, skip_indexed: bool = True) -> Dict[str, Any]:
    """Convenience function for directory indexing."""
    return get_bridge().index_directory(directory_path, recursive, min_confidence, skip_indexed)

def get_statistics() -> Dict[str, Any]:
    """Convenience function for getting statistics."""
    return get_bridge().get_statistics()

def build_index(force_rebuild: bool = False) -> bool:
    """Convenience function for building index."""
    return get_bridge().build_index(force_rebuild)
