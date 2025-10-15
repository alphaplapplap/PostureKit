"""
Ensemble Pose Detection for PostureKit.

Combines predictions from multiple pose estimation models using weighted fusion.
Improves robustness and accuracy through model diversity.

Expected accuracy improvement: 10-15%
Performance impact: ~60% slower (2 models running)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import torch

from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.core.person_detector import YOLOPersonDetector, PersonDetection
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _has_mps_module() -> bool:
    """Check if torch.mps module is available (PyTorch 2.1+)."""
    return hasattr(torch, 'mps')


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble detection.

    Attributes:
        models: List of model specifications
               Each spec: {config: str, checkpoint: str, weight: float}
        fusion_method: Method for combining predictions
                      - 'weighted_average': Average keypoints weighted by model weights
                      - 'confidence_weighted': Weight by per-keypoint visibility
        min_agreement: Minimum overlap between predictions to consider valid (not used yet)
    """

    models: List[Dict[str, any]]
    fusion_method: Literal["weighted_average", "confidence_weighted"] = "weighted_average"
    min_agreement: float = 0.5


class EnsembleDetector:
    """
    Ensemble of multiple pose detection models with weighted fusion.

    Uses multiple models with different architectures or training strategies
    to improve robustness and accuracy. Predictions are fused using weighted
    averaging of keypoint coordinates.

    Benefits:
    - Improved robustness to challenging poses
    - Better keypoint localization through averaging
    - Reduced impact of single-model failure cases
    - Higher overall confidence

    Attributes:
        config: EnsembleConfig with model specifications
        detectors: List of loaded detector instances with weights
    """

    def __init__(self, config: EnsembleConfig, device: Optional[str] = None):
        """
        Initialize ensemble with multiple models.

        Args:
            config: EnsembleConfig with model specifications
            device: Device to use ('mps', 'cuda', 'cpu'). Defaults to settings.DEVICE

        Raises:
            ValueError: If fewer than 2 models specified
            FileNotFoundError: If model files don't exist
        """
        if len(config.models) < 2:
            raise ValueError(
                f"Ensemble requires at least 2 models, got {len(config.models)}"
            )

        self.config = config
        self.detectors = []

        # Determine device for all child detectors
        from src.config.settings import settings
        self.device = device or settings.DEVICE

        logger.info(f"Initializing ensemble with {len(config.models)} models (device={self.device})...")

        # Load all models
        for i, model_spec in enumerate(config.models, 1):
            try:
                detector = RTMWCocktail14Detector(
                    config_file=model_spec["config"],
                    checkpoint_file=model_spec["checkpoint"],
                    device=self.device,  # Explicit device propagation
                )
                weight = model_spec.get("weight", 1.0)

                self.detectors.append({"detector": detector, "weight": weight})

                logger.info(
                    f"  Model {i}/{len(config.models)}: "
                    f"config={Path(model_spec['config']).name}, "
                    f"weight={weight:.2f}"
                )
            except Exception as e:
                logger.error(f"Failed to load model {i}: {e}")
                raise

        # Initialize YOLO person detector for multi-person scenarios
        try:
            self.person_detector = YOLOPersonDetector(
                model_name='yolov8n.pt',  # Nano model (6MB, fast)
                confidence_threshold=0.15,  # Low threshold to catch all people
                device=self.device
            )
            logger.info("YOLO person detector initialized for multi-person ensemble")
        except Exception as e:
            logger.warning(f"Failed to initialize person detector: {e}")
            self.person_detector = None

        total_weight = sum(d["weight"] for d in self.detectors)
        logger.info(
            f"Ensemble initialized successfully: "
            f"{len(self.detectors)} models, "
            f"total_weight={total_weight:.2f}, "
            f"fusion={config.fusion_method}"
        )

    def detect(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect poses using ensemble of models.

        Process:
        1. Run detection on each model independently
        2. Collect all predictions
        3. Fuse predictions using configured method
        4. Return single fused result

        Args:
            image: Input image in RGB format (H, W, 3)

        Returns:
            List containing single fused PoseResult
            (or empty list if no poses detected by any model)
        """
        # Get predictions from all models
        all_predictions = []

        for i, det_spec in enumerate(self.detectors, 1):
            try:
                poses = det_spec["detector"].detect(image)

                if poses:
                    # Take first pose (highest confidence)
                    all_predictions.append(
                        {"pose": poses[0], "weight": det_spec["weight"], "model_id": i}
                    )
                    logger.debug(
                        f"Model {i}: detected pose with confidence {poses[0].overall_confidence:.3f}"
                    )
                else:
                    logger.debug(f"Model {i}: no pose detected")

                # Clear GPU cache between model inferences to prevent accumulation
                if i < len(self.detectors):  # Don't clear after last detector
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        if _has_mps_module():
                            try:
                                torch.mps.empty_cache()
                            except Exception:
                                pass
                        # Else: PyTorch < 2.1, skip MPS cache cleanup
            except Exception as e:
                logger.warning(f"Model {i} detection failed: {e}")
                continue

        if not all_predictions:
            logger.info("No poses detected by any model")
            return []

        logger.info(
            f"Ensemble: {len(all_predictions)}/{len(self.detectors)} models detected poses"
        )

        # Fuse predictions
        try:
            if self.config.fusion_method == "weighted_average":
                fused = self._fuse_weighted_average(all_predictions)
            elif self.config.fusion_method == "confidence_weighted":
                fused = self._fuse_confidence_weighted(all_predictions)
            else:
                raise ValueError(
                    f"Unknown fusion method: {self.config.fusion_method}"
                )

            logger.info(
                f"Fused pose: confidence={fused.overall_confidence:.3f}, "
                f"method={self.config.fusion_method}"
            )

            return [fused]
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            # Fallback: return best single prediction
            best = max(all_predictions, key=lambda p: p["pose"].overall_confidence)
            logger.warning(f"Using fallback (best model): confidence={best['pose'].overall_confidence:.3f}")
            return [best["pose"]]

    def detect_multi_person(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect multiple people using ensemble with per-person fusion.

        This method properly handles multi-person scenarios by:
        1. Detecting all people with YOLO first
        2. For each person, running all ensemble models on their crop
        3. Applying ensemble fusion to each person's predictions
        4. Returning all fused results

        This fixes the bug where ensemble mode only returned the first person.

        Args:
            image: Input image in RGB format (H, W, 3)

        Returns:
            List of PoseResult objects, one per detected person with ensemble fusion applied
        """
        # Fallback to single-person detection if YOLO not available
        if self.person_detector is None:
            logger.warning(
                "YOLO person detector not available, falling back to single-person detection"
            )
            return self.detect(image)

        logger.info("Starting multi-person ensemble detection")

        # Step 1: Detect all people with YOLO
        try:
            person_detections = self.person_detector.detect_people(image, return_crops=True)
        except Exception as e:
            logger.error(f"YOLO person detection failed: {e}, falling back to whole-image")
            return self.detect(image)

        if not person_detections:
            logger.info("No people detected by YOLO")
            return []

        logger.info(f"YOLO detected {len(person_detections)} people")

        # Step 2: For each person, run ensemble detection and fusion
        all_fused_results = []

        for person in person_detections:
            person_id = person.person_id
            logger.debug(f"Processing person {person_id} with ensemble fusion")

            # Collect predictions from all models for this person
            person_predictions = []

            for i, det_spec in enumerate(self.detectors, 1):
                try:
                    # Run pose detection on person's crop
                    pose = det_spec["detector"].detect_from_crop(
                        person.crop,
                        person.bbox
                    )

                    if pose is not None:
                        # Set person ID
                        pose.person_id = person_id

                        person_predictions.append(
                            {"pose": pose, "weight": det_spec["weight"], "model_id": i}
                        )
                        logger.debug(
                            f"Model {i} detected person {person_id} with confidence {pose.overall_confidence:.3f}"
                        )
                    else:
                        logger.debug(f"Model {i} failed to detect person {person_id}")

                    # Clear GPU cache between models
                    if i < len(self.detectors):
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            if _has_mps_module():
                                try:
                                    torch.mps.empty_cache()
                                except Exception:
                                    pass
                            # Else: PyTorch < 2.1, skip MPS cache cleanup

                except Exception as e:
                    logger.warning(f"Model {i} detection failed for person {person_id}: {e}")
                    continue

            # Apply fusion if we have predictions for this person
            if person_predictions:
                try:
                    if self.config.fusion_method == "weighted_average":
                        fused = self._fuse_weighted_average(person_predictions)
                    elif self.config.fusion_method == "confidence_weighted":
                        fused = self._fuse_confidence_weighted(person_predictions)
                    else:
                        raise ValueError(
                            f"Unknown fusion method: {self.config.fusion_method}"
                        )

                    # Ensure person ID is preserved
                    fused.person_id = person_id

                    all_fused_results.append(fused)

                    logger.info(
                        f"Fused person {person_id}: confidence={fused.overall_confidence:.3f}, "
                        f"from {len(person_predictions)} model(s)"
                    )
                except Exception as e:
                    logger.error(f"Fusion failed for person {person_id}: {e}")
                    # Fallback: use best prediction for this person
                    if person_predictions:
                        best = max(person_predictions, key=lambda p: p["pose"].overall_confidence)
                        best["pose"].person_id = person_id
                        all_fused_results.append(best["pose"])
                        logger.warning(
                            f"Using fallback for person {person_id}: best model confidence={best['pose'].overall_confidence:.3f}"
                        )
            else:
                logger.warning(f"No valid predictions for person {person_id}, skipping")

        logger.info(
            f"Multi-person ensemble complete: {len(all_fused_results)}/{len(person_detections)} people processed"
        )

        return all_fused_results

    def _fuse_weighted_average(
        self, predictions: List[Dict]
    ) -> PoseResult:
        """
        Fuse predictions using weighted average of keypoints.

        Each model's keypoints are weighted by its model weight.
        Final confidence is average of all model confidences.

        Args:
            predictions: List of {pose: PoseResult, weight: float, model_id: int}

        Returns:
            Fused PoseResult
        """
        # Calculate total weight for normalization
        total_weight = sum(p["weight"] for p in predictions)

        # Initialize with first prediction's structure
        base_pose = predictions[0]["pose"]
        num_keypoints = len(base_pose.keypoints)

        # Initialize arrays for weighted sum
        fused_keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        fused_visibility = np.zeros(num_keypoints, dtype=np.float32)

        # Weighted average of keypoints
        for pred in predictions:
            pose = pred["pose"]
            weight = pred["weight"] / total_weight  # Normalize weight

            # Extract only x,y coordinates (keypoints is shape (133, 3) with [x, y, confidence])
            fused_keypoints += pose.keypoints[:, :2] * weight
            fused_visibility += pose.visibility * weight

        # Average confidence across models
        avg_confidence = float(
            np.mean([p["pose"].overall_confidence for p in predictions])
        )

        # Average bbox (weighted)
        fused_bbox = np.zeros(4, dtype=np.float32)
        for pred in predictions:
            weight = pred["weight"] / total_weight
            fused_bbox += pred["pose"].bbox * weight

        # CRITICAL: Convert visibility to integers (database expects 0, 1, or 2)
        # Fused visibility is float average, need to round and clip
        fused_visibility_int = np.clip(np.round(fused_visibility), 0, 2).astype(np.int32)

        # Create fused result (pre-allocate for performance)
        fused_keypoints_with_conf = np.empty((num_keypoints, 3), dtype=np.float32)
        fused_keypoints_with_conf[:, :2] = fused_keypoints
        fused_keypoints_with_conf[:, 2] = fused_visibility

        fused = PoseResult(
            keypoints=fused_keypoints_with_conf,
            visibility=fused_visibility_int,  # Use integer visibility
            bbox=fused_bbox,
            overall_confidence=avg_confidence,
            person_id=base_pose.person_id,
        )

        logger.debug(
            f"Weighted average fusion: {len(predictions)} predictions, "
            f"avg_conf={avg_confidence:.3f}"
        )

        return fused

    def _fuse_confidence_weighted(
        self, predictions: List[Dict]
    ) -> PoseResult:
        """
        Fuse using confidence-weighted averaging.

        Each keypoint is weighted by both:
        1. Model weight (global importance)
        2. Keypoint visibility (per-keypoint confidence)

        This gives higher weight to keypoints that models are more confident about.

        Args:
            predictions: List of {pose: PoseResult, weight: float, model_id: int}

        Returns:
            Fused PoseResult
        """
        # Initialize with first prediction's structure
        base_pose = predictions[0]["pose"]
        num_keypoints = len(base_pose.keypoints)

        # Initialize arrays
        fused_keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        total_confidence = np.zeros(num_keypoints, dtype=np.float32)

        # Confidence-weighted averaging
        for pred in predictions:
            pose = pred["pose"]
            model_weight = pred["weight"]

            # Weight by both model weight and per-keypoint visibility
            weights = pose.visibility * model_weight

            # Accumulate weighted keypoints (extract only x,y from (133, 3) array)
            fused_keypoints += pose.keypoints[:, :2] * weights[:, np.newaxis]
            total_confidence += weights

        # Normalize by total confidence per keypoint
        # Add small epsilon to avoid division by zero
        fused_keypoints /= total_confidence[:, np.newaxis] + 1e-6

        # Normalize visibility scores
        total_model_weight = sum(p["weight"] for p in predictions)
        fused_visibility = total_confidence / total_model_weight

        # Overall confidence is mean of fused visibility scores (normalized to [0, 1])
        # Visibility ranges from 0-2, so divide by 2 to get confidence in [0, 1]
        overall_conf = float(np.mean(fused_visibility) / 2.0)

        # CRITICAL: Convert visibility to integers (database expects 0, 1, or 2)
        # Fused visibility is float, need to round and clip
        fused_visibility_int = np.clip(np.round(fused_visibility), 0, 2).astype(np.int32)

        # Average bbox
        fused_bbox = np.mean([p["pose"].bbox for p in predictions], axis=0).astype(
            np.float32
        )

        # Create fused result (pre-allocate for performance)
        fused_keypoints_with_conf = np.empty((num_keypoints, 3), dtype=np.float32)
        fused_keypoints_with_conf[:, :2] = fused_keypoints
        fused_keypoints_with_conf[:, 2] = fused_visibility

        fused = PoseResult(
            keypoints=fused_keypoints_with_conf,
            visibility=fused_visibility_int,  # Use integer visibility
            bbox=fused_bbox,
            overall_confidence=overall_conf,
            person_id=base_pose.person_id,
        )

        logger.debug(
            f"Confidence-weighted fusion: {len(predictions)} predictions, "
            f"overall_conf={overall_conf:.3f}"
        )

        return fused

    def set_fusion_method(
        self, method: Literal["weighted_average", "confidence_weighted"]
    ) -> None:
        """
        Update fusion method.

        Args:
            method: New fusion method

        Raises:
            ValueError: If method is invalid
        """
        valid_methods = ["weighted_average", "confidence_weighted"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid fusion method '{method}'. Must be one of: {valid_methods}"
            )

        self.config.fusion_method = method
        logger.info(f"Updated fusion method to: {method}")

    def get_model_count(self) -> int:
        """Get number of models in ensemble."""
        return len(self.detectors)

    def get_total_weight(self) -> float:
        """Get sum of all model weights."""
        return sum(d["weight"] for d in self.detectors)

    def unload_models(self) -> None:
        """Unload all ensemble models from GPU memory to free resources."""
        for det_spec in self.detectors:
            detector = det_spec["detector"]
            if hasattr(detector, 'unload_model'):
                detector.unload_model()

        # Clear GPU cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            if _has_mps_module():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            # Else: PyTorch < 2.1, skip MPS cache cleanup

        logger.info(f"Unloaded {len(self.detectors)} ensemble models from memory")


def create_ensemble(
    model_configs: List[tuple],
    fusion_method: Literal["weighted_average", "confidence_weighted"] = "weighted_average",
    **kwargs,
) -> EnsembleDetector:
    """
    Factory function to create ensemble detector.

    Args:
        model_configs: List of (config_path, checkpoint_path, weight) tuples
        fusion_method: Method for fusing predictions
        **kwargs: Additional arguments for EnsembleConfig

    Returns:
        Configured EnsembleDetector instance

    Example:
        >>> models = [
        ...     ('config1.py', 'ckpt1.pth', 1.0),
        ...     ('config2.py', 'ckpt2.pth', 1.2)
        ... ]
        >>> ensemble = create_ensemble(models, fusion_method='confidence_weighted')
    """
    model_specs = []
    for config_path, checkpoint_path, weight in model_configs:
        model_specs.append(
            {"config": str(config_path), "checkpoint": str(checkpoint_path), "weight": weight}
        )

    config = EnsembleConfig(models=model_specs, fusion_method=fusion_method, **kwargs)

    return EnsembleDetector(config)
