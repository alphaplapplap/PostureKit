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

# mmpose's inference_topdown natively batches every supplied bbox into one
# model.test_step (collated via pseudo_collate), so the ensemble can run ONE
# batched forward per model over all persons instead of 2N batch-1 crops.
# (torch is patched by the entry point's _torch_patch before this module loads;
# pose_detector below also imports mmpose, so the patch is already applied.)
from mmpose.apis import inference_topdown

from src.core.pose_detector import RTMWCocktail14Detector, PoseResult
from src.core.person_detector import (
    YOLOPersonDetector,
    RTMOPersonDetector,
    PersonDetection,
)
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

        # Bottom-up person detection via RTMO-l body7 (74.8 COCO / 83.8 CrowdPose AP).
        # Replaces YOLOv8x: top-down YOLO cannot separate entangled bodies on this
        # corpus — it fuses overlapping people into a single mega-bbox or drops the
        # occluded partner. RTMO clusters keypoints bottom-up and handles crowds
        # natively. Falls back to YOLOv8x if the ONNX or rtmlib is missing.
        self.person_detector = None
        try:
            self.person_detector = RTMOPersonDetector(device='mps')
            logger.info("RTMO person detector initialized (rtmo-l body7, device=mps)")
        except Exception as e:
            logger.warning(f"RTMO init failed, falling back to YOLOv8x: {e}")
            try:
                self.person_detector = YOLOPersonDetector(
                    model_name='yolov8x.pt',
                    confidence_threshold=0.05,
                    device=self.device,
                )
                logger.info("YOLOv8x fallback initialized (conf=0.05)")
            except Exception as e2:
                logger.warning(f"YOLOv8x fallback also failed: {e2}")
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

                # Per-inference gc.collect() + torch.mps cache wiping removed
                # (finding 12): each cost ~38ms, leaked nothing, and emptying the
                # MPS cache only slowed the next forward on this 128 GB machine.
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

    def _detect_model_batched(
        self,
        detector: "RTMWCocktail14Detector",
        image: np.ndarray,
        person_bboxes_xywh: List[np.ndarray],
    ) -> List[Optional[PoseResult]]:
        """
        Run ONE model over every person in a single batched forward pass.

        mmpose's inference_topdown collates all supplied bboxes into one
        pseudo_collate batch and runs a single model.test_step, so passing the
        N padded person bboxes here issues one batch-N MPS forward instead of N
        batch-1 forwards over per-person crops. Each model is rebuilt once (one
        Compose pipeline) per call rather than once per (person, model) pair.

        Keypoints come back in full-image coordinates (the bbox is the
        full-image padded box, not a crop origin), so no per-person offset-add
        is needed — this reproduces detect_from_crop's geometry without the
        zero-filled crop margins (it samples real pixels in the padded region,
        which is neutral-to-slightly-positive for occluded limbs).

        Args:
            detector: One ensemble member (RTMWCocktail14Detector).
            image: Full RGB image (H, W, 3).
            person_bboxes_xywh: Per-person padded bboxes in [x, y, w, h] order,
                index-aligned with the persons list.

        Returns:
            List index-aligned with person_bboxes_xywh; each entry is a
            PoseResult (keypoints in image coords) or None if that person
            produced no usable sample.
        """
        if detector._model is None:
            detector._load_model()

        # Convert [x, y, w, h] -> [x1, y1, x2, y2] for inference_topdown.
        bboxes_xyxy = np.array(
            [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in person_bboxes_xywh],
            dtype=np.float32,
        )

        # Single batched forward over all persons.
        data_samples = inference_topdown(
            detector._model, image, bboxes=bboxes_xyxy, bbox_format="xyxy"
        )

        results: List[Optional[PoseResult]] = [None] * len(person_bboxes_xywh)
        # inference_topdown returns results in bbox order, index-aligned with persons.
        for idx, sample in enumerate(data_samples):
            if idx >= len(results):
                break
            try:
                pred_instances = sample.pred_instances
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores

                # Strip the per-sample instance dimension if present.
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]
                    keypoint_scores = keypoint_scores[0]

                if torch.is_tensor(keypoints):
                    keypoints = keypoints.detach().cpu().numpy()
                if torch.is_tensor(keypoint_scores):
                    keypoint_scores = keypoint_scores.detach().cpu().numpy()

                # Keypoints are already in full-image coordinates (bbox was the
                # full-image padded box), so NO offset-add — unlike the crop path.
                keypoints_with_conf = np.empty((133, 3), dtype=np.float32)
                keypoints_with_conf[:, :2] = keypoints
                keypoints_with_conf[:, 2] = np.clip(keypoint_scores, 0.0, 1.0)

                overall_confidence = float(np.clip(keypoint_scores.mean(), 0.0, 1.0))

                # Same visibility thresholds as detect_from_crop.
                visibility_array = np.zeros(133, dtype=np.int8)
                visibility_array[keypoint_scores >= 0.5] = 2
                visibility_array[(keypoint_scores >= 0.1) & (keypoint_scores < 0.5)] = 1
                visibility_array[keypoint_scores < 0.1] = 0

                results[idx] = PoseResult(
                    keypoints=keypoints_with_conf,
                    visibility=visibility_array,
                    bbox=person_bboxes_xywh[idx].copy(),
                    overall_confidence=overall_confidence,
                    person_id=0,  # set by caller
                )
            except Exception as e:
                logger.warning(
                    f"Failed to build pose for person index {idx}: {e}", exc_info=True
                )
                results[idx] = None

        return results

    def detect_multi_person(self, image: np.ndarray) -> List[PoseResult]:
        """
        Detect multiple people using ensemble with per-person fusion.

        This method properly handles multi-person scenarios by:
        1. Detecting all people with the person detector first
        2. Running each ensemble model ONCE over all persons (batched forward)
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

        # Step 1: Detect all people with the person detector.
        # Batched pose inference re-crops from the full image via the padded
        # bbox geometry, so we no longer need materialized per-person crops.
        try:
            person_detections = self.person_detector.detect_people(image, return_crops=False)
        except Exception as e:
            logger.error(f"Person detection failed: {e}, falling back to whole-image")
            return self.detect(image)

        if not person_detections:
            logger.info("No people detected by YOLO")
            return []

        logger.info(f"{type(self.person_detector).__name__} detected {len(person_detections)} people (pre-filter)")

        # The mega-bbox filter is a YOLO-specific safety net: YOLO top-down often
        # fuses entangled bodies into one box AND emits the legit single-body
        # boxes alongside. RTMO is bottom-up — it clusters per-person keypoints,
        # so its detections legitimately overlap in tight compositions (center
        # body + partner stacked above/below). Applying the mega-bbox filter to
        # RTMO drops good detections like the center woman in 1362107066.jpg.
        if isinstance(self.person_detector, RTMOPersonDetector):
            logger.info("Skipping mega-bbox filter (RTMO bottom-up detections)")
        else:
            person_detections = self._filter_mega_bboxes(person_detections, image.shape[:2])
            if not person_detections:
                logger.info("All detections filtered out by mega-bbox filter")
                return []
            logger.info(f"After mega-bbox filter: {len(person_detections)} people")

        # Step 2: Run each model ONCE over all persons (batched), then fuse
        # per person. Restructured from 2N batch-1 crop forwards to 2 batch-N
        # forwards (one per ensemble member). Per-inference gc.collect() +
        # torch.mps cache wiping removed: each cost ~38ms and is pointless on a
        # 128 GB machine — they only defeated the MPS allocator cache for the
        # next forward. unload_models() still does one end-of-life cleanup.
        all_fused_results = []

        person_bboxes_xywh = [p.bbox for p in person_detections]

        # model_results[model_index] -> list aligned with persons.
        model_results: List[List[Optional[PoseResult]]] = []
        for i, det_spec in enumerate(self.detectors, 1):
            try:
                per_person = self._detect_model_batched(
                    det_spec["detector"], image, person_bboxes_xywh
                )
            except Exception as e:
                logger.warning(f"Model {i} batched detection failed: {e}")
                per_person = [None] * len(person_detections)
            model_results.append(per_person)

        for p_idx, person in enumerate(person_detections):
            person_id = person.person_id
            logger.debug(f"Processing person {person_id} with ensemble fusion")

            # Collect predictions from all models for this person.
            person_predictions = []
            for i, det_spec in enumerate(self.detectors, 1):
                pose = model_results[i - 1][p_idx]
                if pose is not None:
                    pose.person_id = person_id
                    person_predictions.append(
                        {"pose": pose, "weight": det_spec["weight"], "model_id": i}
                    )
                    logger.debug(
                        f"Model {i} detected person {person_id} with confidence {pose.overall_confidence:.3f}"
                    )
                else:
                    logger.debug(f"Model {i} failed to detect person {person_id}")

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

    def _filter_mega_bboxes(
        self,
        detections: List[PersonDetection],
        image_shape: tuple,
        area_threshold: float = 0.55,
        overlap_threshold: float = 0.30,
    ) -> List[PersonDetection]:
        """
        Drop YOLO "mega-bbox" false positives.

        A mega-bbox covers >area_threshold of the image AND overlaps (IoA >=
        overlap_threshold) with at least one other smaller detection. It is almost always
        YOLO fusing two entangled bodies into a single "person". If no overlapping smaller
        detection exists, the big box is kept (legitimate full-frame solo shot).
        """
        if len(detections) <= 1:
            return detections

        H, W = image_shape
        image_area = float(H * W)

        def area(bbox: np.ndarray) -> float:
            return float(bbox[2] * bbox[3])

        def ioa(inner: np.ndarray, outer: np.ndarray) -> float:
            """Intersection area of inner with outer, normalized by inner's area."""
            ix1 = max(inner[0], outer[0])
            iy1 = max(inner[1], outer[1])
            ix2 = min(inner[0] + inner[2], outer[0] + outer[2])
            iy2 = min(inner[1] + inner[3], outer[1] + outer[3])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inner_area = area(inner)
            if inner_area <= 0:
                return 0.0
            return (iw * ih) / inner_area

        kept: List[PersonDetection] = []
        dropped = 0
        for i, det in enumerate(detections):
            frac = area(det.bbox) / image_area
            if frac >= area_threshold:
                overlaps_smaller = any(
                    j != i
                    and area(other.bbox) < area(det.bbox)
                    and ioa(other.bbox, det.bbox) >= overlap_threshold
                    for j, other in enumerate(detections)
                )
                if overlaps_smaller:
                    logger.info(
                        f"Dropped mega-bbox: person_id={det.person_id} "
                        f"frac={frac:.2f} conf={det.confidence:.2f}"
                    )
                    dropped += 1
                    continue
            kept.append(det)

        if dropped:
            for new_id, d in enumerate(kept):
                d.person_id = new_id
        return kept

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
        fused_kp_conf = np.zeros(num_keypoints, dtype=np.float32)

        # Weighted average of keypoints
        for pred in predictions:
            pose = pred["pose"]
            weight = pred["weight"] / total_weight  # Normalize weight

            # Extract only x,y coordinates (keypoints is shape (133, 3) with [x, y, confidence])
            fused_keypoints += pose.keypoints[:, :2] * weight
            fused_visibility += pose.visibility * weight
            # Real [0,1] model confidence, averaged under the same weights
            fused_kp_conf += pose.keypoints[:, 2] * weight

        # Average confidence across models
        avg_confidence = float(
            np.mean([p["pose"].overall_confidence for p in predictions])
        )

        # Average bbox (weighted)
        fused_bbox = np.zeros(4, dtype=np.float32)
        for pred in predictions:
            weight = pred["weight"] / total_weight
            fused_bbox += pred["pose"].bbox * weight

        # Convert visibility to integers (database expects 0, 1, or 2). Exact
        # halves round DOWN (toward more-occluded) — see _fuse_confidence_weighted.
        fused_visibility_int = np.clip(
            np.ceil(fused_visibility - 0.5), 0, 2
        ).astype(np.int32)

        # Create fused result (pre-allocate for performance)
        # Column 2 carries genuine [0,1] confidence, not [0,2] visibility — see
        # _fuse_confidence_weighted for why this distinction is load-bearing.
        fused_keypoints_with_conf = np.empty((num_keypoints, 3), dtype=np.float32)
        fused_keypoints_with_conf[:, :2] = fused_keypoints
        fused_keypoints_with_conf[:, 2] = np.clip(fused_kp_conf, 0.0, 1.0)

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
        fused_kp_conf = np.zeros(num_keypoints, dtype=np.float32)
        total_confidence = np.zeros(num_keypoints, dtype=np.float32)

        # Confidence-weighted averaging
        for pred in predictions:
            pose = pred["pose"]
            model_weight = pred["weight"]

            # Weight by both model weight and per-keypoint visibility
            weights = pose.visibility * model_weight

            # Accumulate weighted keypoints (extract only x,y from (133, 3) array)
            fused_keypoints += pose.keypoints[:, :2] * weights[:, np.newaxis]
            # Real [0,1] model confidence, fused under the same weights as positions
            fused_kp_conf += pose.keypoints[:, 2] * weights
            total_confidence += weights

        # Normalize by total confidence per keypoint
        # Add small epsilon to avoid division by zero
        fused_keypoints /= total_confidence[:, np.newaxis] + 1e-6
        fused_kp_conf /= total_confidence + 1e-6

        # Keypoints every model missed (zero total weight) would otherwise land
        # at (0,0) — top-left corner — from the 0/eps division. Fall back to the
        # plain mean position; confidence stays ~0 so downstream gates skip them.
        zero_weight = total_confidence <= 1e-6
        if np.any(zero_weight):
            mean_positions = np.mean(
                [p["pose"].keypoints[:, :2] for p in predictions], axis=0
            )
            fused_keypoints[zero_weight] = mean_positions[zero_weight]

        # Normalize visibility scores
        total_model_weight = sum(p["weight"] for p in predictions)
        fused_visibility = total_confidence / total_model_weight

        # Overall confidence is mean of fused visibility scores (normalized to [0, 1])
        # Visibility ranges from 0-2, so divide by 2 to get confidence in [0, 1]
        overall_conf = float(np.mean(fused_visibility) / 2.0)

        # Convert visibility to integers (database expects 0, 1, or 2). Exact
        # halves round DOWN (toward more-occluded): a 1-vs-2 model disagreement
        # is an occlusion vote, and np.round's half-to-even would erase it.
        fused_visibility_int = np.clip(
            np.ceil(fused_visibility - 0.5), 0, 2
        ).astype(np.int32)

        # Average bbox
        fused_bbox = np.mean([p["pose"].bbox for p in predictions], axis=0).astype(
            np.float32
        )

        # Create fused result (pre-allocate for performance)
        # Column 2 carries genuine [0,1] confidence. It was previously overwritten
        # with [0,2] visibility, which silently doubled every downstream
        # conf>=threshold gate (extractor 0.3, masked-distance 0.35, OKS, bbox
        # refinement) and disabled the entire occlusion-handling stack.
        fused_keypoints_with_conf = np.empty((num_keypoints, 3), dtype=np.float32)
        fused_keypoints_with_conf[:, :2] = fused_keypoints
        fused_keypoints_with_conf[:, 2] = np.clip(fused_kp_conf, 0.0, 1.0)

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
