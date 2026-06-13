"""
Person Detector using YOLOv8.
Detects individual people in images before pose estimation.
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import logging
import weakref
import atexit
from concurrent.futures import ThreadPoolExecutor

import cv2
from ultralytics import YOLO
import torch

from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

# Shared, bounded thread pool for concurrent per-tile RTMO inference.
# ONNX Runtime's InferenceSession.run is thread-safe, and the CPU-resident
# partitions of the CoreML-split RTMO graph overlap across threads (measured
# ~1.67x on the 8-tile pass for a 4K image). Tiles are large 1280px inferences,
# so the pool is intentionally bounded (4 workers) rather than scaled to the
# 18-core machine — wider pools oversubscribe the ANE/CPU partitions for no
# additional gain. Module-level so the pool (and its threads) are reused across
# images instead of being torn down per detection.
_TILE_INFERENCE_WORKERS = 4
_tile_executor: Optional[ThreadPoolExecutor] = None


def _get_tile_executor() -> ThreadPoolExecutor:
    """Lazily construct the shared per-tile inference pool."""
    global _tile_executor
    if _tile_executor is None:
        _tile_executor = ThreadPoolExecutor(
            max_workers=_TILE_INFERENCE_WORKERS,
            thread_name_prefix="rtmo-tile",
        )
        atexit.register(_tile_executor.shutdown, wait=False)
    return _tile_executor


def _has_mps_module() -> bool:
    """
    Check if torch.mps module is available (PyTorch 2.1+).

    PyTorch 2.0.x has torch.backends.mps but not torch.mps module.
    PyTorch 2.1+ has both torch.backends.mps and torch.mps.

    Returns:
        True if torch.mps module exists, False otherwise
    """
    return hasattr(torch, 'mps')


@dataclass
class PersonDetection:
    """
    Detected person bounding box.

    Attributes:
        bbox: Bounding box [x, y, width, height]
        confidence: Detection confidence (0-1)
        person_id: Index of person in image
        crop: Cropped image of person (for pose detection)
    """
    bbox: np.ndarray  # (4,) [x, y, w, h]
    confidence: float
    person_id: int
    crop: Optional[np.ndarray] = None  # (H, W, 3)

    def __post_init__(self):
        """Validate detection."""
        assert self.bbox.shape == (4,), f"Expected bbox (4,), got {self.bbox.shape}"
        assert 0.0 <= self.confidence <= 1.0, f"Confidence must be [0,1], got {self.confidence}"


class PersonDetectorError(Exception):
    """Base exception for person detector errors."""
    pass


class YOLOPersonDetector:
    """
    Detects individual people in images using YOLOv8.

    Features:
    - Detects all people in image
    - Returns bounding boxes and cropped images
    - Filters by confidence threshold
    - Handles overlapping detections with NMS

    Attributes:
        model: YOLOv8 model
        confidence_threshold: Minimum confidence for detection
        device: Torch device (mps, cuda, cpu)
    """

    # Class-level model cache with weak references to prevent memory leaks
    _model_cache = weakref.WeakValueDictionary()

    def __init__(
        self,
        model_name: str = 'yolov8n.pt',  # nano model (fastest)
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize person detector.

        Args:
            model_name: YOLOv8 model ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
            confidence_threshold: Minimum confidence for person detection
            device: Device to use ('mps', 'cuda', 'cpu'). Uses settings default if None.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or settings.DEVICE
        self.model = None

        # Model path
        self.model_name = model_name

        logger.info(
            f"YOLOPersonDetector initialized",
            extra={'extra_data': {
                'model': model_name,
                'threshold': confidence_threshold,
                'device': self.device
            }}
        )

    def _load_model(self):
        """Load YOLO model (lazy loading with caching)."""
        if self.model is not None:
            return

        # Check cache first (TOCTOU-safe: use get() to avoid race)
        cache_key = (self.model_name, self.device)
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            self.model = cached_model
            logger.info(f"Reusing cached YOLO model: {self.model_name}")
            return

        logger.info(f"Loading YOLOv8 model: {self.model_name}")

        try:
            # Load model
            self.model = YOLO(self.model_name)

            # Set device
            if self.device == 'mps' and torch.backends.mps.is_available():
                # YOLOv8 automatically handles MPS
                logger.info("Using Apple Silicon MPS")
            elif self.device == 'cuda' and torch.cuda.is_available():
                logger.info("Using CUDA GPU")
            else:
                logger.info("Using CPU")

            # Cache the model
            self._model_cache[cache_key] = self.model

            logger.info("YOLOv8 model loaded successfully")

        except Exception as e:
            raise PersonDetectorError(f"Failed to load YOLO model: {e}") from e

    def detect_people(
        self,
        image: np.ndarray,
        return_crops: bool = True
    ) -> List[PersonDetection]:
        """
        Detect all people in image.

        Args:
            image: Input image (H, W, 3) in RGB format
            return_crops: Whether to return cropped person images

        Returns:
            List of PersonDetection objects, one per detected person
        """
        # Load model if needed
        if self.model is None:
            self._load_model()

        logger.debug(f"Detecting people in image shape: {image.shape}")

        try:
            # Run detection
            # classes=0 means only detect 'person' class
            results = self.model.predict(
                image,
                classes=[0],  # 0 = person in COCO dataset
                conf=self.confidence_threshold,
                verbose=False
            )

            detections = []

            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i, box in enumerate(boxes):
                    # Get bounding box in xyxy format
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Convert to xywh format
                    bbox = np.array([
                        x1,
                        y1,
                        x2 - x1,  # width
                        y2 - y1   # height
                    ], dtype=np.float32)

                    # Get confidence
                    confidence = float(box.conf[0])

                    # Crop person if requested
                    crop = None
                    if return_crops:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Add small padding (5%)
                        h, w = image.shape[:2]
                        pad_x = int((x2 - x1) * 0.05)
                        pad_y = int((y2 - y1) * 0.05)

                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(w, x2 + pad_x)
                        y2 = min(h, y2 + pad_y)

                        crop = image[y1:y2, x1:x2].copy()

                    detection = PersonDetection(
                        bbox=bbox,
                        confidence=confidence,
                        person_id=i,
                        crop=crop
                    )

                    detections.append(detection)

            # CRITICAL: Delete YOLO results to free GPU memory
            del results
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                if _has_mps_module():
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception:
                        # Ignore cache cleanup errors
                        pass
                # Else: PyTorch < 2.1, skip MPS cache cleanup
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(
                f"Detected {len(detections)} people",
                extra={'extra_data': {
                    'count': len(detections),
                    'confidences': [d.confidence for d in detections]
                }}
            )

            return detections

        except Exception as e:
            raise PersonDetectorError(f"Person detection failed: {e}") from e

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def unload_model(self) -> None:
        """Unload YOLO model from memory to free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None

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

            logger.info("YOLO model unloaded from memory")


class RTMOPersonDetector:
    """
    Bottom-up person detector using RTMO-l (body7, 83.8 AP CrowdPose).

    Replaces YOLOv8 for entangled/crowded scenes: RTMO is one-stage bottom-up and
    clusters keypoints into people directly, avoiding YOLO's failure mode of fusing
    overlapping bodies into a single mega-bbox. Returns PersonDetection objects
    whose bboxes are derived from the 17-keypoint RTMO output:
        bbox = tight_box(kpts[scores >= min_kp_score]) + bbox_pad
    """

    _model_cache = weakref.WeakValueDictionary()

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        nms_score_thr: float = 0.1,
        min_max_kp_score: float = 0.5,
        min_kp_score: float = 0.3,
        min_visible_kps: int = 4,
        iou_dedup_thr: float = 0.45,
        containment_thr: float = 0.7,
        bbox_pad: float = 0.10,
        min_bbox_side: int = 40,
        device: str = 'mps',
        tile_min_long_side: int = 1600,
        tile_size: int = 1280,
        tile_overlap: float = 0.25,
        tile_min_visible_kps: int = 8,
        tile_iou_dedup_thr: float = 0.35,
        tile_containment_thr: float = 0.5,
    ):
        default_path = settings.PROJECT_ROOT / "data" / "models" / "rtmo-l_body7.onnx"
        self.onnx_path = onnx_path or str(default_path)
        # nms_score_thr is RTMO's per-box NMS floor (lower = more candidates).
        # We post-filter with min_max_kp_score, so keep this permissive.
        self.nms_score_thr = nms_score_thr
        # A detection is accepted if it has at least one keypoint with score
        # >= min_max_kp_score AND at least min_visible_kps keypoints with
        # score >= min_kp_score. Mean-score is a poor signal for entangled
        # poses (many keypoints are occluded → mean collapses) so we use
        # max instead.
        self.min_max_kp_score = min_max_kp_score
        self.min_kp_score = min_kp_score
        self.min_visible_kps = min_visible_kps
        # IoU/containment dedup: RTMO's internal NMS sometimes keeps a body
        # fragment (partial crop) alongside the full-body detection; we
        # prefer the detection with more visible keypoints.
        self.iou_dedup_thr = iou_dedup_thr
        self.containment_thr = containment_thr
        self.bbox_pad = bbox_pad
        # Minimum keypoint-span side (pre-padding) in absolute pixels. Smaller
        # than this and downstream RTMW cannot fit 133 keypoints reliably — the
        # detection is a face fragment or a patch of skin that happens to have
        # a few high-confidence keypoints clustered together.
        self.min_bbox_side = min_bbox_side
        # RTMO resizes its input to 640x640, so a 4K image leaves ~150px per body —
        # below the model's effective resolution. For images whose long side is
        # >= tile_min_long_side, we ALSO run detection on overlapping tile_size
        # crops (tile_overlap fraction of overlap) and pool all candidates through
        # the same greedy dedup: large/close people come from the full-image pass,
        # small/distant people from the tiles. tile_min_long_side <= 0 disables.
        self.tile_min_long_side = tile_min_long_side
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        # Tiles zoomed into a large body see featureless skin/texture patches and
        # hallucinate 4-6 confident keypoints on them (verified on the 4K entangled
        # test image). Tile candidates therefore need >= tile_min_visible_kps —
        # a small person worth recovering from a tile is fully visible there, so
        # most of their 17 keypoints score; heavily occluded people stay the
        # full-image pass's job under its permissive min_visible_kps gate.
        self.tile_min_visible_kps = tile_min_visible_kps
        # Tile candidates also dedup under stricter overlap thresholds than the
        # full pass: a tile seeing part of a large body emits blob detections
        # ~50-60% contained in the best detection — under the loose containment
        # bar (tuned to keep entangled full-pass partners) but a duplicate.
        # Tile recoveries are a bonus, so they must be clearly distinct to win.
        self.tile_iou_dedup_thr = tile_iou_dedup_thr
        self.tile_containment_thr = tile_containment_thr
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        cache_key = (self.onnx_path, self.device)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            self.model = cached
            logger.info(f"Reusing cached RTMO model: {self.onnx_path}")
            return
        try:
            from rtmlib import RTMO
        except ImportError as e:
            raise PersonDetectorError(
                "rtmlib not installed. Run: pip install rtmlib"
            ) from e
        if not Path(self.onnx_path).exists():
            raise PersonDetectorError(
                f"RTMO ONNX not found: {self.onnx_path}. "
                f"Download from https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
                f"rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
            )
        try:
            self.model = RTMO(
                onnx_model=self.onnx_path,
                model_input_size=(640, 640),
                nms_thr=0.45,
                score_thr=self.nms_score_thr,
                backend='onnxruntime',
                device=self.device,
            )
            self._model_cache[cache_key] = self.model
            logger.info(f"Loaded RTMO-l body7 ({self.onnx_path}, device={self.device})")
        except Exception as e:
            raise PersonDetectorError(f"Failed to load RTMO: {e}") from e

    def detect_people(
        self,
        image: np.ndarray,
        return_crops: bool = True,
    ) -> List[PersonDetection]:
        if self.model is None:
            self._load_model()

        # rtmlib/RTMO was trained on BGR (OpenCV convention); caller passes RGB.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        H, W = image.shape[:2]
        detections: List[PersonDetection] = []

        # High-res tiling pass: each tile is a pixel crop (no resize), so RTMO's
        # 640x640 input sees small people at tile_size/640 ≈ 2x the detail of the
        # full-image pass. Tile fragments of people the full pass already found
        # have fewer visible keypoints and get dropped by the containment dedup.
        tiles: List[tuple] = []
        if self.tile_min_long_side > 0 and max(H, W) >= self.tile_min_long_side:
            tiles = self._generate_tiles(W, H)

        if tiles:
            # Run the full-image pass and all tile passes concurrently through a
            # bounded shared pool: InferenceSession.run is thread-safe and the
            # CPU-resident partitions of the CoreML-split RTMO graph overlap
            # across threads (~1.67x on the tile portion). Futures are collected
            # in deterministic order — full-image candidates first, then tiles in
            # tile order — so the input to the two-phase dedup (and thus the final
            # result) is bit-identical to the previous serial implementation.
            executor = _get_tile_executor()
            full_future = executor.submit(self._collect_candidates, bgr, (0, 0))
            tile_futures = [
                executor.submit(
                    self._collect_candidates,
                    bgr[ty1:ty2, tx1:tx2], (tx1, ty1),
                    self.tile_min_visible_kps, True,
                )
                for (tx1, ty1, tx2, ty2) in tiles
            ]
            candidates = full_future.result()
            for tf in tile_futures:
                candidates.extend(tf.result())
            logger.info(
                f"RTMO tiled detection: {len(tiles)} tiles for {W}x{H}, "
                f"{len(candidates)} candidates pre-dedup"
            )
        else:
            # Full-image pass only: finds large/close people regardless of size.
            candidates = self._collect_candidates(bgr, (0, 0))

        if not candidates:
            logger.info("RTMO detected 0 people")
            return detections

        # Two-phase greedy dedup. Phase 1: the legacy walk — full-pass candidates
        # dedup among themselves under the tuned thresholds. Losers are
        # permanently suppressed: a blob that lost to a full-pass box is a
        # duplicate regardless of what tiling finds, and letting it re-compete
        # against a different (tile) winner revives it — observed, not
        # hypothetical. With tiling off this IS the final result, bit-identical
        # to pre-tiling behavior.
        full_cands = [c for c in candidates if not c[4]]
        full_cands.sort(key=lambda c: c[0], reverse=True)
        legacy: List[tuple] = []
        for cand in full_cands:
            if self._suppressed_by(cand, legacy,
                                   self.iou_dedup_thr, self.containment_thr):
                continue
            legacy.append(cand)

        # Phase 2: legacy survivors and tile candidates compete head-to-head, so
        # a sharper tile detection (more visible keypoints) can replace a merged
        # full-pass mega-box — tile recoveries of people inside mega-boxes are
        # the point of tiling. Tile candidates are held to the stricter tile_*
        # overlap thresholds.
        pool = legacy + [c for c in candidates if c[4]]
        pool.sort(key=lambda c: c[0], reverse=True)
        kept: List[tuple] = []
        for cand in pool:
            from_tile = cand[4]
            iou_thr = self.tile_iou_dedup_thr if from_tile else self.iou_dedup_thr
            cont_thr = self.tile_containment_thr if from_tile else self.containment_thr
            if self._suppressed_by(cand, kept, iou_thr, cont_thr):
                continue
            kept.append(cand)

        # Materialize PersonDetection with padded bbox + crop.
        for new_id, (_, max_score, _, (x1, y1, x2, y2), _, kpts, sc) in enumerate(kept):
            # The keypoint-span bbox excludes occluded body regions entirely,
            # so downstream RTMW hallucinates them inside a truncated crop.
            # Extend toward regions whose keypoints are missing using an
            # anatomical prior, so the crop at least contains the area where
            # the occluded limbs are.
            y1, y2 = self._directional_extend(kpts, sc, y1, y2)
            bw, bh = x2 - x1, y2 - y1
            x1p = max(0.0, x1 - bw * self.bbox_pad)
            y1p = max(0.0, y1 - bh * self.bbox_pad)
            x2p = min(float(W), x2 + bw * self.bbox_pad)
            y2p = min(float(H), y2 + bh * self.bbox_pad)
            bbox = np.array([x1p, y1p, x2p - x1p, y2p - y1p], dtype=np.float32)

            crop = None
            if return_crops:
                crop = image[int(y1p):int(y2p), int(x1p):int(x2p)].copy()

            detections.append(PersonDetection(
                bbox=bbox,
                confidence=min(max(max_score, 0.0), 1.0),
                person_id=new_id,
                crop=crop,
            ))

        logger.info(
            f"RTMO detected {len(detections)} people",
            extra={'extra_data': {
                'count': len(detections),
                'confidences': [d.confidence for d in detections],
            }}
        )
        return detections

    # COCO-17 keypoint groups (RTMO output) for the directional bbox prior
    HEAD_IDXS = (0, 1, 2, 3, 4)
    SHOULDER_IDXS = (5, 6)
    HIP_IDXS = (11, 12)
    KNEE_IDXS = (13, 14)
    ANKLE_IDXS = (15, 16)

    def _directional_extend(
        self,
        kpts: np.ndarray,
        sc: np.ndarray,
        y1: float,
        y2: float,
    ) -> tuple:
        """Extend the bbox vertically toward missing body regions.

        Anatomical prior in torso-height units (shoulder line to hip line):
        full legs ~1.2 below the hips, shins ~0.8 below the knees, head ~0.5
        above the shoulders. Fires only when a region's keypoints are entirely
        missing AND the pose is upright enough to measure a positive vertical
        torso extent (lying poses keep their span). Clamping to image bounds
        happens at materialization, so frame-cropped people are unaffected.
        """
        vis = sc >= self.min_kp_score

        def have(idxs):
            return bool(np.any(vis[list(idxs)]))

        def mean_y(idxs):
            ys = [float(kpts[i, 1]) for i in idxs if vis[i]]
            return float(np.mean(ys)) if ys else None

        shoulder_y = mean_y(self.SHOULDER_IDXS)
        hip_y = mean_y(self.HIP_IDXS)
        if shoulder_y is None or hip_y is None:
            return y1, y2
        torso_h = hip_y - shoulder_y
        if torso_h <= 0:
            return y1, y2

        if not have(self.KNEE_IDXS) and not have(self.ANKLE_IDXS):
            y2 = max(y2, hip_y + 1.2 * torso_h)
        elif not have(self.ANKLE_IDXS):
            knee_y = mean_y(self.KNEE_IDXS)
            if knee_y is not None:
                y2 = max(y2, knee_y + 0.8 * torso_h)
        if not have(self.HEAD_IDXS):
            y1 = min(y1, shoulder_y - 0.5 * torso_h)

        return y1, y2

    def _collect_candidates(
        self,
        bgr_region: np.ndarray,
        offset: tuple,
        min_visible: Optional[int] = None,
        from_tile: bool = False,
    ) -> List[tuple]:
        """Run RTMO on a BGR region and return quality-gated candidates with
        bboxes translated into full-image coordinates by `offset` (x, y).

        Candidate tuple: (quality, max_score, visible_count, bbox_xyxy,
        from_tile, kpts_full_coords, scores).
        Tiles are pixel crops, so keypoint coords are already at full-image scale;
        the quality gates (min_kp_score, min_bbox_side) keep their pixel semantics.
        `min_visible` overrides min_visible_kps (tiles use the stricter
        tile_min_visible_kps to suppress skin-patch hallucinations).
        """
        if min_visible is None:
            min_visible = self.min_visible_kps
        if bgr_region.shape[0] < 2 or bgr_region.shape[1] < 2:
            return []
        try:
            keypoints, scores = self.model(bgr_region)
        except Exception as e:
            raise PersonDetectorError(f"RTMO inference failed: {e}") from e

        if keypoints is None or len(keypoints) == 0:
            return []

        ox, oy = float(offset[0]), float(offset[1])
        candidates = []
        for i in range(keypoints.shape[0]):
            kpts = keypoints[i]
            sc = scores[i]
            max_score = float(sc.max())
            visible_mask = sc >= self.min_kp_score
            n_visible = int(visible_mask.sum())

            # Entangled/occluded poses have low *mean* score (many hidden
            # keypoints) but at least one very confident keypoint. Gate on
            # max_score + visible_count rather than mean.
            if max_score < self.min_max_kp_score:
                continue
            if n_visible < min_visible:
                continue

            xs = kpts[visible_mask, 0]
            ys = kpts[visible_mask, 1]
            x1, y1 = float(xs.min()) + ox, float(ys.min()) + oy
            x2, y2 = float(xs.max()) + ox, float(ys.max()) + oy
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            if (x2 - x1) < self.min_bbox_side or (y2 - y1) < self.min_bbox_side:
                continue

            # quality = visible_count (primary) + max_score (tie-breaker)
            quality = n_visible * 10.0 + max_score
            kpts_full = kpts.copy()
            kpts_full[:, 0] += ox
            kpts_full[:, 1] += oy
            candidates.append((quality, max_score, n_visible,
                               (x1, y1, x2, y2), from_tile, kpts_full, sc))
        return candidates

    def _generate_tiles(self, W: int, H: int) -> List[tuple]:
        """Sliding-window tile boxes (x1, y1, x2, y2) of tile_size with
        tile_overlap fractional overlap. The last tile per axis is shifted to
        end exactly at the image edge, so coverage is complete without
        spilling out of bounds. An axis shorter than tile_size yields one
        full-span tile."""
        def starts(length: int) -> List[int]:
            if length <= self.tile_size:
                return [0]
            step = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
            s = list(range(0, length - self.tile_size + 1, step))
            if s[-1] != length - self.tile_size:
                s.append(length - self.tile_size)
            return s

        return [
            (x, y, min(x + self.tile_size, W), min(y + self.tile_size, H))
            for y in starts(H)
            for x in starts(W)
        ]

    def _suppressed_by(
        self,
        cand: tuple,
        kept: List[tuple],
        iou_thr: float,
        containment_thr: float,
    ) -> bool:
        """Return True if `cand` duplicates a kept detection: IoU >= iou_thr,
        OR contained (intersect/cand_area) >= containment_thr AND its keypoints
        lie on the keeper's body. The keypoint test disambiguates the two cases
        a nested bbox can mean — an RTMO NMS fragment of the keeper (keypoints
        coincide with the keeper's: suppress) vs a distinct occluded partner
        whose visible span falls inside the occluder's box (keypoints on a
        different body: keep). Bbox containment alone cannot tell them apart,
        which previously suppressed nested partners silently.
        """
        _, _, _, (cx1, cy1, cx2, cy2), _, c_kpts, c_sc = cand
        cand_area = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
        for k in kept:
            _, _, _, (kx1, ky1, kx2, ky2), _, k_kpts, k_sc = k
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue
            kept_area = max(1e-6, (kx2 - kx1) * (ky2 - ky1))
            iou = inter / (cand_area + kept_area - inter)
            if iou >= iou_thr:
                return True
            containment = inter / cand_area
            if containment >= containment_thr:
                if self._shared_kp_fraction(
                    c_kpts, c_sc, k_kpts, k_sc, (kx1, ky1, kx2, ky2)
                ) >= 0.5:
                    return True
                # Chimeric clusters (keypoints straddling two adjacent bodies)
                # also share few keypoints with the keeper — the revival
                # privilege additionally requires a person-shaped span.
                # Observed FP: a 97x790 sliver (aspect 8:1) between two bodies.
                bw, bh = cx2 - cx1, cy2 - cy1
                if max(bw, bh) / max(1e-6, min(bw, bh)) > 4.5:
                    return True
        return False

    def _shared_kp_fraction(
        self,
        c_kpts: np.ndarray,
        c_sc: np.ndarray,
        k_kpts: np.ndarray,
        k_sc: np.ndarray,
        k_box: tuple,
    ) -> float:
        """Fraction of the candidate's visible keypoints that coincide with a
        visible keeper keypoint (within 5% of the keeper's bbox diagonal).
        Duplicate NMS fragments score near 1.0 (same body, near-identical
        coordinates); a distinct nested partner scores near 0.0."""
        c_vis = c_kpts[c_sc >= self.min_kp_score][:, :2]
        k_vis = k_kpts[k_sc >= self.min_kp_score][:, :2]
        if len(c_vis) == 0 or len(k_vis) == 0:
            return 1.0  # no signal — fall back to legacy containment behavior
        kx1, ky1, kx2, ky2 = k_box
        radius = 0.05 * float(np.hypot(kx2 - kx1, ky2 - ky1))
        dists = np.linalg.norm(c_vis[:, None, :] - k_vis[None, :, :], axis=2)
        nearest = dists.min(axis=1)
        return float((nearest <= radius).mean())

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
            logger.info("RTMO model unloaded from memory")
