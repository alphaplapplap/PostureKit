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

import cv2
from ultralytics import YOLO
import torch

from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


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
        try:
            keypoints, scores = self.model(bgr)
        except Exception as e:
            raise PersonDetectorError(f"RTMO inference failed: {e}") from e

        H, W = image.shape[:2]
        detections: List[PersonDetection] = []

        if keypoints is None or len(keypoints) == 0:
            logger.info("RTMO detected 0 people")
            return detections

        # Collect candidate detections that pass the per-person quality bar.
        candidates = []  # (quality, max_score, visible_count, bbox_xyxy, raw_idx)
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
            if n_visible < self.min_visible_kps:
                continue

            xs = kpts[visible_mask, 0]
            ys = kpts[visible_mask, 1]
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            if (x2 - x1) < self.min_bbox_side or (y2 - y1) < self.min_bbox_side:
                continue

            # quality = visible_count (primary) + max_score (tie-breaker)
            quality = n_visible * 10.0 + max_score
            candidates.append((quality, max_score, n_visible,
                               (x1, y1, x2, y2), i))

        # Greedy IoU/containment dedup: sort best-first, drop any whose bbox
        # overlaps too much with one already kept.
        candidates.sort(key=lambda c: c[0], reverse=True)
        kept: List[tuple] = []
        for cand in candidates:
            _, _, _, bx, _ = cand
            if self._overlaps_kept(bx, [k[3] for k in kept],
                                   self.iou_dedup_thr, self.containment_thr):
                continue
            kept.append(cand)

        # Materialize PersonDetection with padded bbox + crop.
        for new_id, (_, max_score, _, (x1, y1, x2, y2), _) in enumerate(kept):
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

    @staticmethod
    def _overlaps_kept(
        cand: tuple,
        kept_boxes: List[tuple],
        iou_thr: float,
        containment_thr: float,
    ) -> bool:
        """Return True if `cand` (x1,y1,x2,y2) overlaps any box in `kept_boxes`
        by IoU >= iou_thr OR is contained (intersect/cand_area) >= containment_thr.
        """
        cx1, cy1, cx2, cy2 = cand
        cand_area = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
        for kx1, ky1, kx2, ky2 in kept_boxes:
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
                return True
        return False

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
            logger.info("RTMO model unloaded from memory")
