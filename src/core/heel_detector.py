"""
Heel detection via zero-shot fashion-CLIP on a pose-localized foot crop.

Validated in scripts/eval_heel_detection.py on the `irl` profile (ground truth =
the `P2 Legs and Heels` folder): marqo-fashionCLIP on a foot crop separates heels
from non-heels at precision 0.91 @0.7 / 0.98 @0.9 (per person). Recall is bounded
by foot RESOLUTION, not the classifier — only ~46% of heel photos have a foot
crop >1% of frame; on that recoverable subset recall is high. See the
project_heel_detection memory for the full measurement.

Design notes:
- Mirrors VisualFeatureExtractor's lazy-load + device pattern (src/core/
  visual_feature_extractor.py). Model loads once per instance (one detector per
  subprocess, like the NudeNet _body_detector), so there is no cross-instance
  weakref cache here — it would complicate holding the coupled preprocess/text
  features for marginal benefit.
- open_clip is imported LAZILY (it is a runtime dep of this feature only), so
  importing src.core.* never fails just because open_clip is absent.
- STDOUT DISCIPLINE: logs go to stderr via logging_config; never print() here.
"""
from __future__ import annotations

import os
from typing import Optional, List, Tuple

import numpy as np

from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# COCO-WholeBody foot layout (src/core/models.py:206-213); keypoints are (133,3) [x,y,conf].
FEET = {
    "L": dict(knee=13, ankle=15, heel=19, big_toe=17, small_toe=18),
    "R": dict(knee=14, ankle=16, heel=22, big_toe=20, small_toe=21),
}

# Prompt set covers "any and all heels" per the feature request.
POS_PROMPTS = [
    "a high-heeled shoe", "a stiletto heel", "a high-heeled pump",
    "a heeled sandal", "a wedge heel", "a heeled ankle boot", "a platform heel",
]
NEG_PROMPTS = [
    "a sneaker", "a flat shoe", "a flat sandal", "a flat boot",
    "a bare foot", "a sock", "a flat ballet shoe",
]

_HF_ID = {"marqo-fashionclip": "Marqo/marqo-fashionCLIP",
          "marqo-fashionsiglip": "Marqo/marqo-fashionSigLIP"}


class HeelDetectorError(Exception):
    """Raised when the heel classifier cannot be constructed."""


def _valid(kp, i, min_conf):
    return float(kp[i, 2]) >= min_conf and not (kp[i, 0] == 0 and kp[i, 1] == 0)


def _dist(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def foot_boxes(keypoints: np.ndarray, img_w: int, img_h: int,
               min_conf: float = 0.3, pad: float = 0.35,
               max_frac: float = 0.45) -> List[Tuple[str, np.ndarray]]:
    """Ankle-anchored, hallucination-gated pixel boxes for each visible foot.

    RTMW foot keypoints (17-22) hallucinate when feet aren't visible, so: require
    a confident ankle anchor, scale a plausible foot reach by shin length, only
    include heel/toe points within that reach, and reject boxes > max_frac of the
    image. Returns list of (side, np.array([x1,y1,x2,y2] float)); empty if none.
    (Same logic as scripts/eval_heel_detection.foot_box, kept in sync.)
    """
    kp = keypoints
    diag = (img_w ** 2 + img_h ** 2) ** 0.5
    out: List[Tuple[str, np.ndarray]] = []
    for side, f in FEET.items():
        if not _valid(kp, f["ankle"], min_conf):
            continue
        a = (float(kp[f["ankle"], 0]), float(kp[f["ankle"], 1]))
        shin = _dist(a, (float(kp[f["knee"], 0]), float(kp[f["knee"], 1]))) \
            if _valid(kp, f["knee"], min_conf) else 0.0
        reach = max(1.8 * shin, 0.10 * diag) if shin > 1 else 0.14 * diag
        pts = [a]
        for j in (f["heel"], f["big_toe"], f["small_toe"]):
            if _valid(kp, j, min_conf):
                p = (float(kp[j, 0]), float(kp[j, 1]))
                if _dist(p, a) <= reach:
                    pts.append(p)
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        w, h = x1 - x0, y1 - y0
        padx, pady = w * pad + 12, h * pad + 12
        bx0 = max(0, int(x0 - padx))
        by0 = max(0, int(y0 - pady))
        bx1 = min(img_w, int(x1 + padx))
        by1 = min(img_h, int(y1 + pady))
        if bx1 - bx0 < 10 or by1 - by0 < 10:
            continue
        if (bx1 - bx0) * (by1 - by0) > max_frac * img_w * img_h:
            continue
        out.append((side, np.array([bx0, by0, bx1, by1], dtype=float)))
    return out


class HeelDetector:
    """Zero-shot fashion-CLIP heel classifier over pose-localized foot crops."""

    def __init__(self, device: Optional[str] = None, model: Optional[str] = None,
                 threshold: Optional[float] = None, min_foot_conf: Optional[float] = None):
        self.device = device or settings.DEVICE
        self.model_name = (model or getattr(settings, "HEEL_MODEL", "marqo-fashionclip")).lower()
        self.threshold = threshold if threshold is not None else getattr(settings, "HEEL_THRESHOLD", 0.7)
        self.min_foot_conf = min_foot_conf if min_foot_conf is not None \
            else getattr(settings, "HEEL_MIN_FOOT_CONF", 0.3)
        self._model = None
        self._device_obj = None
        self._preprocess = None
        self._text = None
        self._logit_scale = 100.0
        self._n_pos = len(POS_PROMPTS)
        logger.info(f"HeelDetector init (model={self.model_name}, device={self.device}, "
                    f"threshold={self.threshold})")

    def _initialize_device(self):
        import torch
        if self.device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                return torch.device("cpu")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # CLIP ops lack some MPS kernels
            return torch.device("mps")
        if self.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            import open_clip
        except ImportError as e:
            raise HeelDetectorError(
                "open_clip_torch is required for heel detection. "
                "Install: venv/bin/pip install open_clip_torch"
            ) from e
        hf = _HF_ID.get(self.model_name, self.model_name)
        self._device_obj = self._initialize_device()
        model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{hf}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf}")
        self._model = model.to(self._device_obj).eval()
        self._preprocess = preprocess
        with torch.inference_mode():
            tokens = tokenizer(POS_PROMPTS + NEG_PROMPTS).to(self._device_obj)
            txt = self._model.encode_text(tokens)
            self._text = txt / txt.norm(dim=-1, keepdim=True)
        self._logit_scale = self._model.logit_scale.exp().item() \
            if hasattr(self._model, "logit_scale") else 100.0
        logger.info(f"HeelDetector loaded '{hf}' on {self._device_obj}")

    def score_crop_rgb(self, crop_rgb: np.ndarray) -> float:
        """Heel probability in [0,1] for an (h,w,3) uint8 RGB foot crop.

        Softmax over the positive+negative prompt set; heel = sum of positive
        probs (the fashionSigLIP native sigmoid head saturated, so we score both
        encoders the same softmax way — see eval notes)."""
        import torch
        from PIL import Image as PILImage
        self._load()
        pil = PILImage.fromarray(np.ascontiguousarray(crop_rgb))
        with torch.inference_mode():
            img = self._preprocess(pil).unsqueeze(0).to(self._device_obj)
            feat = self._model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = (feat @ self._text.T).squeeze(0)
            probs = torch.softmax(logits * self._logit_scale, dim=-1)
            heel = float(probs[:self._n_pos].sum().item())
        return max(0.0, min(1.0, heel))

    def detect(self, image_rgb: np.ndarray, keypoints: np.ndarray,
               min_foot_conf: Optional[float] = None) -> Tuple[float, Optional[np.ndarray]]:
        """Per-person heel verdict: max score over the person's (<=2) foot crops.

        Returns (best_score, best_bbox np.array([x1,y1,x2,y2] float)) or (0.0, None)
        if no foot could be localized. Caller compares best_score to self.threshold."""
        h, w = image_rgb.shape[:2]
        mfc = min_foot_conf if min_foot_conf is not None else self.min_foot_conf
        best_score, best_box = 0.0, None
        for _side, box in foot_boxes(keypoints, w, h, min_conf=mfc):
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            s = self.score_crop_rgb(crop)
            if s > best_score:
                best_score, best_box = s, box
        return best_score, best_box
