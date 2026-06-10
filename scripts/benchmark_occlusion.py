"""
Occlusion robustness benchmark.

Measures self-similarity of poses under synthetic occlusion (flat gray boxes)
using the production pipeline: RTMO person detection -> RTMW-L + RTMW-X
ensemble (confidence_weighted fusion) -> 52-dim geometric features.

For each single-person test image:
  baseline -> LEGS variant (gray over bottom 40% of person bbox)
           -> SIDE variant (gray over left half of person bbox)
and reports keypoint/dim validity, plain-L2 self-similarity exp(-d/2), masked
self-similarity (mutually-valid dims at conf >= 0.35, min overlap 12 — exact
semantics of SimilarityEngine._compute_masked_distance), and the squared-L2
decomposition (occlusion-flag dims 45-51 vs imputed/low-conf dims vs valid).

A perfectly occlusion-robust engine scores an occluded variant ~100% similar
to its own baseline under masked distance.

Usage: venv/bin/python3 scripts/benchmark_occlusion.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.core._torch_patch  # noqa: F401  (must precede mmpose imports)

import glob
import numpy as np
import cv2

from src.core.ensemble_detector import EnsembleDetector, EnsembleConfig
from src.core.geometric_feature_extractor import GeometricFeatureExtractor

SINGLE_PERSON = [
    "test-photos/1116194662.jpg",
    "test-photos/151217076.jpg",
    "test-photos/5543268.jpg",
    "test-photos/802472233.jpg",
]
MULTI_PERSON = "test-photos/1470901390.jpg"

MIN_KP_CONF = 0.3       # GeometricFeatureExtractor keypoint gate
MIN_DIM_CONF = 0.35     # masked-distance dimension gate (live default)
MIN_VALID_OVERLAP = 12  # masked-distance overlap floor (live default)
FLAG_DIMS = slice(45, 52)
GRAY = (128, 128, 128)


def build_detector():
    models = []
    for cfg, ckpt in [
        ("rtmw-l_8xb320-270e_cocktail14-384x288.py",
         "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"),
        ("rtmw-x_8xb320-270e_cocktail14-384x288.py",
         "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"),
    ]:
        models.append({
            "config": str(PROJECT_ROOT / "data" / "models" / cfg),
            "checkpoint": str(PROJECT_ROOT / "data" / "models" / ckpt),
            "weight": 1.0,
        })
    config = EnsembleConfig(models=models, fusion_method="confidence_weighted")
    return EnsembleDetector(config, device="mps")


def load_rgb(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def largest_person(poses):
    return max(poses, key=lambda p: float(p.bbox[2]) * float(p.bbox[3]))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_person(poses, ref_bbox):
    """Best-IoU match against the baseline person's bbox."""
    best, best_iou = None, 0.0
    for p in poses:
        iou = bbox_iou(p.bbox, ref_bbox)
        if iou > best_iou:
            best, best_iou = p, iou
    return best, best_iou


def occlude(rgb, bbox, mode):
    x, y, w, h = [int(v) for v in bbox]
    out = rgb.copy()
    if mode == "legs":
        out[y + int(h * 0.6): y + h, x: x + w] = GRAY
    elif mode == "side":
        out[y: y + h, x: x + int(w * 0.5)] = GRAY
    return out


def masked_similarity(v0, c0, v1, c1):
    valid = (c0 >= MIN_DIM_CONF) & (c1 >= MIN_DIM_CONF)
    n = int(valid.sum())
    if n < MIN_VALID_OVERLAP:
        return float("nan"), n
    d = float(np.linalg.norm(v0[valid] - v1[valid]))
    return float(np.exp(-d / 2.0)), n


def analyze(pose, extractor):
    feats = extractor.extract(pose)
    kp = pose.keypoints
    return {
        "feats": feats,
        "body_kps": int((kp[:17, 2] >= MIN_KP_CONF).sum()),
        "all_kps": int((kp[:, 2] >= MIN_KP_CONF).sum()),
        "vis2": int((pose.visibility == 2).sum()),
        "valid_dims": int((feats.feature_confidence >= MIN_DIM_CONF).sum()),
    }


def main():
    detector = build_detector()
    extractor = GeometricFeatureExtractor()
    rows = []

    for pattern in SINGLE_PERSON:
        paths = glob.glob(pattern)
        if not paths:
            print(f"SKIP (missing): {pattern}", file=sys.stderr)
            continue
        path = paths[0]
        rgb = load_rgb(path)
        base_poses = detector.detect_multi_person(rgb)
        if not base_poses:
            print(f"SKIP (no baseline detection): {path}", file=sys.stderr)
            continue
        base = largest_person(base_poses)
        b = analyze(base, extractor)
        v0, c0 = b["feats"].feature_vector, b["feats"].feature_confidence

        for mode in ("legs", "side"):
            occ_rgb = occlude(rgb, base.bbox, mode)
            occ_poses = detector.detect_multi_person(occ_rgb)
            name = Path(path).name[:22]
            if not occ_poses:
                rows.append((name, mode, "PERSON LOST (0 detections)"))
                continue
            person, iou = match_person(occ_poses, base.bbox)
            if person is None or iou < 0.05:
                rows.append((name, mode, f"PERSON LOST (best IoU {iou:.2f}, "
                                         f"count {len(base_poses)}->{len(occ_poses)})"))
                continue
            o = analyze(person, extractor)
            v1, c1 = o["feats"].feature_vector, o["feats"].feature_confidence

            d = float(np.linalg.norm(v0 - v1))
            plain_sim = float(np.exp(-d / 2.0))
            masked_sim, n_valid = masked_similarity(v0, c0, v1, c1)

            sq = (v0 - v1) ** 2
            total_sq = float(sq.sum()) or 1e-12
            flags_pct = 100.0 * float(sq[FLAG_DIMS].sum()) / total_sq
            lowconf = (np.minimum(c0, c1) < MIN_DIM_CONF)
            lowconf_pct = 100.0 * float(sq[lowconf].sum()) / total_sq

            rows.append((
                name, mode,
                f"kps {b['all_kps']}->{o['all_kps']} vis2 {b['vis2']}->{o['vis2']} "
                f"dims {b['valid_dims']}->{o['valid_dims']} | "
                f"plain {plain_sim:.1%} masked {masked_sim:.1%} (n={n_valid}) | "
                f"sq%: flags {flags_pct:.0f} lowconf {lowconf_pct:.0f} "
                f"valid {100 - flags_pct - lowconf_pct:.0f} | "
                f"count {len(base_poses)}->{len(occ_poses)} iou {iou:.2f}"
            ))

    print("\n=== OCCLUSION BENCHMARK ===")
    for name, mode, result in rows:
        print(f"{name:<24} {mode:<5} {result}")

    # Multi-person: occlude 50% of one person, does the count drop?
    paths = glob.glob(MULTI_PERSON)
    if paths:
        rgb = load_rgb(paths[0])
        base_poses = detector.detect_multi_person(rgb)
        if len(base_poses) >= 2:
            target = sorted(base_poses,
                            key=lambda p: float(p.bbox[2]) * float(p.bbox[3]))[-2]
            occ_poses = detector.detect_multi_person(occlude(rgb, target.bbox, "side"))
            _, iou = match_person(occ_poses, target.bbox)
            print(f"\nMULTI-PERSON {Path(paths[0]).name}: count "
                  f"{len(base_poses)}->{len(occ_poses)}, occluded-person best IoU {iou:.2f}")


if __name__ == "__main__":
    main()
