"""
Focused multi-person pipeline test over a directory of images.

Loads the ensemble once, runs detect_multi_person() on each image, and prints
one line per person: person_id, confidence, bbox, keypoint spread. Much faster
than diagnose_multiperson.py since it skips YOLO-at-multiple-thresholds probes
and per-crop-per-model probes.

Usage:
    venv/bin/python scripts/test_entangled_corpus.py
    venv/bin/python scripts/test_entangled_corpus.py path/to/specific/image.jpg
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
import torch

import src.core._torch_patch as _tp
_tp._apply_mmengine_patch()

from src.core.ensemble_detector import EnsembleDetector, EnsembleConfig


def kp_summary(kpts: np.ndarray, vis: np.ndarray) -> str:
    visible = vis >= 1
    n = int(visible.sum())
    if n == 0:
        return "no visible kps"
    xs, ys = kpts[visible, 0], kpts[visible, 1]
    return (f"kps={n}/{len(kpts)} "
            f"x=[{xs.min():.0f},{xs.max():.0f}]({xs.max()-xs.min():.0f}px) "
            f"y=[{ys.min():.0f},{ys.max():.0f}]({ys.max()-ys.min():.0f}px)")


def build_ensemble(device: str) -> EnsembleDetector:
    models_dir = PROJECT_ROOT / "data" / "models"
    cfg = EnsembleConfig(
        models=[
            {"config": str(models_dir / "rtmw-l_8xb320-270e_cocktail14-384x288.py"),
             "checkpoint": str(models_dir / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"),
             "weight": 1.0},
            {"config": str(models_dir / "rtmw-x_8xb320-270e_cocktail14-384x288.py"),
             "checkpoint": str(models_dir / "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"),
             "weight": 1.0},
        ],
        fusion_method="confidence_weighted",
    )
    return EnsembleDetector(cfg, device=device)


def test_image(ensemble: EnsembleDetector, path: Path) -> None:
    print(f"\n== {path.name}")
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"   FAILED: cv2.imread returned None")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    print(f"   {W}x{H}")

    try:
        poses = ensemble.detect_multi_person(rgb)
    except Exception as e:
        print(f"   RAISED: {e}")
        return

    if not poses:
        print(f"   -> 0 poses")
        return

    print(f"   -> {len(poses)} pose(s)")
    for p in poses:
        bx, by, bw, bh = p.bbox.tolist()
        print(f"      id={p.person_id} conf={p.overall_confidence:.3f} "
              f"bbox=[{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}] "
              f"{kp_summary(p.keypoints, p.visibility)}")


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        photos = PROJECT_ROOT / "test-photos"
        paths = sorted([
            *photos.glob("*.jpg"),
            *photos.glob("*.jpeg"),
            *photos.glob("*.png"),
        ])
    if not paths:
        print("No images found")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  {len(paths)} image(s)")
    ensemble = build_ensemble(device)
    print(f"Person detector: {type(ensemble.person_detector).__name__}")

    for p in paths:
        test_image(ensemble, p)


if __name__ == "__main__":
    main()
