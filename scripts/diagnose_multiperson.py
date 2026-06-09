"""
Multi-person ensemble detection diagnostic.

Traces every stage of the multi-person pipeline and prints what each component
actually returns. Use to identify where the pipeline drops people / produces
scrambled keypoints.

Usage:
    cd /Users/linuxbabe/Hardware-Aware/PostureKit
    python scripts/diagnose_multiperson.py [image_path] [image_path2 ...]

If no paths passed, defaults to test-photos/*.jpg.
"""
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
import torch

# Apply the same torch.load monkey-patch Swift uses (via swift_bridge) so the
# Cocktail14 checkpoints load under PyTorch 2.8's weights_only=True default.
import src.core._torch_patch as _torch_patch  # noqa: E402,F401
_torch_patch._apply_mmengine_patch()


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def kpstats(keypoints: np.ndarray, visibility: np.ndarray) -> str:
    """Summarize keypoint array so we can tell if they're actually on the person."""
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    visible = visibility >= 1
    n_vis = int(visible.sum())
    if n_vis == 0:
        return "no visible keypoints"
    return (
        f"visible={n_vis}/{len(keypoints)} "
        f"x_range=[{xs[visible].min():.0f}, {xs[visible].max():.0f}] "
        f"y_range=[{ys[visible].min():.0f}, {ys[visible].max():.0f}] "
        f"x_spread={(xs[visible].max() - xs[visible].min()):.0f}px "
        f"y_spread={(ys[visible].max() - ys[visible].min()):.0f}px"
    )


def yolo_probe(image_rgb: np.ndarray, conf: float, device: str, model: str = "yolov8n.pt") -> list:
    """Run YOLO at a given confidence threshold and print the raw results."""
    from src.core.person_detector import YOLOPersonDetector

    det = YOLOPersonDetector(
        model_name=model,
        confidence_threshold=conf,
        device=device,
    )
    people = det.detect_people(image_rgb, return_crops=True)
    print(f"YOLO[{model}] conf={conf:.2f}: detected {len(people)} person(s)")
    for i, p in enumerate(people):
        x, y, w, h = p.bbox.tolist()
        print(
            f"  [{i}] conf={p.confidence:.3f} "
            f"bbox=[x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f}] "
            f"crop_shape={p.crop.shape if p.crop is not None else None}"
        )
    return people


def build_ensemble(device: str):
    """Build the exact ensemble Swift asks for: rtmw-l + rtmw-x + confidence_weighted."""
    from src.core.ensemble_detector import EnsembleDetector, EnsembleConfig

    models_dir = PROJECT_ROOT / "data" / "models"
    specs = [
        {
            "config": str(models_dir / "rtmw-l_8xb320-270e_cocktail14-384x288.py"),
            "checkpoint": str(models_dir / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"),
            "weight": 1.0,
        },
        {
            "config": str(models_dir / "rtmw-x_8xb320-270e_cocktail14-384x288.py"),
            "checkpoint": str(models_dir / "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"),
            "weight": 1.0,
        },
    ]
    # Verify files exist first — print clear error instead of stacktrace
    for s in specs:
        for k in ("config", "checkpoint"):
            if not Path(s[k]).exists():
                print(f"MISSING: {s[k]}")
                raise FileNotFoundError(s[k])

    cfg = EnsembleConfig(models=specs, fusion_method="confidence_weighted")
    return EnsembleDetector(cfg, device=device)


def diagnose(image_path: Path, device: str = "mps") -> None:
    banner(f"DIAGNOSING {image_path}")

    section("Image load")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"cv2.imread returned None for {image_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"Shape: {rgb.shape}  W={w} H={h}  dtype={rgb.dtype}")

    section("YOLO probes — yolov8n (nano, 6MB, current)")
    people_015 = yolo_probe(rgb, conf=0.15, device=device, model="yolov8n.pt")
    yolo_probe(rgb, conf=0.50, device=device, model="yolov8n.pt")
    yolo_probe(rgb, conf=0.05, device=device, model="yolov8n.pt")

    section("YOLO probes — yolov8x (extra-large, 130MB)")
    yolo_probe(rgb, conf=0.15, device=device, model="yolov8x.pt")
    yolo_probe(rgb, conf=0.30, device=device, model="yolov8x.pt")
    people_x_05 = yolo_probe(rgb, conf=0.05, device=device, model="yolov8x.pt")

    section("YOLO probes — yolo11x (latest, most capable)")
    yolo_probe(rgb, conf=0.15, device=device, model="yolo11x.pt")
    yolo_probe(rgb, conf=0.30, device=device, model="yolo11x.pt")
    yolo_probe(rgb, conf=0.05, device=device, model="yolo11x.pt")

    section("EnsembleDetector construction")
    try:
        ensemble = build_ensemble(device=device)
    except Exception as e:
        print(f"ENSEMBLE CONSTRUCTION FAILED: {e}")
        traceback.print_exc()
        return
    print(f"ensemble.person_detector is None? {ensemble.person_detector is None}")
    print(f"ensemble.detectors count: {len(ensemble.detectors)}")
    print(f"ensemble.config.fusion_method: {ensemble.config.fusion_method}")

    section("EnsembleDetector.detect_multi_person() — full pipeline")
    try:
        poses = ensemble.detect_multi_person(rgb)
    except Exception as e:
        print(f"detect_multi_person RAISED: {e}")
        traceback.print_exc()
        poses = []
    print(f"Returned {len(poses)} pose(s)")
    for i, p in enumerate(poses):
        bx, by, bw, bh = p.bbox.tolist()
        print(
            f"  [{i}] person_id={p.person_id} overall_conf={p.overall_confidence:.3f} "
            f"bbox=[x={bx:.0f} y={by:.0f} w={bw:.0f} h={bh:.0f}]"
        )
        print(f"       {kpstats(p.keypoints, p.visibility)}")

    section("Per-crop, per-model detection (what fusion is actually seeing)")
    if not people_015:
        print("No YOLO people at 0.15, skipping per-crop probe")
    else:
        for idx, person in enumerate(people_015):
            print(f"\nPerson {idx}: bbox={person.bbox.tolist()}  conf={person.confidence:.3f}")
            for mi, det_spec in enumerate(ensemble.detectors, 1):
                try:
                    pose = det_spec["detector"].detect_from_crop(person.crop, person.bbox)
                except Exception as e:
                    print(f"  Model {mi}: detect_from_crop RAISED: {e}")
                    continue
                if pose is None:
                    print(f"  Model {mi}: detect_from_crop returned None")
                else:
                    print(
                        f"  Model {mi}: conf={pose.overall_confidence:.3f}  "
                        f"{kpstats(pose.keypoints, pose.visibility)}"
                    )

    section("EnsembleDetector.detect() — whole-image fallback (no YOLO)")
    try:
        poses_single = ensemble.detect(rgb)
    except Exception as e:
        print(f"detect RAISED: {e}")
        traceback.print_exc()
        poses_single = []
    print(f"Whole-image detect() returned {len(poses_single)} pose(s)")
    for i, p in enumerate(poses_single):
        print(
            f"  [{i}] person_id={p.person_id} overall_conf={p.overall_confidence:.3f}  "
            f"{kpstats(p.keypoints, p.visibility)}"
        )


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        paths = sorted((PROJECT_ROOT / "test-photos").glob("*.jpg"))
    if not paths:
        print("No images found. Pass paths on the command line.")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    for p in paths:
        try:
            diagnose(p, device=device)
        except Exception as e:
            print(f"\nFATAL for {p}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
