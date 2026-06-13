"""
Backfill 576-dim MobileNetV3-Small visual embeddings for every stored pose that
lacks a VisualFeatures row, persisting them per DB_PROFILE.

Finding 36 (visual rerank) blends a stored visual-cosine signal into the top-k of
the geometric search. Today only ~5% of poses carry visual embeddings, and those
predate the letterbox transform (commit 6114e82) so they are not comparable with
freshly extracted query crops. This script extracts the embedding from each
pose's bbox crop — the same convention the live ingest path uses
(swift_bridge._extract_person_features) — batched on MPS, and writes it to the
visual_features table.

Scope: visual_features rows ONLY. Does NOT touch geometric features, does NOT
rebuild the FAISS index, does NOT change any stored vector format — so
scripts/reextract_features.py is NOT required after this runs.

Resumable: by default skips poses that already have a visual_features row, so it
can be re-run after an interruption. Pass --force-all to re-extract every pose
(overwriting existing rows) — use this once to replace the pre-letterbox 5% with
embeddings comparable to the rest of the corpus.

Usage:
    DB_PROFILE=irl venv/bin/python3 scripts/backfill_visual_features.py
    DB_PROFILE=irl venv/bin/python3 scripts/backfill_visual_features.py --force-all
    DB_PROFILE=irl venv/bin/python3 scripts/backfill_visual_features.py --limit 500
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config.settings import settings
from src.core.visual_feature_extractor import (
    VisualFeatureExtractor,
    FeatureExtractionError,
)
from src.storage.storage_manager import StorageManager
from src.storage.models import Image, PoseDetection, VisualFeatures

# Poses pulled from the DB per page (keyset pagination, mirrors reextract_features).
DB_PAGE = 1000
# Crops accumulated before a single batched MobileNetV3 forward pass on MPS.
# Persons after the first in any image amortize per-call dispatch overhead.
VISUAL_BATCH = 32
# Minimum bbox crop size, mirrors _extract_person_features in swift_bridge.
MIN_CROP_PX = 10


def _crop_for_pose(image_rgb: np.ndarray, bbox) -> np.ndarray:
    """Crop the person bbox the same way the live ingest path does.

    Returns the RGB crop, or None if the bbox is missing/degenerate/too small.
    bbox is [x, y, w, h] in pixels (PoseDetection.bbox convention).
    """
    if bbox is None or len(bbox) != 4:
        return None
    h_img, w_img = image_rgb.shape[0], image_rgb.shape[1]
    x, y, w, h = [int(v) for v in bbox]
    # Clamp to image bounds (identical to _extract_person_features)
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= MIN_CROP_PX or h <= MIN_CROP_PX:
        return None
    crop = image_rgb[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return crop


def _load_image_rgb(file_path: str):
    """Decode an image from disk as RGB uint8, or None if unreadable."""
    if not file_path or not Path(file_path).exists():
        return None
    bgr = cv2.imread(str(file_path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _flush_batch(extractor, storage, pending, stats, profile, force_all):
    """Run one batched forward pass over accumulated crops and persist results.

    pending: list of (pose_id, crop_ndarray). Empty list is a no-op.
    On batch failure, retries each crop singly so one bad crop cannot lose a
    whole batch (mirrors the per-image fallback the live path keeps).
    In force_all mode an existing visual_features row is updated in place
    (pose_id is unique), otherwise a new row is inserted.
    """
    if not pending:
        return

    pose_ids = [p for p, _ in pending]
    crops = [c for _, c in pending]

    try:
        feats = extractor.extract_features_batch(crops)
    except FeatureExtractionError as e:
        print(f"[{profile}] batch of {len(crops)} failed ({e}); retrying singly",
              file=sys.stderr)
        feats = []
        for pid, crop in pending:
            try:
                feats.append(extractor.extract(crop))
            except Exception as single_e:
                feats.append(None)
                stats['errors'] += 1
                print(f"  ERROR pose {pid}: {single_e}", file=sys.stderr)

    # Persist in one transaction per flushed batch.
    with storage.session_scope() as session:
        existing = {}
        if force_all:
            present = (
                session.query(VisualFeatures)
                .filter(VisualFeatures.pose_id.in_(pose_ids))
                .all()
            )
            existing = {row.pose_id: row for row in present}
        for pid, vf in zip(pose_ids, feats):
            if vf is None:
                continue
            try:
                vec = vf.feature_vector.tolist()
                row = existing.get(pid) if force_all else None
                if row is not None:
                    row.feature_vector = vec
                    row.model_name = vf.model_name
                    row.normalization = vf.normalization
                else:
                    session.add(VisualFeatures(
                        pose_id=pid,
                        feature_vector=vec,
                        model_name=vf.model_name,
                        normalization=vf.normalization,
                    ))
                stats['stored'] += 1
            except Exception as e:
                stats['errors'] += 1
                print(f"  ERROR persisting pose {pid}: {e}", file=sys.stderr)

    pending.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backfill visual embeddings per DB_PROFILE")
    parser.add_argument('--force-all', action='store_true',
                        help="Re-extract every pose, overwriting existing visual rows "
                             "(use to replace pre-letterbox embeddings).")
    parser.add_argument('--limit', type=int, default=None,
                        help="Process at most this many poses (smoke test).")
    args = parser.parse_args(argv)

    profile = os.environ.get("DB_PROFILE", settings.DB_PROFILE)
    storage = StorageManager(database_profile=profile)

    if settings.VISUAL_MODEL == 'disabled':
        # The extractor still works; warn that ingest has visual off, so this is
        # an explicit backfill for the rerank experiment.
        print(f"[{profile}] note: settings.VISUAL_MODEL=disabled; "
              f"backfilling with {VisualFeatureExtractor.MODEL_NAME} regardless",
              file=sys.stderr)

    extractor = VisualFeatureExtractor(device=settings.DEVICE)

    # Count the work up front (poses needing a visual row).
    with storage.session_scope() as session:
        total_poses = session.query(PoseDetection).count()
        if args.force_all:
            todo = total_poses
        else:
            todo = (
                session.query(PoseDetection.id)
                .outerjoin(VisualFeatures, VisualFeatures.pose_id == PoseDetection.id)
                .filter(VisualFeatures.id.is_(None))
                .count()
            )
    mode = "all (force)" if args.force_all else "missing-only"
    print(f"[{profile}] backfilling visual features: {todo} poses [{mode}] "
          f"of {total_poses} total", file=sys.stderr)

    stats = {'stored': 0, 'errors': 0, 'skipped': 0, 'unreadable': 0}
    processed = 0          # poses pulled from DB (candidates)
    pending = []           # accumulated (pose_id, crop) awaiting a batched forward
    last_id = None

    while True:
        if args.limit is not None and processed >= args.limit:
            break

        with storage.session_scope() as session:
            q = (
                session.query(
                    PoseDetection.id,
                    PoseDetection.image_id,
                    PoseDetection.bbox,
                    Image.file_path,
                    VisualFeatures.id.label('vf_id'),
                )
                .join(Image, Image.id == PoseDetection.image_id)
                .outerjoin(VisualFeatures, VisualFeatures.pose_id == PoseDetection.id)
                .order_by(PoseDetection.id)
            )
            if last_id is not None:
                q = q.filter(PoseDetection.id > last_id)
            rows = q.limit(DB_PAGE).all()

        if not rows:
            break

        # Group this page's poses by image so each source image is decoded once.
        # Poses are ordered by id, not image, so accumulate then process per image.
        page_by_image = {}
        for row in rows:
            last_id = row.id
            processed += 1
            if not args.force_all and row.vf_id is not None:
                stats['skipped'] += 1
                continue
            page_by_image.setdefault((row.image_id, row.file_path), []).append(
                (row.id, row.bbox)
            )
            if args.limit is not None and processed >= args.limit:
                break

        for (image_id, file_path), poses in page_by_image.items():
            image_rgb = _load_image_rgb(file_path)
            if image_rgb is None:
                stats['unreadable'] += 1
                print(f"[{profile}] unreadable image, skipping {len(poses)} pose(s): "
                      f"{file_path}", file=sys.stderr)
                continue
            for pose_id, bbox in poses:
                crop = _crop_for_pose(image_rgb, bbox)
                if crop is None:
                    stats['skipped'] += 1
                    continue
                pending.append((pose_id, crop))
                if len(pending) >= VISUAL_BATCH:
                    _flush_batch(extractor, storage, pending, stats, profile, args.force_all)

        print(f"[{profile}] processed {processed}/{todo} candidates "
              f"(stored={stats['stored']} skipped={stats['skipped']} "
              f"unreadable={stats['unreadable']} errors={stats['errors']})",
              file=sys.stderr)

    # Flush any remaining crops.
    _flush_batch(extractor, storage, pending, stats, profile, args.force_all)

    print(f"[{profile}] backfill complete: stored={stats['stored']} "
          f"skipped={stats['skipped']} unreadable_images={stats['unreadable']} "
          f"errors={stats['errors']}", file=sys.stderr)
    # Final summary line to stdout for any standalone runner (NOT Swift-parsed).
    print(f"RESULT {profile}: stored={stats['stored']} skipped={stats['skipped']} "
          f"unreadable={stats['unreadable']} errors={stats['errors']}")


if __name__ == "__main__":
    main()
