"""
Re-extract geometric features for every stored pose and rebuild the FAISS index.

Run after any change to GeometricFeatureExtractor's vector semantics (the
2026-06-10 batch: per-block normalization, signed body angles, occlusion-flag
weighting, visibility-scaled confidence, symmetry/height-fallback fixes) —
stored vectors are otherwise incompatible with newly-extracted queries.

Legacy confidence normalization: ensemble-era rows stored [0,2] visibility in
the keypoint confidence channel (fixed in ensemble_detector on 2026-06-10).
The real scores were discarded at detection time and cannot be recovered, so
those rows get the visibility/2 approximation (0, 0.5, 1.0). Exact confidences
arrive only with a full re-detection pass.

Scope: geometric_features rows + the geometric FAISS index. FusedFeatures rows
are NOT touched — their visual half predates the letterbox transform and needs
re-detection from images to be meaningful; the live SEARCH_FEATURE_MODE
('geometric') never reads them.

Usage:
    DB_PROFILE=irl venv/bin/python3 scripts/reextract_features.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.core.pose_detector import PoseResult
from src.core.geometric_feature_extractor import GeometricFeatureExtractor
from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures
from src.intelligence.similarity_engine import SimilarityEngine

BATCH = 1000


def to_pose_result(row) -> PoseResult:
    kp = np.array(row.keypoints, dtype=np.float32).reshape(133, 3)

    if row.keypoint_visibility is not None:
        vis = np.array(row.keypoint_visibility, dtype=np.int64)
    else:
        # Derive with the same binning pose_detector uses on raw scores
        conf = kp[:, 2]
        vis = np.where(conf >= 0.5, 2, np.where(conf >= 0.1, 1, 0)).astype(np.int64)

    # Legacy ensemble rows: confidence channel held [0,2] visibility
    if float(kp[:, 2].max()) > 1.001:
        kp[:, 2] = vis.astype(np.float32) / 2.0

    if row.bbox is not None and len(row.bbox) == 4:
        bbox = np.array(row.bbox, dtype=np.float64)
    else:
        confident = kp[kp[:, 2] >= 0.3]
        if confident.shape[0] >= 2:
            x0, y0 = confident[:, 0].min(), confident[:, 1].min()
            bbox = np.array([x0, y0,
                             confident[:, 0].max() - x0,
                             confident[:, 1].max() - y0], dtype=np.float64)
        else:
            bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    return PoseResult(
        keypoints=kp,
        visibility=vis,
        bbox=bbox,
        overall_confidence=float(np.clip(row.overall_confidence, 0.0, 1.0)),
        person_id=int(row.person_id),
    )


def main():
    profile = os.environ.get("DB_PROFILE", "irl")
    storage = StorageManager()
    extractor = GeometricFeatureExtractor()

    with storage.session_scope() as session:
        total = session.query(GeometricFeatures).count()
    print(f"[{profile}] re-extracting {total} poses", file=sys.stderr)

    done, legacy, errors = 0, 0, 0
    last_id = None
    while True:
        with storage.session_scope() as session:
            q = (
                session.query(PoseDetection, GeometricFeatures)
                .join(GeometricFeatures, GeometricFeatures.pose_id == PoseDetection.id)
                .order_by(PoseDetection.id)
            )
            if last_id is not None:
                q = q.filter(PoseDetection.id > last_id)
            rows = q.limit(BATCH).all()
            if not rows:
                break

            for pose_row, feat_row in rows:
                last_id = pose_row.id
                try:
                    kp_raw = np.array(pose_row.keypoints, dtype=np.float32)
                    if float(kp_raw[2::3].max()) > 1.001:
                        legacy += 1
                    pose = to_pose_result(pose_row)
                    feats = extractor.extract(pose)
                    feat_row.feature_vector = feats.feature_vector.tolist()
                    feat_row.feature_confidence = feats.feature_confidence.tolist()
                    feat_row.joint_angles = feats.joint_angles
                    feat_row.limb_ratios = feats.limb_ratios
                    feat_row.body_angles = feats.body_angles
                    feat_row.symmetry_scores = feats.symmetry_scores
                    feat_row.occlusion_pattern = feats.occlusion_pattern.tolist()
                except Exception as e:
                    errors += 1
                    print(f"  ERROR pose {pose_row.id}: {e}", file=sys.stderr)
            done += len(rows)
        print(f"[{profile}] {done}/{total} ({legacy} legacy-conf rows)", file=sys.stderr)

    print(f"[{profile}] extraction complete: {done} poses, {legacy} legacy rows "
          f"normalized, {errors} errors", file=sys.stderr)

    print(f"[{profile}] rebuilding FAISS index...", file=sys.stderr)
    engine = SimilarityEngine(storage, database_profile=profile)
    engine.build_index(force_rebuild=True)
    n = engine.index.ntotal if engine.index is not None else 0
    print(f"[{profile}] index rebuilt: {n} vectors at {engine.index_path}", file=sys.stderr)
    print(f"RESULT {profile}: poses={done} legacy={legacy} errors={errors} index={n}")


if __name__ == "__main__":
    main()
