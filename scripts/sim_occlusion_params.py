#!/usr/bin/env python3
"""
sim_occlusion_params.py — measure how vis-trust exponent, occlusion-flag
weight, and the masked-search gate change real pair comparisons, BEFORE
committing to a re-extract.

Samples N query poses, takes each one's top-K L2 neighbors (realistic match
candidates), then re-extracts both sides' features under a parameter grid and
reports per combo:
  - median mutually-valid dims (of 52)
  - % of pairs falling to the unmasked-L2 fallback (< min_valid_overlap)
  - median valid LEG dims (of 17 leg-related dims)
  - occlusion-flag share of masked divergence

Usage: DB_PROFILE=irl venv/bin/python3 scripts/sim_occlusion_params.py [N=50]
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import src.core._torch_patch  # noqa: F401
from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, GeometricFeatures
from src.intelligence.similarity_engine import SimilarityEngine
from src.core.geometric_feature_extractor import GeometricFeatureExtractor

LEG_DIMS = [4, 5, 6, 7, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31]
FLAG_DIMS = list(range(45, 52))
MIN_VALID_OVERLAP = 12

EXPONENTS = [2.0, 1.5, 1.0]
FLAG_WEIGHTS = [0.25, 0.15, 0.10]
GATES = [0.35, 0.30]

rng = np.random.default_rng(42)


class TunableExtractor(GeometricFeatureExtractor):
    """Extractor with a configurable visibility-trust exponent."""

    vis_trust_exponent = 2.0

    def _get_keypoint_confidence(self, keypoints, indices):
        confidences = []
        for idx in indices:
            if self._current_visibility is not None:
                vis = float(self._current_visibility[idx])
                conf = keypoints[idx, 2]
                vis_trust = (min(vis, 2.0) / 2.0) ** self.vis_trust_exponent
                if self.use_occluded_keypoints:
                    if vis >= 1.0 and conf >= self.confidence_threshold:
                        confidences.append(conf * vis_trust)
                    else:
                        return 0.0
                else:
                    if vis >= 1.5 and conf >= self.confidence_threshold:
                        confidences.append(conf * vis_trust)
                    else:
                        return 0.0
            else:
                conf = keypoints[idx, 2]
                if conf >= self.confidence_threshold:
                    confidences.append(conf)
                else:
                    return 0.0
        return float(np.min(confidences)) if confidences else 0.0


def load_pose(storage, pose_id):
    with storage.session_scope() as session:
        row = session.query(PoseDetection).filter(PoseDetection.id == pose_id).first()
        if row is None or row.keypoint_visibility is None:
            return None
        visibility = np.array(row.keypoint_visibility, dtype=np.float32)
        ns = SimpleNamespace(
            keypoints=np.array(row.keypoints, dtype=np.float32).reshape(133, 3),
            visibility=visibility,
            bbox=np.array(row.bbox, dtype=np.float32) if row.bbox else np.zeros(4, dtype=np.float32),
            overall_confidence=float(row.overall_confidence),
            person_id=row.person_id,
        )
        ns.count_visible = lambda v=visibility: int((v >= 1.5).sum())
        ns.count_occluded = lambda v=visibility: int(((v >= 1.0) & (v < 1.5)).sum())
        return ns


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    storage = StorageManager(settings.DATABASE_URL)
    engine = SimilarityEngine(storage, database_profile=settings.DB_PROFILE)
    if not engine.load_index():
        sys.exit("ERROR: index not loadable")

    # Sample query poses and collect realistic candidate pairs (top-5 neighbors)
    with storage.session_scope() as session:
        ids = [r[0] for r in session.query(GeometricFeatures.pose_id).all()]
    sample = rng.choice(len(ids), size=min(n_queries, len(ids)), replace=False)
    pairs = []
    for i in sample:
        qid = ids[int(i)]
        with storage.session_scope() as session:
            feat = session.query(GeometricFeatures).filter(GeometricFeatures.pose_id == qid).first()
            if feat is None or feat.feature_confidence is None:
                continue
            vec = np.array(feat.feature_vector, dtype=np.float32)
            conf = np.array(feat.feature_confidence, dtype=np.float32)
        engine.clear_search_cache()
        res = engine.search_by_feature(vec, query_confidence=conf, k=6, min_confidence=0.0,
                                       deduplicate_images=True, exclude_pose_id=str(qid))
        import uuid as uuid_mod
        for r in res[:5]:
            pairs.append((qid, uuid_mod.UUID(r["pose_id"])))
    print(f"Sampled {len(pairs)} realistic near-neighbor pairs from {n_queries} queries\n")

    # Cache duck-typed poses
    poses = {}
    for a, b in pairs:
        for pid in (a, b):
            if pid not in poses:
                poses[pid] = load_pose(storage, pid)
    pairs = [(a, b) for a, b in pairs if poses[a] is not None and poses[b] is not None]

    print(f"{'exp':>4s} {'flagW':>6s} {'gate':>5s} | {'med dims':>8s} {'fallback%':>9s} "
          f"{'med leg dims':>12s} {'flag share%':>11s}")
    print("-" * 70)

    for exponent in EXPONENTS:
        for flag_w in FLAG_WEIGHTS:
            extractor = TunableExtractor()
            extractor.vis_trust_exponent = exponent
            extractor.OCCLUSION_FLAG_WEIGHT = flag_w

            feats = {}
            for pid, pose in poses.items():
                f = extractor.extract(pose)
                feats[pid] = (np.array(f.feature_vector, dtype=np.float32),
                              np.array(f.feature_confidence, dtype=np.float32))

            for gate in GATES:
                stats = {"all": {"dims": [], "legs": [], "flags": [], "fb": 0, "n": 0},
                         "occluded": {"dims": [], "legs": [], "flags": [], "fb": 0, "n": 0}}
                for a, b in pairs:
                    qv, qc = feats[a]
                    cv, cc = feats[b]
                    # Occluded-tail pair: either side has >= 2 occluded (vis==1)
                    # body keypoints — the pairs the user's complaints live in.
                    occ = (int((poses[a].visibility[:17] == 1).sum()) >= 2
                           or int((poses[b].visibility[:17] == 1).sum()) >= 2)
                    buckets = ["all", "occluded"] if occ else ["all"]

                    valid = (qc >= gate) & (cc >= gate)
                    vc = int(valid.sum())
                    sq = (qv - cv) ** 2
                    flag_valid = np.zeros(52, dtype=bool)
                    flag_valid[FLAG_DIMS] = True
                    for bk in buckets:
                        s = stats[bk]
                        s["n"] += 1
                        s["dims"].append(vc)
                        s["legs"].append(int(valid[LEG_DIMS].sum()))
                        if vc < MIN_VALID_OVERLAP:
                            s["fb"] += 1
                            continue
                        total = float(sq[valid].sum()) or 1e-9
                        s["flags"].append(100.0 * float(sq[valid & flag_valid].sum()) / total)

                for bk in ("all", "occluded"):
                    s = stats[bk]
                    if not s["n"]:
                        continue
                    print(f"{exponent:4.1f} {flag_w:6.2f} {gate:5.2f} | "
                          f"{np.median(s['dims']):8.0f} {100.0 * s['fb'] / s['n']:9.1f} "
                          f"{np.median(s['legs']):12.0f} "
                          f"{np.mean(s['flags']) if s['flags'] else 0:11.1f}"
                          f"   [{bk} n={s['n']}]")
        print()


if __name__ == "__main__":
    main()
