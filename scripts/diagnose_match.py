#!/usr/bin/env python3
"""
diagnose_match.py — explain why a photo does (or doesn't) match a query pose.

Given a QUERY and a CANDIDATE (image paths in the library, or pose UUIDs),
recomputes exactly what the live search computes, stage by stage, and reports
where the candidate gets lost: never indexed, no stored poses, confidence
masking, below the similarity floor, plausibility demotion, OKS demotion,
dedup shadowing, excluded folder, or missing file.

Each diagnosed pair is a labeled example — append it to eval pairs for the
retrieval benchmark.

Usage:
  DB_PROFILE=irl venv/bin/python3 scripts/diagnose_match.py QUERY CANDIDATE \
      [--min-similarity 0.5] [--flip] [--show-dims]

  QUERY / CANDIDATE: absolute image path, a unique filename substring, or a
  pose UUID. For multi-person images the highest-confidence pose is used for
  the query and every candidate pose is analyzed.

Run with the same DB_PROFILE as the app.
"""

import argparse
import math
import sys
import uuid as uuid_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import src.core._torch_patch  # noqa: F401  (must precede mmengine imports)
from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.storage.models import Image, PoseDetection, GeometricFeatures
from src.intelligence.similarity_engine import SimilarityEngine

# Live search parameters (mirrors swift_bridge defaults / the app's sliders)
LIVE_MIN_FEATURE_CONFIDENCE = 0.35
LIVE_MIN_VALID_OVERLAP = 12

# 52-dim layout (geometric_feature_extractor.py construction order)
DIM_NAMES = [
    (0, "left_elbow_angle", "joint-angles"), (1, "right_elbow_angle", "joint-angles"),
    (2, "left_shoulder_angle", "joint-angles"), (3, "right_shoulder_angle", "joint-angles"),
    (4, "left_hip_angle", "joint-angles"), (5, "right_hip_angle", "joint-angles"),
    (6, "left_knee_angle", "joint-angles"), (7, "right_knee_angle", "joint-angles"),
    (8, "left_armpit_angle", "joint-angles"), (9, "right_armpit_angle", "joint-angles"),
    (10, "left_leg_spread", "joint-angles"), (11, "right_leg_spread", "joint-angles"),
    (12, "left_upper_arm_ratio", "limb-ratios"), (13, "right_upper_arm_ratio", "limb-ratios"),
    (14, "left_forearm_ratio", "limb-ratios"), (15, "right_forearm_ratio", "limb-ratios"),
    (16, "left_thigh_ratio", "limb-ratios"), (17, "right_thigh_ratio", "limb-ratios"),
    (18, "left_shin_ratio", "limb-ratios"), (19, "right_shin_ratio", "limb-ratios"),
    (20, "shoulder_width_ratio", "limb-ratios"), (21, "hip_width_ratio", "limb-ratios"),
    (22, "torso_lean", "orientation"), (23, "head_tilt", "orientation"),
    (24, "left_upper_arm_angle", "orientation"), (25, "right_upper_arm_angle", "orientation"),
    (26, "left_forearm_angle", "orientation"), (27, "right_forearm_angle", "orientation"),
    (28, "left_thigh_angle", "orientation"), (29, "right_thigh_angle", "orientation"),
    (30, "left_shin_angle", "orientation"), (31, "right_shin_angle", "orientation"),
    (32, "shoulder_line_angle", "orientation"), (33, "hip_line_angle", "orientation"),
    (34, "left_arm_spread", "orientation"), (35, "right_arm_spread", "orientation"),
    (36, "body_twist[DEAD:always-masked]", "orientation"),
    (37, "elbow_symmetry", "symmetry"), (38, "shoulder_symmetry", "symmetry"),
    (39, "hip_symmetry", "symmetry"), (40, "knee_symmetry", "symmetry"),
    (41, "upper_arm_len_symmetry", "symmetry"), (42, "forearm_len_symmetry", "symmetry"),
    (43, "thigh_len_symmetry", "symmetry"), (44, "shin_len_symmetry", "symmetry"),
    (45, "head_visible", "occlusion-flags"), (46, "left_arm_visible", "occlusion-flags"),
    (47, "right_arm_visible", "occlusion-flags"), (48, "torso_visible", "occlusion-flags"),
    (49, "left_leg_visible", "occlusion-flags"), (50, "right_leg_visible", "occlusion-flags"),
    (51, "feet_visible", "occlusion-flags"),
]


def hr(title=""):
    print("\n" + "=" * 26 + f" {title} " + "=" * max(0, 26 - len(title)) if title else "=" * 60)


class PoseRecord:
    def __init__(self, pose_row, image_row, feat_row):
        self.pose_id = str(pose_row.id)
        self.person_id = pose_row.person_id
        self.confidence = float(pose_row.overall_confidence)
        self.keypoints = np.array(pose_row.keypoints, dtype=np.float32).reshape(133, 3)
        self.bbox = list(pose_row.bbox) if pose_row.bbox else None
        self.is_corrected = bool(pose_row.is_corrected)
        self.image_path = image_row.file_path
        self.image_id = str(image_row.id)
        self.vector = np.array(feat_row.feature_vector, dtype=np.float32)
        self.conf = (np.array(feat_row.feature_confidence, dtype=np.float32)
                     if feat_row.feature_confidence is not None else np.ones(52, dtype=np.float32))
        self.pose_row = pose_row
        self.feat_row = feat_row


def resolve(storage, token):
    """Resolve a CLI token to (image_row_or_None, [PoseRecord]) with diagnostics."""
    with storage.session_scope() as session:
        session.expire_on_commit = False
        # Pose UUID?
        try:
            pid = uuid_mod.UUID(token)
            pose = session.query(PoseDetection).filter(PoseDetection.id == pid).first()
            if pose is None:
                sys.exit(f"ERROR: no pose with id {token}")
            image = session.query(Image).filter(Image.id == pose.image_id).first()
            feat = session.query(GeometricFeatures).filter(GeometricFeatures.pose_id == pid).first()
            if feat is None:
                sys.exit(f"ERROR: pose {token} has no geometric features (re-extraction needed?)")
            return image, [PoseRecord(pose, image, feat)]
        except ValueError:
            pass

        # Exact path, then unique substring match
        image = session.query(Image).filter(Image.file_path == token).first()
        if image is None:
            matches = session.query(Image).filter(Image.file_path.like(f"%{token}%")).limit(5).all()
            if len(matches) == 1:
                image = matches[0]
            elif len(matches) > 1:
                print(f"ERROR: '{token}' matches multiple library photos:")
                for m in matches:
                    print(f"  {m.file_path}")
                sys.exit(1)

        if image is None:
            return None, []

        poses = session.query(PoseDetection).filter(PoseDetection.image_id == image.id).all()
        records = []
        for p in poses:
            feat = session.query(GeometricFeatures).filter(GeometricFeatures.pose_id == p.id).first()
            if feat is not None:
                records.append(PoseRecord(p, image, feat))
        records.sort(key=lambda r: -r.confidence)
        return image, records


def group_blame(q, c):
    """Per-group divergence over mutually valid dims + masked-out listing."""
    valid = (q.conf >= LIVE_MIN_FEATURE_CONFIDENCE) & (c.conf >= LIVE_MIN_FEATURE_CONFIDENCE)
    groups = {}
    for idx, name, group in DIM_NAMES:
        g = groups.setdefault(group, {"sqdiff": 0.0, "valid": 0, "masked": []})
        if valid[idx]:
            g["sqdiff"] += float((q.vector[idx] - c.vector[idx]) ** 2)
            g["valid"] += 1
        else:
            g["masked"].append(name)
    return valid, groups


def analyze_pair(engine, q, c, floor, flip_on, show_dims):
    print(f"\nCandidate pose {c.pose_id[:8]} (person {c.person_id}, det conf {c.confidence:.2f})"
          + (" [CORRECTED]" if c.is_corrected else ""))

    valid, groups = group_blame(q, c)
    valid_count = int(valid.sum())

    # --- masked distance exactly as the live search computes it
    dist, vc = engine._compute_masked_distance(
        q.vector, q.conf, c.vector, c.conf,
        min_confidence=LIVE_MIN_FEATURE_CONFIDENCE, min_valid_overlap=LIVE_MIN_VALID_OVERLAP)

    if math.isinf(dist):
        raw = float(np.sum((q.vector - c.vector) ** 2))  # FAISS squared-L2 fallback
        fallback_dist = math.sqrt(max(raw, 0.0))
        base_l2 = engine._distance_to_similarity(fallback_dist)
        print(f"  !! MASKED FALLBACK: only {vc} mutually-confident dims (< {LIVE_MIN_VALID_OVERLAP}).")
        print(f"     Search falls back to UNMASKED distance over all 52 dims — including")
        print(f"     defaults/garbage in the masked ones. Fallback base similarity: {base_l2:.3f}")
    else:
        base_l2 = engine._distance_to_similarity(dist)
        print(f"  Masked L2: {dist:.4f} over {vc}/52 mutually-confident dims -> base similarity {base_l2:.3f}")

    # --- OKS leg (would rank results only if enabled AND flip search is OFF)
    oks_enabled = bool(settings.ENABLE_OKS_METRIC)
    oks_sim = None
    if not oks_enabled:
        print("  OKS: disabled (.env ENABLE_OKS_METRIC=false, deliberate) — L2 ranking applies")
    elif q.bbox and c.bbox:
        _, oks_sim = engine._compute_oks_distance(
            q.keypoints, c.keypoints, np.array(q.bbox, dtype=np.float32),
            np.array(c.bbox, dtype=np.float32), min_confidence=LIVE_MIN_FEATURE_CONFIDENCE)
        print(f"  OKS similarity: {oks_sim:.3f}"
              + ("  (ACTIVE: ranks results — flip search off)" if not flip_on
                 else "  (INACTIVE: flip search ON strips OKS — L2 ranking applies)"))
    else:
        print("  OKS: unavailable (missing stored bbox)")

    # --- plausibility boost (applied BEFORE the similarity floor)
    plaus = engine._compute_plausibility_score(c.pose_row, c.feat_row)
    boost = 0.8 + 0.2 * plaus * settings.PLAUSIBILITY_WEIGHT
    print(f"  Plausibility {plaus:.2f} -> boost x{boost:.3f} (applied BEFORE your floor)")

    use_oks = oks_enabled and oks_sim is not None and not flip_on
    base = oks_sim if use_oks else base_l2
    final = base * boost
    mode = "OKS" if use_oks else "L2"
    verdict = "PASSES" if final >= floor else "FAILS"
    margin = final - floor
    print(f"  => PROJECTED SCORE ({mode} mode): {final:.3f} vs floor {floor:.2f} -> {verdict} ({margin:+.3f})")
    if base >= floor > final:
        print(f"  !! PLAUSIBILITY DEMOTION: base {base:.3f} clears your floor; the x{boost:.3f}"
              f" boost pushes it under. Raise PLAUSIBILITY tolerance or lower the floor.")

    # --- blame table
    print("  Divergence by feature group (masked dims excluded):")
    total_sq = sum(g["sqdiff"] for g in groups.values()) or 1e-9
    for group, g in sorted(groups.items(), key=lambda kv: -kv[1]["sqdiff"]):
        share = 100.0 * g["sqdiff"] / total_sq
        masked = f", masked: {len(g['masked'])}" if g["masked"] else ""
        print(f"    {group:18s} {share:5.1f}% of divergence ({g['valid']} valid dims{masked})")
    all_masked = [n for g in groups.values() for n in g["masked"]]
    if all_masked:
        print(f"  Masked out ({len(all_masked)}): {', '.join(all_masked)}")

    if show_dims:
        print(f"  {'dim':28s} {'query':>7s} {'cand':>7s} {'|d|':>6s} {'qconf':>6s} {'cconf':>6s} ok")
        for idx, name, _ in DIM_NAMES:
            d = abs(float(q.vector[idx] - c.vector[idx]))
            print(f"  {name:28s} {q.vector[idx]:7.3f} {c.vector[idx]:7.3f} {d:6.3f}"
                  f" {q.conf[idx]:6.2f} {c.conf[idx]:6.2f} {'Y' if valid[idx] else '-'}")

    return final


def rank_check(engine, storage, q, c_records, floor, flip_on):
    """Run the real engine search and locate the candidate's poses in it."""
    hr("LIVE RANK CHECK")
    engine.clear_search_cache()
    kwargs = dict(
        query_confidence=q.conf, k=engine.index.ntotal if engine.index is not None else 100000,
        min_confidence=0.0, min_feature_confidence=LIVE_MIN_FEATURE_CONFIDENCE,
        min_valid_overlap=LIVE_MIN_VALID_OVERLAP, deduplicate_images=False,
        min_similarity=0.0, exclude_pose_id=q.pose_id,
    )
    if settings.ENABLE_OKS_METRIC and not flip_on and q.bbox is not None:
        kwargs["query_keypoints"] = q.keypoints
        kwargs["query_bbox"] = np.array(q.bbox, dtype=np.float32)
    results = engine.search_by_feature(q.vector, **kwargs)

    by_pose = {r["pose_id"]: r for r in results}
    cand_ids = {r.pose_id for r in c_records}
    cand_image_id = c_records[0].image_id if c_records else None

    found_any = False
    for cr in c_records:
        r = by_pose.get(cr.pose_id)
        if r is None:
            print(f"  pose {cr.pose_id[:8]}: ABSENT from full unfiltered ranking — killed by a hard filter:")
            p = Path(cr.image_path)
            excluded = [f['folder_path'] for f in storage.get_excluded_folders()]
            if excluded and storage.is_path_excluded(cr.image_path, excluded):
                print(f"    -> EXCLUDED FOLDER: {cr.image_path}")
            elif not p.exists():
                print(f"    -> FILE MISSING ON DISK: {cr.image_path}")
            else:
                print(f"    -> likely OKS drop (no mutually-confident keypoints) or stale index entry")
            continue
        found_any = True
        n_above = sum(1 for x in results if x["similarity_score"] > r["similarity_score"])
        print(f"  pose {cr.pose_id[:8]}: rank #{r['rank']} of {len(results)},"
              f" score {r['similarity_score']:.3f} (base {r.get('base_similarity', 0):.3f})"
              f" -> {'VISIBLE' if r['similarity_score'] >= floor else 'BELOW FLOOR'} at floor {floor:.2f}")
        # dedup shadowing: would another pose of the same image outrank it?
        shadow = [x for x in results
                  if x.get("image_id") == cand_image_id and x["pose_id"] not in cand_ids
                  and x["similarity_score"] > r["similarity_score"]]
        if shadow:
            print(f"    !! DEDUP SHADOW: with 'show multiple people' OFF, pose {shadow[0]['pose_id'][:8]}"
                  f" (score {shadow[0]['similarity_score']:.3f}) represents this image instead.")
        _ = n_above
    if not found_any and not c_records:
        print("  (no candidate poses to rank)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="image path / filename fragment / pose UUID")
    ap.add_argument("candidate", help="image path / filename fragment / pose UUID")
    ap.add_argument("--min-similarity", type=float, default=0.5, help="your slider floor (default 0.5)")
    ap.add_argument("--flip", action="store_true", help="diagnose as if 'Include flipped poses' is ON")
    ap.add_argument("--show-dims", action="store_true", help="print the full 52-dim comparison table")
    args = ap.parse_args()

    storage = StorageManager(settings.DATABASE_URL)
    # database_profile must be passed EXPLICITLY for the profile-aware index
    # path (data/indices/{profile}/) — mirrors the bridge's construction.
    engine = SimilarityEngine(storage, database_profile=settings.DB_PROFILE)
    if not engine.load_index():
        sys.exit("ERROR: could not load FAISS index for this profile")

    q_image, q_records = resolve(storage, args.query)
    c_image, c_records = resolve(storage, args.candidate)

    hr("RESOLUTION")
    for label, image, records, token in (("QUERY", q_image, q_records, args.query),
                                         ("CANDIDATE", c_image, c_records, args.candidate)):
        if image is None:
            print(f"  {label}: '{token}' is NOT IN THE LIBRARY.")
            print(f"    -> It can never appear in results. Run Indexed Photos > Update on its folder.")
            sys.exit(0 if label == "CANDIDATE" else 1)
        flag = ""
        if not records:
            flag = "  !! ZERO POSES STORED (pose-less photo: detection found nothing here)"
        print(f"  {label}: {image.file_path}")
        print(f"    {len(records)} pose(s) with features{flag}")

    if not q_records:
        sys.exit("\nQUERY photo has no stored poses — re-detect it (skip-off Update on its folder).")
    if not c_records:
        print("\nVERDICT: candidate photo is in the library but detection stored NO poses for it.")
        print("It cannot match anything. Re-run detection on it: Indexed Photos > Update with")
        print("'Skip photos already in library' OFF (scoped re-detect), then re-check.")
        sys.exit(0)

    q = q_records[0]
    print(f"\n  Using query pose {q.pose_id[:8]} (person {q.person_id}, det conf {q.confidence:.2f})")
    if args.flip:
        print("  NOTE: flip mode — OKS re-ranking is stripped by flip search; L2 ordering applies.")

    hr("PAIR ANALYSIS")
    for cr in c_records:
        analyze_pair(engine, q, cr, args.min_similarity, args.flip, args.show_dims)

    rank_check(engine, storage, q, c_records, args.min_similarity, args.flip)

    print("\nTip: append this pair to scripts/eval_pairs.json to grow the retrieval benchmark.")


if __name__ == "__main__":
    main()
