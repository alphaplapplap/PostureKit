#!/usr/bin/env python3
"""
benchmark_retrieval.py — self-supervised retrieval ground truth, no manual labels.

The eval_pairs.json benchmark needs hand-judged pairs and currently has ~2 of
them. This harness manufactures objective ground truth from the indexed corpus
itself, so every extractor or scoring change gets a single regression number
without any human labeling.

For each of K deterministically-sampled indexed images (seed-controlled) it runs
two self-retrieval probes against the LIVE engine the app uses (same FAISS index,
same masking params, same plausibility/OKS settings):

  (a) OCCLUDED SELF-RETRIEVAL — render a legs/side occlusion with
      benchmark_occlusion.occlude(), re-run the production detector +
      GeometricFeatureExtractor on the occluded image, query the engine, and
      check whether the SOURCE image's stored pose is retrieved @1 / @10. Also
      records the score margin of the source over the best non-source distractor.
      A perfectly occlusion-robust engine returns the source pose at rank 1.

  (b) MIRRORED POSITIVE — horizontally flip the source pose's stored feature
      vector and run flip search (search_with_flip), expecting the original pose
      back. A left/right-symmetric engine returns the source at rank 1.

Aggregates recall@1, recall@10, MRR, and false-admit-rate-at-the-default-floor
(fraction of occluded probes whose top non-source result clears the floor while
the source pose itself does not — i.e. a confident wrong answer), printed as ONE
JSON line to stdout. All progress and diagnostics go to stderr.

Deterministic given --seed: the same seed samples the same images every run.

Usage:
  DB_PROFILE=irl venv/bin/python3 scripts/benchmark_retrieval.py \
      [--k 300] [--seed 42] [--floor 0.5] [--mode legs|side|both] \
      [--no-occlusion] [--no-mirror] [--json-only]

  --k          number of source images to sample (default 300, capped ~1000)
  --seed       RNG seed for deterministic sampling (default 42)
  --floor      similarity floor for false-admit accounting (default 0.5)
  --mode       occlusion variant(s) to render (default both)
  --no-*       skip a probe family
  --json-only  suppress the human-readable stderr summary table

NOTE: a full K=300-1000 run drives the dual-RTMW ensemble per image on MPS and
takes a while; Wave 4 owns the full pass. K=3 proves the harness end-to-end.
"""

import argparse
import json
import sys
import time
import uuid as uuid_mod
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

import src.core._torch_patch  # noqa: F401  (must precede mmpose imports)
from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.storage.models import Image, PoseDetection, GeometricFeatures
from src.intelligence.similarity_engine import SimilarityEngine
from src.core.geometric_feature_extractor import GeometricFeatureExtractor

# Reuse the occlusion synthesis + production-detector helpers so the benchmark
# detects exactly what the occlusion benchmark and the app's ingest do.
from benchmark_occlusion import (  # noqa: E402
    build_detector, load_rgb, largest_person, match_person, occlude,
)
# Live masking params (mirror the app's sliders / swift_bridge defaults).
from diagnose_match import LIVE_MIN_FEATURE_CONFIDENCE, LIVE_MIN_VALID_OVERLAP  # noqa: E402

MAX_K = 1000


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def sample_sources(storage, k, seed):
    """Deterministically pick k indexed images that have a stored pose+features
    and an on-disk file. Returns a list of dicts describing each source."""
    # Pull the full candidate id list once, sort for determinism, then a seeded
    # permutation picks the sample — identical for a given (seed, corpus).
    with storage.session_scope() as session:
        rows = session.query(GeometricFeatures.pose_id).all()
    pose_ids = sorted(str(r[0]) for r in rows)
    if not pose_ids:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pose_ids))

    sources = []
    seen_images = set()
    for idx in order:
        pid = pose_ids[int(idx)]
        with storage.session_scope() as session:
            session.expire_on_commit = False
            pose = session.query(PoseDetection).filter(
                PoseDetection.id == uuid_mod.UUID(pid)).first()
            if pose is None or pose.bbox is None:
                continue
            image = session.query(Image).filter(Image.id == pose.image_id).first()
            if image is None:
                continue
            feat = session.query(GeometricFeatures).filter(
                GeometricFeatures.pose_id == uuid_mod.UUID(pid)).first()
            if feat is None or feat.feature_vector is None:
                continue
            image_id = str(image.id)
            file_path = image.file_path
            # bbox stored as [x, y, w, h]
            bbox = [float(v) for v in pose.bbox]
            vector = np.array(feat.feature_vector, dtype=np.float32)
            confidence = (np.array(feat.feature_confidence, dtype=np.float32)
                          if feat.feature_confidence is not None
                          else np.ones(52, dtype=np.float32))
            # All stored pose_ids for this image — any of them is a valid
            # ground-truth "self" hit (multi-person images, re-detection drift).
            image_pose_ids = {
                str(r[0]) for r in session.query(PoseDetection.id).filter(
                    PoseDetection.image_id == image.id).all()
            }
        # Dedup by image so one photo can't dominate the sample.
        if image_id in seen_images:
            continue
        if not Path(file_path).exists():
            continue
        seen_images.add(image_id)
        sources.append({
            "pose_id": pid,
            "image_id": image_id,
            "file_path": file_path,
            "bbox": bbox,
            "vector": vector,
            "confidence": confidence,
            "self_pose_ids": image_pose_ids,
            "self_image_ids": {image_id},
        })
        if len(sources) >= k:
            break
    return sources


def rank_of_self(results, self_pose_ids, self_image_ids):
    """Best (lowest) rank at which any pose/image of the source appears, plus
    that result's similarity, and the best score among NON-self results."""
    self_rank = None
    self_score = 0.0
    best_distractor_score = 0.0
    for r in results:
        is_self = (r["pose_id"] in self_pose_ids
                   or r.get("image_id") in self_image_ids)
        if is_self:
            if self_rank is None:
                self_rank = r["rank"]
                self_score = r["similarity_score"]
        elif best_distractor_score == 0.0:
            # results are best-first, so the first non-self is the top distractor
            best_distractor_score = r["similarity_score"]
    return self_rank, self_score, best_distractor_score


def detect_occluded(detector, extractor, rgb, bbox, mode):
    """Render the occlusion, re-detect, match the source person by bbox IoU, and
    extract its 52-dim feature vector. Returns (vector, confidence) or None."""
    occ_rgb = occlude(rgb, bbox, mode)
    poses = detector.detect_multi_person(occ_rgb)
    if not poses:
        return None
    person, iou = match_person(poses, bbox)
    if person is None or iou < 0.05:
        return None
    feats = extractor.extract(person)
    return (np.array(feats.feature_vector, dtype=np.float32),
            np.array(feats.feature_confidence, dtype=np.float32))


def query_engine(engine, vector, confidence):
    """Full-index search with live masking params, identical to eval_pairs.py.

    The SOURCE pose is the ground-truth target, so it is NOT excluded: the query
    here is a freshly re-detected OCCLUDED vector, not the stored clean vector,
    so retrieving the source is a real occlusion-robustness test, not a trivial
    identity match."""
    engine.clear_search_cache()
    return engine.search_by_feature(
        vector, query_confidence=confidence,
        k=engine.index.ntotal, min_confidence=0.0,
        min_feature_confidence=LIVE_MIN_FEATURE_CONFIDENCE,
        min_valid_overlap=LIVE_MIN_VALID_OVERLAP,
        deduplicate_images=False, min_similarity=0.0,
        exclude_pose_id=None)


def query_engine_mirror(engine, extractor, vector, confidence):
    """Mirrored-positive probe: build the source pose's horizontally-flipped
    feature vector (and confidence) with GeometricFeatureExtractor.flip_features
    — the same left/right swap the engine's flip search uses — and query the
    index with it. A left/right-symmetric retrieval system should return the
    ORIGINAL (unflipped) source pose for a mirrored query.

    This is the deterministic, detection-free form of the brief's "flip the
    image, run flip-search, expect the original back": it isolates symmetry from
    re-detection noise and avoids identity-match dominance (the flipped query
    differs from every stored vector, including the source's unflipped one)."""
    flipped_vec = extractor.flip_features(np.asarray(vector, dtype=np.float32))
    # feature_confidence shares the exact 52-dim left/right layout, so the same
    # swap mirrors the confidence mask to align with the flipped vector.
    flipped_conf = extractor.flip_features(np.asarray(confidence, dtype=np.float32))
    engine.clear_search_cache()
    return engine.search_by_feature(
        flipped_vec, query_confidence=flipped_conf,
        k=engine.index.ntotal, min_confidence=0.0,
        min_feature_confidence=LIVE_MIN_FEATURE_CONFIDENCE,
        min_valid_overlap=LIVE_MIN_VALID_OVERLAP,
        deduplicate_images=False, min_similarity=0.0,
        exclude_pose_id=None)


def aggregate(records):
    """recall@1, recall@10, MRR over probes where the source was retrievable."""
    n = len(records)
    if n == 0:
        return {"n": 0, "recall@1": None, "recall@10": None,
                "mrr": None, "retrieved": 0}
    r1 = sum(1 for x in records if x["rank"] is not None and x["rank"] <= 1)
    r10 = sum(1 for x in records if x["rank"] is not None and x["rank"] <= 10)
    retrieved = sum(1 for x in records if x["rank"] is not None)
    mrr = sum((1.0 / x["rank"]) for x in records if x["rank"] is not None) / n
    return {
        "n": n,
        "retrieved": retrieved,
        "recall@1": round(r1 / n, 4),
        "recall@10": round(r10 / n, 4),
        "mrr": round(mrr, 4),
        "median_self_margin": round(float(np.median(
            [x["margin"] for x in records if x["margin"] is not None])), 4)
            if any(x["margin"] is not None for x in records) else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=300,
                    help="number of source images to sample (default 300)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic sampling (default 42)")
    ap.add_argument("--floor", type=float, default=0.5,
                    help="similarity floor for false-admit accounting (0.5)")
    ap.add_argument("--mode", choices=["legs", "side", "both"], default="both",
                    help="occlusion variant(s) to render (default both)")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="skip the occluded self-retrieval probe")
    ap.add_argument("--no-mirror", action="store_true",
                    help="skip the mirrored-positive probe")
    ap.add_argument("--json-only", action="store_true",
                    help="suppress the human-readable stderr summary table")
    args = ap.parse_args()

    k = max(1, min(args.k, MAX_K))
    modes = ["legs", "side"] if args.mode == "both" else [args.mode]

    t0 = time.time()
    storage = StorageManager(settings.DATABASE_URL)
    engine = SimilarityEngine(storage, database_profile=settings.DB_PROFILE)
    if not engine.load_index():
        print(json.dumps({"status": "error",
                          "error": "could not load FAISS index for this profile"}))
        return
    ntotal = engine.index.ntotal if engine.index is not None else 0
    log(f"[setup] profile={settings.DB_PROFILE} index_ntotal={ntotal} "
        f"k={k} seed={args.seed} floor={args.floor} modes={modes}")

    sources = sample_sources(storage, k, args.seed)
    log(f"[setup] sampled {len(sources)} source image(s) in {time.time()-t0:.1f}s")
    if not sources:
        print(json.dumps({"status": "error",
                          "error": "no sampleable indexed images for this profile"}))
        return

    # The extractor is cheap and needed by both probes (occlusion: feature
    # extraction from re-detected poses; mirror: flip_features on stored
    # vectors). The ensemble detector is heavy and only built for the occlusion
    # probe, which re-detects images.
    detector = None
    extractor = GeometricFeatureExtractor()
    if not args.no_occlusion:
        log("[setup] building ensemble detector (RTMW-L + RTMW-X) ...")
        detector = build_detector()
        log(f"[setup] detector ready in {time.time()-t0:.1f}s")

    occ_records = []      # per (source, mode) occluded probe
    mirror_records = []   # per source mirror probe
    occ_person_lost = 0
    occ_false_admits = 0

    for i, src in enumerate(sources, 1):
        log(f"[{i}/{len(sources)}] {Path(src['file_path']).name[:40]} "
            f"pose={src['pose_id'][:8]}")

        # --- (a) occluded self-retrieval ------------------------------------
        if not args.no_occlusion:
            try:
                rgb = load_rgb(src["file_path"])
            except Exception as exc:  # unreadable/corrupt image -> skip probe
                log(f"      occlusion: unreadable image ({exc}) — skip")
                rgb = None
            if rgb is not None:
                for mode in modes:
                    probe = detect_occluded(detector, extractor, rgb,
                                            src["bbox"], mode)
                    if probe is None:
                        occ_person_lost += 1
                        occ_records.append({"rank": None, "margin": None,
                                            "mode": mode, "lost": True})
                        log(f"      occlusion[{mode}]: PERSON LOST")
                        continue
                    qvec, qconf = probe
                    results = query_engine(engine, qvec, qconf)
                    rank, self_score, distractor = rank_of_self(
                        results, src["self_pose_ids"], src["self_image_ids"])
                    margin = (self_score - distractor) if rank is not None else None
                    # false admit: a confident wrong top result while the true
                    # source pose failed to clear the floor (or wasn't retrieved)
                    if distractor >= args.floor and (
                            rank is None or self_score < args.floor):
                        occ_false_admits += 1
                    occ_records.append({"rank": rank, "margin": margin,
                                        "mode": mode, "lost": False})
                    if margin is not None:
                        log(f"      occlusion[{mode}]: self rank={rank} "
                            f"score={self_score:.3f} margin={margin:.3f}")
                    else:
                        log(f"      occlusion[{mode}]: self rank={rank} "
                            f"(not retrieved)")

        # --- (b) mirrored positive -----------------------------------------
        if not args.no_mirror:
            results = query_engine_mirror(
                engine, extractor, src["vector"], src["confidence"])
            rank, self_score, distractor = rank_of_self(
                results, src["self_pose_ids"], src["self_image_ids"])
            margin = (self_score - distractor) if rank is not None else None
            mirror_records.append({"rank": rank, "margin": margin})
            if margin is not None:
                log(f"      mirror: self rank={rank} "
                    f"score={self_score:.3f} margin={margin:.3f}")
            else:
                log(f"      mirror: self rank={rank} (not retrieved)")

    out = {
        "status": "ok",
        "profile": settings.DB_PROFILE,
        "index_ntotal": ntotal,
        "k_requested": k,
        "k_sampled": len(sources),
        "seed": args.seed,
        "floor": args.floor,
        "modes": modes,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if not args.no_occlusion:
        agg = aggregate(occ_records)
        agg["person_lost"] = occ_person_lost
        denom = len(occ_records) or 1
        agg["false_admit_rate_at_floor"] = round(occ_false_admits / denom, 4)
        out["occluded_self_retrieval"] = agg
    if not args.no_mirror:
        out["mirrored_positive"] = aggregate(mirror_records)

    if not args.json_only:
        log("\n=== RETRIEVAL BENCHMARK SUMMARY ===")
        if "occluded_self_retrieval" in out:
            o = out["occluded_self_retrieval"]
            log(f"occluded:  n={o['n']} recall@1={o['recall@1']} "
                f"recall@10={o['recall@10']} mrr={o['mrr']} "
                f"person_lost={o['person_lost']} "
                f"false_admit@floor={o['false_admit_rate_at_floor']}")
        if "mirrored_positive" in out:
            m = out["mirrored_positive"]
            log(f"mirror:    n={m['n']} recall@1={m['recall@1']} "
                f"recall@10={m['recall@10']} mrr={m['mrr']}")

    # ONE final JSON line on stdout — stdout discipline.
    print(json.dumps(out))


if __name__ == "__main__":
    main()
