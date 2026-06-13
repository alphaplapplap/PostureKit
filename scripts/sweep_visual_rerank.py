#!/usr/bin/env python3
"""
sweep_visual_rerank.py — does finding #36 (visual rerank) help retrieval?

This is benchmark_retrieval.py's occluded/mirrored self-retrieval, run at MULTIPLE
visual-rerank blend weights from a SINGLE expensive re-detection per probe. The
re-detection + feature extraction (~75s/probe, dominated by the dual-RTMW ensemble)
is the cost; the engine search itself is cheap. So for each sampled probe we:

  1. render the occlusion (benchmark_occlusion.occlude),
  2. re-detect with the production ensemble + match the source person by bbox IoU,
  3. extract the query's 52-dim GEOMETRIC vector + confidence ONCE, AND
  4. crop the matched person from the occluded image and extract its 576-dim
     MobileNetV3-Small VISUAL embedding ONCE (same crop convention as
     scripts/backfill_visual_features.py / the live ingest path),

then query the LIVE engine at EACH weight in --weights (default 0.0, 0.15, 0.3, 0.5)
by monkeypatching settings.ENABLE_VISUAL_RERANK / settings.VISUAL_RERANK_WEIGHT
between calls. search_by_feature reads those settings FRESH per call, so the same
re-detected query is reranked at every weight WITHOUT re-detecting. Weight 0.0 is
the geometric baseline (rerank disabled, the visual embedding is ignored).

For each weight we record the source pose's rank and aggregate recall@1, recall@10,
MRR for the occluded probes (and, as a symmetry control, the mirrored probes).

The MIRRORED probe is a detection-free symmetry control: it flips the source pose's
STORED geometric vector and searches. There is no query image crop for it (nothing
was re-detected), so no query visual embedding exists — supplying the source's own
stored embedding would trivially boost the source. So the mirror probe runs ONCE
with visual rerank off and is reported identically across weights: it isolates
geometric symmetry and confirms the sweep does not perturb the non-visual path.

Output is ONE JSON line on stdout; all progress + diagnostics go to stderr.
Deterministic given --seed (same seed samples the same images every run).

  visual_coverage in the output is the fraction of indexed poses that carried a
  stored VisualFeatures embedding during the sweep (count(VisualFeatures)/ntotal).
  A backfill (scripts/backfill_visual_features.py) may still be filling these in,
  so coverage < 1.0 means the blend only moved candidates that already had one.

Usage:
  DB_PROFILE=irl venv/bin/python3 scripts/sweep_visual_rerank.py \
      [--k 12] [--seed 42] [--weights 0.0,0.15,0.3,0.5] \
      [--mode legs|side|both] [--no-mirror] [--json-only]

  --k        number of source images to sample (default 12, capped ~1000)
  --seed     RNG seed for deterministic sampling (default 42)
  --weights  comma-separated blend weights to sweep (default 0.0,0.15,0.3,0.5)
  --mode     occlusion variant(s) to render (default both)
  --no-mirror suppress the mirrored-positive symmetry control
  --json-only suppress the human-readable stderr summary table
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

# --- IMPORT ORDER (load-bearing) ------------------------------------------
# Mirror benchmark_retrieval.py EXACTLY: _torch_patch first (force
# weights_only=False for mmpose checkpoints), then settings/storage/models,
# then the SimilarityEngine + extractors. benchmark_occlusion (imported below)
# pulls in the ensemble detector builder; building the detector BEFORE faiss is
# fully exercised avoids the faiss+torch libomp double-init abort.
import src.core._torch_patch  # noqa: F401  (must precede mmpose imports)
from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.storage.models import Image, PoseDetection, GeometricFeatures, VisualFeatures
from src.intelligence.similarity_engine import SimilarityEngine
from src.core.geometric_feature_extractor import GeometricFeatureExtractor
from src.core.visual_feature_extractor import VisualFeatureExtractor

# Reuse benchmark_retrieval's harness + benchmark_occlusion's synthesis/detector
# so we sweep over EXACTLY the same probes, detector, and scoring it validates.
from benchmark_occlusion import (  # noqa: E402
    build_detector, load_rgb, match_person, occlude,
)
from benchmark_retrieval import (  # noqa: E402
    sample_sources, rank_of_self, aggregate,
    query_engine, query_engine_mirror,
)
# Crop convention shared with the visual backfill (live ingest path).
from backfill_visual_features import _crop_for_pose  # noqa: E402
# Live masking params (mirror the app's sliders / swift_bridge defaults).
from diagnose_match import LIVE_MIN_FEATURE_CONFIDENCE, LIVE_MIN_VALID_OVERLAP  # noqa: E402

MAX_K = 1000
DEFAULT_WEIGHTS = [0.0, 0.15, 0.3, 0.5]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_weights(raw):
    """Parse a comma-separated weight override into a sorted, deduped float list.

    Clamps to [0, 1]; always includes 0.0 so the geometric baseline is present.
    """
    if not raw:
        weights = list(DEFAULT_WEIGHTS)
    else:
        weights = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            w = float(tok)
            if w < 0.0 or w > 1.0:
                raise ValueError(f"weight out of range [0,1]: {w}")
            weights.append(w)
    if 0.0 not in weights:
        weights.append(0.0)
    # Stable, deterministic order; dedup.
    return sorted(set(round(w, 6) for w in weights))


def detect_occluded_with_visual(detector, geo_extractor, vis_extractor,
                                rgb, bbox, mode):
    """Render the occlusion, re-detect, match the source person by bbox IoU, and
    extract BOTH its 52-dim geometric vector and its 576-dim visual embedding
    from the SAME matched-person crop of the occluded image — once.

    Returns (geo_vec, geo_conf, visual_embedding) or None if the person was lost.
    visual_embedding may be None if the matched crop was degenerate/too small or
    the visual forward pass failed (the geometric probe still proceeds; the
    sweep just can't blend a query embedding for that probe).
    """
    occ_rgb = occlude(rgb, bbox, mode)
    poses = detector.detect_multi_person(occ_rgb)
    if not poses:
        return None
    person, iou = match_person(poses, bbox)
    if person is None or iou < 0.05:
        return None

    feats = geo_extractor.extract(person)
    geo_vec = np.array(feats.feature_vector, dtype=np.float32)
    geo_conf = np.array(feats.feature_confidence, dtype=np.float32)

    # Visual embedding from the MATCHED person's bbox crop of the OCCLUDED image,
    # cropped exactly like the backfill (live ingest convention). person.bbox is
    # [x, y, w, h] in the occluded image's pixel space.
    visual_embedding = None
    try:
        crop = _crop_for_pose(occ_rgb, list(person.bbox))
        if crop is not None:
            vf = vis_extractor.extract(crop)
            visual_embedding = np.asarray(vf.feature_vector, dtype=np.float32)
    except Exception as exc:  # never let a visual hiccup kill the geometric probe
        log(f"      visual-extract failed ({exc}) — probe runs geometric-only")
        visual_embedding = None

    return geo_vec, geo_conf, visual_embedding


def query_engine_at_weight(engine, vector, confidence, weight, visual_embedding):
    """Run the live full-index occlusion query at a single rerank weight.

    weight == 0.0  -> baseline: disable visual rerank entirely (the embedding is
                      ignored even if present), reproducing the geometric path.
    weight  > 0.0  -> enable rerank and pass the query visual embedding; the
                      engine reads ENABLE_VISUAL_RERANK / VISUAL_RERANK_WEIGHT
                      FRESH per call, so monkeypatching here takes effect.

    The cache key already includes the weight + a hash of the embedding, but we
    clear it anyway so weights never alias and timing reflects a real search.
    """
    rerank_on = weight > 0.0 and visual_embedding is not None
    settings.ENABLE_VISUAL_RERANK = bool(rerank_on)
    settings.VISUAL_RERANK_WEIGHT = float(weight)
    engine.clear_search_cache()
    return engine.search_by_feature(
        vector, query_confidence=confidence,
        k=engine.index.ntotal, min_confidence=0.0,
        min_feature_confidence=LIVE_MIN_FEATURE_CONFIDENCE,
        min_valid_overlap=LIVE_MIN_VALID_OVERLAP,
        deduplicate_images=False, min_similarity=0.0,
        exclude_pose_id=None,
        query_visual_embedding=(visual_embedding if rerank_on else None))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=12,
                    help="number of source images to sample (default 12)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic sampling (default 42)")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated blend weights (default 0.0,0.15,0.3,0.5)")
    ap.add_argument("--mode", choices=["legs", "side", "both"], default="both",
                    help="occlusion variant(s) to render (default both)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="suppress the mirrored-positive symmetry control")
    ap.add_argument("--json-only", action="store_true",
                    help="suppress the human-readable stderr summary table")
    args = ap.parse_args()

    k = max(1, min(args.k, MAX_K))
    weights = parse_weights(args.weights)
    modes = ["legs", "side"] if args.mode == "both" else [args.mode]

    t0 = time.time()
    storage = StorageManager(settings.DATABASE_URL)
    engine = SimilarityEngine(storage, database_profile=settings.DB_PROFILE)
    if not engine.load_index():
        print(json.dumps({"status": "error",
                          "error": "could not load FAISS index for this profile"}))
        return
    ntotal = engine.index.ntotal if engine.index is not None else 0

    # Visual coverage during the sweep: stored VisualFeatures rows / index size.
    with storage.session_scope() as session:
        visual_rows = session.query(VisualFeatures).count()
    visual_coverage = round(visual_rows / ntotal, 4) if ntotal else 0.0

    log(f"[setup] profile={settings.DB_PROFILE} index_ntotal={ntotal} "
        f"k={k} seed={args.seed} weights={weights} modes={modes}")
    log(f"[setup] visual coverage: {visual_rows}/{ntotal} = {visual_coverage:.3f} "
        f"of indexed poses carry a stored embedding")

    sources = sample_sources(storage, k, args.seed)
    log(f"[setup] sampled {len(sources)} source image(s) in {time.time()-t0:.1f}s")
    if not sources:
        print(json.dumps({"status": "error",
                          "error": "no sampleable indexed images for this profile"}))
        return

    log("[setup] building ensemble detector (RTMW-L + RTMW-X) ...")
    detector = build_detector()
    geo_extractor = GeometricFeatureExtractor()
    vis_extractor = VisualFeatureExtractor(device=settings.DEVICE)
    log(f"[setup] detector + extractors ready in {time.time()-t0:.1f}s")

    # Per-weight record buckets: occluded probes blend the query embedding; mirror
    # probes are a weight-invariant symmetry control (recorded once, copied).
    occ_records = {w: [] for w in weights}
    mirror_records = []  # computed once, identical across weights
    occ_person_lost = 0
    no_visual_probes = 0  # geometric-only probes (no query embedding available)

    for i, src in enumerate(sources, 1):
        log(f"[{i}/{len(sources)}] {Path(src['file_path']).name[:40]} "
            f"pose={src['pose_id'][:8]}")

        # --- occluded self-retrieval, swept over weights from ONE detection ---
        try:
            rgb = load_rgb(src["file_path"])
        except Exception as exc:  # unreadable/corrupt image -> skip occlusion
            log(f"      occlusion: unreadable image ({exc}) — skip")
            rgb = None

        if rgb is not None:
            for mode in modes:
                probe = detect_occluded_with_visual(
                    detector, geo_extractor, vis_extractor,
                    rgb, src["bbox"], mode)
                if probe is None:
                    occ_person_lost += 1
                    for w in weights:
                        occ_records[w].append({"rank": None, "margin": None,
                                               "mode": mode, "lost": True})
                    log(f"      occlusion[{mode}]: PERSON LOST")
                    continue
                qvec, qconf, qvisual = probe
                if qvisual is None:
                    no_visual_probes += 1

                # Sweep weights on the SAME re-detected query (no re-detection).
                for w in weights:
                    results = query_engine_at_weight(
                        engine, qvec, qconf, w, qvisual)
                    rank, self_score, distractor = rank_of_self(
                        results, src["self_pose_ids"], src["self_image_ids"])
                    margin = (self_score - distractor) if rank is not None else None
                    occ_records[w].append({"rank": rank, "margin": margin,
                                           "mode": mode, "lost": False})
                ranks_by_w = ", ".join(
                    f"w{w}:r{occ_records[w][-1]['rank']}" for w in weights)
                vtag = "" if qvisual is not None else " [no-visual]"
                log(f"      occlusion[{mode}]{vtag}: {ranks_by_w}")

        # --- mirrored positive (symmetry control, weight-invariant) ----------
        if not args.no_mirror:
            # Restore the default path for the control query (rerank off).
            settings.ENABLE_VISUAL_RERANK = False
            settings.VISUAL_RERANK_WEIGHT = 0.0
            results = query_engine_mirror(
                engine, geo_extractor, src["vector"], src["confidence"])
            rank, self_score, distractor = rank_of_self(
                results, src["self_pose_ids"], src["self_image_ids"])
            margin = (self_score - distractor) if rank is not None else None
            mirror_records.append({"rank": rank, "margin": margin})
            log(f"      mirror: self rank={rank}")

    # Aggregate per weight. Mirror is identical across weights (control).
    mirror_agg = aggregate(mirror_records) if not args.no_mirror else None
    weights_out = {}
    for w in weights:
        entry = {"occluded": aggregate(occ_records[w])}
        if mirror_agg is not None:
            entry["mirror"] = mirror_agg
        weights_out[str(w)] = entry

    out = {
        "status": "ok",
        "profile": settings.DB_PROFILE,
        "index_ntotal": ntotal,
        "k_requested": k,
        "k_sampled": len(sources),
        "seed": args.seed,
        "modes": modes,
        "weights": weights_out,
        "visual_coverage": visual_coverage,
        "occ_person_lost": occ_person_lost,
        "occ_no_visual_probes": no_visual_probes,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    if not args.json_only:
        log("\n=== VISUAL RERANK SWEEP SUMMARY ===")
        log(f"visual_coverage={visual_coverage}  person_lost={occ_person_lost}  "
            f"no_visual_probes={no_visual_probes}")
        for w in weights:
            o = weights_out[str(w)]["occluded"]
            line = (f"w={w:<5} occluded: n={o['n']} recall@1={o['recall@1']} "
                    f"recall@10={o['recall@10']} mrr={o['mrr']}")
            if mirror_agg is not None and w == 0.0:
                line += (f"  |  mirror(control): recall@1={mirror_agg['recall@1']} "
                         f"recall@10={mirror_agg['recall@10']} mrr={mirror_agg['mrr']}")
            log(line)

    # ONE final JSON line on stdout — stdout discipline.
    print(json.dumps(out))


if __name__ == "__main__":
    main()
