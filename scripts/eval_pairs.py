#!/usr/bin/env python3
"""
eval_pairs.py — retrieval benchmark over labeled pose pairs.

Consumes a pairs file (default scripts/eval_pairs.json) of the form:

  [
    {"query": "<path|fragment|pose-uuid>", "candidate": "<...>",
     "label": "match", "note": "same kneeling pose, different model"},
    {"query": "...", "candidate": "...", "label": "non-match"}
  ]

For every pair it runs the REAL engine search (live parameters) and reports:
  - the candidate's rank and score for the query
  - hit@1 / hit@10 / hit@50 for "match" pairs
  - false-admit rate at the floor for "non-match" pairs
  - a masking-sensitivity sweep: how each pair's masked base similarity and
    valid-dim count respond to min_feature_confidence in {0.25, 0.30, 0.35, 0.40}

Grow the pairs file by running scripts/diagnose_match.py on photo pairs you
know should (or should not) match. Re-run this after every accuracy change.

Usage:
  DB_PROFILE=irl venv/bin/python3 scripts/eval_pairs.py [pairs.json] [--floor 0.5]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import src.core._torch_patch  # noqa: F401
from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.intelligence.similarity_engine import SimilarityEngine
from src.core.geometric_feature_extractor import GeometricFeatureExtractor

from diagnose_match import resolve, LIVE_MIN_FEATURE_CONFIDENCE, LIVE_MIN_VALID_OVERLAP  # noqa: E402

SWEEP_CONFIDENCES = [0.25, 0.30, 0.35, 0.40]

# Shared extractor for mirrored-positive pairs ("flip": true). flip_features
# swaps the 52-dim vector's (and confidence's) left/right dimensions, the same
# transform the engine's flip search uses.
_EXTRACTOR = GeometricFeatureExtractor()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pairs_path = Path(args[0]) if args else Path(__file__).parent / "eval_pairs.json"
    floor = 0.5
    for a in sys.argv[1:]:
        if a.startswith("--floor"):
            floor = float(a.split("=", 1)[1]) if "=" in a else float(sys.argv[sys.argv.index(a) + 1])

    if not pairs_path.exists():
        sys.exit(f"No pairs file at {pairs_path}. Diagnose pairs with diagnose_match.py and record them there.")
    pairs = json.loads(pairs_path.read_text())
    if not pairs:
        sys.exit("Pairs file is empty.")

    storage = StorageManager(settings.DATABASE_URL)
    engine = SimilarityEngine(storage, database_profile=settings.DB_PROFILE)
    if not engine.load_index():
        sys.exit("ERROR: could not load FAISS index for this profile")

    hits = {1: 0, 10: 0, 50: 0}
    n_match = 0
    n_nonmatch = 0
    false_admits = 0

    print(f"Evaluating {len(pairs)} pair(s) at floor {floor:.2f} "
          f"(min_feature_confidence={LIVE_MIN_FEATURE_CONFIDENCE}, min_valid_overlap={LIVE_MIN_VALID_OVERLAP})\n")

    for i, pair in enumerate(pairs, 1):
        _, q_records = resolve(storage, pair["query"])
        _, c_records = resolve(storage, pair["candidate"])
        label = pair.get("label", "match")
        note = pair.get("note", "")
        if not q_records or not c_records:
            print(f"{i:3d}. SKIP ({label}): unresolved or pose-less — {pair['query']} vs {pair['candidate']}")
            continue
        q = q_records[0]
        cand_ids = {c.pose_id for c in c_records}

        # Mirrored-positive pair ("flip": true): query the index with the
        # horizontally-flipped feature vector and expect the (unflipped)
        # candidate back — a label-free left/right-symmetry test. The source is
        # the target here, so it is NOT excluded.
        flip = bool(pair.get("flip"))
        if flip:
            q_vector = _EXTRACTOR.flip_features(np.asarray(q.vector, dtype=np.float32))
            q_conf = _EXTRACTOR.flip_features(np.asarray(q.conf, dtype=np.float32))
            exclude = None
        else:
            q_vector, q_conf = q.vector, q.conf
            exclude = q.pose_id

        engine.clear_search_cache()
        results = engine.search_by_feature(
            q_vector, query_confidence=q_conf,
            k=engine.index.ntotal, min_confidence=0.0,
            min_feature_confidence=LIVE_MIN_FEATURE_CONFIDENCE,
            min_valid_overlap=LIVE_MIN_VALID_OVERLAP,
            deduplicate_images=False, min_similarity=0.0, exclude_pose_id=exclude)
        best = next((r for r in results if r["pose_id"] in cand_ids), None)

        if label == "match":
            n_match += 1
            if best is None:
                print(f"{i:3d}. MISS  (match): candidate absent from ranking entirely  {note}")
                continue
            for k in hits:
                if best["rank"] <= k:
                    hits[k] += 1
            status = "ok" if best["similarity_score"] >= floor else "BELOW FLOOR"
            print(f"{i:3d}. rank #{best['rank']:>5d}  score {best['similarity_score']:.3f}  [{status}]  {note}")
        else:
            n_nonmatch += 1
            if best is not None and best["similarity_score"] >= floor:
                false_admits += 1
                print(f"{i:3d}. FALSE ADMIT (non-match): rank #{best['rank']} score {best['similarity_score']:.3f}  {note}")
            else:
                print(f"{i:3d}. correctly rejected (non-match)  {note}")

        # masking-sensitivity sweep for this pair
        sweep = []
        for mc in SWEEP_CONFIDENCES:
            best_sim, best_valid = 0.0, 0
            for c in c_records:
                d, vc = engine._compute_masked_distance(
                    q.vector, q.conf, c.vector, c.conf,
                    min_confidence=mc, min_valid_overlap=LIVE_MIN_VALID_OVERLAP)
                sim = 0.0 if d == float("inf") else engine._distance_to_similarity(d)
                if sim > best_sim:
                    best_sim, best_valid = sim, vc
            sweep.append(f"mc={mc:.2f}: {best_sim:.3f} ({best_valid}d)")
        print(f"      sweep  {'  '.join(sweep)}")

    print()
    if n_match:
        print(f"match pairs:     {n_match}  |  hit@1 {hits[1]}/{n_match}  hit@10 {hits[10]}/{n_match}  hit@50 {hits[50]}/{n_match}")
    if n_nonmatch:
        print(f"non-match pairs: {n_nonmatch}  |  false admits at floor: {false_admits}")


if __name__ == "__main__":
    main()
