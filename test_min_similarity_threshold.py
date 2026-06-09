#!/usr/bin/env python3
"""
Verify the server-side similarity threshold (min_similarity) added to
SimilarityEngine.search_by_feature / swift_bridge.search_similar.

Checks, against a populated profile:
  1. Floor:        every threshold-run result has similarity_score >= threshold.
  2. Ordering:     threshold-run results are sorted best-first.
  3. Completeness: the threshold run misses no pose that a full (no-threshold) scan
                   would itself place at/above the threshold.

Run it against whichever profile actually has data indexed, e.g.:
    DB_PROFILE=irl venv/bin/python3 test_min_similarity_threshold.py
    DB_PROFILE=2d  venv/bin/python3 test_min_similarity_threshold.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.swift_bridge import PostureKitBridge
from src.storage.models import GeometricFeatures

THRESHOLD = 0.5  # 50% similarity floor for the test


def main() -> int:
    profile = os.environ.get("DB_PROFILE", "irl")
    print(f"[TEST] DB_PROFILE={profile}, threshold={THRESHOLD}")

    bridge = PostureKitBridge(num_threads=4, device="cpu", skip_models=True)
    engine = bridge.similarity_engine

    if not engine.load_index():
        print("[TEST] No index for this profile — index something first. SKIP.")
        return 0

    ntotal = engine.index.ntotal if engine.index is not None else 0
    print(f"[TEST] index ntotal = {ntotal}")
    if ntotal == 0:
        print("[TEST] Empty index. SKIP.")
        return 0

    # Use a real indexed feature vector (+ confidence) as the query.
    with engine.storage.session_scope() as session:
        row = session.query(
            GeometricFeatures.feature_vector,
            GeometricFeatures.feature_confidence,
        ).first()
    qvec = np.array(row[0], dtype=np.float32).tolist()
    qconf = np.array(row[1], dtype=np.float32).tolist() if row[1] is not None else None

    common = dict(
        feature_vector=qvec,
        query_confidence=qconf,
        k=max(ntotal, 500),
        min_confidence=0.0,
        deduplicate_images=False,
    )

    t0 = time.time()
    thr = bridge.search_similar(min_similarity=THRESHOLD, **common)
    t_thr = time.time() - t0

    t0 = time.time()
    full = bridge.search_similar(min_similarity=0.0, **common)
    t_full = time.time() - t0

    print(f"[TEST] threshold run: {len(thr):>6} results in {t_thr:6.2f}s")
    print(f"[TEST] full run:      {len(full):>6} results in {t_full:6.2f}s")

    ok = True

    below = [r for r in thr if r["similarity_score"] < THRESHOLD - 1e-6]
    if below:
        ok = False
        worst = min(r["similarity_score"] for r in below)
        print(f"[FAIL] {len(below)} threshold results below {THRESHOLD} (min={worst:.3f})")
    else:
        print(f"[PASS] all {len(thr)} threshold results >= {THRESHOLD}")

    sims = [r["similarity_score"] for r in thr]
    if sims != sorted(sims, reverse=True):
        ok = False
        print("[FAIL] threshold results are not sorted best-first")
    else:
        print("[PASS] threshold results sorted best-first")

    full_ge = {r["pose_id"] for r in full if r["similarity_score"] >= THRESHOLD - 1e-6}
    thr_ids = {r["pose_id"] for r in thr}
    missing = full_ge - thr_ids
    if missing:
        ok = False
        print(f"[FAIL] threshold run missing {len(missing)} poses the full run placed >= {THRESHOLD}")
    else:
        print(f"[PASS] completeness: threshold run covers all {len(full_ge)} poses >= {THRESHOLD}")

    print("[RESULT]", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
