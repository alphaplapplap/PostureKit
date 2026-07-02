"""
Backfill HEELS_HIGH body-part tags for already-indexed poses (no re-detection).

For every stored pose: crop each foot from stored keypoints, score with the
fashion-CLIP HeelDetector, and INSERT one HEELS_HIGH body_parts row per person
when the per-person max score >= threshold. "Find photos with heels" then works
via the existing browse_by_body_parts / required_regions=["HEELS_HIGH"] path.

Writes body_parts only — does NOT touch geometric features or FAISS, so
reextract_features.py is NOT required after.

Idempotency: body_parts has no unique key, so re-runs would duplicate. Default
skips a (image_id, person_index) that already has a HEELS_HIGH row; --force
deletes existing HEELS_HIGH for the persons it re-tags before inserting.

Stale paths: the corpus predates folder moves; pass --remap 'OLD=>NEW' (repeatable)
to resolve relocated images (e.g. the P2 Legs and Heels folder moved under
Favorites). Unresolvable images are counted and skipped.

Usage:
    DB_PROFILE=irl venv/bin/python3 scripts/backfill_heels.py \
        --threshold 0.7 \
        --remap "References/irl/P2 Legs and Heels (irl)/=>References/irl/Favorites (irl)/P2 Legs and Heels (irl)/"
    # smoke test: add --limit 500 ; re-tag from scratch: add --force
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config.settings import settings
from src.storage.storage_manager import StorageManager
from src.storage.models import Image, PoseDetection, BodyPart
from src.core.heel_detector import HeelDetector

DB_PAGE = 1000


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def resolve_path(p, remaps=()):
    cand = Path(p).expanduser()
    if cand.exists():
        return cand
    for old, new in remaps:
        if old in p:
            rp = Path(p.replace(old, new, 1)).expanduser()
            if rp.exists():
                return rp
    alt = (PROJECT_ROOT / p)
    return alt if alt.exists() else None


def repair_keypoints(kp_raw, vis_raw):
    """(133,3) [x,y,conf] with the legacy-confidence repair from reextract_features."""
    kp = np.array(kp_raw, dtype=np.float32).reshape(133, 3)
    if vis_raw is not None:
        vis = np.array(vis_raw, dtype=np.int64)
    else:
        conf = kp[:, 2]
        vis = np.where(conf >= 0.5, 2, np.where(conf >= 0.1, 1, 0)).astype(np.int64)
    if float(kp[:, 2].max()) > 1.001:  # ensemble-era rows stored [0,2] visibility here
        kp[:, 2] = vis.astype(np.float32) / 2.0
    return kp


def load_rgb(path):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"))


def folder_prior_pass(profile, args):
    """Curation-prior tagging: the folder membership is the label (user-curated).

    For each image in scope with no HEELS_HIGH row, insert ONE row on its
    highest-confidence pose at exactly args.folder_prior. Pure DB pass — no
    image loading, no classifier. Prior rows are distinguishable from
    classifier rows by their exact confidence value; the browse sensitivity
    slider at > prior shows visually-confirmed tags only. Idempotent: images
    with any existing HEELS_HIGH row are skipped.
    """
    conf = float(args.folder_prior)
    if not (0.0 < conf <= 1.0):
        log(f"--folder-prior must be in (0,1], got {conf}")
        return 1
    storage = StorageManager(database_profile=profile)
    stats = dict(images=0, tagged=0, already=0, no_pose_bbox=0)
    with storage.session_scope() as s:
        scoped = (s.query(Image.id)
                  .filter(Image.file_path.like(f"%{args.path_contains}%")).subquery())
        tagged_imgs = set(r[0] for r in s.query(BodyPart.image_id)
                          .filter(BodyPart.part_name == "HEELS_HIGH",
                                  BodyPart.image_id.in_(scoped.select())).all())
        rows = (s.query(PoseDetection)
                .filter(PoseDetection.image_id.in_(scoped.select()))
                .order_by(PoseDetection.image_id,
                          PoseDetection.overall_confidence.desc()).all())
        best_per_image = {}
        for pose in rows:  # first per image = highest confidence (ordered desc)
            best_per_image.setdefault(pose.image_id, pose)
        stats["images"] = len(best_per_image)
        for image_id, pose in best_per_image.items():
            if args.limit is not None and stats["tagged"] >= args.limit:
                break
            if image_id in tagged_imgs:
                stats["already"] += 1
                continue
            # bbox: pose box [x,y,w,h] -> [x1,y1,x2,y2]; fall back to confident keypoints
            bbox = None
            if pose.bbox is not None and len(pose.bbox) == 4 and pose.bbox[2] > 0 and pose.bbox[3] > 0:
                x, y, w, h = pose.bbox
                bbox = [float(x), float(y), float(x + w), float(y + h)]
            else:
                kp = np.array(pose.keypoints, dtype=np.float32).reshape(133, 3)
                pts = kp[kp[:, 2] >= 0.3]
                if pts.shape[0] >= 2:
                    bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max() + 1), float(pts[:, 1].max() + 1)]
            if bbox is None:
                stats["no_pose_bbox"] += 1
                continue
            s.add(BodyPart(
                id=uuid.uuid4(), image_id=image_id, person_index=int(pose.person_id),
                part_name="HEELS_HIGH", canonical_region="feet",
                confidence=conf, bbox=bbox, is_exposed=False,
            ))
            stats["tagged"] += 1
    log(f"[{profile}] folder-prior pass: {stats}")
    print(f"RESULT {profile}: prior_tagged={stats['tagged']} images={stats['images']} "
          f"already={stats['already']} no_pose_bbox={stats['no_pose_bbox']} conf={conf}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backfill HEELS_HIGH body-part tags per DB_PROFILE")
    ap.add_argument("--threshold", type=float, default=settings.HEEL_THRESHOLD)
    ap.add_argument("--min-foot-conf", type=float, default=settings.HEEL_MIN_FOOT_CONF)
    ap.add_argument("--model", default=settings.HEEL_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="process at most N poses (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-tag: delete existing HEELS_HIGH first")
    ap.add_argument("--remap", action="append", default=None, help="stale-path fix 'OLD=>NEW' (repeatable)")
    ap.add_argument("--path-contains", default=None,
                    help="only process poses whose image path contains this substring (subset backfill)")
    ap.add_argument("--folder-prior", type=float, default=None, metavar="CONF",
                    help="curation-prior mode: no pixels/classifier — for every scoped image with NO "
                         "HEELS_HIGH row, tag its highest-confidence pose at exactly CONF (e.g. 0.70). "
                         "Use when the folder itself is the label. Requires --path-contains.")
    args = ap.parse_args(argv)

    profile = os.environ.get("DB_PROFILE", settings.DB_PROFILE)
    remaps = []
    for r in (args.remap or []):
        if "=>" in r:
            a, b = r.split("=>", 1)
            remaps.append((a, b))

    if args.folder_prior is not None:
        if not args.path_contains:
            log("--folder-prior requires --path-contains (refusing to prior-tag the whole DB)")
            return 1
        return folder_prior_pass(profile, args)

    storage = StorageManager(database_profile=profile)
    detector = HeelDetector(device=settings.DEVICE, model=args.model,
                            threshold=args.threshold, min_foot_conf=args.min_foot_conf)

    with storage.session_scope() as s:
        tq = s.query(PoseDetection)
        if args.path_contains:
            tq = tq.join(Image, Image.id == PoseDetection.image_id).filter(
                Image.file_path.like(f"%{args.path_contains}%"))
        total = tq.count()
    log(f"[{profile}] backfill heels over {total} poses "
        f"(threshold={args.threshold}, force={args.force}, remaps={len(remaps)})")

    stats = dict(seen=0, tagged=0, skipped_existing=0, no_foot=0, below_thr=0,
                 unreadable=0, errors=0)
    last_id = None

    while True:
        if args.limit is not None and stats["seen"] >= args.limit:
            break
        # --- read a page (detach data before slow inference) ---
        with storage.session_scope() as s:
            q = (s.query(PoseDetection, Image)
                 .join(Image, Image.id == PoseDetection.image_id)
                 .order_by(PoseDetection.id))
            if args.path_contains:
                q = q.filter(Image.file_path.like(f"%{args.path_contains}%"))
            if last_id is not None:
                q = q.filter(PoseDetection.id > last_id)
            page = q.limit(DB_PAGE).all()
            if not page:
                break
            rows = []
            img_ids = set()
            for pose, img in page:
                last_id = pose.id
                rows.append(dict(
                    image_id=img.id, person_index=int(pose.person_id),
                    file_path=img.file_path, w=int(img.width), h=int(img.height),
                    kp=repair_keypoints(pose.keypoints, pose.keypoint_visibility),
                ))
                img_ids.add(img.id)
            # idempotency: which (image_id, person_index) already have HEELS_HIGH
            existing = set(
                (bp.image_id, bp.person_index) for bp in
                s.query(BodyPart.image_id, BodyPart.person_index)
                .filter(BodyPart.part_name == "HEELS_HIGH", BodyPart.image_id.in_(img_ids)).all()
            )

        # --- score outside the session (no DB connection held during inference) ---
        pending = []  # dicts of BodyPart kwargs
        img_cache = {}
        for r in rows:
            if args.limit is not None and stats["seen"] >= args.limit:
                break
            stats["seen"] += 1
            key = (r["image_id"], r["person_index"])
            if not args.force and key in existing:
                stats["skipped_existing"] += 1
                continue
            path = resolve_path(r["file_path"], remaps)
            if path is None:
                stats["unreadable"] += 1
                continue
            try:
                if r["image_id"] not in img_cache:
                    img_cache[r["image_id"]] = load_rgb(path)
                image_rgb = img_cache[r["image_id"]]
                score, box = detector.detect(image_rgb, r["kp"])
            except Exception as e:
                stats["errors"] += 1
                log(f"  ERROR pose img={r['image_id']} p{r['person_index']}: {e}")
                continue
            if box is None:
                stats["no_foot"] += 1
                continue
            if score < args.threshold:
                stats["below_thr"] += 1
                continue
            pending.append(dict(
                image_id=r["image_id"], person_index=r["person_index"],
                confidence=float(max(0.0, min(1.0, score))),
                bbox=[float(v) for v in box],
            ))

        # --- write the page's tags in one transaction ---
        if pending:
            with storage.session_scope() as s:
                if args.force:
                    for pair in set((p["image_id"], p["person_index"]) for p in pending):
                        (s.query(BodyPart)
                         .filter(BodyPart.part_name == "HEELS_HIGH",
                                 BodyPart.image_id == pair[0],
                                 BodyPart.person_index == pair[1])
                         .delete(synchronize_session=False))
                for p in pending:
                    s.add(BodyPart(
                        id=uuid.uuid4(), image_id=p["image_id"], person_index=p["person_index"],
                        part_name="HEELS_HIGH", canonical_region="feet",
                        confidence=p["confidence"], bbox=p["bbox"], is_exposed=False,
                    ))
                    stats["tagged"] += 1

        log(f"[{profile}] seen={stats['seen']}/{total} tagged={stats['tagged']} "
            f"skip_existing={stats['skipped_existing']} no_foot={stats['no_foot']} "
            f"below_thr={stats['below_thr']} unreadable={stats['unreadable']} err={stats['errors']}")

    print(f"RESULT {profile}: tagged={stats['tagged']} seen={stats['seen']} "
          f"skip_existing={stats['skipped_existing']} no_foot={stats['no_foot']} "
          f"below_thr={stats['below_thr']} unreadable={stats['unreadable']} errors={stats['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
