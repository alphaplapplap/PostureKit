"""
Offline feasibility eval for high-heel detection (step 1 of 2).

Localizes each foot from the COCO-WholeBody foot keypoints PostureKit already
detects + stores (idx 17-22), crops a padded foot region, and scores it with
every available classifier so we can pick a model + operating point on REAL
photos before wiring anything into the index. Read-only: never writes the DB,
never touches FAISS, never re-detects (reads stored poses for the active
DB_PROFILE).

Scorers (each degrades gracefully if its dep/weights are missing):
  - fashionSigLIP : Marqo/marqo-fashionSigLIP via open_clip; per-prompt sigmoid,
                    heel_score = max over positive prompts.
  - fashionCLIP   : Marqo/marqo-fashionCLIP via open_clip; softmax over the
                    positive+negative prompt set, heel_score = sum positive probs.
  - yoloworld     : ultralytics YOLOWorld (already installed); open-vocab boxes,
                    heel_score = max box conf among heel classes. (Weak on tiny
                    crops -- logged.)
  - angle         : pure-geometry control. Plantar-flexion / foot inclination from
                    ankle->heel->toe. Expected to fire on barefoot tiptoe -- that's
                    the point: it shows geometry under-discriminates vs a shoe.

Two modes:
  GENERATE (default): scan stored poses -> crop feet -> score -> write
    data/reports/heel_eval/{crops/, results.csv, review.html}
  SCORE  (--score-labels results.csv): read the hand-labeled CSV (fill the
    `label` column with heel/not/ambiguous) -> print precision/recall +
    confusion per model at 0.5/0.7/0.9, split by crop-size bucket and an
    optional `neg_class` column. Also writes metrics.json.

Usage:
    # generate a review set (random 300-pose sample) for the irl profile
    DB_PROFILE=irl venv/bin/python3 scripts/eval_heel_detection.py --profile irl --sample 300 --seed 42

    # after hand-labeling the `label` column in results.csv:
    venv/bin/python3 scripts/eval_heel_detection.py --score-labels data/reports/heel_eval/results.csv

Stdout discipline: progress -> stderr; only the final RESULT/summary -> stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# COCO-WholeBody keypoint layout (src/core/models.py:206-212). kp = (133,3) [x,y,conf].
# ---------------------------------------------------------------------------
KP = dict(
    L_KNEE=13, R_KNEE=14, L_ANKLE=15, R_ANKLE=16,
    L_BIG_TOE=17, L_SMALL_TOE=18, L_HEEL=19,
    R_BIG_TOE=20, R_SMALL_TOE=21, R_HEEL=22,
)
FEET = {
    "L": dict(knee=KP["L_KNEE"], ankle=KP["L_ANKLE"], heel=KP["L_HEEL"],
              big_toe=KP["L_BIG_TOE"], small_toe=KP["L_SMALL_TOE"],
              box=[KP["L_ANKLE"], KP["L_BIG_TOE"], KP["L_SMALL_TOE"], KP["L_HEEL"]]),
    "R": dict(knee=KP["R_KNEE"], ankle=KP["R_ANKLE"], heel=KP["R_HEEL"],
              big_toe=KP["R_BIG_TOE"], small_toe=KP["R_SMALL_TOE"],
              box=[KP["R_ANKLE"], KP["R_BIG_TOE"], KP["R_SMALL_TOE"], KP["R_HEEL"]]),
}

# Prompt sets cover "any and all heels" per the user request.
POS_PROMPTS = [
    "a high-heeled shoe", "a stiletto heel", "a high-heeled pump",
    "a heeled sandal", "a wedge heel", "a heeled ankle boot", "a platform heel",
]
NEG_PROMPTS = [
    "a sneaker", "a flat shoe", "a flat sandal", "a flat boot",
    "a bare foot", "a sock", "a flat ballet shoe",
]
YOLO_POS = ["high heel", "stiletto", "high-heeled shoe", "heeled sandal", "wedge heel"]
YOLO_NEG = ["sneaker", "flat shoe", "boot", "sandal", "bare foot"]

SCORE_COLS = ["fashionSigLIP", "fashionCLIP", "yoloworld", "angle"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _kp_ok(kp, idx, min_conf):
    return float(kp[idx, 2]) >= min_conf and not (kp[idx, 0] == 0 and kp[idx, 1] == 0)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def foot_box(kp, foot, min_conf, img_w, img_h, pad=0.35, max_frac=0.45):
    """Padded, clamped pixel box [x0,y0,x1,y1] around a foot, ANKLE-ANCHORED.

    RTMW foot keypoints (17-22) are routinely hallucinated when feet aren't
    visible (the eval surfaced a "foot crop" that landed on a face at frac=0.49).
    Guard against that:
      - require a confident ankle as the anchor (no ankle -> can't localize),
      - scale a plausible foot reach by shin length (knee->ankle), and only
        include heel/toe points within that reach of the ankle,
      - reject boxes larger than max_frac of the image (residual hallucination).
    Box/clamp pattern follows swift_bridge.py:729-738.
    """
    ankle = foot["ankle"]
    if not _kp_ok(kp, ankle, min_conf):
        return None
    a = (float(kp[ankle, 0]), float(kp[ankle, 1]))
    knee = foot["knee"]
    shin = _dist(a, (float(kp[knee, 0]), float(kp[knee, 1]))) if _kp_ok(kp, knee, min_conf) else 0.0
    diag = (img_w ** 2 + img_h ** 2) ** 0.5
    reach = max(1.8 * shin, 0.10 * diag) if shin > 1 else 0.14 * diag

    pts = [a]
    for i in (foot["heel"], foot["big_toe"], foot["small_toe"]):
        if _kp_ok(kp, i, min_conf):
            p = (float(kp[i, 0]), float(kp[i, 1]))
            if _dist(p, a) <= reach:
                pts.append(p)
    if len(pts) < 2:  # ankle + >=1 real foot point
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w, h = x1 - x0, y1 - y0
    padx, pady = w * pad + 12, h * pad + 12
    bx0 = max(0, int(x0 - padx))
    by0 = max(0, int(y0 - pady))
    bx1 = min(img_w, int(x1 + padx))
    by1 = min(img_h, int(y1 + pady))
    if bx1 - bx0 < 10 or by1 - by0 < 10:
        return None
    if (bx1 - bx0) * (by1 - by0) > max_frac * img_w * img_h:
        return None  # implausibly large for a foot -> hallucinated keypoints
    return (bx0, by0, bx1, by1)


def foot_inclination_deg(kp, foot, min_conf):
    """Plantar-flexion proxy: downward tilt of the heel->big_toe vector vs horizontal.
    Flat foot ~0-15deg; high heel / tiptoe ~25-50deg (toe below raised heel).
    Returns (inclination_deg, pseudo_score in [0,1]) or (None, None)."""
    import math
    h, t = foot["heel"], foot["big_toe"]
    if not (_kp_ok(kp, h, min_conf) and _kp_ok(kp, t, min_conf)):
        return None, None
    dx = float(kp[t, 0] - kp[h, 0])
    dy = float(kp[t, 1] - kp[h, 1])  # image y grows downward; toe below heel => dy>0
    incl = math.degrees(math.atan2(dy, abs(dx) + 1e-6))
    score = max(0.0, min(1.0, incl / 45.0))  # crude monotonic map, documented
    return round(incl, 1), round(score, 4)


# ---------------------------------------------------------------------------
# Scorers (lazy-load; device mirrors settings.DEVICE, mps fallback like
# visual_feature_extractor.py:169-195)
# ---------------------------------------------------------------------------
class _OpenClipScorer:
    """Base for the two Marqo fashion-CLIP encoders via open_clip."""
    hf_id = ""
    col = ""
    mode = "softmax"  # "softmax" (fashionCLIP) or "sigmoid" (SigLIP)

    def __init__(self, device):
        self.device = device
        self.model = None
        self.preprocess = None
        self._txt = None
        self._n_pos = len(POS_PROMPTS)
        self.failed = False

    def load(self):
        import torch
        import open_clip
        log(f"  loading {self.hf_id} on {self.device} ...")
        model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{self.hf_id}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{self.hf_id}")
        model = model.eval().to(self.device)
        self.model, self.preprocess = model, preprocess
        with torch.no_grad():
            tokens = tokenizer(POS_PROMPTS + NEG_PROMPTS).to(self.device)
            txt = model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            self._txt = txt
        self._logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
        self._logit_bias = float(getattr(model, "logit_bias", 0.0) or 0.0)

    def score(self, pil_crop):
        import torch
        if self.failed:
            return ""
        if self.model is None:
            try:
                self.load()
            except Exception as e:
                self.failed = True
                log(f"  [disabled] {self.col}: load failed ({type(e).__name__}: {e}); skipping for the rest of the run")
                return ""
        with torch.no_grad():
            img = self.preprocess(pil_crop).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = (feat @ self._txt.T).squeeze(0)  # (n_prompts,)
            if self.mode == "sigmoid":
                probs = torch.sigmoid(logits * self._logit_scale + self._logit_bias)
                heel = float(probs[: self._n_pos].max().item())
            else:
                probs = torch.softmax(logits * self._logit_scale, dim=-1)
                heel = float(probs[: self._n_pos].sum().item())
        return round(heel, 4)


class FashionSigLIP(_OpenClipScorer):
    hf_id = "Marqo/marqo-fashionSigLIP"
    col = "fashionSigLIP"
    # Native SigLIP sigmoid head saturated here (cos x large logit_scale + bias,
    # max over 7 positives -> ~1 for every crop). Use the same softmax-over-prompt-set
    # scoring as fashionCLIP so the comparison isolates the ENCODER, not the head.
    mode = "softmax"


class FashionCLIP(_OpenClipScorer):
    hf_id = "Marqo/marqo-fashionCLIP"
    col = "fashionCLIP"
    mode = "softmax"


class YoloWorld:
    col = "yoloworld"

    def __init__(self, device, weights="yolov8x-worldv2.pt"):
        self.device = device
        self.weights = weights
        self.model = None
        self._names = None
        self._pos = set(YOLO_POS)
        self.failed = False

    def load(self):
        from ultralytics import YOLOWorld as _YW
        log(f"  loading YOLO-World {self.weights} on {self.device} ...")
        model = _YW(self.weights)
        model.set_classes(YOLO_POS + YOLO_NEG)  # needs the `clip` package; raises if absent
        # assign only after set_classes succeeds, so a failure leaves model unset
        self.model = model
        self._names = YOLO_POS + YOLO_NEG

    def score(self, pil_crop):
        if self.failed:
            return ""
        if self.model is None:
            try:
                self.load()
            except Exception as e:
                self.failed = True
                log(f"  [disabled] {self.col}: load failed ({type(e).__name__}: {e}); skipping for the rest of the run")
                return ""
        res = self.model.predict(pil_crop, conf=0.01, verbose=False, device=self.device)
        best = 0.0
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = self._names[int(b.cls.item())]
                if cls in self._pos:
                    best = max(best, float(b.conf.item()))
        return round(best, 4)


def select_device(requested):
    try:
        import torch
        if requested == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                return "mps"
            return "cpu"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_scorers(which, device):
    """Instantiate requested scorers, skipping any whose dependency is absent."""
    want = set(which)
    scorers = {}
    if "fashionSigLIP" in want or "fashionCLIP" in want:
        try:
            import open_clip  # noqa: F401
            if "fashionSigLIP" in want:
                scorers["fashionSigLIP"] = FashionSigLIP(device)
            if "fashionCLIP" in want:
                scorers["fashionCLIP"] = FashionCLIP(device)
        except Exception as e:
            log(f"  [skip] open_clip unavailable ({e}); fashion classifiers disabled")
    if "yoloworld" in want:
        try:
            import ultralytics  # noqa: F401
            scorers["yoloworld"] = YoloWorld(device)
        except Exception as e:
            log(f"  [skip] ultralytics unavailable ({e}); YOLO-World disabled")
    return scorers


# ---------------------------------------------------------------------------
# DB read (mirrors scripts/reextract_features.py:41-98)
# ---------------------------------------------------------------------------
def repair_keypoints(row):
    """(133,3) keypoints with the legacy-confidence repair from to_pose_result()."""
    import numpy as np
    kp = np.array(row.keypoints, dtype=np.float32).reshape(133, 3)
    if row.keypoint_visibility is not None:
        vis = np.array(row.keypoint_visibility, dtype=np.int64)
    else:
        conf = kp[:, 2]
        vis = np.where(conf >= 0.5, 2, np.where(conf >= 0.1, 1, 0)).astype(np.int64)
    if float(kp[:, 2].max()) > 1.001:  # ensemble-era rows stored [0,2] visibility here
        kp[:, 2] = vis.astype(np.float32) / 2.0
    return kp


def iter_pose_rows(storage, profile, sample, limit, seed):
    """Yield plain dicts {pose_id, image_path, kp, img_w, img_h, person_id} read
    inside the session so the FK (pose.image) is hydrated before detachment."""
    import numpy as np
    from src.storage.models import PoseDetection, Image

    if sample:
        import random
        with storage.session_scope() as s:
            ids = [r[0] for r in s.query(PoseDetection.id).all()]
        rng = random.Random(seed)
        if len(ids) > sample:
            ids = rng.sample(ids, sample)
        log(f"[{profile}] sampling {len(ids)} of stored poses (seed={seed})")
        CH = 500
        for i in range(0, len(ids), CH):
            chunk = ids[i:i + CH]
            with storage.session_scope() as s:
                rows = (s.query(PoseDetection, Image)
                        .join(Image, Image.id == PoseDetection.image_id)
                        .filter(PoseDetection.id.in_(chunk)).all())
                for pose, img in rows:
                    yield _row_dict(pose, img, repair_keypoints(pose))
        return

    # full / first-N scan
    last_id = None
    seen = 0
    BATCH = 1000
    while True:
        with storage.session_scope() as s:
            q = (s.query(PoseDetection, Image)
                 .join(Image, Image.id == PoseDetection.image_id)
                 .order_by(PoseDetection.id))
            if last_id is not None:
                q = q.filter(PoseDetection.id > last_id)
            rows = q.limit(BATCH).all()
            if not rows:
                break
            for pose, img in rows:
                last_id = pose.id
                yield _row_dict(pose, img, repair_keypoints(pose))
                seen += 1
                if limit and seen >= limit:
                    return


def _row_dict(pose, img, kp):
    return {
        "pose_id": str(pose.id), "image_path": img.file_path,
        "img_w": int(img.width), "img_h": int(img.height),
        "person_id": int(pose.person_id), "kp": kp,
    }


def resolve_path(p, remaps=()):
    """Stored paths may be absolute, relative, ~-prefixed, or STALE (folder moved
    after indexing). `remaps` is a list of (old, new) substring swaps tried in order."""
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


def iter_labeled_rows(storage, pos_substr, neg_substrs, per_class, seed):
    """Ground-truth selection by folder: images whose stored path contains
    `pos_substr` are label='heel'; negatives are either images matching any
    `neg_substrs`, or (default) a random sample of images NOT under pos_substr.
    Each yielded row carries gt_label so no manual labeling is needed."""
    import random
    from sqlalchemy import or_
    from src.storage.models import PoseDetection, Image
    rng = random.Random(seed)
    with storage.session_scope() as s:
        base = s.query(PoseDetection.id).join(Image, Image.id == PoseDetection.image_id)
        pos_ids = [r[0] for r in base.filter(Image.file_path.like(f"%{pos_substr}%")).all()]
        if neg_substrs:
            neg_ids = [r[0] for r in base.filter(
                or_(*[Image.file_path.like(f"%{n}%") for n in neg_substrs])).all()]
        else:
            neg_ids = [r[0] for r in base.filter(~Image.file_path.like(f"%{pos_substr}%")).all()]
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)
    if per_class:
        pos_ids, neg_ids = pos_ids[:per_class], neg_ids[:per_class]
    label = {**{i: "heel" for i in pos_ids}, **{i: "not" for i in neg_ids}}
    ids = pos_ids + neg_ids
    log(f"labeled selection: {len(pos_ids)} heel + {len(neg_ids)} not = {len(ids)} poses")
    CH = 500
    from src.storage.models import PoseDetection as PD, Image as IM
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        with storage.session_scope() as s:
            rows = (s.query(PD, IM).join(IM, IM.id == PD.image_id)
                    .filter(PD.id.in_(chunk)).all())
            for pose, img in rows:
                d = _row_dict(pose, img, repair_keypoints(pose))
                d["gt_label"] = label.get(pose.id, "")
                yield d


# ---------------------------------------------------------------------------
# GENERATE mode
# ---------------------------------------------------------------------------
def generate(args):
    os.environ["DB_PROFILE"] = args.profile  # set BEFORE importing settings/StorageManager
    from src.storage.storage_manager import StorageManager
    from PIL import Image as PILImage

    device = select_device(args.device)
    log(f"[{args.profile}] device={device}; models={args.models}")
    scorers = build_scorers(args.models.split(","), device)
    do_angle = "angle" in args.models.split(",")
    if not scorers and not do_angle:
        log("no scorers available; nothing to do"); return 1

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    remaps = []
    for r in (args.remap or []):
        if "=>" in r:
            a, b = r.split("=>", 1)
            remaps.append((a, b))
    negs = [x.strip() for x in args.negative_substr.split(",")] if args.negative_substr else None

    storage = StorageManager(database_profile=args.profile)
    rows_out = []
    n_pose = n_foot = n_skip_path = n_skip_foot = 0
    missing = set()

    if args.positive_substr:
        row_iter = iter_labeled_rows(storage, args.positive_substr, negs, args.per_class, args.seed)
    else:
        row_iter = iter_pose_rows(storage, args.profile, args.sample, args.limit, args.seed)

    for rec in row_iter:
        n_pose += 1
        if n_pose % 100 == 0:
            log(f"  ...{n_pose} poses, {n_foot} foot crops")
        path = resolve_path(rec["image_path"], remaps)
        if path is None:
            n_skip_path += 1
            missing.add(rec["image_path"])
            continue
        pil = None
        kp = rec["kp"]
        for side, foot in FEET.items():
            box = foot_box(kp, foot, args.min_foot_conf, rec["img_w"], rec["img_h"])
            if box is None:
                n_skip_foot += 1
                continue
            if pil is None:
                try:
                    pil = PILImage.open(path).convert("RGB")
                except Exception as e:
                    log(f"  [skip] cannot open {path}: {e}")
                    n_skip_path += 1
                    break
            x0, y0, x1, y1 = box
            # keypoints can extend past stored W/H if image was re-saved; clamp to real size
            x1 = min(x1, pil.width); y1 = min(y1, pil.height)
            if x1 - x0 < 10 or y1 - y0 < 10:
                n_skip_foot += 1
                continue
            crop = pil.crop((x0, y0, x1, y1))
            crop_w, crop_h = crop.size
            crop_frac = round((crop_w * crop_h) / max(1, pil.width * pil.height), 5)

            crop_name = f"{rec['pose_id']}_{rec['person_id']}_{side}.jpg"
            crop.save(crops_dir / crop_name, quality=88)

            incl, angle_score = foot_inclination_deg(kp, foot, args.min_foot_conf) if do_angle else (None, None)
            row = {
                "pose_id": rec["pose_id"], "person_id": rec["person_id"], "foot": side,
                "image_path": str(path), "crop": f"crops/{crop_name}",
                "crop_w": crop_w, "crop_h": crop_h, "crop_frac": crop_frac,
                "angle_inclination_deg": incl if incl is not None else "",
                "label": rec.get("gt_label", ""), "neg_class": "",
            }
            for col in SCORE_COLS:
                row[col] = ""
            if angle_score is not None:
                row["angle"] = angle_score
            for col, sc in scorers.items():
                try:
                    row[col] = sc.score(crop)
                except Exception as e:
                    log(f"  [score-err] {col} on {crop_name}: {e}")
            rows_out.append(row)
            n_foot += 1

    # write CSV
    fields = ["pose_id", "person_id", "foot", "image_path", "crop",
              "crop_w", "crop_h", "crop_frac", "angle_inclination_deg",
              *SCORE_COLS, "label", "neg_class"]
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # sort key for the review sheet: best available appearance model, else angle
    rank_col = next((c for c in ["fashionSigLIP", "fashionCLIP", "yoloworld", "angle"]
                     if any(isinstance(r.get(c), (int, float)) for r in rows_out)), None)
    write_html(out, rows_out, rank_col)

    if missing:
        log(f"[{args.profile}] {len(missing)} image paths unresolved (e.g. {list(missing)[:3]})")
    summary = (f"RESULT {args.profile}: poses={n_pose} foot_crops={n_foot} "
               f"skipped_no_path={n_skip_path} skipped_low_foot_conf={n_skip_foot} "
               f"models={'+'.join([*scorers.keys()] + (['angle'] if do_angle else []))} "
               f"out={csv_path}")
    log(f"[{args.profile}] wrote {csv_path} and review.html ({n_foot} crops)")
    print(summary)
    return 0


def write_html(out, rows, rank_col):
    if rank_col:
        rows = sorted(rows, key=lambda r: r[rank_col] if isinstance(r.get(rank_col), (int, float)) else -1,
                      reverse=True)
    tiles = []
    for r in rows:
        scores = " ".join(
            f"<b>{c}</b>={r[c]}" if c == rank_col else f"{c}={r[c]}"
            for c in SCORE_COLS if isinstance(r.get(c), (int, float))
        )
        cap = (f"{scores}<br>frac={r['crop_frac']} incl={r['angle_inclination_deg']}"
               f"<br><span class=p title='{r['image_path']}'>{Path(r['image_path']).name}</span>")
        tiles.append(
            f"<div class=t><img src='{r['crop']}' loading=lazy>"
            f"<div class=c>{cap}</div></div>"
        )
    html = f"""<!doctype html><meta charset=utf-8>
<title>Heel detection eval</title>
<style>
 body{{font:12px/1.4 -apple-system,sans-serif;background:#111;color:#ddd;margin:16px}}
 h1{{font-size:15px}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .t{{width:150px;background:#1c1c1c;border-radius:6px;padding:6px}}
 .t img{{width:138px;height:138px;object-fit:contain;background:#000;border-radius:4px}}
 .c{{font-size:11px;margin-top:4px;word-break:break-word}} .p{{color:#888}}
 b{{color:#7fd}}
</style>
<h1>Heel detection eval &mdash; {len(rows)} foot crops, sorted by {rank_col or 'n/a'} (desc)</h1>
<p>Eyeball: precision = how clean the top is; recall failures = real heels sunk to the bottom.
Then fill the <code>label</code> column (heel/not/ambiguous) in results.csv and run with --score-labels.</p>
<div class=grid>{''.join(tiles)}</div>
"""
    (out / "review.html").write_text(html)


# ---------------------------------------------------------------------------
# SCORE mode
# ---------------------------------------------------------------------------
def bucket(frac):
    try:
        f = float(frac)
    except (TypeError, ValueError):
        return "?"
    if f < 0.01:
        return "tiny(<1%)"
    if f < 0.05:
        return "small(1-5%)"
    return "large(>=5%)"


def score_labels(args):
    path = Path(args.score_labels)
    rows = list(csv.DictReader(open(path)))
    labeled = [r for r in rows if (r.get("label") or "").strip().lower() in ("heel", "not")]
    if not labeled:
        log(f"no usable labels in {path}; fill the `label` column with heel/not/ambiguous")
        return 1

    def is_pos(r):
        return (r["label"].strip().lower() == "heel")

    models = [c for c in SCORE_COLS
              if any((r.get(c) or "").strip() not in ("", "nan") for r in labeled)]
    thresholds = [0.5, 0.7, 0.9]
    n_pos = sum(is_pos(r) for r in labeled)
    n_neg = len(labeled) - n_pos
    log(f"labeled: {len(labeled)} ({n_pos} heel / {n_neg} not); models={models}")

    report = {"n_labeled": len(labeled), "n_pos": n_pos, "n_neg": n_neg, "models": {}}
    lines = [f"\n=== Heel-eval metrics ({len(labeled)} labeled: {n_pos} heel / {n_neg} not) ==="]

    for m in models:
        def val(r):
            try:
                return float(r[m])
            except (TypeError, ValueError):
                return None
        usable = [r for r in labeled if val(r) is not None]
        report["models"][m] = {"n": len(usable), "thresholds": {}}
        lines.append(f"\n[{m}]  (scored on {len(usable)} of {len(labeled)})")
        lines.append(f"  {'thr':>4} {'P':>6} {'R':>6} {'F1':>6}   TP/FP/FN/TN")
        for t in thresholds:
            tp = sum(1 for r in usable if is_pos(r) and val(r) >= t)
            fp = sum(1 for r in usable if not is_pos(r) and val(r) >= t)
            fn = sum(1 for r in usable if is_pos(r) and val(r) < t)
            tn = sum(1 for r in usable if not is_pos(r) and val(r) < t)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            report["models"][m]["thresholds"][str(t)] = dict(
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                tp=tp, fp=fp, fn=fn, tn=tn)
            lines.append(f"  {t:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}   {tp}/{fp}/{fn}/{tn}")

        # crop-size breakout at the middle threshold (recall on positives by size)
        t = 0.7
        by_size = {}
        for r in usable:
            if not is_pos(r):
                continue
            b = bucket(r.get("crop_frac"))
            by_size.setdefault(b, [0, 0])
            by_size[b][1] += 1
            if val(r) >= t:
                by_size[b][0] += 1
        if by_size:
            lines.append(f"  recall@{t} by crop size: " +
                         "  ".join(f"{b}={hit}/{tot}" for b, (hit, tot) in sorted(by_size.items())))
            report["models"][m]["recall_by_size@0.7"] = {b: f"{h}/{t2}" for b, (h, t2) in by_size.items()}

        # false-positive breakout by neg_class (if provided)
        t = 0.7
        by_neg = {}
        for r in usable:
            if is_pos(r):
                continue
            nc = (r.get("neg_class") or "").strip() or "(unspecified)"
            by_neg.setdefault(nc, [0, 0])
            by_neg[nc][1] += 1
            if val(r) >= t:
                by_neg[nc][0] += 1
        if any(k != "(unspecified)" for k in by_neg):
            lines.append(f"  false-pos@{t} by neg_class: " +
                         "  ".join(f"{nc}={fp}/{tot}" for nc, (fp, tot) in sorted(by_neg.items())))
            report["models"][m]["fp_by_negclass@0.7"] = {nc: f"{fp}/{t2}" for nc, (fp, t2) in by_neg.items()}

    out_json = path.parent / "metrics.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("\n".join(lines))
    print(f"\nmetrics.json -> {out_json}")
    return 0


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="High-heel detection feasibility eval (read-only)")
    ap.add_argument("--profile", default=os.environ.get("DB_PROFILE", "irl"),
                    choices=["irl", "2d", "3d"])
    ap.add_argument("--sample", type=int, default=None, help="random N poses (diverse review set)")
    ap.add_argument("--limit", type=int, default=None, help="first N poses by id (no --sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positive-substr", default=None,
                    help="GROUND-TRUTH mode: stored-path substring; matching images -> label=heel")
    ap.add_argument("--negative-substr", default=None,
                    help="comma-list of path substrings for negatives (default: random non-positive)")
    ap.add_argument("--per-class", type=int, default=None, help="sample N per class in ground-truth mode")
    ap.add_argument("--remap", action="append", default=None,
                    help="stale-path fix 'OLD=>NEW' substring swap (repeatable)")
    ap.add_argument("--min-foot-conf", type=float, default=0.3,
                    help="foot-keypoint confidence gate (matches GeometricFeatureExtractor default)")
    ap.add_argument("--models", default=",".join(SCORE_COLS),
                    help="comma list of: fashionSigLIP,fashionCLIP,yoloworld,angle")
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (default: settings.DEVICE)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "reports" / "heel_eval"))
    ap.add_argument("--score-labels", default=None,
                    help="SCORE mode: path to hand-labeled results.csv")
    args = ap.parse_args(argv)

    if args.score_labels:
        return score_labels(args)

    if args.device is None:
        os.environ["DB_PROFILE"] = args.profile
        from src.config.settings import settings
        args.device = settings.DEVICE
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
