"""
Local before/after (A/B) report generator for model-quality comparisons.

Renders side-by-side annotated images + metrics into a single self-contained
HTML file under data/reports/ab/<name>/report.html. Everything stays on disk
(images embedded as base64 data URIs) — nothing is uploaded anywhere.

Used as a library by comparison scripts; also demoable standalone. Layout is
deliberately fixed so every proposed model change is judged the same way:
summary verdict table on top, then per-image rows with pane A (incumbent) vs
pane B (candidate), each with its own metric strip.

Usage (library):
    from scripts.ab_report import ABReport, draw_boxes, draw_keypoints
    rep = ABReport("person-conf-sweep", "YOLOv8x conf 0.05 vs 0.25",
                   label_a="conf=0.05 (current)", label_b="conf=0.25")
    rep.add_row(image_path, img_a, img_b, metrics_a={...}, metrics_b={...}, note="...")
    rep.add_summary({"metric": ..., "A": ..., "B": ...})
    path = rep.write()   # -> data/reports/ab/person-conf-sweep/report.html
"""
from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# COCO-WholeBody skeleton (body subset) for pose overlays; feet drawn emphasized.
_BODY_LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
               (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
_FOOT_LINKS = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
               (17, 18), (19, 17), (20, 21), (22, 20)]


def _to_pil(img):
    from PIL import Image as PILImage
    if isinstance(img, np.ndarray):
        return PILImage.fromarray(img.astype(np.uint8))
    return img


def draw_boxes(image_rgb, boxes, color=(80, 220, 120), width=4, labels=None):
    """boxes: iterable of [x1,y1,x2,y2]; labels: optional per-box strings."""
    from PIL import ImageDraw, ImageFont
    im = _to_pil(image_rgb).convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels and i < len(labels) and labels[i]:
            txt = str(labels[i])
            ty = max(0, y1 - 18)
            d.rectangle([x1, ty, x1 + 8 + 8 * len(txt), ty + 18], fill=color)
            d.text((x1 + 4, ty + 2), txt, fill=(0, 0, 0))
    return im


def draw_keypoints(image_rgb, keypoints, min_conf=0.3, radius=4,
                   body_color=(90, 200, 255), foot_color=(255, 120, 90)):
    """keypoints: (133,3) [x,y,conf]. Feet (17-22) drawn larger in foot_color."""
    from PIL import ImageDraw
    im = _to_pil(image_rgb).convert("RGB").copy()
    d = ImageDraw.Draw(im)
    kp = np.asarray(keypoints, dtype=float).reshape(-1, 3)

    def ok(i):
        return kp[i, 2] >= min_conf and not (kp[i, 0] == 0 and kp[i, 1] == 0)

    for links, color, w in ((_BODY_LINKS, body_color, 3), (_FOOT_LINKS, foot_color, 4)):
        for a, b in links:
            if a < len(kp) and b < len(kp) and ok(a) and ok(b):
                d.line([kp[a, 0], kp[a, 1], kp[b, 0], kp[b, 1]], fill=color, width=w)
    for i in range(min(23, len(kp))):  # body + feet points only (hands/face too dense)
        if ok(i):
            r = radius + (2 if 17 <= i <= 22 else 0)
            c = foot_color if 17 <= i <= 22 else body_color
            d.ellipse([kp[i, 0] - r, kp[i, 1] - r, kp[i, 0] + r, kp[i, 1] + r], fill=c)
    return im


def _b64(img, max_w=680):
    im = _to_pil(img).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _metric_strip(metrics):
    if not metrics:
        return ""
    cells = "".join(
        f"<span class=m><b>{html.escape(str(k))}</b> {html.escape(str(v))}</span>"
        for k, v in metrics.items())
    return f"<div class=ms>{cells}</div>"


class ABReport:
    def __init__(self, name, title, label_a="A (current)", label_b="B (candidate)"):
        self.name = name
        self.title = title
        self.label_a = label_a
        self.label_b = label_b
        self.rows = []
        self.summary = []   # list of dicts (columns free-form, rendered as table)
        self.verdict = ""

    def add_row(self, image_path, img_a, img_b, metrics_a=None, metrics_b=None, note=""):
        self.rows.append(dict(path=str(image_path), a=_b64(img_a), b=_b64(img_b),
                              ma=metrics_a or {}, mb=metrics_b or {}, note=note))

    def add_summary(self, row: dict):
        self.summary.append(row)

    def set_verdict(self, text):
        self.verdict = text

    def write(self, open_in_browser=False):
        out_dir = PROJECT_ROOT / "data" / "reports" / "ab" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "report.html"

        srows = ""
        if self.summary:
            cols = list(self.summary[0].keys())
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                for r in self.summary)
            srows = f"<table class=sum><tr>{head}</tr>{body}</table>"

        tiles = []
        for r in self.rows:
            fname = html.escape(Path(r["path"]).name)
            note = f"<div class=note>{html.escape(r['note'])}</div>" if r["note"] else ""
            tiles.append(f"""
<div class=row>
  <div class=fn title="{html.escape(r['path'])}">{fname}</div>
  <div class=panes>
    <div class=pane><div class=pl>{html.escape(self.label_a)}</div>
      <img src="{r['a']}" loading=lazy>{_metric_strip(r['ma'])}</div>
    <div class=pane><div class=pl pb>{html.escape(self.label_b)}</div>
      <img src="{r['b']}" loading=lazy>{_metric_strip(r['mb'])}</div>
  </div>{note}
</div>""")

        verdict = f"<div class=verdict>{html.escape(self.verdict)}</div>" if self.verdict else ""
        doc = f"""<!doctype html><meta charset=utf-8>
<title>A/B: {html.escape(self.title)}</title>
<style>
 body{{font:13px/1.45 -apple-system,sans-serif;background:#101014;color:#ddd;margin:20px;max-width:1480px}}
 h1{{font-size:17px}} .sub{{color:#888;margin-bottom:14px}}
 .verdict{{background:#1c2a1c;border:1px solid #2f5d2f;border-radius:8px;padding:10px 14px;margin:12px 0;font-size:14px}}
 table.sum{{border-collapse:collapse;margin:10px 0 18px}}
 table.sum th,table.sum td{{border:1px solid #333;padding:5px 12px;font-size:12px}}
 table.sum th{{background:#1b1b22;text-align:left}}
 .row{{margin:20px 0;padding:12px;background:#17171c;border-radius:10px}}
 .fn{{color:#9aa;font-size:12px;margin-bottom:8px}}
 .panes{{display:flex;gap:14px;flex-wrap:wrap}}
 .pane{{flex:1;min-width:340px}}
 .pane img{{width:100%;border-radius:6px;background:#000}}
 .pl{{font-weight:600;font-size:12px;margin-bottom:6px;color:#8ec7ff}}
 .pl[pb]{{color:#8fe3a1}}
 .ms{{margin-top:6px}} .m{{display:inline-block;background:#22222a;border-radius:5px;padding:2px 8px;margin:2px 4px 0 0;font-size:11px}}
 .m b{{color:#7fd}}
 .note{{margin-top:8px;color:#c9b458;font-size:12px}}
</style>
<h1>Before / After — {html.escape(self.title)}</h1>
<div class=sub>{len(self.rows)} comparison(s) · pane A = {html.escape(self.label_a)} · pane B = {html.escape(self.label_b)} · local file, nothing uploaded</div>
{verdict}
{srows}
{''.join(tiles)}
"""
        out.write_text(doc)
        if open_in_browser:
            import subprocess
            subprocess.run(["open", str(out)], check=False)
        print(f"AB report: {out}", file=sys.stderr)
        return out
