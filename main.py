import os
import io
import base64
import csv
import json
import datetime
import subprocess
import math
from typing import Optional
from urllib.parse import urlparse, unquote

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import traceback

load_dotenv(override=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
PYTHON_EXE  = os.path.join(BASE_DIR, "my_env", "bin", "python3")

LARAVEL_SECRET        = os.getenv("LARAVEL_SECRET", "")
MODEL_PATH            = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "Annotator", "Model", "best.pt"))
CONFIDENCE_THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
DECISION_WEBHOOK      = os.getenv("DECISION_WEBHOOK", "")
DECISIONS_LOG         = os.getenv("DECISIONS_LOG", os.path.join(BASE_DIR, "decisions.jsonl"))

# Annotator / Browser / Review — resolved relative to this file
ANNOTATOR_SCRIPT = os.getenv("ANNOTATOR_SCRIPT", os.path.join(SCRIPTS_DIR, "annotate.py"))
BROWSER_SCRIPT   = os.getenv("BROWSER_SCRIPT",   os.path.join(SCRIPTS_DIR, "browser.py"))
REVIEW_SCRIPT    = os.getenv("REVIEW_SCRIPT",    os.path.join(SCRIPTS_DIR, "review.py"))
ANNOTATOR_PYTHON = os.getenv("ANNOTATOR_PYTHON", PYTHON_EXE)

# Data paths
DATA_DIR     = os.getenv("DATA_DIR",     os.path.join(BASE_DIR, "data"))
INDEX_CSV    = os.getenv("INDEX_CSV",    os.path.join(DATA_DIR, "index.csv"))
ANNOTATIONS  = os.getenv("ANNOTATIONS", os.path.join(DATA_DIR, "annotations.json"))
RAW_DIR      = os.getenv("RAW_DIR",     os.path.join(DATA_DIR, "raw"))
REVIEW_DIR   = os.getenv("REVIEW_DIR",  os.path.join(DATA_DIR, "review_queue"))

app = FastAPI(title="onesvs YOLO Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load model once at startup — not on every request
model = YOLO(MODEL_PATH)


# Ensure data directories exist
for _d in [DATA_DIR, RAW_DIR, REVIEW_DIR,
           os.path.join(DATA_DIR, "approved"),
           os.path.join(DATA_DIR, "recycle"),
           os.path.join(DATA_DIR, "skipped")]:
    os.makedirs(_d, exist_ok=True)


def _fix_orientation(pil_image: Image.Image) -> Image.Image:
    """Rotate image to match EXIF orientation tag so phone photos aren't sideways."""
    try:
        from PIL import ExifTags
        exif = pil_image.getexif()
        orientation_key = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            pil_image = pil_image.rotate(rotations[orientation], expand=True)
    except Exception:
        pass  # No EXIF or no orientation tag — leave image as-is
    return pil_image


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x_laravel_secret: Optional[str] = Header(None),
):
    if LARAVEL_SECRET and x_laravel_secret != LARAVEL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        contents = await image.read()
        pil_image = _fix_orientation(Image.open(io.BytesIO(contents)).convert("RGB"))
        results = model(pil_image)

        detections = []
        for r in results:
            boxes = r.boxes
            masks = r.masks

            for i, box in enumerate(boxes):
                conf      = float(box.conf)
                class_idx = int(box.cls)

                polygon = []
                if masks is not None and i < len(masks.xy):
                    polygon = [[round(float(x), 2), round(float(y), 2)]
                               for x, y in masks.xy[i]]

                detections.append({
                    "class":      model.names[class_idx],
                    "confidence": round(conf, 3),
                    "box":        [round(x, 1) for x in box.xyxy[0].tolist()],
                    "polygon":    polygon,
                })

        top_conf = max((d["confidence"] for d in detections), default=0)

        return {
            "detections":           detections,
            "review_needed":        top_conf < CONFIDENCE_THRESHOLD or len(detections) == 0,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _load_coco() -> dict:
    """Load annotations.json (COCO format). Returns empty structure if missing."""
    if os.path.exists(ANNOTATIONS):
        with open(ANNOTATIONS) as f:
            return json.load(f)
    return {"images": [], "annotations": [], "categories": []}


def _draw_annotations(img: Image.Image, fname: str, coco: dict) -> Image.Image:
    """Draw bounding boxes + labels onto a PIL image. Returns a copy."""
    from PIL import ImageDraw, ImageFont

    # Build lookups
    img_record = next((i for i in coco["images"] if i["file_name"] == fname), None)
    if not img_record:
        return img

    cat_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns = [a for a in coco.get("annotations", []) if a["image_id"] == img_record["id"]]

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    colors = ["#00d4ff","#1abc9c","#e67e22","#e74c3c","#9b59b6","#3498db",
              "#f1c40f","#2ecc71","#e91e63","#ff5722"]

    for i, ann in enumerate(anns):
        color = colors[i % len(colors)]
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = cat_by_id.get(ann.get("category_id"), "?")
        bbox_text = draw.textbbox((x + 4, y + 4), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x + 4, y + 4), label, fill="#fff", font=font)

    return img


def _queue_photos() -> list[str]:
    """Return sorted list of image filenames in the review queue."""
    if not os.path.isdir(REVIEW_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        f for f in os.listdir(REVIEW_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )


def _decision_counts() -> dict:
    """Count approved / rejected / skipped from decisions.jsonl."""
    counts = {"approved": 0, "reject": 0, "skipped": 0}
    if not os.path.exists(DECISIONS_LOG):
        return counts
    try:
        with open(DECISIONS_LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    action = entry.get("action", "")
                    if action == "approve":
                        counts["approved"] += 1
                    elif action == "reject":
                        counts["reject"] += 1
                    elif action == "skip":
                        counts["skipped"] += 1
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return counts


# ── Index CSV helpers ─────────────────────────────────────────────────────────

INDEX_FIELDNAMES = [
    "image_id", "company_id", "company_name",
    "job_id", "url", "status", "annotated_by", "date", "notes",
]

def _load_index() -> list[dict]:
    """Load index.csv. Returns empty list if missing."""
    if not os.path.exists(INDEX_CSV):
        return []
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_index(rows: list[dict]):
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _index_counts(rows: list[dict]) -> dict:
    counts = {"approved": 0, "rejected": 0, "pending": 0}
    for r in rows:
        s = r.get("status", "pending")
        if s in counts:
            counts[s] += 1
    return counts


def _fetch_image_from_url(url: str) -> tuple[str, str]:
    """
    Fetch image from a cloud URL, run YOLO inference, return (base64_jpeg, labels_html).
    Returns ("", error_message) on failure.
    """
    import urllib.request
    import html as _html

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "onesvs-review/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
        from PIL import ImageOps
        img = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw)).convert("RGB")
        )
    except Exception as e:
        return "", f"Could not fetch image: {_html.escape(str(e))}"

    # Run YOLO inference
    labels = []
    try:
        from PIL import ImageDraw, ImageFont
        results = model(img)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        colors = ["#00d4ff","#1abc9c","#e67e22","#e74c3c","#9b59b6",
                  "#3498db","#f1c40f","#2ecc71","#e91e63","#ff5722"]
        for res in results:
            masks = res.masks
            for i, box in enumerate(res.boxes):
                color        = colors[i % len(colors)]
                x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf         = float(box.conf)
                label        = model.names[int(box.cls)]
                if masks is not None and i < len(masks.xy):
                    pts = [(float(x), float(y)) for x, y in masks.xy[i]]
                    if len(pts) >= 3:
                        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        odraw   = ImageDraw.Draw(overlay)
                        rv, gv, bv = tuple(int(color.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                        odraw.polygon(pts, fill=(rv, gv, bv, 80))
                        odraw.line(pts + [pts[0]], fill=color, width=3)
                        img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                        draw = ImageDraw.Draw(img)
                else:
                    draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
                tag       = f"{label} {conf:.0%}"
                bbox_text = draw.textbbox((x1+4, y1+4), tag, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1+4, y1+4), tag, fill="#fff", font=font)
                labels.append(label)
        labels = list(set(labels))
    except Exception:
        pass  # Show image without annotations if inference fails

    max_w = 1000
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    label_pills = "".join(
        f'<span class="pill">{l}</span>' for l in sorted(labels)
    ) or '<span style="color:#888">No detections</span>'

    return img_b64, label_pills


@app.get("/", response_class=HTMLResponse)
def home():
    """Simple approve / reject review page — driven by index.csv cloud URLs."""
    import html as _html

    rows   = _load_index()
    counts = _index_counts(rows)
    queue  = [r for r in rows if r.get("status", "pending") == "pending"]

    approved_count = counts["approved"]
    rejected_count = counts["rejected"]

    # ── Empty state ───────────────────────────────────────────────────────────
    if not queue:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pii Review</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{ font-family: -apple-system, sans-serif; background: #0f0f1a;
                       color: #fff; display: flex; align-items: center;
                       justify-content: center; min-height: 100vh; text-align: center; }}
                .card {{ padding: 48px; }}
                h1 {{ font-size: 28px; margin-bottom: 12px; }}
                p  {{ color: #888; font-size: 16px; margin-top: 8px; }}
                .stats {{ display: flex; gap: 24px; justify-content: center;
                          margin-top: 24px; }}
                .stat {{ background: #1a1a2e; border-radius: 10px; padding: 14px 28px; }}
                .stat-num {{ font-size: 26px; font-weight: 700; }}
                .stat-lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
                .green {{ color: #06d6a0; }}
                .red   {{ color: #e63946; }}
                .countdown {{ margin-top: 28px; font-size: 13px; color: #555; }}
                a  {{ color: #00d4ff; text-decoration: none; font-size: 14px;
                     display: inline-block; margin-top: 32px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>✅ All caught up!</h1>
                <p>No photos waiting for review.</p>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-num green">{approved_count}</div>
                        <div class="stat-lbl">Approved</div>
                    </div>
                    <div class="stat">
                        <div class="stat-num red">{rejected_count}</div>
                        <div class="stat-lbl">Rejected</div>
                    </div>
                </div>
                <div class="countdown">Checking for new photos in <span id="cd">30</span>s…</div>
                <a href="/tools">Go to tools →</a>
            </div>
            <script>
                let secs = 30;
                const el = document.getElementById('cd');
                setInterval(() => {{
                    secs--;
                    el.textContent = secs;
                    if (secs <= 0) location.reload();
                }}, 1000);
            </script>
        </body>
        </html>
        """

    # ── Next pending photo from index.csv ─────────────────────────────────────
    row         = queue[0]
    total       = len(queue)
    image_id    = str(row.get("image_id", ""))
    url         = row.get("url", "")
    company     = _html.escape(row.get("company_name", ""))
    job_id      = _html.escape(str(row.get("job_id", "")))
    notes       = _html.escape(row.get("notes", ""))

    img_b64, label_pills = _fetch_image_from_url(url)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pii Review</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    background: #0f0f1a; color: #fff; min-height: 100vh;
                    display: flex; flex-direction: column; align-items: center;
                    padding: 24px 16px; }}

            .topbar {{ width: 100%; max-width: 960px; display: flex;
                       justify-content: space-between; align-items: center;
                       margin-bottom: 20px; }}
            .topbar h1 {{ font-size: 20px; color: #00d4ff; }}
            .progress {{ display: flex; gap: 16px; align-items: center; }}
            .progress span {{ font-size: 13px; }}
            .p-approved {{ color: #06d6a0; font-weight: 600; }}
            .p-rejected {{ color: #e63946; font-weight: 600; }}
            .p-remaining {{ color: #888; }}

            .photo-wrap {{ width: 100%; max-width: 960px; border-radius: 12px;
                           overflow: hidden; background: #1a1a2e;
                           box-shadow: 0 4px 24px rgba(0,0,0,0.5); }}
            .photo-wrap img {{ width: 100%; display: block; }}
            .no-img {{ padding: 80px; text-align: center; color: #888; }}

            .meta {{ width: 100%; max-width: 960px; margin-top: 16px;
                     display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
            .pill {{ background: #1e3a5f; color: #00d4ff; border-radius: 20px;
                     padding: 4px 14px; font-size: 13px; font-weight: 600; }}

            .actions {{ display: flex; gap: 12px; margin-top: 28px;
                        width: 100%; max-width: 960px; }}
            .btn {{ flex: 1; padding: 20px; font-size: 18px; font-weight: 700;
                    border: none; border-radius: 12px; cursor: pointer;
                    transition: transform 0.1s, opacity 0.1s; letter-spacing: 0.5px; }}
            .btn:active {{ transform: scale(0.97); opacity: 0.85; }}
            .approve {{ background: #06d6a0; color: #0f0f1a; }}
            .reject  {{ background: #e63946; color: #fff; }}
            .skip    {{ background: #2a2a3e; color: #aaa; flex: 0 0 auto;
                        padding: 20px 28px; font-size: 16px; }}
            .skip:hover {{ background: #3a3a5e; color: #ccc; }}

            .shortcuts {{ width: 100%; max-width: 960px; margin-top: 10px;
                          text-align: center; font-size: 11px; color: #444; }}
            .kbd {{ background: #1a1a2e; border: 1px solid #333; border-radius: 4px;
                    padding: 1px 5px; font-family: monospace; color: #666; }}

            .fname {{ font-size: 11px; color: #444; margin-top: 14px; }}
            a.tools-link {{ color: #555; font-size: 12px; margin-top: 20px;
                            text-decoration: none; }}
            a.tools-link:hover {{ color: #888; }}
        </style>
    </head>
    <body>

        <div class="topbar">
            <h1>📋 Pii Review</h1>
            <div class="progress">
                <span class="p-approved">✅ {approved_count} approved</span>
                <span class="p-rejected">❌ {rejected_count} rejected</span>
                <span class="p-remaining">{total} remaining</span>
            </div>
        </div>

        <div class="photo-wrap">
            {"<img src='data:image/jpeg;base64," + img_b64 + "' alt='Review photo'>" if img_b64
             else '<div class="no-img">Could not load image</div>'}
        </div>

        <div class="meta">{label_pills}</div>

        <div class="actions">
            <form method="post" action="/decision" style="flex:1;display:contents;" id="form-approve">
                <input type="hidden" name="image_id" value="{image_id}">
                <input type="hidden" name="source"   value="index">
                <button class="btn approve" name="action" value="approve">✅ Approve</button>
            </form>
            <form method="post" action="/decision" style="flex:1;display:contents;" id="form-reject">
                <input type="hidden" name="image_id" value="{image_id}">
                <input type="hidden" name="source"   value="index">
                <button class="btn reject" name="action" value="reject">❌ Reject</button>
            </form>
            <form method="post" action="/decision" id="form-skip">
                <input type="hidden" name="image_id" value="{image_id}">
                <input type="hidden" name="source"   value="index">
                <button class="btn skip" name="action" value="skip">⏭ Skip</button>
            </form>
        </div>

        <div class="shortcuts">
            <span class="kbd">A</span> approve &nbsp;
            <span class="kbd">R</span> reject &nbsp;
            <span class="kbd">S</span> skip
        </div>

        <p class="fname">{company} · Job {job_id}{" · " + notes if notes else ""}</p>
        <div style="display:flex;gap:20px;margin-top:16px;">
            <a class="tools-link" href="/gallery">🖼 Browse all</a>
            <a class="tools-link" href="/tools">⚙ Tools</a>
        </div>

    </body>
    <script>
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const key = e.key.toLowerCase();
            if (key === 'a') document.querySelector('#form-approve button').click();
            if (key === 'r') document.querySelector('#form-reject button').click();
            if (key === 's') document.querySelector('#form-skip button').click();
        }});
    </script>
    </html>
    """

@app.get("/inference", response_class=HTMLResponse)
def inference_home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>onesvs — YOLO Inference</title>
        <style>
            body  {{ font-family: sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }}
            h2    {{ color: #1a1a2e; }}
            .card {{ background: #fff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 24px; }}
            label {{ display: block; font-weight: bold; margin-bottom: 8px; color: #1a1a2e; }}
            input[type=file], input[type=url] {{ width: 100%; box-sizing: border-box;
                padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; margin-bottom: 12px; }}
            input[type=submit] {{ background: #00b4d8; color: #fff; border: none; padding: 10px 28px;
                                 border-radius: 6px; cursor: pointer; font-size: 15px; }}
            input[type=submit]:hover {{ background: #0096c7; }}
            .section-title {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
                               color: #888; margin-bottom: 16px; }}
            hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 0 0 20px 0; }}
        </style>
    </head>
    <body>
        <h2>onesvs YOLO Inference</h2>

        <div class="card">
            <p class="section-title">Continuous Learning — Review Flagged Image</p>
            <hr>
            <form action="/review" method="get">
                <label for="url">Cloud image URL (flagged for review)</label>
                <input id="url" name="url" type="url"
                       placeholder="https://storage.example.com/jobs/photo.jpg" required>
                <input type="submit" value="Review Image">
            </form>
        </div>

        <div class="card">
            <p class="section-title">Local Test — Upload a Photo</p>
            <hr>
            <form action="/preview" method="post" enctype="multipart/form-data">
                <label for="image">Upload a photo to run inference and see detections.</label>
                <input id="image" name="image" type="file" accept="image/*" required>
                <input type="submit" value="Run Detection">
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/preview", response_class=HTMLResponse)
async def preview(image: UploadFile = File(...)):
    """Browser-friendly endpoint — returns the annotated image + detection table."""
    try:
        contents = await image.read()
        pil_image = _fix_orientation(Image.open(io.BytesIO(contents)).convert("RGB"))
        results = model(pil_image)

        # Draw detections on the image using YOLO's built-in renderer
        annotated = results[0].plot()  # numpy BGR array
        annotated_rgb = Image.fromarray(annotated[..., ::-1])  # BGR → RGB
        buf = io.BytesIO()
        annotated_rgb.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Build detection rows
        detections = []
        for r in results:
            for i, box in enumerate(r.boxes):
                conf      = float(box.conf)
                class_idx = int(box.cls)
                detections.append({
                    "class":      model.names[class_idx],
                    "confidence": round(conf * 100, 1),
                    "review":     conf < CONFIDENCE_THRESHOLD,
                })

        rows = ""
        for d in detections:
            color = "#ff4444" if d["review"] else "#06d6a0"
            rows += f"""
            <tr>
                <td>{d['class']}</td>
                <td><b style="color:{color}">{d['confidence']}%</b></td>
                <td style="color:{color}">{"⚠ Review" if d['review'] else "✓ Pass"}</td>
            </tr>"""

        if not detections:
            rows = "<tr><td colspan='3' style='color:#aaa'>No detections above threshold</td></tr>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detection Result</title>
            <style>
                body  {{ font-family: sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }}
                img   {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff;
                         border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                th    {{ background: #1a1a2e; color: #fff; padding: 12px 16px; text-align: left; }}
                td    {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
                a     {{ color: #00b4d8; }}
            </style>
        </head>
        <body>
            <h2>Detection Result</h2>
            <img src="data:image/jpeg;base64,{img_b64}" alt="Annotated prediction">
            <table>
                <thead><tr><th>Class</th><th>Confidence</th><th>Status</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <p><a href="/">← Upload another image</a></p>
        </body>
        </html>
        """

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_image(url: str) -> tuple[bytes, "Image.Image"]:
    """Synchronous helper — fetches an image URL and returns (raw_bytes, PIL image)."""
    import html as _html
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="url must be an http or https link")
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "onesvs-review/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    pil = _fix_orientation(Image.open(io.BytesIO(raw)).convert("RGB"))
    return raw, pil


def _run_inference(pil_image):
    """Run YOLO and return (results, img_b64, detections_list)."""
    results = model(pil_image)

    # Polygon / mask overlay — boxes=False keeps it clean for seg models;
    # falls back gracefully to bboxes if the model is detection-only.
    annotated     = results[0].plot(boxes=False)
    annotated_rgb = Image.fromarray(annotated[..., ::-1])
    buf = io.BytesIO()
    annotated_rgb.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    detections = []
    for r in results:
        masks = r.masks
        for i, box in enumerate(r.boxes):
            conf      = float(box.conf)
            class_idx = int(box.cls)
            polygon   = []
            if masks is not None and i < len(masks.xy):
                polygon = [[round(float(x), 2), round(float(y), 2)]
                           for x, y in masks.xy[i]]
            detections.append({
                "class":      model.names[class_idx],
                "confidence": round(conf * 100, 1),
                "review":     conf < CONFIDENCE_THRESHOLD,
                "polygon":    polygon,
            })

    return results, img_b64, detections


def _detection_rows_html(detections: list) -> str:
    if not detections:
        return "<tr><td colspan='3' style='color:#aaa'>No detections above threshold</td></tr>"
    rows = ""
    for d in detections:
        color = "#ff4444" if d["review"] else "#06d6a0"
        poly_note = f" ({len(d['polygon'])} pts)" if d["polygon"] else " (bbox)"
        rows += f"""
        <tr>
            <td>{d['class']}</td>
            <td><b style="color:{color}">{d['confidence']}%</b></td>
            <td style="color:{color}">{"⚠ Review" if d['review'] else "✓ Pass"}{poly_note}</td>
        </tr>"""
    return rows


@app.get("/review", response_class=HTMLResponse)
async def review(url: str):
    """
    Correct & Approve review page.
    Fetches the cloud image, runs YOLO, then presents an interactive canvas
    where the reviewer can move/resize/relabel/delete boxes and draw new ones
    before approving or rejecting.
    """
    import html as _html

    try:
        _, pil_image = _fetch_image(url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {e}")

    try:
        img_w, img_h = pil_image.size
        _, img_b64, detections = _run_inference(pil_image)
        safe_url   = _html.escape(url, quote=True)
        class_names = list(model.names.values())
        classes_js  = json.dumps(class_names)
        dets_js     = json.dumps([
            {"label": d["class"], "conf": d["confidence"],
             "x1": d["box"][0], "y1": d["box"][1],
             "x2": d["box"][2], "y2": d["box"][3]}
            for d in detections
        ])
        top_conf    = max((d["confidence"] for d in detections), default=0)
        banner_color = "#ff4444" if top_conf < CONFIDENCE_THRESHOLD * 100 else "#06d6a0"
        banner_text  = f"Model confidence: {top_conf}% — {len(detections)} detection(s)"

        return f"""<!DOCTYPE html><html>
<head>
<title>Review &amp; Correct</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:sans-serif;background:#0f0f1a;color:#fff;display:flex;flex-direction:column;height:100vh}}
#topbar{{background:#16213e;padding:10px 16px;display:flex;align-items:center;gap:12px;flex-shrink:0}}
#topbar h2{{font-size:16px;color:#00d4ff;flex:1}}
.banner{{padding:6px 14px;border-radius:6px;font-size:13px;font-weight:bold;background:{banner_color}}}
.source{{font-size:11px;color:#aaa;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
#main{{display:flex;flex:1;overflow:hidden}}
#canvas-wrap{{flex:1;position:relative;overflow:hidden;background:#1a1a2e;display:flex;align-items:center;justify-content:center}}
canvas{{cursor:crosshair}}
#sidebar{{width:280px;background:#16213e;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}}
#sidebar-header{{padding:10px 12px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#aaa;border-bottom:1px solid #2a2a4a}}
#det-list{{flex:1;overflow-y:auto;padding:6px}}
.det-item{{background:#1a1a2e;border-radius:6px;padding:8px 10px;margin-bottom:6px;border:2px solid transparent;cursor:pointer}}
.det-item.selected{{border-color:#00d4ff}}
.det-item .conf{{font-size:11px;color:#aaa;margin-bottom:4px}}
.det-item select{{width:100%;background:#0f0f1a;color:#fff;border:1px solid #2a2a4a;border-radius:4px;padding:4px;font-size:13px}}
.det-item .del{{float:right;background:#e74c3c;border:none;color:#fff;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px;margin-left:6px}}
#actions{{padding:10px 12px;border-top:1px solid #2a2a4a;display:flex;flex-direction:column;gap:8px}}
.btn{{padding:12px;font-size:14px;font-weight:bold;border:none;border-radius:8px;cursor:pointer;color:#fff;width:100%}}
.approve{{background:#06d6a0}}.approve:hover{{background:#05b88a}}
.reject{{background:#e74c3c}}.reject:hover{{background:#c0392b}}
#hint{{padding:6px 12px;font-size:11px;color:#888;border-top:1px solid #2a2a4a}}
a{{color:#00b4d8}}
</style>
</head>
<body>
<div id="topbar">
  <h2>&#9998; Review &amp; Correct</h2>
  <span class="banner">{banner_text}</span>
  <span class="source" title="{safe_url}">{safe_url}</span>
  <a href="/tools" style="color:#aaa;font-size:12px">&#8592; Tools</a>
</div>
<div id="main">
  <div id="canvas-wrap"><canvas id="c"></canvas></div>
  <div id="sidebar">
    <div id="sidebar-header">Detections &nbsp;<span id="count" style="color:#00d4ff"></span></div>
    <div id="det-list"></div>
    <div id="hint">Drag canvas to draw box &nbsp;|&nbsp; Click box to select &nbsp;|&nbsp; Del to remove</div>
    <div id="actions">
      <button class="btn approve" onclick="submit('approve')">&#10003; Correct &amp; Approve</button>
      <button class="btn reject"  onclick="submit('reject')">&#10007; Reject</button>
    </div>
  </div>
</div>

<script>
const IMG_SRC   = "data:image/jpeg;base64,{img_b64}";
const IMG_W     = {img_w};
const IMG_H     = {img_h};
const CLASSES   = {classes_js};
const SOURCE_URL = "{safe_url}";
const DECISION_URL = "/decision";

const COLORS = ["#00d4ff","#1abc9c","#e67e22","#e74c3c","#9b59b6","#3498db",
                "#f1c40f","#2ecc71","#e91e63","#ff5722","#00bcd4","#8bc34a"];
function colorFor(label) {{
  const i = CLASSES.indexOf(label);
  return COLORS[(i < 0 ? 0 : i) % COLORS.length];
}}

let boxes = {dets_js}.map((d,i) => ({{...d, id:i}}));
let nextId = boxes.length;
let selectedId = null;

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
const img    = new Image();
let sx = 1, sy = 1, offX = 0, offY = 0;

img.onload = () => {{ resize(); render(); }};
img.src = IMG_SRC;

function resize() {{
  const wrap = document.getElementById("canvas-wrap");
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  sx = Math.min(cw / IMG_W, ch / IMG_H);
  canvas.width  = Math.round(IMG_W * sx);
  canvas.height = Math.round(IMG_H * sx);
}}
window.addEventListener("resize", () => {{ resize(); render(); }});

// img → canvas
function ic(x,y) {{ return [x*sx, y*sx]; }}
// canvas → img
function ci(x,y) {{ return [x/sx, y/sx]; }}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0,0, canvas.width, canvas.height);
  boxes.forEach(b => drawBox(b));
  updateSidebar();
}}

function drawBox(b) {{
  const [cx1,cy1] = ic(b.x1,b.y1);
  const [cx2,cy2] = ic(b.x2,b.y2);
  const color = colorFor(b.label);
  const sel   = b.id === selectedId;

  ctx.strokeStyle = sel ? "#ffff00" : color;
  ctx.lineWidth   = sel ? 3 : 2;
  ctx.strokeRect(cx1,cy1,cx2-cx1,cy2-cy1);

  // fill
  ctx.fillStyle = (sel ? "rgba(255,255,0,0.08)" : hexAlpha(color,0.1));
  ctx.fillRect(cx1,cy1,cx2-cx1,cy2-cy1);

  // label pill
  ctx.fillStyle = color;
  const label = b.label + " " + b.conf + "%";
  ctx.font = "bold 12px sans-serif";
  const tw = ctx.measureText(label).width + 10;
  ctx.fillRect(cx1, cy1 - 20, tw, 20);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, cx1+5, cy1-5);

  // handles if selected
  if (sel) drawHandles(cx1,cy1,cx2,cy2);
}}

function hexAlpha(hex,a) {{
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

const H=6;
function handles(cx1,cy1,cx2,cy2) {{
  const mx=(cx1+cx2)/2, my=(cy1+cy2)/2;
  return [
    {{x:cx1,y:cy1,cursor:"nw-resize",dx1:1,dy1:1,dx2:0,dy2:0}},
    {{x:mx, y:cy1,cursor:"n-resize", dx1:0,dy1:1,dx2:0,dy2:0}},
    {{x:cx2,y:cy1,cursor:"ne-resize",dx1:0,dy1:1,dx2:1,dy2:0}},
    {{x:cx2,y:my, cursor:"e-resize", dx1:0,dy1:0,dx2:1,dy2:0}},
    {{x:cx2,y:cy2,cursor:"se-resize",dx1:0,dy1:0,dx2:1,dy2:1}},
    {{x:mx, y:cy2,cursor:"s-resize", dx1:0,dy1:0,dx2:0,dy2:1}},
    {{x:cx1,y:cy2,cursor:"sw-resize",dx1:1,dy1:0,dx2:0,dy2:1}},
    {{x:cx1,y:my, cursor:"w-resize", dx1:1,dy1:0,dx2:0,dy2:0}},
  ];
}}

function drawHandles(cx1,cy1,cx2,cy2) {{
  ctx.fillStyle="#ffff00";
  handles(cx1,cy1,cx2,cy2).forEach(h => {{
    ctx.fillRect(h.x-H/2, h.y-H/2, H, H);
  }});
}}

// ── Mouse ─────────────────────────────────────────────────────────────────────
let drag = null; // {{type:'move'|'resize'|'draw', ...}}

canvas.addEventListener("mousedown", e => {{
  const {{ox,oy}} = canvasXY(e);

  // Check handle hit on selected box
  if (selectedId !== null) {{
    const b = boxes.find(b=>b.id===selectedId);
    if (b) {{
      const [cx1,cy1]=[b.x1*sx,b.y1*sx],[cx2,cy2]=[b.x2*sx,b.y2*sx];
      const hs = handles(cx1,cy1,cx2,cy2);
      for (let i=0;i<hs.length;i++) {{
        const h=hs[i];
        if (Math.abs(ox-h.x)<H+2 && Math.abs(oy-h.y)<H+2) {{
          drag={{type:"resize",box:b,handle:h,sx0:ox,sy0:oy,
                 ox1:b.x1,oy1:b.y1,ox2:b.x2,oy2:b.y2}};
          return;
        }}
      }}
    }}
  }}

  // Check box hit
  const hit = [...boxes].reverse().find(b => {{
    return ox>=b.x1*sx && ox<=b.x2*sx && oy>=b.y1*sx && oy<=b.y2*sx;
  }});
  if (hit) {{
    selectedId = hit.id;
    drag = {{type:"move",box:hit,sx0:ox,sy0:oy,ox1:hit.x1,oy1:hit.y1,ox2:hit.x2,oy2:hit.y2}};
    render(); return;
  }}

  // Draw new box
  selectedId = null;
  drag = {{type:"draw",sx0:ox,sy0:oy,ex:ox,ey:oy}};
  render();
}});

canvas.addEventListener("mousemove", e => {{
  if (!drag) return;
  const {{ox,oy}} = canvasXY(e);
  const dx=ox-drag.sx0, dy=oy-drag.sy0;

  if (drag.type==="move") {{
    const b=drag.box;
    const nw=(drag.ox2-drag.ox1), nh=(drag.oy2-drag.oy1);
    b.x1=Math.max(0,(drag.ox1+dx/sx)); b.y1=Math.max(0,(drag.oy1+dy/sx));
    b.x2=b.x1+nw; b.y2=b.y1+nh;
  }} else if (drag.type==="resize") {{
    const b=drag.box, h=drag.handle;
    if(h.dx1) b.x1=Math.max(0,drag.ox1+dx/sx);
    if(h.dy1) b.y1=Math.max(0,drag.oy1+dy/sx);
    if(h.dx2) b.x2=Math.min(IMG_W,drag.ox2+dx/sx);
    if(h.dy2) b.y2=Math.min(IMG_H,drag.oy2+dy/sx);
  }} else if (drag.type==="draw") {{
    drag.ex=ox; drag.ey=oy;
  }}
  render();
  if (drag.type==="draw") drawPreview(drag.sx0,drag.sy0,drag.ex,drag.ey);
}});

canvas.addEventListener("mouseup", e => {{
  if (drag?.type==="draw") {{
    const {{ox,oy}}=canvasXY(e);
    const [ix1,iy1]=ci(Math.min(drag.sx0,ox),Math.min(drag.sy0,oy));
    const [ix2,iy2]=ci(Math.max(drag.sx0,ox),Math.max(drag.sy0,oy));
    if (Math.abs(ix2-ix1)>5 && Math.abs(iy2-iy1)>5) {{
      const b={{id:nextId++,label:CLASSES[0],conf:100,x1:ix1,y1:iy1,x2:ix2,y2:iy2}};
      boxes.push(b); selectedId=b.id;
    }}
  }}
  drag=null; render();
}});

function drawPreview(x1,y1,x2,y2) {{
  ctx.strokeStyle="#00d4ff"; ctx.lineWidth=2; ctx.setLineDash([6,3]);
  ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
}}

function canvasXY(e) {{
  const r=canvas.getBoundingClientRect();
  return {{ox:e.clientX-r.left, oy:e.clientY-r.top}};
}}

document.addEventListener("keydown", e => {{
  if ((e.key==="Delete"||e.key==="Backspace") && selectedId!==null) {{
    boxes=boxes.filter(b=>b.id!==selectedId); selectedId=null; render();
  }}
}});

// ── Sidebar ───────────────────────────────────────────────────────────────────
function updateSidebar() {{
  const list = document.getElementById("det-list");
  document.getElementById("count").textContent = boxes.length;
  list.innerHTML = "";
  boxes.forEach(b => {{
    const div = document.createElement("div");
    div.className = "det-item" + (b.id===selectedId?" selected":"");
    div.innerHTML = `
      <div class="conf" style="color:${{colorFor(b.label)}}">
        <button class="del" onclick="delBox(${{b.id}})">&#10005;</button>
        ${{b.conf}}% confidence
      </div>
      <select onchange="relabel(${{b.id}},this.value)">
        ${{CLASSES.map(c=>`<option value="${{c}}" ${{c===b.label?"selected":""}}>${{c}}</option>`).join("")}}
      </select>`;
    div.addEventListener("click", ev => {{
      if (ev.target.tagName==="SELECT"||ev.target.tagName==="BUTTON"||ev.target.tagName==="OPTION") return;
      selectedId=b.id; render();
    }});
    list.appendChild(div);
  }});
}}

function delBox(id)   {{ boxes=boxes.filter(b=>b.id!==id); if(selectedId===id) selectedId=null; render(); }}
function relabel(id,v){{ const b=boxes.find(b=>b.id===id); if(b) b.label=v; render(); }}

// ── Submit ────────────────────────────────────────────────────────────────────
function submit(action) {{
  const payload = {{
    url:    SOURCE_URL,
    action: action,
    corrected_detections: boxes.map(b=>({{"class":b.label,"confidence":b.conf,
      "box":[Math.round(b.x1),Math.round(b.y1),Math.round(b.x2),Math.round(b.y2)]}}))
  }};
  fetch(DECISION_URL, {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify(payload)
  }}).then(r=>r.json()).then(d=>{{
    window.location = d.redirect || "/tools";
  }}).catch(err => alert("Error: "+err));
}}
</script>
</body></html>"""

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/launch-annotator", response_class=HTMLResponse)
async def launch_annotator(request: Request):
    """
    Downloads the image from `url`, pre-populates polygon annotations from YOLO,
    writes a COCO JSON to REVIEW_DIR, then launches annotate.py in the background.
    The reviewer corrects anything missing in the Tkinter tool before the image
    is committed to the training set.
    """
    import html as _html

    if not ANNOTATOR_SCRIPT:
        raise HTTPException(status_code=501,
                            detail="ANNOTATOR_SCRIPT is not configured in .env")

    form = await request.form()
    url  = str(form.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")

    # ── Download image ────────────────────────────────────────────────────────
    try:
        raw_bytes, pil_image = _fetch_image(url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")

    # ── Build review dir and save image ──────────────────────────────────────
    images_dir = os.path.join(REVIEW_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Derive a safe filename from the URL
    raw_name   = os.path.basename(unquote(urlparse(url).path)) or "image.jpg"
    ts         = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename   = f"{ts}_{raw_name}"
    image_path = os.path.join(images_dir, filename)
    with open(image_path, "wb") as f:
        f.write(raw_bytes)

    w, h = pil_image.size

    # ── Run inference and build COCO JSON ─────────────────────────────────────
    results = model(pil_image)
    json_path = os.path.join(REVIEW_DIR, "labels.json")

    # Load or init COCO structure
    if os.path.exists(json_path):
        with open(json_path) as f:
            coco = json.load(f)
    else:
        coco = {
            "info": {"description": "onesvs review queue"},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name}
                for i, name in enumerate(model.names.values())
            ],
        }

    # Add image record
    image_id = max((img["id"] for img in coco["images"]), default=0) + 1
    coco["images"].append({"id": image_id, "file_name": filename, "width": w, "height": h})

    # Add polygon annotations
    ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    for r in results:
        masks = r.masks
        for i, box in enumerate(r.boxes):
            conf      = float(box.conf)
            class_idx = int(box.cls)
            cat_id    = class_idx + 1

            if masks is not None and i < len(masks.xy):
                pts      = masks.xy[i]                          # [[x,y], ...]
                flat     = [coord for pt in pts for coord in (float(pt[0]), float(pt[1]))]
                xs       = flat[0::2]; ys = flat[1::2]
                bx, by   = min(xs), min(ys)
                bw, bh   = max(xs) - bx, max(ys) - by
                area     = abs(sum(
                    xs[j] * ys[(j+1) % len(xs)] - xs[(j+1) % len(xs)] * ys[j]
                    for j in range(len(xs))
                )) / 2
                segmentation = [flat]
            else:
                # Fallback to bbox-derived polygon if no mask
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                flat = [x1, y1, x2, y1, x2, y2, x1, y2]
                segmentation = [flat]
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
                area = bw * bh

            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": segmentation,
                "bbox":         [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "area":         round(area, 1),
                "iscrowd":      0,
                "score":        round(conf, 3),
            })
            ann_id += 1

    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    # ── Launch annotate.py (non-blocking subprocess) ──────────────────────────
    try:
        subprocess.Popen(
            [ANNOTATOR_PYTHON, ANNOTATOR_SCRIPT,
             "--images", images_dir,
             "--json",   json_path],
            start_new_session=True,
        )
        launched = True
        launch_err = ""
    except Exception as e:
        launched = False
        launch_err = str(e)

    safe_url    = _html.escape(url, quote=True)
    status_color = "#06d6a0" if launched else "#ff4444"
    status_text  = "Annotator launched — fix any missing polygons, then save." if launched else f"Failed to launch annotator: {launch_err}"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Review & Annotate</title>
        <style>
            body  {{ font-family: sans-serif; max-width: 700px; margin: 80px auto; padding: 0 20px;
                     background: #f4f4f4; text-align: center; }}
            .card {{ background: #fff; border-radius: 10px; padding: 40px;
                     box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
            .status {{ font-size: 18px; font-weight: bold; color: {status_color}; margin: 16px 0; }}
            .detail {{ font-size: 13px; color: #555; margin-bottom: 8px; }}
            a {{ color: #00b4d8; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Review &amp; Annotate</h2>
            <p class="status">{status_text}</p>
            <p class="detail">Image saved to: <code>{_html.escape(image_path)}</code></p>
            <p class="detail">COCO JSON: <code>{_html.escape(json_path)}</code></p>
            <p class="detail">Source: <a href="{safe_url}" target="_blank">{safe_url}</a></p>
            <p><a href="/">← Review another image</a></p>
        </div>
    </body>
    </html>
    """


@app.post("/decision", response_class=HTMLResponse)
async def decision(request: Request):
    """
    Records an approve / reject / correct-and-approve decision.

    Simple review queue (front page):
      - form fields: fname, source="review_queue", action
      - Moves the file to data/approved/ or data/recycle/
      - Redirects back to / for the next photo

    Canvas review (technical, /review?url=...):
      - JSON body: url, action, corrected_detections
      - Logs + forwards to Laravel webhook
    """
    from fastapi.responses import RedirectResponse
    import shutil

    content_type = request.headers.get("content-type", "")

    # ── Simple queue review (front page form) ─────────────────────────────────
    if "application/json" not in content_type:
        form   = await request.form()
        action = str(form.get("action", "")).strip()
        fname  = str(form.get("fname",  "")).strip()
        source = str(form.get("source", "")).strip()

        if source == "index":
            image_id = str(form.get("image_id", "")).strip()
            if image_id:
                rows = _load_index()
                new_status = "approved" if action == "approve" else ("pending" if action == "skip" else "rejected")
                for r in rows:
                    if str(r.get("image_id", "")) == image_id:
                        r["status"] = new_status
                        r["date"]   = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                        break
                _save_index(rows)

                entry = {
                    "image_id": image_id,
                    "action":   action,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
                with open(DECISIONS_LOG, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                if DECISION_WEBHOOK:
                    try:
                        async with httpx.AsyncClient(timeout=10) as client:
                            headers = {"X-Laravel-Secret": LARAVEL_SECRET} if LARAVEL_SECRET else {}
                            await client.post(DECISION_WEBHOOK, json=entry, headers=headers)
                    except Exception:
                        pass

            return RedirectResponse(url="/", status_code=303)

        if source == "review_queue" and fname:
            src_path = os.path.join(REVIEW_DIR, fname)
            if action == "approve":
                dest_dir = os.path.join(DATA_DIR, "approved")
            elif action == "skip":
                dest_dir = os.path.join(DATA_DIR, "skipped")
            else:
                dest_dir = os.path.join(DATA_DIR, "recycle")
            os.makedirs(dest_dir, exist_ok=True)

            if os.path.exists(src_path):
                shutil.move(src_path, os.path.join(dest_dir, fname))

            entry = {
                "fname":     fname,
                "action":    action,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(DECISIONS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if DECISION_WEBHOOK:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        headers = {"X-Laravel-Secret": LARAVEL_SECRET} if LARAVEL_SECRET else {}
                        await client.post(DECISION_WEBHOOK, json=entry, headers=headers)
                except Exception:
                    pass

            return RedirectResponse(url="/", status_code=303)

    # ── Canvas / JSON review (/review?url=...) ────────────────────────────────
    if "application/json" in content_type:
        body   = await request.json()
        url    = str(body.get("url", "")).strip()
        action = str(body.get("action", "")).strip()
        corrected_detections = body.get("corrected_detections", None)
    else:
        form   = await request.form()
        url    = str(form.get("url", "")).strip()
        action = str(form.get("action", "")).strip()
        corrected_detections = None

    if not url or action not in ("approve", "reject", "annotate"):
        raise HTTPException(status_code=400, detail="Missing or invalid url / action")

    entry = {
        "url":       url,
        "action":    action,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if corrected_detections is not None:
        entry["corrected_detections"] = corrected_detections

    # Append to local log
    with open(DECISIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Forward to Laravel webhook if configured
    webhook_status = None
    if DECISION_WEBHOOK:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                headers = {"X-Laravel-Secret": LARAVEL_SECRET} if LARAVEL_SECRET else {}
                r = await client.post(DECISION_WEBHOOK, json=entry, headers=headers)
                webhook_status = r.status_code
        except Exception as e:
            webhook_status = f"error: {e}"

    import html as _html
    safe_url = _html.escape(url, quote=True)
    color    = "#06d6a0" if action == "approve" else "#ff4444"
    if action == "approve":
        label = ("Corrected &amp; Approved — queued for training"
                 if corrected_detections is not None
                 else "Approved — queued for training")
    else:
        label = "Rejected — discarded"
    webhook_note = ""
    if DECISION_WEBHOOK:
        webhook_note = f"<p style='color:#888;font-size:13px'>Webhook response: {webhook_status}</p>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision recorded</title>
        <style>
            body  {{ font-family: sans-serif; max-width: 700px; margin: 80px auto; padding: 0 20px;
                     background: #f4f4f4; text-align: center; }}
            .card {{ background: #fff; border-radius: 10px; padding: 40px;
                     box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
            .status {{ font-size: 22px; font-weight: bold; color: {color}; margin: 16px 0; }}
            a {{ color: #00b4d8; }}
            .source {{ font-size: 12px; color: #888; word-break: break-all; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Decision Recorded</h2>
            <p class="status">{label}</p>
            <p class="source">{safe_url}</p>
            {webhook_note}
            <p><a href="/">← Review another image</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/gallery", response_class=HTMLResponse)
def gallery():
    """Browse all photos in the review queue with model predictions — no decisions."""
    queue = _queue_photos()

    if not queue:
        return """
        <!DOCTYPE html><html>
        <head><title>Gallery</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, sans-serif; background: #0f0f1a;
                   color: #fff; display: flex; align-items: center;
                   justify-content: center; min-height: 100vh; text-align: center; }
            a { color: #00d4ff; text-decoration: none; display: block; margin-top: 24px; }
        </style></head>
        <body><div><h1>No photos in queue</h1><a href="/">← Back</a></div></body>
        </html>
        """

    # Build thumbnail cards — run inference on each photo
    coco   = _load_coco()
    cards  = []

    for fname in queue:
        path = os.path.join(REVIEW_DIR, fname)
        try:
            img = _fix_orientation(Image.open(path).convert("RGB"))

            # Check annotations.json first, fall back to live inference
            img_record  = next((i for i in coco["images"] if i["file_name"] == fname), None)
            has_anns    = img_record and any(
                a["image_id"] == img_record["id"] for a in coco.get("annotations", [])
            )
            labels = []

            if has_anns:
                img = _draw_annotations(img, fname, coco)
                cat_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
                labels = list({cat_by_id.get(a["category_id"], "?")
                               for a in coco.get("annotations", [])
                               if a["image_id"] == img_record["id"]})
            else:
                results = model(img)
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
                except Exception:
                    font = ImageFont.load_default()
                colors = ["#00d4ff","#1abc9c","#e67e22","#e74c3c","#9b59b6",
                          "#3498db","#f1c40f","#2ecc71","#e91e63","#ff5722"]
                for r in results:
                    masks = r.masks
                    for i, box in enumerate(r.boxes):
                        color        = colors[i % len(colors)]
                        x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                        conf         = float(box.conf)
                        label        = model.names[int(box.cls)]
                        if masks is not None and i < len(masks.xy):
                            pts = [(float(x), float(y)) for x, y in masks.xy[i]]
                            if len(pts) >= 3:
                                overlay = Image.new("RGBA", img.size, (0,0,0,0))
                                odraw   = ImageDraw.Draw(overlay)
                                rv,gv,bv = tuple(int(color.lstrip("#")[j:j+2],16) for j in (0,2,4))
                                odraw.polygon(pts, fill=(rv,gv,bv,80))
                                odraw.line(pts + [pts[0]], fill=color, width=3)
                                img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                                draw = ImageDraw.Draw(img)
                        else:
                            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
                        tag       = f"{label} {conf:.0%}"
                        bbox_text = draw.textbbox((x1+4,y1+4), tag, font=font)
                        draw.rectangle(bbox_text, fill=color)
                        draw.text((x1+4,y1+4), tag, fill="#fff", font=font)
                        labels.append(label)
                labels = list(set(labels))

            # Thumbnail — small version for grid
            thumb = img.copy()
            thumb.thumbnail((400, 400), Image.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=75)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()

            # Full size for lightbox
            full = img.copy()
            if full.width > 1000:
                full = full.resize((1000, int(full.height * 1000 / full.width)), Image.LANCZOS)
            buf2 = io.BytesIO()
            full.save(buf2, format="JPEG", quality=85)
            full_b64 = base64.b64encode(buf2.getvalue()).decode()

            import html as _html
            pills = " ".join(f'<span class="pill">{_html.escape(l)}</span>' for l in sorted(labels))
            cards.append((fname, thumb_b64, full_b64, pills))

        except Exception:
            continue

    # Build grid HTML
    card_html = ""
    for idx, (fname, thumb_b64, full_b64, pills) in enumerate(cards):
        import html as _html
        safe_fname = _html.escape(fname)
        card_html += f"""
        <div class="card" onclick="openLight({idx})">
            <img src="data:image/jpeg;base64,{thumb_b64}" alt="{safe_fname}">
            <div class="meta">{pills if pills else '<span class="no-det">No detections</span>'}</div>
        </div>
        """

    lightbox_data = json.dumps([
        {"fname": fname, "src": f"data:image/jpeg;base64,{full_b64}"}
        for fname, _, full_b64, _ in cards
    ])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gallery — Label Forge</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, sans-serif; background: #0f0f1a;
                    color: #fff; padding: 24px 16px; }}

            .topbar {{ display: flex; justify-content: space-between; align-items: center;
                       margin-bottom: 24px; }}
            .topbar h1 {{ font-size: 20px; color: #00d4ff; }}
            .topbar a  {{ color: #555; font-size: 13px; text-decoration: none; }}
            .topbar a:hover {{ color: #888; }}
            .count {{ font-size: 14px; color: #888; }}

            .grid {{ display: grid;
                     grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                     gap: 16px; }}

            .card {{ background: #1a1a2e; border-radius: 10px; overflow: hidden;
                     cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }}
            .card:hover {{ transform: translateY(-3px);
                           box-shadow: 0 8px 24px rgba(0,212,255,0.15); }}
            .card img {{ width: 100%; display: block; }}
            .meta {{ padding: 8px 10px; display: flex; flex-wrap: wrap; gap: 5px;
                     min-height: 36px; align-items: center; }}
            .pill {{ background: #1e3a5f; color: #00d4ff; border-radius: 20px;
                     padding: 2px 10px; font-size: 11px; font-weight: 600; }}
            .no-det {{ color: #444; font-size: 11px; }}

            /* Lightbox */
            #lightbox {{ display: none; position: fixed; inset: 0;
                          background: rgba(0,0,0,0.92); z-index: 100;
                          flex-direction: column; align-items: center; justify-content: center; }}
            #lightbox.open {{ display: flex; }}
            #lightbox img {{ max-width: 95vw; max-height: 80vh;
                              border-radius: 8px; object-fit: contain; }}
            #lb-fname {{ color: #888; font-size: 12px; margin-top: 12px; }}
            #lb-close {{ position: absolute; top: 20px; right: 28px;
                          font-size: 32px; cursor: pointer; color: #fff;
                          line-height: 1; background: none; border: none; }}
            #lb-prev, #lb-next {{ position: absolute; top: 50%; transform: translateY(-50%);
                                   font-size: 36px; cursor: pointer; color: #fff;
                                   background: none; border: none; padding: 0 20px; }}
            #lb-prev {{ left: 0; }}
            #lb-next {{ right: 0; }}
        </style>
    </head>
    <body>

        <div class="topbar">
            <h1>🖼 Gallery</h1>
            <span class="count">{len(cards)} photos</span>
            <a href="/">← Review queue</a>
        </div>

        <div class="grid">{card_html}</div>

        <!-- Lightbox -->
        <div id="lightbox">
            <button id="lb-close" onclick="closeLight()">✕</button>
            <button id="lb-prev"  onclick="moveLight(-1)">‹</button>
            <img id="lb-img" src="" alt="">
            <p id="lb-fname"></p>
            <button id="lb-next" onclick="moveLight(1)">›</button>
        </div>

        <script>
        const PHOTOS = {lightbox_data};
        let cur = 0;

        function openLight(idx) {{
            cur = idx;
            showLight();
            document.getElementById('lightbox').classList.add('open');
        }}
        function closeLight() {{
            document.getElementById('lightbox').classList.remove('open');
        }}
        function moveLight(dir) {{
            cur = (cur + dir + PHOTOS.length) % PHOTOS.length;
            showLight();
        }}
        function showLight() {{
            document.getElementById('lb-img').src   = PHOTOS[cur].src;
            document.getElementById('lb-fname').textContent = PHOTOS[cur].fname;
        }}
        document.addEventListener('keydown', e => {{
            if (!document.getElementById('lightbox').classList.contains('open')) return;
            if (e.key === 'Escape')      closeLight();
            if (e.key === 'ArrowRight')  moveLight(1);
            if (e.key === 'ArrowLeft')   moveLight(-1);
        }});
        </script>

    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


# ── Tool Launcher Helpers ──────────────────────────────────────────────────────

def _launch(script: str, extra_args: list[str] = []) -> tuple[bool, str]:
    """Launch a script as a detached background process. Returns (success, error)."""
    if not os.path.exists(script):
        return False, f"Script not found: {script}"
    try:
        env = os.environ.copy()
        # Use DISPLAY from .env if set, otherwise default to :0 for WSLg
        env["DISPLAY"] = os.getenv("DISPLAY", ":0")
        subprocess.Popen(
            [ANNOTATOR_PYTHON, script] + extra_args,
            start_new_session=True,
            env=env,
        )
        return True, ""
    except Exception as e:
        return False, str(e)


def _tool_response(title: str, launched: bool, error: str, detail: str = "") -> str:
    color = "#06d6a0" if launched else "#ff4444"
    msg   = f"✓ {title} launched successfully." if launched else f"✗ Failed to launch: {error}"
    return f"""<!DOCTYPE html><html>
    <head><title>{title}</title>
    <style>body{{font-family:sans-serif;max-width:600px;margin:80px auto;padding:0 20px;
    background:#f4f4f4;text-align:center}}
    .card{{background:#fff;border-radius:10px;padding:40px;box-shadow:0 2px 12px rgba(0,0,0,.1)}}
    .status{{font-size:18px;font-weight:bold;color:{color};margin:16px 0}}
    .detail{{font-size:13px;color:#555;margin:8px 0}}a{{color:#00b4d8}}</style></head>
    <body><div class="card"><h2>{title}</h2>
    <p class="status">{msg}</p>
    {"<p class='detail'>" + detail + "</p>" if detail else ""}
    <p><a href="/">← Home</a></p></div></body></html>"""


# ── Annotator ─────────────────────────────────────────────────────────────────

@app.get("/launch-annotator-local", response_class=HTMLResponse)
async def launch_annotator_local():
    """Open the annotation tool against the local raw dataset."""
    os.makedirs(RAW_DIR, exist_ok=True)
    ok, err = _launch(ANNOTATOR_SCRIPT, [
        "--images", RAW_DIR,
        "--json",   ANNOTATIONS,
    ])
    return _tool_response(
        "Annotation Tool",
        ok, err,
        f"Images: {RAW_DIR}<br>Labels: {ANNOTATIONS}"
    )


@app.get("/launch-annotator-index", response_class=HTMLResponse)
async def launch_annotator_index(status: str = "pending"):
    """Open the annotation tool loaded from the index CSV (web photos)."""
    if not os.path.exists(INDEX_CSV):
        raise HTTPException(status_code=404, detail=f"index.csv not found: {INDEX_CSV}")
    ok, err = _launch(ANNOTATOR_SCRIPT, [
        "--index",  INDEX_CSV,
        "--json",   ANNOTATIONS,
        "--status", status,
    ])
    return _tool_response(
        "Annotation Tool (Index)",
        ok, err,
        f"Index: {INDEX_CSV}<br>Status filter: {status}<br>Labels: {ANNOTATIONS}"
    )


# ── Browser ───────────────────────────────────────────────────────────────────

@app.get("/launch-browser", response_class=HTMLResponse)
async def launch_browser():
    """Open the photo/job browser."""
    ok, err = _launch(BROWSER_SCRIPT)
    return _tool_response("Photo Browser", ok, err)


# ── Review ────────────────────────────────────────────────────────────────────

@app.get("/launch-review", response_class=HTMLResponse)
async def launch_review_tool(project: str = "svs_plumbing"):
    """Run the model review script and display output in the browser."""
    if not os.path.exists(REVIEW_SCRIPT):
        raise HTTPException(status_code=404, detail=f"review.py not found: {REVIEW_SCRIPT}")
    try:
        result = subprocess.run(
            [ANNOTATOR_PYTHON, REVIEW_SCRIPT, "--project", project],
            capture_output=True, text=True, timeout=60,
            cwd=BASE_DIR,
        )
        output = result.stdout + (result.stderr if result.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        output = "Review timed out after 60 seconds."
    except Exception as e:
        output = f"Error: {e}"

    lines = output.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html><html>
    <head><title>Label Forge{f" — {project}" if project else ""}</title>
    <style>body{{font-family:sans-serif;max-width:960px;margin:40px auto;padding:0 20px;background:#f4f4f4}}
    h2{{color:#1a1a2e}}pre{{background:#1a1a2e;color:#00d4ff;padding:24px;border-radius:8px;
    overflow-x:auto;font-size:13px;line-height:1.6}}a{{color:#00b4d8}}</style></head>
    <body><h2>🔨 Label Forge{f" — {project}" if project else ""}</h2>
    <pre>{lines}</pre>
    <p><a href="/tools">← Tools</a></p>
    </body></html>"""


# ── Updated Home Page ─────────────────────────────────────────────────────────

@app.get("/tools", response_class=HTMLResponse)
def tools():
    """Internal tools hub — launch annotation, browsing, and review tools."""
    tools_html = ""
    tool_list = [
        ("/launch-browser",           "📂 Photo Browser",            "Browse all photos and job data from the database."),
        ("/launch-annotator-local",   "✏️ Annotate Local Dataset",    "Open the annotation tool on the local raw image dataset."),
        ("/launch-annotator-index",   "🌐 Annotate from Index",      "Load pending photos from index.csv for annotation."),
        ("/launch-review",            "🔨 Label Forge",              "Run model validation — shows per-class precision, recall, and mAP in the browser."),
    ]
    for href, label, desc in tool_list:
        tools_html += f"""
        <div class="card">
            <strong>{label}</strong>
            <p style="color:#555;font-size:13px;margin:6px 0 12px">{desc}</p>
            <a href="{href}"><button>Launch</button></a>
        </div>"""

    return f"""<!DOCTYPE html><html>
    <head><title>onesvs — Internal Tools</title>
    <style>body{{font-family:sans-serif;max-width:860px;margin:40px auto;padding:0 20px;background:#f4f4f4}}
    h2{{color:#1a1a2e}}.card{{background:#fff;padding:20px 24px;border-radius:8px;
    box-shadow:0 2px 8px rgba(0,0,0,.1);margin-bottom:16px}}
    button{{background:#00b4d8;color:#fff;border:none;padding:10px 24px;border-radius:6px;
    cursor:pointer;font-size:14px}}button:hover{{background:#0096c7}}a{{text-decoration:none}}</style>
    </head><body>
    <h2>onesvs — Internal ML Tools</h2>
    <p style="color:#666">These tools launch on the server desktop. Requires an active display session.</p>
    {tools_html}
    <p><a href="/" style="color:#00b4d8">← Back to inference home</a></p>
    </body></html>"""
