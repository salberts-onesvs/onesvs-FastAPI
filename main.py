import os
import io
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import traceback

load_dotenv()

LARAVEL_SECRET = os.getenv("LARAVEL_SECRET", "")
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

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


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x_laravel_secret: Optional[str] = Header(None),
):
    if LARAVEL_SECRET and x_laravel_secret != LARAVEL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>onesvs — YOLO Inference</title>
        <style>
            body { font-family: sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
            h2   { color: #1a1a2e; }
            form { background: #fff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            input[type=file]   { margin: 12px 0; }
            input[type=submit] { background: #00b4d8; color: #fff; border: none; padding: 10px 28px;
                                 border-radius: 6px; cursor: pointer; font-size: 15px; }
            input[type=submit]:hover { background: #0096c7; }
        </style>
    </head>
    <body>
        <h2>onesvs YOLO — Visual Test</h2>
        <form action="/preview" method="post" enctype="multipart/form-data">
            <p>Upload a photo to run inference and see detections drawn on the image.</p>
            <input name="image" type="file" accept="image/*" required><br>
            <input type="submit" value="Run Detection">
        </form>
    </body>
    </html>
    """


@app.post("/preview", response_class=HTMLResponse)
async def preview(image: UploadFile = File(...)):
    """Browser-friendly endpoint — returns the annotated image + detection table."""
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
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


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}
