# onesvs-FastAPI

YOLOv8 object detection inference API. Runs independently from the main website — receives images, returns JSON predictions, no database connection.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your values
```

## Run locally

```bash
uvicorn main:app --reload --port 8001
```

## Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/predict` | Run inference on an uploaded image |
| GET | `/health` | Health check |

### POST /predict

**Request:** `multipart/form-data` with `image` field + `x-laravel-secret` header

**Response:**
```json
{
  "detections": [
    { "class": "ice_machine", "confidence": 0.87, "box": [x1, y1, x2, y2] }
  ],
  "review_needed": false,
  "confidence_threshold": 0.6
}
```

`review_needed: true` when top confidence < threshold or no detections found.
