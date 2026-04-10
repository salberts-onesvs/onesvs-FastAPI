#!/bin/bash
# onesvs-FastAPI — first-time setup
# Run: bash setup.sh

set -e

echo "=== onesvs FastAPI Setup ==="

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "my_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv my_env
fi

source my_env/bin/activate
echo "Virtual environment active."

# ── pip dependencies ──────────────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip -q
pip install -r requirements.txt

# ── Git-hosted packages ───────────────────────────────────────────────────────
echo "Installing MobileSAM..."
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# ── Environment file ──────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo ">>> .env created from .env.example"
    echo ">>> Edit .env and set LARAVEL_SECRET and DECISION_WEBHOOK before running."
fi

# ── Data directories ──────────────────────────────────────────────────────────
mkdir -p data/raw data/review_queue data/recycle Annotator/Model

# ── Model weights reminder ────────────────────────────────────────────────────
echo ""
echo "=== Model Weights ==="
if [ ! -f "Annotator/Model/best.pt" ]; then
    echo "  WARNING: Annotator/Model/best.pt not found."
    echo "  Copy your trained YOLO model to: Annotator/Model/best.pt"
fi
if [ ! -f "Annotator/Model/mobile_sam.pt" ]; then
    echo "  Downloading MobileSAM weights (~40MB)..."
    wget -q --show-progress \
        -O Annotator/Model/mobile_sam.pt \
        https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
    && echo "  ✓  mobile_sam.pt downloaded." \
    || echo "  WARNING: Download failed. Get it manually:
        wget -O Annotator/Model/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the server:"
echo "  source my_env/bin/activate"
echo "  uvicorn main:app --reload --port 8000"
echo ""
echo "Then open: http://localhost:8000"
