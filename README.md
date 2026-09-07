# LIGHTNOTEAI

AI-powered video object editor prototype for the LightnoteAI Full-Stack AI Developer assignment.

## Architecture

`Next/HTML-style frontend (Stitch design) -> FastAPI -> Gemini instruction parser -> YOLO segmentation/tracking -> OpenCV inpainting/compositing -> MP4 output`

The frontend is intentionally kept lightweight so the focus remains on the AI/video workflow. The Stitch-generated `code.html` was adapted into `frontend/index.html` and connected to real backend APIs.

## AI approach

1. Gemini converts natural language into a structured operation such as `replace`, target object, and replacement.
2. Ultralytics YOLO segmentation detects the target object. ByteTrack-backed persistence is used by `model.track` where available.
3. OpenCV inpainting removes the original object from each frame.
4. A supplied reference image is alpha-extracted and composited into the tracked bounding region.
5. The edited frames are rendered into an MP4.

If `GEMINI_API_KEY` is absent or temporarily unavailable, a deterministic local parser handles common commands such as `Replace the Coca-Cola bottle with Pepsi.` This makes local demos possible while retaining the real Gemini integration path.

## Setup

Requirements: Python 3.10+ and FFmpeg recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/app/requirements.txt
Copy-Item .env.example .env
# Optional: add GEMINI_API_KEY to .env
uvicorn backend.app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

On first AI-processing run, Ultralytics may download the small segmentation model configured by `YOLO_MODEL`. A network connection is therefore required for that first model download unless the weight is already cached.

## Demo asset

A short Coca-Cola bottle sample is included at `demo/coca-cola-sample.mp4` for local testing. Use a clean, watermark-free Pepsi reference image for the final demo.

## Demo prompt

`Replace the Coca-Cola bottle with Pepsi while preserving the original motion and background.`

Upload the Coca-Cola sample video and a clean Pepsi bottle reference image.

## Known limitations

- Best suited to short clips with one prominent target object.
- Heavy occlusion, fast camera movement, reflections, and extreme perspective changes can reduce mask quality.
- The prototype uses bounding-region compositing rather than a full generative video model, so replacement realism is intentionally limited.
- Local storage is used for simplicity; production would use object storage and a real queue/worker system.
