import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .ai.parser import parse_instruction
from .video.pipeline import process_video

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
STORAGE = ROOT / "storage"
UPLOADS = STORAGE / "uploads"; OUTPUTS = STORAGE / "outputs"; JOBS = STORAGE / "jobs"
for p in (UPLOADS, OUTPUTS, JOBS): p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LIGHTNOTEAI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "LIGHTNOTEAI", "ai": "Gemini optional / local fallback"}

def _save_job(job_id, data):
    (JOBS / f"{job_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def _read_job(job_id):
    p = JOBS / f"{job_id}.json"
    if not p.exists(): raise HTTPException(404, "Job not found")
    return json.loads(p.read_text(encoding="utf-8"))

def _run(job_id, video_path, ref_path, prompt):
    job = _read_job(job_id)
    try:
        job["status"] = "processing"; job["progress"] = 4; job["stage"] = "Understanding instruction"
        _save_job(job_id, job)
        instruction = parse_instruction(prompt)
        job["instruction"] = instruction; job["progress"] = 12; job["stage"] = "Detecting target object"; _save_job(job_id, job)
        out = OUTPUTS / f"{job_id}.mp4"
        def progress(p, stage):
            job.update(progress=p, stage=stage); _save_job(job_id, job)
        meta = process_video(video_path, str(out), ref_path, instruction, progress)
        job.update(status="completed", progress=100, stage="Completed", output_url=f"/api/jobs/{job_id}/output", metadata=meta)
        _save_job(job_id, job)
    except Exception as e:
        job.update(status="failed", progress=100, stage="Failed", error=str(e)); _save_job(job_id, job)

@app.post("/api/process")
async def create_process(background_tasks: BackgroundTasks, video: UploadFile = File(...), reference_image: UploadFile | None = File(None), prompt: str = Form(...)):
    if not video.filename: raise HTTPException(400, "Video is required")
    job_id = uuid.uuid4().hex[:12]
    video_path = UPLOADS / f"{job_id}_{Path(video.filename).name}"
    with video_path.open("wb") as f: shutil.copyfileobj(video.file, f)
    ref_path = None
    if reference_image and reference_image.filename:
        ref_path = str(UPLOADS / f"{job_id}_reference_{Path(reference_image.filename).name}")
        with open(ref_path, "wb") as f: shutil.copyfileobj(reference_image.file, f)
    job = {"id": job_id, "status": "queued", "progress": 0, "stage": "Queued", "prompt": prompt}
    _save_job(job_id, job)
    background_tasks.add_task(_run, job_id, str(video_path), ref_path, prompt)
    return job

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str): return _read_job(job_id)

@app.get("/api/jobs/{job_id}/output")
def output(job_id: str):
    job = _read_job(job_id); path = OUTPUTS / f"{job_id}.mp4"
    if job.get("status") != "completed" or not path.exists(): raise HTTPException(404, "Output is not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"lightnoteai-{job_id}.mp4")

FRONTEND = ROOT / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
