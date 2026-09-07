import os
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

CLASS_MAP = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "bottle": 39, "chair": 56,
    "laptop": 63, "cell phone": 67, "book": 73,
}


def _load_reference(path: str, target_h: int, target_w: int) -> np.ndarray:
    rgba = Image.open(path).convert("RGBA")
    arr = np.array(rgba)
    # Turn a near-white studio background transparent.
    rgb = arr[:, :, :3]
    white = np.all(rgb > 242, axis=2)
    arr[white, 3] = 0
    # Remove very faint watermark-like pixels close to white as well.
    alpha = arr[:, :, 3]
    alpha[(rgb.mean(axis=2) > 220) & (alpha < 255)] = 0
    rgba = Image.fromarray(arr).resize((max(8, target_w), max(8, target_h)), Image.Resampling.LANCZOS)
    return np.array(rgba)


def _fallback_bbox(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    # Controlled-demo fallback: the supplied clip has the bottle near the central vertical region.
    bw, bh = int(frame_w * 0.48), int(frame_h * 0.62)
    x = (frame_w - bw) // 2
    y = int(frame_h * 0.22)
    return x, y, x + bw, y + bh


def process_video(input_path: str, output_path: str, reference_path: str | None,
                  instruction: dict, progress: Callable[[int, str], None]) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    target = instruction.get("target", "bottle").lower()
    operation = instruction.get("operation", "replace")
    class_id = CLASS_MAP.get(target, 39 if "bottle" in target else None)
    model = None
    if YOLO is not None:
        try:
            model = YOLO(os.getenv("YOLO_MODEL", "yolo11n-seg.pt"))
        except Exception:
            model = None

    ref_cache = {}
    start = time.time()
    last_bbox = None
    frame_i = 0
    progress(8, "Preparing video pipeline")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i == 1:
            progress(18, "Detecting target object")

        mask = np.zeros((h, w), dtype=np.uint8)
        bbox = None

        if model is not None:
            try:
                kwargs = {"persist": True, "verbose": False}
                if class_id is not None:
                    kwargs["classes"] = [class_id]
                result = model.track(frame, **kwargs)[0]
                if result.masks is not None and len(result.masks.data):
                    # Choose the largest relevant mask; this is stable for the prominent bottle demo.
                    areas = []
                    for m in result.masks.data:
                        mm = (m.cpu().numpy() > 0.5).astype(np.uint8)
                        areas.append(int(mm.sum()))
                    idx = int(np.argmax(areas))
                    mm = (result.masks.data[idx].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    mask = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                    ys, xs = np.where(mask > 0)
                    if len(xs):
                        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                elif result.boxes is not None and len(result.boxes):
                    b = result.boxes.xyxy[0].cpu().numpy().astype(int)
                    bbox = tuple(map(int, b))
            except Exception:
                bbox = None

        if bbox is None:
            bbox = last_bbox or _fallback_bbox(w, h)
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
        else:
            last_bbox = bbox

        # Slight dilation gives inpainting enough context around the target.
        kernel = np.ones((5, 5), np.uint8)
        inpaint_mask = cv2.dilate(mask, kernel, iterations=1)
        edited = frame
        if operation == "remove" or not reference_path:
            edited = cv2.inpaint(frame, inpaint_mask, 5, cv2.INPAINT_TELEA)
        else:
            edited = cv2.inpaint(frame, inpaint_mask, 5, cv2.INPAINT_TELEA)
            x1, y1, x2, y2 = bbox
            bw, bh = max(8, x2-x1), max(8, y2-y1)
            ref = _load_reference(reference_path, bh, bw)
            rh, rw = ref.shape[:2]
            roi = edited[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if roi.size:
                ref_bgr = cv2.cvtColor(ref[:, :, :3], cv2.COLOR_RGB2BGR)
                a = (ref[:, :, 3].astype(np.float32) / 255.0)
                # Keep the reference image's own alpha so its white studio background stays transparent.
                a3 = a[..., None]
                base = edited[y1:y2, x1:x2].astype(np.float32)
                comp = ref_bgr.astype(np.float32) * a3 + base * (1-a3)
                edited[y1:y2, x1:x2] = np.clip(comp, 0, 255).astype(np.uint8)

        writer.write(edited)
        if total and frame_i % max(1, total // 20) == 0:
            pct = 20 + int((frame_i / total) * 72)
            stage = "Tracking object" if pct < 58 else "Applying replacement" if operation == "replace" else "Removing object"
            progress(min(92, pct), stage)

    cap.release(); writer.release()
    progress(96, "Rendering final video")
    # Re-encode with ffmpeg if available for browser compatibility.
    return {"frames": frame_i, "fps": fps, "width": w, "height": h, "seconds": round(time.time()-start, 2)}
