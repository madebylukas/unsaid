"""Local Auto-AVSR bridge for unsaid.

Inference plumbing is adapted from Chaplin (MIT) and Auto-AVSR (Apache-2.0).
The model and language-model weights are downloaded separately and retain
their upstream terms.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
CHAPLIN = ROOT / ".cache" / "chaplin"
CONFIG = ROOT / "backend" / "config.ini"
MODEL = CHAPLIN / "benchmarks" / "LRS3" / "models" / "LRS3_V_WER19.1" / "model.pth"
LM = CHAPLIN / "benchmarks" / "LRS3" / "language_models" / "lm_en_subword" / "model.pth"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(CHAPLIN))

app = FastAPI(title="unsaid visual speech engine", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

pipeline = None
pipeline_lock = threading.Lock()
inference_lock = threading.Lock()
loading = False
load_error: str | None = None


def selected_device() -> str:
    return os.environ.get("UNSAID_DEVICE", "cpu").lower()


def assets_exist() -> bool:
    return CHAPLIN.exists() and MODEL.exists() and LM.exists()


def ensure_video_reader() -> None:
    """Restore torchvision's retired read_video API with the existing PyAV dependency."""
    import av
    import numpy as np
    import torchvision

    if hasattr(torchvision.io, "read_video"):
        return

    def read_video(filename, pts_unit="sec"):
        del pts_unit
        container = av.open(filename)
        try:
            frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
        finally:
            container.close()
        if not frames:
            raise RuntimeError("video contains no decodable frames")
        video = torch.from_numpy(np.stack(frames))
        return video, torch.empty((1, 0)), {}

    torchvision.io.read_video = read_video


def load_pipeline():
    global pipeline, loading, load_error
    if pipeline is not None:
        return pipeline
    if not assets_exist():
        raise RuntimeError("run `npm run setup:vsr` first")
    with pipeline_lock:
        if pipeline is not None:
            return pipeline
        loading = True
        load_error = None
        try:
            ensure_video_reader()
            from pipelines.pipeline import InferencePipeline

            pipeline = InferencePipeline(
                str(CONFIG),
                device=torch.device(selected_device()),
                detector="mediapipe",
                face_track=True,
            )
            return pipeline
        except Exception as error:
            load_error = str(error)
            raise
        finally:
            loading = False


def clean_text(text: str) -> str:
    return text.replace("▁", " ").replace("<eos>", "").strip().lower()


def decode_nbest(engine, filename: str, limit: int = 5):
    from espnet.asr.asr_utils import add_results_to_json

    landmarks = engine.process_landmarks(filename, None)
    data = engine.dataloader.load_data(filename, landmarks)
    model = engine.model
    with torch.inference_mode():
        encoded = model.model.encode(data.to(model.device))
        hypotheses = model.beam_search(encoded)

    rows = []
    for hypothesis in hypotheses[:limit]:
        raw = hypothesis.asdict()
        text = clean_text(add_results_to_json([raw], model.token_list))
        if not text or any(row["text"] == text for row in rows):
            continue
        raw_score = raw["score"]
        score = float(raw_score.detach().cpu()) if hasattr(raw_score, "detach") else float(raw_score)
        rows.append({"text": text, "score": score})

    if not rows:
        return []
    highest = max(row["score"] for row in rows)
    weights = [math.exp(max(-40, row["score"] - highest)) for row in rows]
    total = sum(weights) or 1
    for row, weight in zip(rows, weights):
        row["probability"] = weight / total
    return rows


def run_decode(filename: str):
    # ESPnet keeps mutable decoder state. Serial inference is deliberate.
    with inference_lock:
        engine = load_pipeline()
        return decode_nbest(engine, filename)


@app.get("/health")
def health():
    if not assets_exist():
        return {
            "status": "setup_required",
            "model_loaded": False,
            "device": selected_device(),
            "engine": "auto-avsr",
            "detail": "run npm run setup:vsr",
        }
    if loading:
        status = "loading"
    else:
        status = "ready"
    return {
        "status": status,
        "model_loaded": pipeline is not None,
        "device": selected_device(),
        "engine": "auto-avsr",
        "detail": load_error,
    }


@app.post("/infer")
async def infer(video: UploadFile = File(...)):
    if not assets_exist():
        raise HTTPException(status_code=503, detail="run `npm run setup:vsr` first")
    suffix = ".webm" if "webm" in (video.content_type or "") else ".mp4"
    payload = await video.read()
    if not payload or len(payload) > 60 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="video must be between 1 byte and 60 MB")

    started = time.perf_counter()
    temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        temporary.write(payload)
        temporary.close()
        candidates = await run_in_threadpool(run_decode, temporary.name)
        if not candidates:
            raise HTTPException(status_code=422, detail="the model could not decode visible speech")
        return {
            "transcript": candidates[0]["text"],
            "candidates": candidates,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "duration_seconds": 0,
            "device": selected_device(),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"inference failed: {error}") from error
    finally:
        Path(temporary.name).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8787, log_level="info")
