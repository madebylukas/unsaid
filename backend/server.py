"""Small HTTP bridge around Auto-AVSR.

The process deliberately starts with only Python's standard library. Vercel
containers have a short initialization window, while the ML stack is large.
Torch, MediaPipe, and ESPnet therefore load only after an inference request.
"""

from __future__ import annotations

import gc
import json
import math
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CHAPLIN = ROOT / ".cache" / "chaplin"
CONFIG = ROOT / "backend" / "config.ini"
SOURCE_MODEL = CHAPLIN / "benchmarks" / "LRS3" / "models" / "LRS3_V_WER19.1" / "model.pth"
SOURCE_LM = CHAPLIN / "benchmarks" / "LRS3" / "language_models" / "lm_en_subword" / "model.pth"
TEMP_ASSETS = Path(tempfile.gettempdir()) / "unsaid-assets"
DIST = ROOT / "dist"
MODEL_PARTS = sorted((ROOT / ".cache" / "model-parts").glob("model-state-*.pth"))
LM_PARTS = sorted((ROOT / ".cache" / "model-parts").glob("lm-state-*.pth"))
MODEL = SOURCE_MODEL
LM = SOURCE_LM
RUNTIME_CONFIG = CONFIG if SOURCE_MODEL.exists() and SOURCE_LM.exists() else TEMP_ASSETS / "config.ini"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
LOCAL_ORIGIN = re.compile(r"^http://(?:127\.0\.0\.1|localhost):\d+$")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(CHAPLIN))

pipeline = None
pipeline_lock = threading.Lock()
inference_lock = threading.Lock()
loading = False
load_error: str | None = None


def materialize_assets() -> None:
    if RUNTIME_CONFIG != CONFIG and not RUNTIME_CONFIG.exists():
        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        config = CONFIG.read_text()
        config = re.sub(r"(?m)^model_path=.*$", f"model_path={MODEL}", config)
        config = re.sub(r"(?m)^rnnlm=.*$", f"rnnlm={LM}", config)
        RUNTIME_CONFIG.write_text(config)


def selected_device() -> str:
    return os.environ.get("UNSAID_DEVICE", "cpu").lower()


def assets_exist() -> bool:
    model_ready = SOURCE_MODEL.exists() or bool(MODEL_PARTS)
    language_model_ready = SOURCE_LM.exists() or bool(LM_PARTS)
    return CHAPLIN.exists() and model_ready and language_model_ready


def packaged_assets_exist() -> bool:
    return CHAPLIN.exists() and bool(MODEL_PARTS) and bool(LM_PARTS)


class ShardedStateDict:
    def __init__(self, parts: list[Path]):
        self.parts = parts


def install_sharded_torch_load(torch) -> None:
    if SOURCE_MODEL.exists() or getattr(torch.load, "_unsaid_sharded", False):
        return
    original_load = torch.load
    original_load_state_dict = torch.nn.Module.load_state_dict

    def sharded_load(source, *args, **kwargs):
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if path == SOURCE_MODEL:
                return ShardedStateDict(MODEL_PARTS)
            if path == SOURCE_LM:
                return ShardedStateDict(LM_PARTS)
        return original_load(source, *args, **kwargs)

    def incremental_load_state_dict(module, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, ShardedStateDict):
            return original_load_state_dict(module, state_dict, strict=strict, assign=assign)
        for part in state_dict.parts:
            shard = original_load(part, map_location="cpu", weights_only=True)
            original_load_state_dict(module, shard, strict=False, assign=assign)
            del shard
            gc.collect()
        # The shards were produced from one strict upstream state dict. The
        # caller ignores this return value; an empty check keeps that contract.
        return original_load_state_dict(module, {}, strict=False, assign=assign)

    sharded_load._unsaid_sharded = True
    torch.load = sharded_load
    torch.nn.Module.load_state_dict = incremental_load_state_dict


def ensure_video_reader() -> None:
    """Restore torchvision's retired read_video API with the existing PyAV dependency."""
    import av
    import numpy as np
    import torch
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
    with pipeline_lock:
        if pipeline is not None:
            return pipeline
        loading = True
        load_error = None
        try:
            materialize_assets()
            if not assets_exist():
                raise RuntimeError("run `npm run setup:vsr` first")

            import torch

            install_sharded_torch_load(torch)
            ensure_video_reader()
            from pipelines.pipeline import InferencePipeline

            pipeline = InferencePipeline(
                str(RUNTIME_CONFIG),
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
    import torch
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
        return decode_nbest(load_pipeline(), filename)


def health_payload() -> dict:
    if not assets_exist() and not packaged_assets_exist():
        return {
            "status": "setup_required",
            "model_loaded": False,
            "device": selected_device(),
            "engine": "auto-avsr",
            "detail": "run npm run setup:vsr",
        }
    return {
        "status": "loading" if loading else "ready",
        "model_loaded": pipeline is not None,
        "device": selected_device(),
        "engine": "auto-avsr",
        "detail": load_error,
    }


def extract_video(content_type: str, body: bytes) -> tuple[bytes, str]:
    boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
    if not boundary_match:
        raise ValueError("expected a multipart video upload")
    boundary = (boundary_match.group(1) or boundary_match.group(2)).encode()
    for part in body.split(b"--" + boundary):
        headers, separator, payload = part.lstrip(b"\r\n").partition(b"\r\n\r\n")
        if not separator or b'name="video"' not in headers:
            continue
        payload = payload.removesuffix(b"\r\n").removesuffix(b"--")
        content_header = re.search(br"(?im)^Content-Type:\s*([^\r\n]+)", headers)
        mime = content_header.group(1).decode().lower() if content_header else "video/mp4"
        return payload, ".webm" if "webm" in mime else ".mp4"
    raise ValueError("multipart upload is missing the video field")


def read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    length_header = handler.headers.get("Content-Length")
    if length_header:
        try:
            length = int(length_header)
        except ValueError as error:
            raise ValueError("invalid upload length") from error
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("video must be under 4 MB")
        return handler.rfile.read(length)

    if "chunked" not in handler.headers.get("Transfer-Encoding", "").lower():
        raise ValueError("upload length is missing")
    body = bytearray()
    while True:
        size_line = handler.rfile.readline(128).split(b";", 1)[0].strip()
        try:
            size = int(size_line, 16)
        except ValueError as error:
            raise ValueError("invalid chunked upload") from error
        if size == 0:
            handler.rfile.readline()
            break
        if len(body) + size > MAX_UPLOAD_BYTES:
            raise ValueError("video must be under 4 MB")
        body.extend(handler.rfile.read(size))
        handler.rfile.read(2)
    return bytes(body)


class UnsaidHandler(BaseHTTPRequestHandler):
    server_version = "unsaid/0.3"

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def end_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if LOCAL_ORIGIN.match(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/health", "/api/health"}:
            self.send_json(HTTPStatus.OK, health_payload())
            return
        self.serve_static(path)

    def do_HEAD(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/health", "/api/health"}:
            encoded = json.dumps(health_payload(), separators=(",", ":")).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            return
        self.serve_static(path, head_only=True)

    def do_POST(self) -> None:
        if urlsplit(self.path).path not in {"/infer", "/api/infer"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
            return
        if not assets_exist() and not packaged_assets_exist():
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "run `npm run setup:vsr` first"})
            return
        try:
            body = read_request_body(self)
            payload, suffix = extract_video(self.headers.get("Content-Type", ""), body)
            if not payload:
                raise ValueError("video is empty")
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"detail": str(error)})
            return

        started = time.perf_counter()
        temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            temporary.write(payload)
            temporary.close()
            candidates = run_decode(temporary.name)
            if not candidates:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": "the model could not decode visible speech"})
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "transcript": candidates[0]["text"],
                    "candidates": candidates,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "duration_seconds": 0,
                    "device": selected_device(),
                },
            )
        except Exception as error:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"inference failed: {error}"})
        finally:
            Path(temporary.name).unlink(missing_ok=True)

    def serve_static(self, request_path: str, head_only: bool = False) -> None:
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (DIST / relative).resolve()
        try:
            candidate.relative_to(DIST.resolve())
        except ValueError:
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
            return
        if not candidate.is_file():
            candidate = DIST / "index.html"
        if not candidate.is_file():
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "web build missing"})
            return
        payload = candidate.read_bytes()
        mime, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8787"))
    print(f"unsaid listening on http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), UnsaidHandler).serve_forever()
