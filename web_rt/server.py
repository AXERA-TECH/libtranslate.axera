#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import json
import os
import socket
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
AX_MEETING_ROOT = Path(os.getenv("AX_MEETING_ROOT", "/home/axera/AXERA-TECH/3D-Speaker-MT.Axera"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pytranslate"))
sys.path.insert(0, str(AX_MEETING_ROOT))

from ax_meeting.model_bundle import ModelBundle  # noqa: E402
from ax_meeting.config import SAMPLE_RATE, VAD_CHECK_INTERVAL_SEC, PAUSE_MS, MIN_SEGMENT_MS  # noqa: E402

from pytranslate import AXTranslate  # noqa: E402

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"

DEFAULT_ASR_MODEL_DIR = os.getenv(
    "ASR_MODEL_DIR",
    str(AX_MEETING_ROOT / "ax_meeting" / "ax_model"),
)
DEFAULT_TRANS_MODEL_DIR = os.getenv(
    "TRANS_MODEL_DIR",
    "/home/axera/ax-llm/build_650/axllm-models/HY-MT1.5-1.8B_GPTQ_INT4",
)


def is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Require at least one letter/number or CJK character
    for ch in stripped:
        if ch.isalnum():
            return True
        # Basic CJK Unified Ideographs
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


@dataclass
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    translation: str


@dataclass
class StreamingTranslateSession:
    model_bundle: Any
    translator: AXTranslate
    translate_lock: threading.Lock
    target_language: str
    sample_rate: int = SAMPLE_RATE
    pause_ms: int = PAUSE_MS
    min_segment_ms: int = MIN_SEGMENT_MS
    vad_check_interval_sec: float = VAD_CHECK_INTERVAL_SEC

    audio_chunks: List[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    last_processed_ms: int = 0
    last_vad_check_ts: float = field(default_factory=lambda: 0.0)
    processing: bool = False

    async def add_audio(self, pcm_bytes: bytes, websocket):
        if not pcm_bytes:
            return
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        self.audio_chunks.append(audio)
        self.total_samples += audio.size

        now = asyncio.get_event_loop().time()
        if (now - self.last_vad_check_ts) >= self.vad_check_interval_sec and not self.processing:
            self.last_vad_check_ts = now
            self.processing = True
            asyncio.create_task(self._run_vad_asr_translate(websocket))

    def _current_ms(self) -> int:
        return int(self.total_samples / self.sample_rate * 1000)

    def _audio_all(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.audio_chunks, axis=0)

    async def _run_vad_asr_translate(self, websocket):
        try:
            segments = await asyncio.to_thread(self._detect_ready_segments)
            for seg in segments:
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "start_ms": seg.start_ms,
                        "end_ms": seg.end_ms,
                        "text": seg.text,
                        "translation": seg.translation,
                        "language": self.target_language,
                    }
                )
        finally:
            self.processing = False

    def _detect_ready_segments(self) -> List[TranscriptSegment]:
        audio = self._audio_all()
        if audio.size == 0:
            return []

        current_ms = self._current_ms()
        vad_segments = self.model_bundle.vad_infer(audio)
        ready_segments: List[TranscriptSegment] = []

        for start_ms, end_ms in vad_segments:
            if end_ms <= self.last_processed_ms:
                continue
            if (end_ms - start_ms) < self.min_segment_ms:
                continue
            if end_ms > current_ms - self.pause_ms:
                continue

            start_sample = int(start_ms / 1000 * self.sample_rate)
            end_sample = int(end_ms / 1000 * self.sample_rate)
            seg_audio = audio[start_sample:end_sample]
            text, _ = self.model_bundle.asr_infer(seg_audio, output_timestamp=False, key="live")
            text = text.strip()
            if text and is_meaningful_text(text):
                with self.translate_lock:
                    translation = self.translator.translate(text, self.target_language)
                ready_segments.append(
                    TranscriptSegment(start_ms, end_ms, text, translation)
                )
            self.last_processed_ms = max(self.last_processed_ms, int(end_ms))

        return ready_segments

    async def finalize(self) -> Dict[str, Any]:
        # Flush remaining segments by running VAD+ASR on whole audio
        full_text = await asyncio.to_thread(self._full_asr)
        full_text = full_text.strip()
        full_translation = ""
        if full_text and is_meaningful_text(full_text):
            def _do_translate():
                with self.translate_lock:
                    return self.translator.translate(full_text, self.target_language)
            full_translation = await asyncio.to_thread(_do_translate)
        return {"text": full_text, "translation": full_translation}

    def _full_asr(self) -> str:
        audio = self._audio_all()
        if audio.size == 0:
            return ""
        vad_segments = self.model_bundle.vad_infer(audio)
        parts: List[str] = []
        for start_ms, end_ms in vad_segments:
            start_sample = int(start_ms / 1000 * self.sample_rate)
            end_sample = int(end_ms / 1000 * self.sample_rate)
            seg_audio = audio[start_sample:end_sample]
            text, _ = self.model_bundle.asr_infer(seg_audio, output_timestamp=False, key="final")
            if text and text.strip():
                parts.append(text.strip())
        return " ".join(parts)


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

models = ModelBundle(ax_model_dir=DEFAULT_ASR_MODEL_DIR)
translator: AXTranslate | None = None
translator_lock = threading.Lock()
app_ready = False
app_error: str | None = None


@app.on_event("startup")
async def startup_event():
    global translator, app_ready, app_error
    try:
        await asyncio.to_thread(models.ensure_loaded)
        translator = AXTranslate(model_dir=DEFAULT_TRANS_MODEL_DIR)
        app_ready = True
    except Exception as e:
        app_error = str(e)
        app_ready = False


@app.get("/")
def index():
    if not app_ready:
        msg = app_error or "模型加载中，请稍后刷新。"
        html = f"""
        <html><head><meta charset="utf-8"><title>Loading</title></head>
        <body style="font-family:sans-serif;padding:20px;">
        <h2>模型未就绪</h2><pre>{msg}</pre></body></html>
        """
        return HTMLResponse(html, status_code=503)
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not app_ready or translator is None:
        await ws.send_json({"type": "error", "message": "模型未就绪，请稍后再试"})
        await ws.close()
        return
    session = StreamingTranslateSession(
        models, translator, translator_lock, target_language="English"
    )
    await ws.send_json({"type": "ready"})

    try:
        while True:
            message = await ws.receive()

            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type")
                if msg_type == "config":
                    lang = data.get("language")
                    if lang:
                        session.target_language = lang
                        await ws.send_json({"type": "config_ack", "language": lang})
                elif msg_type == "end":
                    await ws.send_json({"type": "end_ack"})
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

            if "bytes" in message and message["bytes"]:
                await session.add_audio(message["bytes"], ws)

    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    def _get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # No packets are sent; this is used to resolve outbound interface.
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    ssl_cert = os.getenv("SSL_CERT")
    ssl_key = os.getenv("SSL_KEY")
    use_ssl = bool(
        ssl_cert
        and ssl_key
        and Path(ssl_cert).is_file()
        and Path(ssl_key).is_file()
    )

    display_host = _get_local_ip() if host in {"0.0.0.0", "::"} else host
    scheme = "https" if use_ssl else "http"
    print(f"Web RT Translate: {scheme}://{display_host}:{port}")

    uvicorn_kwargs = {
        "app": "web_rt.server:app",
        "host": host,
        "port": port,
        "reload": False,
    }
    if use_ssl:
        uvicorn_kwargs["ssl_certfile"] = ssl_cert
        uvicorn_kwargs["ssl_keyfile"] = ssl_key

    uvicorn.run(**uvicorn_kwargs)
