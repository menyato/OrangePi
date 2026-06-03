#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi
─────────────────────────────────────────────
Does ONLY:
  • Record voice with Silero VAD
  • Transcribe with faster-whisper (tiny, CPU)
  • Send text to Windows PC over TCP
  • Receive TTS text back and speak via espeak-ng

No camera, no YOLO, no torch GPU — pure voice I/O.

Install:
    sudo apt install espeak-ng portaudio19-dev libsndfile1
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install faster-whisper sounddevice numpy silero-vad pyserial
"""

import socket
import json
import threading
import subprocess
import collections
import argparse
import warnings

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.80.64.159"   # ← your Windows PC IP
SERVER_PORT = 9000

SAMPLE_RATE        = 16000
VAD_CHUNK          = 512
VAD_THRESHOLD      = 0.5
VAD_MIN_SILENCE_MS = 600
VAD_SPEECH_PAD_MS  = 200
VAD_MAX_CHUNKS     = 400
VAD_PREROLL_CHUNKS = 10
VAD_INTERRUPT_ENERGY = 0.04

# ── GLOBALS ───────────────────────────────────────────────────────────────────
whisper_model = None
vad_model     = None
_tts_lock     = threading.Lock()
_tts_proc     = None

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    global _tts_proc
    print(f">> {text}")
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()
            _tts_proc.wait()
        _tts_proc = subprocess.Popen(
            ["espeak-ng", "-s", "145", "-v", "en", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def is_speaking() -> bool:
    with _tts_lock:
        return _tts_proc is not None and _tts_proc.poll() is None

def stop_speaking():
    global _tts_proc
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()
            _tts_proc.wait()

def wait_speaking():
    """Block until TTS finishes."""
    if _tts_proc:
        _tts_proc.wait()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_models():
    global whisper_model, vad_model
    print("Loading Silero VAD...")
    vad_model = load_silero_vad()
    print("Loading Whisper tiny...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Models ready.")

# ── RECORDING ─────────────────────────────────────────────────────────────────
def record_with_vad() -> np.ndarray | None:
    iterator = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )
    preroll   = collections.deque(maxlen=VAD_PREROLL_CHUNKS)
    recording = []
    speaking  = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=VAD_CHUNK) as stream:
        while True:
            chunk, _ = stream.read(VAD_CHUNK)
            chunk    = chunk.flatten()
            result   = iterator(torch.from_numpy(chunk), return_seconds=False)

            if result:
                if "start" in result:
                    speaking = True
                    recording.extend(preroll)
                    if is_speaking():
                        if float(np.abs(chunk).mean()) > VAD_INTERRUPT_ENERGY:
                            stop_speaking()
                if "end" in result and speaking:
                    recording.append(chunk.copy())
                    break

            if speaking:
                recording.append(chunk.copy())
                if len(recording) >= VAD_MAX_CHUNKS:
                    break
            else:
                preroll.append(chunk.copy())

    return np.concatenate(recording).astype("float32") if recording else None

def transcribe(audio: np.ndarray | None) -> str:
    if audio is None:
        return ""
    segments, _ = whisper_model.transcribe(
        audio, language="en", beam_size=1, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = " ".join(s.text for s in segments).strip().lower()
    print(f"Heard: {text!r}")
    return text

def listen(retries: int = 5) -> str:
    for _ in range(retries):
        text = transcribe(record_with_vad())
        if text.strip():
            return text
        print("Nothing heard, retrying...")
    return ""

# ── NETWORK ───────────────────────────────────────────────────────────────────
def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def send_message(payload: dict) -> dict | None:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=20) as sock:
            data = json.dumps(payload).encode()
            sock.sendall(len(data).to_bytes(4, "big") + data)
            raw_len = _recv_exact(sock, 4)
            if not raw_len:
                return None
            resp_len = int.from_bytes(raw_len, "big")
            raw_resp = _recv_exact(sock, resp_len)
            if not raw_resp:
                return None
            return json.loads(raw_resp.decode())
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"[NET] {e}")
        return None

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global SERVER_HOST, SERVER_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=SERVER_HOST, help="Windows PC IP")
    ap.add_argument("--port", default=SERVER_PORT, type=int)
    args = ap.parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port

    load_models()
    speak("Orange Pi ready. Connecting to server.")

    resp = send_message({"type": "hello"})
    if not resp:
        speak("Cannot reach server. Check network and PC IP.")
        return
    tts = resp.get("tts", "Connected.")
    speak(tts)
    wait_speaking()

    while True:
        text = listen()
        if not text:
            continue

        resp = send_message({"type": "voice", "text": text})
        if not resp:
            speak("Server not responding.")
            continue

        tts = resp.get("tts", "")
        if tts:
            speak(tts)
            wait_speaking()

        if resp.get("quit"):
            break

    speak("Goodbye.")

if __name__ == "__main__":
    main()