#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi
───────────────────────────────────────────
Responsibilities:
  • Record voice with Silero VAD
  • Transcribe with faster-whisper (tiny, CPU)
  • Send text command to Windows server over TCP
  • Receive TTS text back and speak it via espeak/piper
  • (Later) capture frame and send as JPEG for YOLO analysis

Install on OrangePi:
    pip install faster-whisper sounddevice numpy torch silero-vad

TTS engine (choose one):
    sudo apt install espeak-ng          # simple, always works
    # or install piper for better quality
"""

import socket
import json
import time
import threading
import subprocess
import collections
import numpy as np
import sounddevice as sd
import torch
import warnings
import argparse

warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.80.64.159"   # ← set to your Windows PC IP
SERVER_PORT = 9000
BUFFER_SIZE = 65536

SAMPLE_RATE          = 16000
VAD_CHUNK            = 512
VAD_THRESHOLD        = 0.5
VAD_MIN_SILENCE_MS   = 600
VAD_SPEECH_PAD_MS    = 200
VAD_MAX_CHUNKS       = 400
VAD_PREROLL_CHUNKS   = 10
VAD_INTERRUPT_ENERGY = 0.04

TTS_ENGINE = "espeak"   # "espeak" or "piper"
PIPER_EXE  = "/usr/local/bin/piper"
PIPER_MODEL= "/home/orangepi/piper/en_US-lessac-medium.onnx"

# ── GLOBALS ───────────────────────────────────────────────────────────────────
whisper_model = None
vad_model     = None
_tts_lock     = threading.Lock()
_tts_proc     = None

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    """Non-blocking TTS — kills previous utterance first."""
    global _tts_proc
    print(f">> {text}")
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()
        if TTS_ENGINE == "piper":
            _tts_proc = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_MODEL, "--output-raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
            play = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                stdin=_tts_proc.stdout
            )
            _tts_proc.stdin.write(text.encode())
            _tts_proc.stdin.close()
        else:
            _tts_proc = subprocess.Popen(
                ["espeak-ng", "-s", "145", "-v", "en", text]
            )

def is_speaking() -> bool:
    with _tts_lock:
        return _tts_proc is not None and _tts_proc.poll() is None

def stop_speaking():
    global _tts_proc
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_models():
    global whisper_model, vad_model
    print("Loading Silero VAD...")
    vad_model = load_silero_vad()
    print("Loading Whisper tiny...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Models ready")

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

def listen_once() -> str:
    return transcribe(record_with_vad())

def listen(retries: int = 5) -> str:
    for _ in range(retries):
        text = listen_once()
        if text.strip():
            return text
        print("Nothing heard, retrying...")
    return ""

# ── NETWORK ───────────────────────────────────────────────────────────────────
def send_message(payload: dict) -> dict | None:
    """Send JSON payload to server, return JSON response."""
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=15) as sock:
            data = json.dumps(payload).encode()
            # Send length-prefixed message
            sock.sendall(len(data).to_bytes(4, "big") + data)
            # Read response length
            raw_len = _recv_exact(sock, 4)
            if not raw_len:
                return None
            resp_len = int.from_bytes(raw_len, "big")
            raw_resp = _recv_exact(sock, resp_len)
            if not raw_resp:
                return None
            return json.loads(raw_resp.decode())
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"[NET] Error: {e}")
        return None

def send_image(jpeg_bytes: bytes, extra: dict = {}) -> dict | None:
    """Send image + metadata to server."""
    import base64
    payload = {"type": "image", "image": base64.b64encode(jpeg_bytes).decode()}
    payload.update(extra)
    return send_message(payload)

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=SERVER_HOST)
    ap.add_argument("--port", default=SERVER_PORT, type=int)
    args = ap.parse_args()

    global SERVER_HOST, SERVER_PORT
    SERVER_HOST = args.host
    SERVER_PORT = args.port

    load_models()
    speak("Orange Pi client ready. Connecting to server.")

    # ── Handshake ──
    resp = send_message({"type": "hello"})
    if not resp:
        speak("Cannot reach server. Check network.")
        return
    speak(resp.get("tts", "Connected."))

    # ── Main voice loop ──
    while True:
        text = listen()
        if not text:
            continue

        # Send voice command to server
        resp = send_message({"type": "voice", "text": text})
        if not resp:
            speak("Server not responding.")
            continue

        # Speak whatever the server wants us to say
        tts = resp.get("tts")
        if tts:
            speak(tts)

        # Server can request a capture
        action = resp.get("action")
        if action == "capture":
            speak("Hold steady. Capturing.")
            frame = capture_frame()
            if frame is not None:
                import cv2
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                resp2 = send_image(jpeg.tobytes(), {"type": "image"})
                if resp2:
                    tts2 = resp2.get("tts")
                    if tts2:
                        speak(tts2)
                else:
                    speak("Could not send image.")
            else:
                speak("Could not capture image.")

        if resp.get("quit"):
            speak("Goodbye.")
            break

# ── CAMERA (placeholder — swap with real capture later) ───────────────────────
def capture_frame():
    """
    Capture best frame from OrangePi camera.
    Right now this is a stub — replace with your actual camera code.
    Returns a BGR numpy array or None.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)   # /dev/video0 — adjust as needed
        if not cap.isOpened():
            print("[CAM] Cannot open camera")
            return None

        best_frame = None
        best_score = -1
        end_time   = time.time() + 3.0

        while time.time() < end_time:
            for _ in range(3):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

        cap.release()
        print(f"[CAM] Best sharpness: {best_score:.1f}")
        return best_frame if best_score > 20 else None
    except Exception as e:
        print(f"[CAM] Error: {e}")
        return None

if __name__ == "__main__":
    main()
