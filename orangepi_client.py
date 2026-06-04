#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi 2W Zero
─────────────────────────────────────────────────────
Flow:
  mic → webrtcvad → faster-whisper → print transcript clearly
  → send text (or JPEG frame) over TCP to Windows PC
  ← receive TTS string back
  → espeak-ng | aplay plughw:3,0 speaks it through Logi headset

Camera is now the Logitech webcam CONNECTED TO the OrangePi.
Frames are captured here and sent to the server as JPEG bytes
for YOLO processing. The phone is no longer involved.

Voice calibration flags let you tune espeak-ng from the command
line without editing this file.

Install:
    sudo apt install -y espeak-ng portaudio19-dev libsndfile1
    pip install faster-whisper sounddevice numpy webrtcvad opencv-python-headless
"""

import socket
import json
import threading
import subprocess
import collections
import argparse
import warnings
import time
import struct
import base64

import cv2
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.254.249.159"   # ← your Windows PC IP
SERVER_PORT = 9000

SAMPLE_RATE        = 16000
FRAME_DURATION     = 30
FRAME_SIZE         = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 480 samples
VAD_AGGRESSIVENESS = 2

VOICED_THRESHOLD   = 0.7
UNVOICED_THRESHOLD = 0.4
RING_BUFFER_MS     = 400
RING_BUFFER_FRAMES = int(RING_BUFFER_MS / FRAME_DURATION)
MAX_RECORD_SEC     = 15
MIN_SPEECH_SEC     = 0.3

WHISPER_MODEL    = "tiny"
WHISPER_COMPUTE  = "int8"
WHISPER_LANGUAGE = "en"

# ── ESPEAK VOICE DEFAULTS (overridable via CLI) ────────────────────────────────
ESPEAK_SPEED       = 130          # WPM — lower = clearer / slower
ESPEAK_VOICE       = "en"         # language/voice code
ESPEAK_PITCH       = 30           # 0-99, lower = deeper
ESPEAK_AMPLITUDE   = 180          # 0-200, louder
ESPEAK_ALSA_DEVICE = "plughw:3,0" # ALSA device for Logi headset

# ── CAMERA CONFIG ─────────────────────────────────────────────────────────────
CAM_DEVICE         = 0            # OpenCV device index for Logi webcam on OrangePi
CAM_WARMUP_FRAMES  = 10           # discard this many frames on open (sensor warmup)
CAM_CAPTURE_SEC    = 4.0          # seconds to sample frames
CAM_FPS            = 4            # frames per second to sample
CAM_JPEG_QUALITY   = 80           # JPEG encode quality 0-100 sent to server
MIN_SHARPNESS      = 10           # Laplacian variance minimum

# ── GLOBALS ───────────────────────────────────────────────────────────────────
whisper_model: WhisperModel | None = None
vad: webrtcvad.Vad | None          = None
AUDIO_DEVICE: int | str | None     = None

_tts_lock   = threading.Lock()
_aplay_proc: subprocess.Popen | None = None

# ── TTS ───────────────────────────────────────────────────────────────────────
def _play_wav_bytes(wav_bytes: bytes) -> subprocess.Popen:
    """Feed raw WAV bytes directly into aplay. Returns the aplay process."""
    proc = subprocess.Popen(
        ["aplay", "-D", ESPEAK_ALSA_DEVICE, "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(wav_bytes)
    proc.stdin.close()
    return proc


def speak(text: str, wav_bytes: bytes | None = None) -> None:
    """
    Play audio through Logi headset.
    • If wav_bytes given (SAPI WAV from PC server) → play that directly.
      Gives the exact same Windows voice as the server.
    • Otherwise fall back to local espeak-ng using current calibration.
    Non-blocking — returns immediately while audio plays in background.
    """
    global _aplay_proc
    print(f"\n[TTS] >> {text}\n")
    with _tts_lock:
        if _aplay_proc and _aplay_proc.poll() is None:
            _aplay_proc.terminate()
            _aplay_proc.wait()

        if wav_bytes:
            _aplay_proc = _play_wav_bytes(wav_bytes)
        else:
            espeak = subprocess.Popen(
                ["espeak-ng",
                 "-s", str(ESPEAK_SPEED),
                 "-v", ESPEAK_VOICE,
                 "-p", str(ESPEAK_PITCH),
                 "-a", str(ESPEAK_AMPLITUDE),
                 "--stdout",
                 text],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            aplay = subprocess.Popen(
                ["aplay", "-D", ESPEAK_ALSA_DEVICE, "-q"],
                stdin=espeak.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            espeak.stdout.close()
            _aplay_proc = aplay


def is_speaking() -> bool:
    with _tts_lock:
        return _aplay_proc is not None and _aplay_proc.poll() is None


def stop_speaking() -> None:
    global _aplay_proc
    with _tts_lock:
        if _aplay_proc and _aplay_proc.poll() is None:
            _aplay_proc.terminate()
            _aplay_proc.wait()


def wait_speaking() -> None:
    """Block until audio finishes, then pause so mic doesn't catch reverb."""
    with _tts_lock:
        proc = _aplay_proc
    if proc:
        proc.wait()
    time.sleep(0.4)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_models() -> None:
    global whisper_model, vad
    print("[INIT] Loading webrtcvad...")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("[INIT] webrtcvad ready.")
    print(f"[INIT] Loading faster-whisper '{WHISPER_MODEL}' int8 CPU...")
    print("[INIT] First run downloads the model — may take 30–60s...")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)
    print("[INIT] All models ready.\n")


# ── VAD RECORDING ─────────────────────────────────────────────────────────────
def _to_pcm(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def record_with_vad() -> np.ndarray | None:
    ring_buffer  = collections.deque(maxlen=RING_BUFFER_FRAMES)
    triggered    = False
    voiced_frames: list[np.ndarray] = []
    pre_roll:     list[np.ndarray]  = []
    max_frames   = int(MAX_RECORD_SEC * 1000 / FRAME_DURATION)
    frame_count  = 0

    print("[MIC] Listening... speak now")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=FRAME_SIZE, device=AUDIO_DEVICE,
    ) as stream:
        while frame_count < max_frames:
            chunk, _ = stream.read(FRAME_SIZE)
            chunk    = chunk.flatten()
            pcm      = _to_pcm(chunk)
            if len(pcm) != FRAME_SIZE * 2:
                continue
            is_speech = vad.is_speech(pcm, SAMPLE_RATE)

            if not triggered:
                pre_roll.append(chunk.copy())
                if len(pre_roll) > RING_BUFFER_FRAMES:
                    pre_roll.pop(0)
                ring_buffer.append((chunk.copy(), is_speech))
                if sum(1 for _, s in ring_buffer if s) / len(ring_buffer) > VOICED_THRESHOLD:
                    triggered = True
                    print("[MIC] Speech detected...")
                    if is_speaking():
                        stop_speaking()
                    voiced_frames.extend(pre_roll)
                    ring_buffer.clear()
            else:
                voiced_frames.append(chunk.copy())
                ring_buffer.append((chunk.copy(), is_speech))
                if sum(1 for _, s in ring_buffer if not s) / len(ring_buffer) > UNVOICED_THRESHOLD:
                    print("[MIC] Speech ended.")
                    break
            frame_count += 1

    if not voiced_frames:
        return None
    audio = np.concatenate(voiced_frames).astype("float32")
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_SEC:
        print("[MIC] Too short, discarding.")
        return None
    return audio


# ── TRANSCRIPTION ─────────────────────────────────────────────────────────────
def transcribe(audio: np.ndarray | None) -> str:
    if audio is None:
        return ""
    print("[STT] Transcribing...")
    segments, _ = whisper_model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100},
    )
    text = " ".join(s.text for s in segments).strip()
    return text.lower()


def listen(retries: int = 5) -> str:
    for attempt in range(1, retries + 1):
        audio = record_with_vad()
        text  = transcribe(audio)
        if text.strip():
            print("\n" + "─" * 50)
            print(f"  YOU SAID  →  {text}")
            print("─" * 50 + "\n")
            return text
        print(f"[STT] Nothing understood ({attempt}/{retries}), retrying...")
    return ""


# ── CAMERA — Logitech webcam connected to OrangePi ───────────────────────────
def capture_best_frame_local() -> bytes | None:
    """
    Open the Logitech webcam on the OrangePi, capture CAM_CAPTURE_SEC seconds
    of frames, pick the sharpest one, and return it as JPEG bytes.
    Returns None if the camera is unavailable or all frames are too blurry.
    """
    print(f"[CAM] Opening Logi webcam (device {CAM_DEVICE})...")
    cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback: try without explicit backend
        cap = cv2.VideoCapture(CAM_DEVICE)
    if not cap.isOpened():
        print("[CAM] ERROR: Could not open webcam.")
        return None

    # Discard warmup frames so the sensor exposure settles
    print(f"[CAM] Warming up — discarding {CAM_WARMUP_FRAMES} frames...")
    for _ in range(CAM_WARMUP_FRAMES):
        cap.grab()

    best_frame = None
    best_score = -1.0
    count      = 0
    interval   = 1.0 / CAM_FPS
    end_time   = time.time() + CAM_CAPTURE_SEC

    print(f"[CAM] Capturing {CAM_CAPTURE_SEC}s of frames at {CAM_FPS} fps...")
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAM] Frame read failed mid-capture.")
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"[CAM] Frame {count + 1}: sharpness={score:.1f}")
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
        count += 1
        time.sleep(interval)

    cap.release()
    print(f"[CAM] Best sharpness: {best_score:.1f} from {count} frames")

    if best_score < MIN_SHARPNESS or best_frame is None:
        print("[CAM] All frames too blurry.")
        return None

    # Encode to JPEG for network transfer
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), CAM_JPEG_QUALITY]
    ok, buf = cv2.imencode(".jpg", best_frame, encode_param)
    if not ok:
        print("[CAM] JPEG encode failed.")
        return None

    jpeg_bytes = buf.tobytes()
    print(f"[CAM] JPEG size: {len(jpeg_bytes) / 1024:.1f} KB")
    return jpeg_bytes


# ── NETWORK — one persistent connection per session ───────────────────────────
def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _send(sock: socket.socket, payload: dict) -> dict | None:
    """
    Serialise payload to JSON and send with a 4-byte big-endian length prefix.
    'frame' key (if present) must be raw bytes — it is base64-encoded here
    so the whole payload stays JSON-compatible.
    """
    try:
        # If a camera frame is attached, base64-encode it into the JSON
        if "frame" in payload and isinstance(payload["frame"], (bytes, bytearray)):
            payload = dict(payload)          # don't mutate caller's dict
            payload["frame"] = base64.b64encode(payload["frame"]).decode("ascii")

        data = json.dumps(payload).encode("utf-8")
        sock.sendall(struct.pack(">I", len(data)) + data)

        raw_len = _recv_exact(sock, 4)
        if raw_len is None:
            return None
        resp_len = struct.unpack(">I", raw_len)[0]
        raw_resp = _recv_exact(sock, resp_len)
        if raw_resp is None:
            return None
        return json.loads(raw_resp.decode("utf-8"))
    except (OSError, TimeoutError) as e:
        print(f"[NET] Error: {e}")
        return None


def connect_to_server() -> socket.socket | None:
    try:
        print(f"[NET] Connecting to {SERVER_HOST}:{SERVER_PORT}...")
        sock = socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=20)
        sock.settimeout(120)   # longer timeout — frame upload can take a moment
        print("[NET] Connected.")
        return sock
    except ConnectionRefusedError:
        print("[NET] Connection refused — is pc_server.py running?")
    except TimeoutError:
        print("[NET] Timeout.")
    except OSError as e:
        print(f"[NET] OS error: {e}")
    return None


def _speak_resp(resp: dict) -> None:
    """Play TTS from server response. Uses SAPI WAV if provided."""
    tts       = resp.get("tts", "")
    if not tts:
        return
    audio_b64 = resp.get("audio")
    wav_bytes  = base64.b64decode(audio_b64) if audio_b64 else None
    speak(tts, wav_bytes)


# ── SCAN TRIGGER WORDS (mirror the server set) ────────────────────────────────
SCAN_TRIGGERS = {
    "scan","check","go","now","yes","okay","ok","capture","take","snap",
    "shoot","photo","picture","frame","analyze","detect","read","process",
    "identify","start","run","next","continue","more","again","ready",
    "do it","let's go","lets go","scan it","do scan","take picture",
    "take photo","scan now","check now","yalla","hayde","sur",
}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    global SERVER_HOST, SERVER_PORT, AUDIO_DEVICE
    global ESPEAK_SPEED, ESPEAK_VOICE, ESPEAK_PITCH, ESPEAK_AMPLITUDE, ESPEAK_ALSA_DEVICE
    global CAM_DEVICE, CAM_WARMUP_FRAMES, CAM_CAPTURE_SEC, CAM_FPS, CAM_JPEG_QUALITY

    ap = argparse.ArgumentParser(description="OrangePi 2W Zero voice client")

    # ── Network ───────────────────────────────────────────────────────────────
    ap.add_argument("--host",   default=SERVER_HOST, help="Windows PC IP")
    ap.add_argument("--port",   default=SERVER_PORT, type=int)

    # ── Whisper ───────────────────────────────────────────────────────────────
    ap.add_argument("--model",  default=WHISPER_MODEL, choices=["tiny", "base"])

    # ── Mic device ────────────────────────────────────────────────────────────
    ap.add_argument("--device", default=None,
                    help="Mic index or hw string. "
                         "List with: python3 -c 'import sounddevice; print(sounddevice.query_devices())'")

    # ── Voice calibration (espeak-ng) ─────────────────────────────────────────
    ap.add_argument("--speed",     type=int,   default=ESPEAK_SPEED,
                    help="espeak-ng speed in words per minute (default: %(default)s). "
                         "Lower = slower and clearer. Try 90-150.")
    ap.add_argument("--pitch",     type=int,   default=ESPEAK_PITCH,
                    help="espeak-ng pitch 0-99 (default: %(default)s). "
                         "Lower = deeper. Try 10-50.")
    ap.add_argument("--amplitude", type=int,   default=ESPEAK_AMPLITUDE,
                    help="espeak-ng amplitude 0-200 (default: %(default)s). "
                         "Higher = louder.")
    ap.add_argument("--voice",     type=str,   default=ESPEAK_VOICE,
                    help="espeak-ng voice code (default: %(default)s). "
                         "Examples: en, en-us, en-gb, en+m3, en+f3. "
                         "Run 'espeak-ng --voices=en' to list all English variants.")
    ap.add_argument("--alsa",      type=str,   default=ESPEAK_ALSA_DEVICE,
                    help="ALSA output device (default: %(default)s). "
                         "Run 'aplay -l' to list devices.")

    # ── Camera (Logi webcam on OrangePi) ─────────────────────────────────────
    ap.add_argument("--cam",        type=int,   default=CAM_DEVICE,
                    help="Logitech webcam OpenCV device index (default: %(default)s). "
                         "Run 'v4l2-ctl --list-devices' to find yours.")
    ap.add_argument("--cam-warmup", type=int,   default=CAM_WARMUP_FRAMES,
                    help="Frames to discard on camera open for sensor warmup (default: %(default)s).")
    ap.add_argument("--cam-sec",    type=float, default=CAM_CAPTURE_SEC,
                    help="Seconds to capture frames per scan (default: %(default)s).")
    ap.add_argument("--cam-fps",    type=int,   default=CAM_FPS,
                    help="Frames per second to sample during capture (default: %(default)s).")
    ap.add_argument("--cam-quality",type=int,   default=CAM_JPEG_QUALITY,
                    help="JPEG encode quality 0-100 sent to server (default: %(default)s).")

    args = ap.parse_args()

    # Apply all settings
    SERVER_HOST       = args.host
    SERVER_PORT       = args.port
    ESPEAK_SPEED      = args.speed
    ESPEAK_PITCH      = args.pitch
    ESPEAK_AMPLITUDE  = args.amplitude
    ESPEAK_VOICE      = args.voice
    ESPEAK_ALSA_DEVICE= args.alsa
    CAM_DEVICE        = args.cam
    CAM_WARMUP_FRAMES = args.cam_warmup
    CAM_CAPTURE_SEC   = args.cam_sec
    CAM_FPS           = args.cam_fps
    CAM_JPEG_QUALITY  = args.cam_quality

    # Print current voice calibration so user can confirm before starting
    print("─" * 60)
    print("  VOICE CALIBRATION")
    print(f"    Speed     : {ESPEAK_SPEED} WPM")
    print(f"    Pitch     : {ESPEAK_PITCH}")
    print(f"    Amplitude : {ESPEAK_AMPLITUDE}")
    print(f"    Voice     : {ESPEAK_VOICE}")
    print(f"    ALSA out  : {ESPEAK_ALSA_DEVICE}")
    print("  CAMERA")
    print(f"    Device    : /dev/video{CAM_DEVICE}  (index {CAM_DEVICE})")
    print(f"    Warmup    : {CAM_WARMUP_FRAMES} frames")
    print(f"    Capture   : {CAM_CAPTURE_SEC}s @ {CAM_FPS} fps")
    print(f"    Quality   : {CAM_JPEG_QUALITY}%")
    print("─" * 60)

    # Resolve mic device
    if args.device is not None:
        try:
            AUDIO_DEVICE = int(args.device)
        except ValueError:
            AUDIO_DEVICE = args.device
        print(f"[AUDIO] Using device: {AUDIO_DEVICE}")
    else:
        for i, dev in enumerate(sd.query_devices()):
            name = dev["name"].lower()
            if dev["max_input_channels"] > 0 and ("logi" in name or "usb" in name):
                AUDIO_DEVICE = i
                print(f"[AUDIO] Auto-detected mic: [{i}] {dev['name']}")
                break
        if AUDIO_DEVICE is None:
            print("[AUDIO] No USB mic found — using system default.")

    load_models()

    speak("Orange Pi ready. Connecting to server.")
    wait_speaking()

    sock = connect_to_server()
    if sock is None:
        speak("Cannot reach server. Check network and I P address.")
        wait_speaking()
        return

    resp = _send(sock, {"type": "hello"})
    if resp is None:
        speak("Server did not respond.")
        wait_speaking()
        sock.close()
        return

    _speak_resp(resp)
    wait_speaking()

    # ── Main voice loop ───────────────────────────────────────────────────────
    while True:
        text = listen()
        if not text:
            continue

        # ── If this is a scan command, capture the frame HERE on the OrangePi ──
        payload: dict = {"type": "voice", "text": text}

        if any(t in text for t in SCAN_TRIGGERS):
            print("[CAM] Scan trigger detected — capturing frame from Logi webcam...")
            jpeg_bytes = capture_best_frame_local()
            if jpeg_bytes is None:
                speak("Camera error. Could not capture a frame. Please try again.")
                wait_speaking()
                continue
            payload["frame"] = jpeg_bytes   # _send() will base64-encode this

        print(f"[NET] Sending → {text!r}"
              + (f"  +{len(payload.get('frame', b'')) / 1024:.1f} KB frame"
                 if "frame" in payload else ""))

        resp = _send(sock, payload)

        if resp is None:
            speak("Connection lost. Reconnecting.")
            wait_speaking()
            sock.close()
            time.sleep(2)
            sock = connect_to_server()
            if sock is None:
                speak("Could not reconnect. Exiting.")
                wait_speaking()
                break
            speak("Reconnected. Session was reset. Please start again.")
            wait_speaking()
            resp = _send(sock, {"type": "hello"})
            if resp:
                _speak_resp(resp)
                wait_speaking()
            continue

        _speak_resp(resp)
        wait_speaking()

        if resp.get("quit"):
            break

    sock.close()
    speak("Goodbye.")
    wait_speaking()

if __name__ == "__main__":
    main()