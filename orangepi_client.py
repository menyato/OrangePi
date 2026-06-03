#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi 2W Zero
─────────────────────────────────────────────────────
OrangePi 2W Zero specs this code respects:
  • Allwinner H618 quad-core ARM Cortex-A53 @ 1.5GHz
  • 512 MB or 1 GB LPDDR4 RAM  ← very tight
  • No GPU, no NPU acceleration for torch
  • Linux (Debian/Ubuntu arm64)

What this script does (ONLY):
  1. Record mic audio with a lightweight WebRTC-based VAD
     (webrtcvad — pure C extension, zero torch dependency)
  2. Transcribe with faster-whisper tiny/int8 (CPU only)
  3. Send transcript to Windows PC over TCP
  4. Receive TTS string back
  5. Speak via espeak-ng

What this script does NOT do:
  • No camera, no YOLO, no image processing
  • No torch (removed — was only needed for Silero VAD)
  • No silero-vad (replaced with webrtcvad — ~10× lighter)

Flow:
  mic → webrtcvad → faster-whisper → TCP → PC brain
                                    TCP ← TTS string
  espeak-ng ←─────────────────────────────────────────

Install (run as root or with sudo):
    sudo apt update
    sudo apt install -y espeak-ng portaudio19-dev libsndfile1 python3-pip
    pip3 install --break-system-packages \
        faster-whisper sounddevice numpy webrtcvad

NOTE: No torch install needed at all for this script.
      faster-whisper on CPU with int8 uses ctranslate2 (C++) — much lighter.
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

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.80.64.159"   # ← Windows PC IP — change this
SERVER_PORT = 9000

# Audio
SAMPLE_RATE    = 16000          # Hz — required by webrtcvad & whisper
FRAME_DURATION = 30             # ms per VAD frame — must be 10, 20, or 30
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # samples per frame = 480
VAD_AGGRESSIVENESS = 2          # 0 (least) … 3 (most aggressive filtering)

# VAD speech detection tunables
VOICED_THRESHOLD   = 0.7        # fraction of frames in ring buffer that must be voiced to start recording
UNVOICED_THRESHOLD = 0.4        # fraction of frames that must be UNvoiced to stop recording
RING_BUFFER_MS     = 400        # ms of context around speech boundaries
RING_BUFFER_FRAMES = int(RING_BUFFER_MS / FRAME_DURATION)  # = 13 frames
MAX_RECORD_SEC     = 15         # hard stop to avoid infinite recording
MIN_SPEECH_SEC     = 0.3        # discard clips shorter than this (avoids noise bursts)

# Whisper model settings — "tiny" is the only sane choice on 512 MB RAM
WHISPER_MODEL    = "tiny"       # tiny=~40 MB RAM, base=~80 MB, small=~250 MB
WHISPER_COMPUTE  = "int8"       # int8 quantised — fastest on CPU, ~half memory
WHISPER_LANGUAGE = "en"         # force English — saves beam-search time

# espeak-ng settings
ESPEAK_SPEED = 145              # words per minute
ESPEAK_VOICE = "en"

# ── GLOBALS ───────────────────────────────────────────────────────────────────
whisper_model: WhisperModel | None = None
vad: webrtcvad.Vad | None = None
AUDIO_DEVICE: str | None = None  # set via --device arg or auto-detected

_tts_lock = threading.Lock()
_tts_proc: subprocess.Popen | None = None

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    """Speak text asynchronously via espeak-ng, killing any current speech."""
    global _tts_proc
    print(f"[TTS] >> {text}")
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()
            _tts_proc.wait()
        _tts_proc = subprocess.Popen(
            ["espeak-ng", "-s", str(ESPEAK_SPEED), "-v", ESPEAK_VOICE, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def is_speaking() -> bool:
    with _tts_lock:
        return _tts_proc is not None and _tts_proc.poll() is None

def stop_speaking() -> None:
    global _tts_proc
    with _tts_lock:
        if _tts_proc and _tts_proc.poll() is None:
            _tts_proc.terminate()
            _tts_proc.wait()

def wait_speaking() -> None:
    """Block until current TTS finishes."""
    with _tts_lock:
        proc = _tts_proc
    if proc:
        proc.wait()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_models() -> None:
    global whisper_model, vad

    print("[INIT] Loading webrtcvad...")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("[INIT] webrtcvad ready.")

    print(f"[INIT] Loading faster-whisper '{WHISPER_MODEL}' (int8, CPU)...")
    print("[INIT] This may take 30–60 s on first run while downloading the model...")
    whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type=WHISPER_COMPUTE,
        # keep model files in ~/.cache/huggingface to avoid re-download
    )
    print("[INIT] Models ready. Waiting for voice input.")

# ── VAD RECORDING ─────────────────────────────────────────────────────────────
def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Convert float32 [-1,1] audio frame to int16 PCM bytes for webrtcvad."""
    pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.tobytes()

def record_with_vad() -> np.ndarray | None:
    """
    Record until speech is detected, then record until silence.
    Returns float32 numpy array at SAMPLE_RATE, or None if nothing captured.

    webrtcvad ring-buffer algorithm:
      • Fill a ring buffer of the last N frames.
      • If >VOICED_THRESHOLD fraction are voiced → speech started.
      • After speech starts, if >UNVOICED_THRESHOLD fraction are unvoiced → speech ended.
    """
    ring_buffer: collections.deque = collections.deque(maxlen=RING_BUFFER_FRAMES)
    triggered   = False
    voiced_frames: list[np.ndarray] = []  # frames that are part of the utterance
    pre_roll: list[np.ndarray] = []       # frames before speech (captured for context)

    max_frames  = int(MAX_RECORD_SEC * 1000 / FRAME_DURATION)
    frame_count = 0

    print("[VAD] Listening... (speak now)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        device=AUDIO_DEVICE,
    ) as stream:

        while frame_count < max_frames:
            audio_chunk, _ = stream.read(FRAME_SIZE)
            audio_chunk = audio_chunk.flatten()

            # webrtcvad needs exact int16 PCM — check length
            pcm_bytes = _frame_to_bytes(audio_chunk)
            if len(pcm_bytes) != FRAME_SIZE * 2:
                continue  # skip malformed frames

            is_speech = vad.is_speech(pcm_bytes, SAMPLE_RATE)

            if not triggered:
                # Accumulate pre-roll for context
                pre_roll.append(audio_chunk.copy())
                if len(pre_roll) > RING_BUFFER_FRAMES:
                    pre_roll.pop(0)

                ring_buffer.append((audio_chunk.copy(), is_speech))
                num_voiced = sum(1 for _, speech in ring_buffer if speech)

                if num_voiced / len(ring_buffer) > VOICED_THRESHOLD:
                    triggered = True
                    print("[VAD] Speech started.")

                    # If TTS is playing and user interrupts, stop it
                    if is_speaking():
                        stop_speaking()

                    # Include pre-roll so we don't cut the start of the word
                    voiced_frames.extend(pre_roll)
                    ring_buffer.clear()

            else:
                voiced_frames.append(audio_chunk.copy())
                ring_buffer.append((audio_chunk.copy(), is_speech))
                num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)

                if num_unvoiced / len(ring_buffer) > UNVOICED_THRESHOLD:
                    print("[VAD] Speech ended.")
                    break

            frame_count += 1

    if not voiced_frames:
        return None

    audio = np.concatenate(voiced_frames).astype("float32")

    # Discard clips that are too short (noise bursts, clicks)
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_SEC:
        print("[VAD] Clip too short, discarding.")
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
        beam_size=1,                  # greedy — fastest on weak CPU
        best_of=1,
        temperature=0.0,              # deterministic
        vad_filter=True,              # whisper's own built-in VAD as a second pass
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 100,
        },
    )
    text = " ".join(s.text for s in segments).strip()
    print(f"[STT] Heard: {text!r}")
    return text.lower()

def listen(retries: int = 5) -> str:
    """Record and transcribe, retrying up to `retries` times on empty result."""
    for attempt in range(1, retries + 1):
        audio = record_with_vad()
        text  = transcribe(audio)
        if text.strip():
            return text
        print(f"[STT] Nothing understood (attempt {attempt}/{retries}), listening again...")
    return ""

# ── NETWORK ───────────────────────────────────────────────────────────────────
def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes from socket, returning None on connection drop."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def send_message(payload: dict) -> dict | None:
    """
    Send a JSON payload to the PC server and return the JSON response.
    Each message is prefixed with a 4-byte big-endian length header.
    Returns None on any network error.
    """
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=20) as sock:
            data = json.dumps(payload).encode("utf-8")
            # Length-prefixed framing: [4 bytes big-endian length][data]
            sock.sendall(struct.pack(">I", len(data)) + data)

            raw_len = _recv_exact(sock, 4)
            if raw_len is None:
                print("[NET] Server closed connection before response.")
                return None
            resp_len = struct.unpack(">I", raw_len)[0]

            raw_resp = _recv_exact(sock, resp_len)
            if raw_resp is None:
                print("[NET] Incomplete response from server.")
                return None

            return json.loads(raw_resp.decode("utf-8"))

    except ConnectionRefusedError:
        print(f"[NET] Connection refused — is pc_server.py running on {SERVER_HOST}:{SERVER_PORT}?")
    except TimeoutError:
        print(f"[NET] Timeout — PC not responding.")
    except OSError as e:
        print(f"[NET] OS error: {e}")
    return None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main() -> None:
    global SERVER_HOST, SERVER_PORT

    ap = argparse.ArgumentParser(description="OrangePi 2W Zero voice client")
    ap.add_argument("--host", default=SERVER_HOST,
                    help=f"Windows PC IP (default: {SERVER_HOST})")
    ap.add_argument("--port", default=SERVER_PORT, type=int,
                    help=f"Server port (default: {SERVER_PORT})")
    ap.add_argument("--model", default=WHISPER_MODEL,
                    choices=["tiny", "base"],
                    help="Whisper model size — 'tiny' recommended for 512 MB RAM")
    ap.add_argument("--device", default=None,
                    help="Input device: index number OR hw string e.g. hw:3,0 (Logi USB Headset default)")
    args = ap.parse_args()

    SERVER_HOST = args.host
    SERVER_PORT = args.port

    # Resolve audio device — prefer CLI arg, fall back to auto-detect Logi headset
    global AUDIO_DEVICE
    if args.device is not None:
        # Accept either integer index or hw:X,Y string
        try:
            AUDIO_DEVICE = int(args.device)
        except ValueError:
            AUDIO_DEVICE = args.device
        print(f"[AUDIO] Using device: {AUDIO_DEVICE}")
    else:
        # Auto-detect: find first device with 'Logi' or 'USB' in name that has inputs
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and ('logi' in dev['name'].lower() or 'usb' in dev['name'].lower()):
                AUDIO_DEVICE = i
                print(f"[AUDIO] Auto-detected USB mic: [{i}] {dev['name']}")
                break
        if AUDIO_DEVICE is None:
            # Last resort: find any device with input channels
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0 and 'hw:' in dev.get('name','').lower():
                    AUDIO_DEVICE = i
                    print(f"[AUDIO] Fallback mic: [{i}] {dev['name']}")
                    break
        if AUDIO_DEVICE is None:
            print("[AUDIO] WARNING: Could not auto-detect mic. Using system default.")

    load_models()

    speak("Orange Pi ready. Connecting to server.")

    # Handshake
    resp = send_message({"type": "hello"})
    if resp is None:
        speak("Cannot reach server. Check network and PC IP address.")
        return

    tts = resp.get("tts", "Connected.")
    speak(tts)
    wait_speaking()

    # Main voice loop
    while True:
        text = listen()
        if not text:
            # Listening timed out with no speech — just keep going
            continue

        resp = send_message({"type": "voice", "text": text})
        if resp is None:
            speak("Server not responding. Please check connection.")
            time.sleep(2)
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