#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi 2W Zero
─────────────────────────────────────────────────────
Flow:
  mic → webrtcvad → faster-whisper → print transcript clearly
  → send text over TCP to Windows PC
  ← receive TTS string back
  → espeak-ng | aplay plughw:3,0 speaks it through Logi headset

No torch, no silero, no YOLO, no camera.

Install:
    sudo apt install -y espeak-ng portaudio19-dev libsndfile1
    pip install faster-whisper sounddevice numpy webrtcvad
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

ESPEAK_SPEED       = 150
ESPEAK_VOICE       = "mb-us1"      # cleaner American English voice
ESPEAK_PITCH       = 45           # slightly lower = more natural
# plughw handles: mono→stereo upmix + 22050→48000 Hz resampling automatically
ESPEAK_ALSA_DEVICE = "plughw:3,0"

# ── GLOBALS ───────────────────────────────────────────────────────────────────
whisper_model: WhisperModel | None = None
vad: webrtcvad.Vad | None          = None
AUDIO_DEVICE: int | str | None     = None

_tts_lock   = threading.Lock()
_aplay_proc: subprocess.Popen | None = None   # track aplay, not espeak

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    """
    Speak text through Logi headset via espeak-ng → aplay plughw:3,0.
    plughw automatically converts mono 22050 Hz → stereo 48000 Hz.
    Non-blocking — returns immediately while audio plays.
    """
    global _aplay_proc
    print(f"\n[TTS] >> {text}\n")
    with _tts_lock:
        # Kill any currently playing audio
        if _aplay_proc and _aplay_proc.poll() is None:
            _aplay_proc.terminate()
            _aplay_proc.wait()

        espeak = subprocess.Popen(
            ["espeak-ng",
             "-s", str(ESPEAK_SPEED),
             "-v", ESPEAK_VOICE,
             "-p", str(ESPEAK_PITCH),
             "--stdout",
             text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay = subprocess.Popen(
            ["aplay",
             "-D", ESPEAK_ALSA_DEVICE,   # plughw:3,0 — auto format conversion
             "-q"],                       # quiet — suppress "Playing WAVE..." line
            stdin=espeak.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        espeak.stdout.close()  # let espeak get SIGPIPE if aplay exits early
        _aplay_proc = aplay    # track aplay — it finishes when audio is done

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
    time.sleep(0.4)   # short silence — prevents mic from catching speaker tail

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
    try:
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
        sock.settimeout(60)
        print(f"[NET] Connected.")
        return sock
    except ConnectionRefusedError:
        print(f"[NET] Connection refused — is pc_server.py running?")
    except TimeoutError:
        print(f"[NET] Timeout.")
    except OSError as e:
        print(f"[NET] OS error: {e}")
    return None

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    global SERVER_HOST, SERVER_PORT, AUDIO_DEVICE

    ap = argparse.ArgumentParser(description="OrangePi 2W Zero voice client")
    ap.add_argument("--host",   default=SERVER_HOST, help="Windows PC IP")
    ap.add_argument("--port",   default=SERVER_PORT, type=int)
    ap.add_argument("--model",  default=WHISPER_MODEL, choices=["tiny", "base"])
    ap.add_argument("--device", default=None,
                    help="Mic index or hw string. List with: python3 -c 'import sounddevice; print(sounddevice.query_devices())'")
    args = ap.parse_args()

    SERVER_HOST = args.host
    SERVER_PORT = args.port

    # Resolve audio input device
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

    # ── Startup message through headset ───────────────────────────────────────
    speak("Orange Pi ready. Connecting to server.")
    wait_speaking()

    # ── One persistent TCP connection — server ties session state to it ───────
    sock = connect_to_server()
    if sock is None:
        speak("Cannot reach server. Check network and IP address.")
        wait_speaking()
        return

    resp = _send(sock, {"type": "hello"})
    if resp is None:
        speak("Server did not respond.")
        wait_speaking()
        sock.close()
        return

    speak(resp.get("tts", "Connected."))
    wait_speaking()

    # ── Main voice loop ───────────────────────────────────────────────────────
    while True:
        text = listen()
        if not text:
            continue

        print(f"[NET] Sending → {text!r}")
        resp = _send(sock, {"type": "voice", "text": text})

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
                speak(resp.get("tts", ""))
                wait_speaking()
            continue

        tts = resp.get("tts", "")
        if tts:
            speak(tts)
            wait_speaking()

        if resp.get("quit"):
            break

    sock.close()
    speak("Goodbye.")
    wait_speaking()

if __name__ == "__main__":
    main()