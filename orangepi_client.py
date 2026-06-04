#!/usr/bin/env python3
"""
orangepi_client.py  —  runs ON the OrangePi 2W Zero
─────────────────────────────────────────────────────
Auto-detects Logitech mic, speaker (ALSA), and webcam by scanning
system devices at startup — no manual config needed.

Flow:
  mic → webrtcvad → faster-whisper → print transcript
  → send text (or JPEG frame) over TCP to Windows PC
  ← receive TTS WAV back
  → aplay through auto-detected Logi speaker

Install:
    sudo apt install -y espeak-ng portaudio19-dev libsndfile1 v4l-utils
    pip install faster-whisper sounddevice numpy webrtcvad opencv-python-headless
"""

import socket, json, threading, subprocess, collections
import argparse, warnings, time, struct, base64, glob, re, os

import cv2
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ── SERVER ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.254.249.159"
SERVER_PORT = 9000

# ── VAD / WHISPER ─────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
FRAME_DURATION     = 30
FRAME_SIZE         = int(SAMPLE_RATE * FRAME_DURATION / 1000)   # 480 samples
VAD_AGGRESSIVENESS = 2
VOICED_THRESHOLD   = 0.7
UNVOICED_THRESHOLD = 0.4
RING_BUFFER_MS     = 400
RING_BUFFER_FRAMES = int(RING_BUFFER_MS / FRAME_DURATION)
MAX_RECORD_SEC     = 15
MIN_SPEECH_SEC     = 0.3
WHISPER_MODEL      = "tiny"
WHISPER_COMPUTE    = "int8"
WHISPER_LANGUAGE   = "en"

# ── ESPEAK DEFAULTS (tunable via CLI) ─────────────────────────────────────────
ESPEAK_SPEED     = 130
ESPEAK_VOICE     = "en"
ESPEAK_PITCH     = 30
ESPEAK_AMPLITUDE = 180

# ── CAMERA ────────────────────────────────────────────────────────────────────
CAM_WARMUP_FRAMES = 10
CAM_CAPTURE_SEC   = 4.0
CAM_FPS           = 4
CAM_JPEG_QUALITY  = 80
MIN_SHARPNESS     = 10

# ── RUNTIME GLOBALS (filled by auto-detect) ───────────────────────────────────
AUDIO_INPUT_DEVICE: int | None  = None   # sounddevice index for mic
ALSA_OUTPUT_DEVICE: str | None  = None   # e.g. "plughw:3,0" for speaker
CAM_DEVICE:         int         = 0      # OpenCV /dev/videoN index

whisper_model: WhisperModel | None = None
vad:           webrtcvad.Vad | None = None

_tts_lock   = threading.Lock()
_aplay_proc: subprocess.Popen | None = None

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

LOGI_KEYWORDS = ["logi", "c270", "c920", "c310", "c525", "c922",
                 "webcam", "usb audio", "usb mic"]

def _is_logi(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in LOGI_KEYWORDS)


def detect_mic() -> int | None:
    """
    Return the sounddevice input-device index for the Logitech mic.
    Falls back to the system default (None) if not found.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and _is_logi(dev["name"]):
            print(f"[DETECT] Mic  → [{i}] {dev['name']}")
            return i
    # second pass: any USB input
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and "usb" in dev["name"].lower():
            print(f"[DETECT] Mic  → [{i}] {dev['name']}  (USB fallback)")
            return i
    print("[DETECT] Mic  → system default (no Logi/USB mic found)")
    return None


def detect_speaker() -> str:
    """
    Return an ALSA plughw string for the Logitech *speaker/headset*.

    Key insight from probe output:
      • card 3 = C270 HD WEBCAM  → capture only, NOT a playback device
      • card 4 = Logi USB Headset → has playback (and capture)
    So we MUST read from 'aplay -l' (playback list), not 'arecord -l',
    and we prefer cards whose short name contains "headset" or "logi"
    OVER cards that look like webcams.

    Priority order:
      1. Any playback card whose bracket-name contains "headset"
      2. Any playback card matching LOGI_KEYWORDS but NOT "webcam"/"c270"
      3. First USB playback card
      4. Hard fallback plughw:0,0
    """
    WEBCAM_WORDS = {"webcam", "c270", "c920", "c310", "c525", "c922"}

    try:
        out = subprocess.check_output(["aplay", "-l"], stderr=subprocess.DEVNULL,
                                      text=True)
    except FileNotFoundError:
        print("[DETECT] Speaker → aplay not found; defaulting to plughw:0,0")
        return "plughw:0,0"

    # aplay -l lines:  card 4: Headset [Logi USB Headset], device 0: USB Audio
    card_pattern = re.compile(
        r"^card\s+(\d+):\s+\S+\s+\[([^\]]+)\].*device\s+(\d+):", re.IGNORECASE
    )

    headset_cards = []   # priority 1
    logi_cards    = []   # priority 2
    usb_cards     = []   # priority 3

    for line in out.splitlines():
        m = card_pattern.match(line.strip())
        if not m:
            continue
        card_num, card_name, dev_num = m.group(1), m.group(2), m.group(3)
        n = card_name.lower()
        alsa = f"plughw:{card_num},{dev_num}"

        if "headset" in n:
            headset_cards.append((alsa, card_name))
        elif _is_logi(n) and not any(w in n for w in WEBCAM_WORDS):
            logi_cards.append((alsa, card_name))
        elif "usb" in n:
            usb_cards.append((alsa, card_name))

    for alsa, name in (headset_cards or logi_cards or usb_cards):
        print(f"[DETECT] Speaker → {alsa}  ({name})")
        return alsa

    print("[DETECT] Speaker → plughw:0,0  (hard fallback)")
    return "plughw:0,0"


def detect_camera() -> int:
    """
    Return the /dev/videoN *index* for the C270 webcam.

    Key insight from probe output:
      • /dev/video0 = 'cedrus' (hardware video codec — NOT a webcam, can't stream)
      • /dev/video1 = C270 HD WEBCAM  ← correct capture node
      • /dev/video2 = C270 HD WEBCAM  ← metadata/secondary node (also uvcvideo)

    Strategy:
      1. v4l2-ctl pass: find nodes whose Card type is Logi AND driver is uvcvideo.
         Among those, take the lowest-numbered node (video1 before video2).
      2. OpenCV open-test: skip index 0, try each remaining node for a real frame.
      3. Hard fallback: 1  (never 0 — cedrus always sits there on OrangePi).
    """
    video_nodes = sorted(glob.glob("/dev/video*"),
                         key=lambda p: int(re.search(r"\d+", p).group()))

    # ── Pass 1: v4l2-ctl — must be uvcvideo AND match Logi keywords ──────────
    uvc_logi_nodes = []
    for node in video_nodes:
        idx = int(re.search(r"\d+", node).group())
        if idx == 0:
            continue   # cedrus is always video0 on OrangePi — skip unconditionally
        try:
            info = subprocess.check_output(
                ["v4l2-ctl", "--device", node, "--info"],
                stderr=subprocess.DEVNULL, text=True, timeout=2
            )
            lines = info.splitlines()
            is_uvc  = any("uvcvideo" in l.lower() for l in lines)
            is_logi = any(_is_logi(l) for l in lines)
            if is_uvc and is_logi:
                uvc_logi_nodes.append(idx)
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    if uvc_logi_nodes:
        idx = min(uvc_logi_nodes)   # lowest node = primary capture node
        print(f"[DETECT] Camera → /dev/video{idx}  (uvcvideo + Logi name)")
        return idx

    # ── Pass 2: OpenCV open-test — skip 0, take first that yields a frame ────
    print("[DETECT] v4l2-ctl found nothing — trying OpenCV open test (skipping video0)...")
    for node in video_nodes:
        idx = int(re.search(r"\d+", node).group())
        if idx == 0:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"[DETECT] Camera → /dev/video{idx}  (first readable node)")
                return idx
        else:
            cap.release()

    print("[DETECT] Camera → /dev/video1  (hard fallback, skipping cedrus)")
    return 1


def auto_detect_all() -> None:
    """Run all three detectors and populate globals."""
    global AUDIO_INPUT_DEVICE, ALSA_OUTPUT_DEVICE, CAM_DEVICE
    print("\n" + "═" * 60)
    print("  AUTO-DETECTING LOGITECH DEVICES")
    print("═" * 60)
    AUDIO_INPUT_DEVICE = detect_mic()
    ALSA_OUTPUT_DEVICE = detect_speaker()
    CAM_DEVICE         = detect_camera()
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────

def _play_wav_bytes(wav_bytes: bytes) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(wav_bytes)
    proc.stdin.close()
    return proc


def speak(text: str, wav_bytes: bytes | None = None) -> None:
    """
    Non-blocking. Plays SAPI WAV from server if provided, else local espeak-ng.
    Interrupts any currently playing audio first.
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
                 "--stdout", text],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            aplay = subprocess.Popen(
                ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-q"],
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
    with _tts_lock:
        proc = _aplay_proc
    if proc:
        proc.wait()
    time.sleep(0.4)   # let reverb die before mic opens


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models() -> None:
    global whisper_model, vad
    print("[INIT] Loading webrtcvad...")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("[INIT] webrtcvad ready.")
    print(f"[INIT] Loading faster-whisper '{WHISPER_MODEL}' int8 CPU...")
    print("[INIT] First run downloads the model — may take 30-60s...")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu",
                                 compute_type=WHISPER_COMPUTE)
    print("[INIT] All models ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# VAD RECORDING
# ─────────────────────────────────────────────────────────────────────────────

def _to_pcm(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def record_with_vad() -> np.ndarray | None:
    """
    Capture audio with VAD gating.

    ALSA xrun fix for C270 on OrangePi:
    ─────────────────────────────────────
    The C270 USB mic on OrangePi triggers PortAudio xrun errors
    ("Broken pipe", ALSA error -32) when blocksize is too small or
    latency is unspecified.  Two fixes applied here:
      1. latency="high"  — tells PortAudio to use a large ALSA buffer
         (≈100 ms) which absorbs scheduling jitter on the ARM CPU.
      2. stream.read() wrapped in try/except — if a transient xrun still
         occurs we discard that frame and continue rather than crashing.
    """
    ring_buffer   = collections.deque(maxlen=RING_BUFFER_FRAMES)
    triggered     = False
    voiced_frames: list[np.ndarray] = []
    pre_roll:     list[np.ndarray]  = []
    max_frames    = int(MAX_RECORD_SEC * 1000 / FRAME_DURATION)
    frame_count   = 0
    xrun_count    = 0
    MAX_XRUNS     = 10   # abort and retry if stream is totally broken

    print("[MIC] Listening... speak now")

    try:
        stream_ctx = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=AUDIO_INPUT_DEVICE,
            latency="high",          # ← key fix: large ALSA buffer avoids xruns
        )
    except Exception as e:
        print(f"[MIC] Could not open input stream: {e}")
        return None

    with stream_ctx as stream:
        while frame_count < max_frames:
            try:
                chunk, overflowed = stream.read(FRAME_SIZE)
                if overflowed:
                    # Buffer overflowed — data may be stale; skip this frame
                    continue
            except sd.PortAudioError as e:
                xrun_count += 1
                print(f"[MIC] xrun #{xrun_count}: {e} — skipping frame")
                if xrun_count >= MAX_XRUNS:
                    print("[MIC] Too many xruns — aborting this listen attempt.")
                    break
                time.sleep(0.02)
                continue

            chunk     = chunk.flatten()
            pcm       = _to_pcm(chunk)
            if len(pcm) != FRAME_SIZE * 2:
                continue
            is_speech = vad.is_speech(pcm, SAMPLE_RATE)

            if not triggered:
                pre_roll.append(chunk.copy())
                if len(pre_roll) > RING_BUFFER_FRAMES:
                    pre_roll.pop(0)
                ring_buffer.append((chunk.copy(), is_speech))
                voiced_ratio = sum(1 for _, s in ring_buffer if s) / len(ring_buffer)
                if voiced_ratio > VOICED_THRESHOLD:
                    triggered = True
                    print("[MIC] Speech detected...")
                    if is_speaking():
                        stop_speaking()
                    voiced_frames.extend(pre_roll)
                    ring_buffer.clear()
            else:
                voiced_frames.append(chunk.copy())
                ring_buffer.append((chunk.copy(), is_speech))
                unvoiced_ratio = sum(1 for _, s in ring_buffer if not s) / len(ring_buffer)
                if unvoiced_ratio > UNVOICED_THRESHOLD:
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


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray | None) -> str:
    if audio is None:
        return ""
    print("[STT] Transcribing...")
    segments, _ = whisper_model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=1, best_of=1, temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100},
    )
    return " ".join(s.text for s in segments).strip().lower()


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


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA — Logitech webcam on OrangePi
# ─────────────────────────────────────────────────────────────────────────────

def capture_best_frame_local() -> bytes | None:
    """
    Open the auto-detected Logi webcam, warm it up, capture CAM_CAPTURE_SEC
    seconds of frames, pick the sharpest, encode as JPEG and return bytes.
    Tries multiple OpenCV backends so it works across kernel/driver versions.
    """
    backends = [
        (cv2.CAP_V4L2,  "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY,   "ANY"),
    ]

    cap = None
    for backend, bname in backends:
        print(f"[CAM] Trying /dev/video{CAM_DEVICE} with backend {bname}...")
        _c = cv2.VideoCapture(CAM_DEVICE, backend)
        if _c.isOpened():
            # Quick sanity: can we grab at least one frame?
            ok, _f = _c.read()
            if ok and _f is not None:
                cap = _c
                print(f"[CAM] Opened with {bname}")
                break
            _c.release()
        else:
            _c.release()

    if cap is None:
        # Last resort: try each /dev/videoN node directly
        print("[CAM] Index open failed — scanning all /dev/video* nodes...")
        for node in sorted(glob.glob("/dev/video*"),
                           key=lambda p: int(re.search(r"\d+", p).group())):
            _c = cv2.VideoCapture(node)
            if _c.isOpened():
                ok, _f = _c.read()
                if ok and _f is not None:
                    cap = _c
                    print(f"[CAM] Opened {node} directly")
                    break
            _c.release()

    if cap is None:
        print("[CAM] ERROR: Could not open any webcam device.")
        return None

    # Warmup — let exposure/AWB settle
    print(f"[CAM] Warming up ({CAM_WARMUP_FRAMES} frames)...")
    for _ in range(CAM_WARMUP_FRAMES):
        cap.grab()

    best_frame, best_score, count = None, -1.0, 0
    interval  = 1.0 / CAM_FPS
    end_time  = time.time() + CAM_CAPTURE_SEC

    print(f"[CAM] Capturing {CAM_CAPTURE_SEC}s @ {CAM_FPS} fps...")
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAM] Frame read failed mid-capture.")
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"[CAM]   frame {count + 1}: sharpness={score:.1f}")
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
        count += 1
        time.sleep(interval)

    cap.release()
    print(f"[CAM] Best sharpness: {best_score:.1f} from {count} frames")

    if best_score < MIN_SHARPNESS or best_frame is None:
        print("[CAM] All frames too blurry — aborting.")
        return None

    ok, buf = cv2.imencode(".jpg", best_frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), CAM_JPEG_QUALITY])
    if not ok:
        print("[CAM] JPEG encode failed.")
        return None

    jpeg_bytes = buf.tobytes()
    print(f"[CAM] JPEG ready: {len(jpeg_bytes) / 1024:.1f} KB")
    return jpeg_bytes


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────────────────────────────────────

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
        if "frame" in payload and isinstance(payload["frame"], (bytes, bytearray)):
            payload = dict(payload)
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
        sock.settimeout(120)
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
    tts = resp.get("tts", "")
    if not tts:
        return
    audio_b64 = resp.get("audio")
    wav_bytes  = base64.b64decode(audio_b64) if audio_b64 else None
    speak(tts, wav_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# SCAN TRIGGER WORDS
# ─────────────────────────────────────────────────────────────────────────────
SCAN_TRIGGERS = {
    "scan","check","go","now","yes","okay","ok","capture","take","snap",
    "shoot","photo","picture","frame","analyze","detect","read","process",
    "identify","start","run","next","continue","more","again","ready",
    "do it","let's go","lets go","scan it","do scan","take picture",
    "take photo","scan now","check now","yalla","hayde","sur",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global SERVER_HOST, SERVER_PORT
    global ESPEAK_SPEED, ESPEAK_VOICE, ESPEAK_PITCH, ESPEAK_AMPLITUDE
    global CAM_WARMUP_FRAMES, CAM_CAPTURE_SEC, CAM_FPS, CAM_JPEG_QUALITY
    global CAM_DEVICE, AUDIO_INPUT_DEVICE, ALSA_OUTPUT_DEVICE

    ap = argparse.ArgumentParser(description="OrangePi 2W Zero voice client")
    # Network
    ap.add_argument("--host",  default=SERVER_HOST)
    ap.add_argument("--port",  default=SERVER_PORT, type=int)
    # Whisper
    ap.add_argument("--model", default=WHISPER_MODEL, choices=["tiny", "base"])
    # Manual overrides (skip auto-detect)
    ap.add_argument("--mic",     default=None, type=int,
                    help="Force sounddevice mic index (skips auto-detect)")
    ap.add_argument("--alsa",    default=None,
                    help="Force ALSA output device string, e.g. plughw:3,0")
    ap.add_argument("--cam",     default=None, type=int,
                    help="Force camera /dev/videoN index (skips auto-detect)")
    # Voice calibration
    ap.add_argument("--speed",      type=int,   default=ESPEAK_SPEED)
    ap.add_argument("--pitch",      type=int,   default=ESPEAK_PITCH)
    ap.add_argument("--amplitude",  type=int,   default=ESPEAK_AMPLITUDE)
    ap.add_argument("--voice",      type=str,   default=ESPEAK_VOICE,
                    help="espeak-ng voice, e.g. en, en-gb, en+m3, en+f3")
    # Camera tuning
    ap.add_argument("--cam-warmup",  type=int,   default=CAM_WARMUP_FRAMES)
    ap.add_argument("--cam-sec",     type=float, default=CAM_CAPTURE_SEC)
    ap.add_argument("--cam-fps",     type=int,   default=CAM_FPS)
    ap.add_argument("--cam-quality", type=int,   default=CAM_JPEG_QUALITY)
    args = ap.parse_args()

    SERVER_HOST      = args.host
    SERVER_PORT      = args.port
    ESPEAK_SPEED     = args.speed
    ESPEAK_PITCH     = args.pitch
    ESPEAK_AMPLITUDE = args.amplitude
    ESPEAK_VOICE     = args.voice
    CAM_WARMUP_FRAMES = args.cam_warmup
    CAM_CAPTURE_SEC   = args.cam_sec
    CAM_FPS           = args.cam_fps
    CAM_JPEG_QUALITY  = args.cam_quality

    # ── Auto-detect all three devices ────────────────────────────────────────
    auto_detect_all()

    # Apply manual overrides if given
    if args.mic  is not None:
        AUDIO_INPUT_DEVICE = args.mic
        print(f"[OVERRIDE] Mic  → device index {AUDIO_INPUT_DEVICE}")
    if args.alsa is not None:
        ALSA_OUTPUT_DEVICE = args.alsa
        print(f"[OVERRIDE] Speaker → {ALSA_OUTPUT_DEVICE}")
    if args.cam  is not None:
        CAM_DEVICE = args.cam
        print(f"[OVERRIDE] Camera → /dev/video{CAM_DEVICE}")

    # ── Print final device summary ────────────────────────────────────────────
    print("─" * 60)
    print("  DEVICES IN USE")
    sd_name = sd.query_devices(AUDIO_INPUT_DEVICE)["name"] \
              if AUDIO_INPUT_DEVICE is not None else "system default"
    print(f"    Mic      : [{AUDIO_INPUT_DEVICE}] {sd_name}")
    print(f"    Speaker  : {ALSA_OUTPUT_DEVICE}")
    print(f"    Camera   : /dev/video{CAM_DEVICE}")
    print("  VOICE CALIBRATION")
    print(f"    Speed    : {ESPEAK_SPEED} WPM")
    print(f"    Pitch    : {ESPEAK_PITCH}")
    print(f"    Amplitude: {ESPEAK_AMPLITUDE}")
    print(f"    Voice    : {ESPEAK_VOICE}")
    print("─" * 60 + "\n")

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

        payload: dict = {"type": "voice", "text": text}

        if any(t in text for t in SCAN_TRIGGERS):
            print("[CAM] Scan trigger — capturing from Logi webcam...")
            jpeg_bytes = capture_best_frame_local()
            if jpeg_bytes is None:
                speak("Camera error. Could not capture a frame. Please try again.")
                wait_speaking()
                continue
            payload["frame"] = jpeg_bytes

        kb = len(payload.get("frame", b"")) / 1024
        print(f"[NET] Sending → {text!r}"
              + (f"  +{kb:.1f} KB frame" if "frame" in payload else ""))

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