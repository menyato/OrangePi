import socket, json, threading, subprocess, collections
import argparse, warnings, time, struct, base64, glob, re, os
import difflib, inspect, shutil, tempfile, wave, io

import cv2
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# Optional software noise cancellation (spectral gating). If installed, only the
# user's speech is cleaned before transcription.  Install:  pip install noisereduce
try:
    import noisereduce as _nr
    _NR_AVAILABLE = True
except Exception:
    _nr = None
    _NR_AVAILABLE = False
NR_ENABLED = True   # honored only when _NR_AVAILABLE; toggle with --no-denoise

# ── SERVER ────────────────────────────────────────────────────────────────────
SERVER_HOST = "10.254.249.159"
SERVER_PORT = 9000

# ── VAD / WHISPER ─────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
FRAME_DURATION     = 30
FRAME_SIZE         = int(SAMPLE_RATE * FRAME_DURATION / 1000)   # 480 samples
# VAD_AGGRESSIVENESS 3 (max) + a 500ms ring buffer meant "speech ended" fired
# after as little as ~300ms of quiet (60% of a 500ms window) -- well inside a
# normal mid-sentence pause ("load... a book"), so multi-word commands kept
# getting cut off after the first word or two and discarded outright as
# "Too short" once the truncated capture fell under MIN_SPEECH_SEC. Widened
# the window to 900ms (needs ~540ms of real quiet to end a phrase) and eased
# aggressiveness from 3 to 2, which also reduces false "unvoiced" frames on
# quieter phonemes/consonants that were contributing to premature cutoffs.
VAD_AGGRESSIVENESS = 2
VOICED_THRESHOLD   = 0.6
UNVOICED_THRESHOLD = 0.6
RING_BUFFER_MS     = 900
RING_BUFFER_FRAMES = int(RING_BUFFER_MS / FRAME_DURATION)
MAX_RECORD_SEC     = 15
MIN_SPEECH_SEC     = 0.4
NOISE_GATE_RMS     = 0.004
WHISPER_MODEL      = "base"
WHISPER_COMPUTE    = "int8"
WHISPER_LANGUAGE   = "en"
WHISPER_BEAM       = 1          # greedy = fastest. Raise (--beam 3/5) for accuracy.
WHISPER_CPU_THREADS = os.cpu_count() or 4   # use all 4 A53 cores on the Zero 2W

# Domain prompt: biases Whisper's decoder toward the words this app actually uses.
# This is the single most effective upstream fix for "Lebanese"→"Japanese" etc.
WHISPER_INITIAL_PROMPT = (
    "Currency scanner voice commands. Words used: scan, rescan, redo, discard, "
    "ready, again, done, quit, yes, no, confirm, change. "
    "Currencies: Lebanese pounds, lira, dollars, USD, LBP. "
    "Numbers: twenty, fifty, eighty, one hundred, two hundred, one thousand."
)

# ── PIPER (LOCAL NEURAL TTS — the natural voice the blind user hears) ─────────
# This runs ON the Pi. It is the preferred engine; espeak below is last-resort.
#   Install:  pip install piper-tts
#   Voice:    download e.g. en_US-amy-medium.onnx (+ .onnx.json beside it) from
#             https://huggingface.co/rhasspy/piper-voices  and pass --piper-model
PIPER_MODEL = ""                       # path to a .onnx voice; empty = disabled
PIPER_TMP   = "/tmp/opi_tts.wav"       # single reusable synth target
_piper_cli  = shutil.which("piper")    # CLI is the most version-stable interface
LAST_TTS_ENGINE = "none"               # for metrics: which engine actually spoke

# ── ESPEAK DEFAULTS (LAST-RESORT fallback only — PC voice is preferred) ────────
ESPEAK_SPEED     = 145
ESPEAK_VOICE     = "en-us+m3"
ESPEAK_PITCH     = 38
ESPEAK_AMPLITUDE = 170

# ── CAMERA ────────────────────────────────────────────────────────────────────
CAM_WARMUP_FRAMES = 10
CAM_CAPTURE_SEC   = 4.0
CAM_FPS           = 4
CAM_JPEG_QUALITY  = 80
MIN_SHARPNESS     = 10

# ── RUNTIME GLOBALS (filled by auto-detect) ───────────────────────────────────
AUDIO_INPUT_DEVICE: int | None  = None
_MIC_OVERRIDE:      int | None  = None   # set via set_mic_override() — wins over auto-detect


def set_mic_override(index: int) -> None:
    """Force a specific sounddevice input index instead of auto-detecting.
    Must be called before auto_detect_all() (hub.py does this right after
    parsing --mic, before any feature loads models)."""
    global _MIC_OVERRIDE
    _MIC_OVERRIDE = index
ALSA_OUTPUT_DEVICE: str | None  = None
CAM_DEVICE:         int         = 0

whisper_model: WhisperModel | None = None
vad:           webrtcvad.Vad | None = None
_supports_hotwords = False   # detected at model-load time

_tts_lock   = threading.Lock()
_aplay_proc: subprocess.Popen | None = None

metrics: "Metrics | None" = None

# ── AUDIO CUES (earcons) — non-verbal feedback for a blind user ───────────────
CUES_ENABLED = True
_CUE_LISTEN: bytes | None = None   # soft beep: "I'm listening, speak now"
_CUE_ACK:    bytes | None = None   # rising two-tone: "got it, working on it"
_CUE_SHOT:   bytes | None = None   # bright blip: "frame captured"
_CUE_THINK:  bytes | None = None   # low blip: "transcribing now"


# ─────────────────────────────────────────────────────────────────────────────
# METRICS / AUDITING
# ─────────────────────────────────────────────────────────────────────────────
class Metrics:
    """
    Lightweight auditing. Appends one JSON object per utterance to a JSONL file
    and keeps running aggregates so we can print a session summary at exit.
    Use this to compare 'before' and 'after' tuning runs quantitatively.
    """
    def __init__(self, path: str):
        self.path = path
        self.records: list[dict] = []
        self.session_start = time.time()
        try:
            # touch the file so we fail fast if the path is bad
            with open(self.path, "a"):
                pass
            print(f"[METRICS] Logging to {self.path}")
        except OSError as e:
            print(f"[METRICS] Could not open log file ({e}); in-memory only.")
            self.path = None

    def log(self, rec: dict) -> None:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        self.records.append(rec)
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    def summary(self) -> None:
        utts = [r for r in self.records if r.get("event") == "utterance"]
        if not utts:
            print("[METRICS] No utterances recorded.")
            return

        def avg(key):
            vals = [r[key] for r in utts if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        corrected = sum(1 for r in utts if r.get("corrections"))
        n = len(utts)
        dur = time.time() - self.session_start

        print("\n" + "═" * 60)
        print("  SESSION METRICS SUMMARY")
        print("═" * 60)
        print(f"  Utterances        : {n}  over {dur/60:.1f} min")
        print(f"  Avg record time   : {avg('t_record'):.2f} s")
        print(f"  Avg transcribe    : {avg('t_transcribe'):.2f} s")
        print(f"  Avg network RTT   : {avg('t_network'):.2f} s")
        print(f"  Avg total latency : {avg('t_total'):.2f} s  (speech end → reply)")
        print(f"  Avg Whisper logp  : {avg('avg_logprob'):.3f}  (closer to 0 = more confident)")
        print(f"  Avg no-speech prob: {avg('no_speech_prob'):.3f}  (lower = better)")
        print(f"  Utterances corrected by fuzzy layer: {corrected}/{n}")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT HELPERS  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

LOGI_KEYWORDS = ["logi", "c270", "c920", "c310", "c525", "c922",
                 "webcam", "usb audio", "usb mic"]

def _is_logi(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in LOGI_KEYWORDS)


def detect_mic() -> int | None:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and _is_logi(dev["name"]):
            print(f"[DETECT] Mic  → [{i}] {dev['name']}")
            return i
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and "usb" in dev["name"].lower():
            print(f"[DETECT] Mic  → [{i}] {dev['name']}  (USB fallback)")
            return i
    print("[DETECT] Mic  → system default (no Logi/USB mic found)")
    return None


def detect_speaker() -> str:
    WEBCAM_WORDS = {"webcam", "c270", "c920", "c310", "c525", "c922"}
    try:
        out = subprocess.check_output(["aplay", "-l"], stderr=subprocess.DEVNULL,
                                      text=True)
    except FileNotFoundError:
        print("[DETECT] Speaker → aplay not found; defaulting to plughw:0,0")
        return "plughw:0,0"

    card_pattern = re.compile(
        r"^card\s+(\d+):\s+\S+\s+\[([^\]]+)\].*device\s+(\d+):", re.IGNORECASE
    )
    headset_cards, logi_cards, usb_cards = [], [], []

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
    video_nodes = sorted(glob.glob("/dev/video*"),
                         key=lambda p: int(re.search(r"\d+", p).group()))
    uvc_logi_nodes = []
    for node in video_nodes:
        idx = int(re.search(r"\d+", node).group())
        if idx == 0:
            continue
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
        idx = min(uvc_logi_nodes)
        print(f"[DETECT] Camera → /dev/video{idx}  (uvcvideo + Logi name)")
        return idx

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
    global AUDIO_INPUT_DEVICE, ALSA_OUTPUT_DEVICE, CAM_DEVICE
    print("\n" + "═" * 60)
    print("  AUTO-DETECTING LOGITECH DEVICES")
    print("═" * 60)
    if _MIC_OVERRIDE is not None:
        AUDIO_INPUT_DEVICE = _MIC_OVERRIDE
        print(f"[DETECT] Mic  → [{AUDIO_INPUT_DEVICE}] (forced via --mic)")
    else:
        AUDIO_INPUT_DEVICE = detect_mic()
    ALSA_OUTPUT_DEVICE = detect_speaker()
    CAM_DEVICE         = detect_camera()
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CUES (earcons)
# ─────────────────────────────────────────────────────────────────────────────

def _tone_wav(segments: list[tuple[float, int]], sr: int = 16000,
              vol: float = 0.22) -> bytes:
    """Build a small mono WAV from (freq_hz, duration_ms) segments, with short
    fades to avoid clicks. Returns complete WAV bytes ready for aplay."""
    pcm_all = []
    for freq, dur_ms in segments:
        n = int(sr * dur_ms / 1000)
        t = np.arange(n) / sr
        wav = np.sin(2 * np.pi * freq * t)
        fade = max(1, int(sr * 0.008))
        env = np.ones(n)
        env[:fade]  = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        pcm_all.append(wav * env * vol)
    data = np.concatenate(pcm_all) if pcm_all else np.zeros(1)
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def init_cues() -> None:
    """Pre-render the earcons once at startup."""
    global _CUE_LISTEN, _CUE_ACK, _CUE_SHOT, _CUE_THINK
    _CUE_LISTEN = _tone_wav([(880, 110)])                 # one soft beep
    _CUE_ACK    = _tone_wav([(660, 90), (990, 110)])      # rising "got it"
    _CUE_SHOT   = _tone_wav([(1320, 70), (1320, 70)])     # bright double blip
    _CUE_THINK  = _tone_wav([(440, 90)])                  # low "thinking" blip


def play_cue(wav_bytes: bytes | None) -> None:
    """Play an earcon synchronously (short). Safe to call before opening the
    mic — it finishes before recording starts, so it won't leak into VAD."""
    if not CUES_ENABLED or not wav_bytes or ALSA_OUTPUT_DEVICE is None:
        return
    try:
        p = subprocess.Popen(
            ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-q"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        p.stdin.write(wav_bytes)
        p.stdin.close()
        p.wait()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND CORRECTION  (deterministic STT safety net)
# ─────────────────────────────────────────────────────────────────────────────
# The vocabulary is tiny and known, so we can snap near-miss tokens back to it.

COMMAND_VOCAB = {
    # scan / redo / discard
    "scan", "rescan", "redo", "capture", "snap", "shoot", "photo", "picture",
    "frame", "again", "over", "discard", "cancel", "remove", "delete",
    # confirm / reject
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "correct", "confirm",
    "confirmed", "right", "sure", "okay", "ok", "wrong", "incorrect", "change",
    # other commands
    "check", "go", "now", "ready", "start", "run", "next", "continue", "more",
    "analyze", "detect", "read", "process", "identify",
    # quit
    "quit", "exit", "stop", "done", "end", "finish", "bye", "goodbye",
    # OCR / book reader (load, add, close a saved session; skip a step)
    "load", "add", "book", "page", "close", "skip",
    # currency
    "dollar", "dollars", "usd", "buck", "bucks",
    "lebanese", "lbp", "pound", "pounds", "lira", "lirah",
    # number words
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million",
}

# High-value, hard-coded fixes for the specific mistakes you reported, plus
# common phonetic neighbours Whisper produces. These run first (exact).
KNOWN_CONFUSIONS = {
    "japanese": "lebanese", "javanese": "lebanese", "lebenese": "lebanese",
    "lebanon": "lebanese",  "libanese": "lebanese", "leb": "lebanese",
    "scun": "scan", "scam": "scan", "scanned": "scan", "skan": "scan",
    "scane": "scan", "scen": "scan", "scon": "scan", "skin": "scan",
    "scant": "scan", "scance": "scan",
    "leera": "lira", "leira": "lira", "lera": "lira",
    "dollor": "dollar", "dolar": "dollar", "dollers": "dollars",
    "yas": "yes", "yus": "yes", "redu": "redo", "ridu": "redo",
    "riddo": "redo", "diskard": "discard", "kancel": "cancel",
    "conform": "confirm", "no.": "no",
    "lord": "load", "loud": "load", "lode": "load", "lowd": "load",
    "ad": "add", "att": "add",
    "clothes": "close", "clos": "close",
}

FUZZY_CUTOFF = 0.74   # difflib ratio threshold for snapping unknown tokens

# Never fuzzy-snap these: ordinary function words that sit right at the
# cutoff against vocabulary entries — "the"→"three" scores 0.75 and was
# live-corrupting nearly every sentence ("load the page"→"load three page",
# a session named "that's the one" saved as "that's three one").
_NEVER_SNAP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is",
    "it", "its", "for", "my", "me", "this", "that", "was", "are", "be",
}


def correct_transcript(text: str) -> tuple[str, list]:
    """
    Return (corrected_text, list_of_(orig, fixed) corrections).
    1. exact KNOWN_CONFUSIONS replacement, then
    2. fuzzy snap of remaining unknown tokens to COMMAND_VOCAB.
    Digits, already-known words, function words, and words too short to
    fuzzy-match safely are left untouched (short tokens only get fixed via
    the exact KNOWN_CONFUSIONS table, e.g. "ad"→"add").
    """
    corrections: list[tuple[str, str]] = []
    out_tokens: list[str] = []
    for tok in text.split():
        low = tok.lower()
        if low in KNOWN_CONFUSIONS:
            fixed = KNOWN_CONFUSIONS[low]
            corrections.append((tok, fixed))
            out_tokens.append(fixed)
            continue
        if low in COMMAND_VOCAB or low.isdigit():
            out_tokens.append(low)
            continue
        if low in _NEVER_SNAP or len(low) < 4:
            out_tokens.append(low)
            continue
        match = difflib.get_close_matches(low, COMMAND_VOCAB, n=1, cutoff=FUZZY_CUTOFF)
        if match:
            corrections.append((tok, match[0]))
            out_tokens.append(match[0])
        else:
            out_tokens.append(low)
    return " ".join(out_tokens), corrections


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_tts(text: str) -> str:
    """Clean text before espeak so the LAST-RESORT fallback sounds less awful."""
    import re as _re
    abbrevs = {
        r'\bI\.?P\.?\b':         'I P',
        r'\bI\.?P\.? address\b': 'I P address',
        r'\bOK\b':               'okay',
        r'\be\.g\.\b':           'for example',
        r'\bi\.e\.\b':           'that is',
        r'\betc\.\b':            'et cetera',
        r'\bvs\.\b':             'versus',
        r'\bDr\.\b':             'Doctor',
        r'\bMr\.\b':             'Mister',
        r'\bMrs\.\b':            'Missus',
        r'\bst\.\b':             'street',
        r'\bapprox\.\b':         'approximately',
    }
    for pat, rep in abbrevs.items():
        text = _re.sub(pat, rep, text, flags=_re.IGNORECASE)
    text = _re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(1).title(), text)
    text = _re.sub(r'\([^)]*\)', '', text)
    text = _re.sub(r'(\w+)\s*/\s*(\w+)', r'\1 or \2', text)
    for word in ('However', 'Therefore', 'Meanwhile', 'Otherwise',
                 'Furthermore', 'Additionally', 'Nevertheless'):
        text = _re.sub(rf'\b{word}\b', f'{word},', text)
    text = _re.sub(r'  +', ' ', text).strip()
    return text


def _play_wav_bytes(wav_bytes: bytes) -> subprocess.Popen:
    """
    RESTORED FUNCTION (was missing in the original — this is the bug that made
    the natural PC voice silently fail and forced robotic espeak fallback).
    Pipes a complete WAV (with header) into aplay on the detected speaker.

    aplay can die before/during the write (e.g. a Ctrl-C's SIGINT reaches it
    too, since it's in the same foreground process group; or the ALSA device
    briefly rejects the open) — a bare .write() then raises BrokenPipeError,
    which previously crashed the whole hub instead of just this one TTS line.
    """
    proc = subprocess.Popen(
        ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.stdin.write(wav_bytes)
        proc.stdin.close()
    except (BrokenPipeError, OSError) as e:
        print(f"[TTS] aplay pipe closed early ({e}) — skipping this line.")
    return proc


def _piper_ready() -> bool:
    return bool(PIPER_MODEL and _piper_cli and os.path.exists(PIPER_MODEL))


def _speak_piper(text: str) -> subprocess.Popen | None:
    """
    Synthesize `text` to a WAV with local Piper, then play it via aplay (async).
    Returns the aplay Popen (so it can be interrupted for barge-in), or None on
    failure so the caller can fall back to the PC WAV / espeak.
    """
    try:
        subprocess.run(
            [_piper_cli, "--model", PIPER_MODEL, "--output_file", PIPER_TMP],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        if not os.path.exists(PIPER_TMP) or os.path.getsize(PIPER_TMP) < 64:
            return None
        return subprocess.Popen(
            ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-q", PIPER_TMP],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[TTS] Piper local error: {e}")
        return None


def speak(text: str, wav_bytes: bytes | None = None) -> None:
    """
    Non-blocking. Plays the PC's natural-voice WAV if provided, else falls back
    to local espeak-ng. Interrupts any currently-playing audio first.
    """
    global _aplay_proc, LAST_TTS_ENGINE
    clean = _preprocess_tts(text)
    print(f"\n[TTS] >> {clean}\n")
    with _tts_lock:
        if _aplay_proc and _aplay_proc.poll() is None:
            _aplay_proc.terminate()
            _aplay_proc.wait()

        # 1) Local Piper — natural neural voice, runs on the Pi, no PC needed
        if _piper_ready():
            proc = _speak_piper(clean)
            if proc is not None:
                _aplay_proc = proc
                LAST_TTS_ENGINE = "piper_local"
                return

        # 2) PC-rendered WAV (Piper/SAPI on the server) if Piper isn't on the Pi
        if wav_bytes:
            _aplay_proc = _play_wav_bytes(wav_bytes)
            LAST_TTS_ENGINE = "pc_voice"
            return

        # 3) espeak-ng — robotic last resort
        espeak = subprocess.Popen(
            ["espeak-ng",
             "-s", str(ESPEAK_SPEED),
             "-v", ESPEAK_VOICE,
             "-p", str(ESPEAK_PITCH),
             "-a", str(ESPEAK_AMPLITUDE),
             "-g", "6",
             "--stdout", clean],
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
        LAST_TTS_ENGINE = "espeak_local"


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
    global whisper_model, vad, _supports_hotwords
    print("[INIT] Loading webrtcvad...")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("[INIT] webrtcvad ready.")
    print(f"[INIT] Loading faster-whisper '{WHISPER_MODEL}' int8, {WHISPER_CPU_THREADS} threads...")
    print("[INIT] First run downloads the model — 'base' is ~150 MB, may take a minute...")
    whisper_model = WhisperModel(
        WHISPER_MODEL, device="cpu",
        compute_type=WHISPER_COMPUTE,
        cpu_threads=WHISPER_CPU_THREADS,   # ← use every core; biggest speed win
        num_workers=1,
    )
    _supports_hotwords = "hotwords" in inspect.signature(
        WhisperModel.transcribe).parameters
    print(f"[INIT] hotwords supported: {_supports_hotwords}")
    print("[INIT] All models ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# VAD RECORDING  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def _to_pcm(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# When patient=True, a phrase only ends after this much CONTINUOUS quiet, and a
# single noisy/voiced blip resets the timer — so the speaker can pause to think
# mid-sentence ("is he a ... male or female?") without being cut off. Used by
# Environmental Awareness, where the user asks longer, more deliberate questions.
PATIENT_END_MS = 1500


def record_with_vad(stop_ev=None, patient: bool = False) -> np.ndarray | None:
    ring_buffer   = collections.deque(maxlen=RING_BUFFER_FRAMES)
    triggered     = False
    voiced_frames: list[np.ndarray] = []
    pre_roll:     list[np.ndarray]  = []
    xrun_count    = 0
    MAX_XRUNS     = 10
    consec_silence = 0   # consecutive unvoiced frames (patient mode only)

    print("[MIC] Listening... speak now")
    play_cue(_CUE_LISTEN)   # tells the blind user: speak now
    try:
        stream_ctx = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SIZE, device=AUDIO_INPUT_DEVICE, latency="high",
        )
    except Exception as e:
        print(f"[MIC] Could not open input stream: {e}")
        return None

    # Wall-clock deadline, not a frame counter: below the noise gate, the old
    # code `continue`'d before ever reaching its frame_count += 1, so
    # MAX_RECORD_SEC was only actually enforced once enough frames had
    # cleared the noise gate -- in a quiet room this let one listen attempt
    # run far longer than 15s (33s+ observed live), tying up the mic (and
    # blocking anything waiting on it, e.g. _voice_off()'s join) well past
    # what any caller expected.
    deadline = time.time() + MAX_RECORD_SEC
    with stream_ctx as stream:
        while time.time() < deadline:
            # A caller asked us to stop and no speech has started yet — bail
            # out immediately instead of sitting in the (up to 15 s) VAD wait.
            # This is what actually makes _voice_off() responsive: without it,
            # "turn voice off, then speak a prompt" had to wait out the full
            # record window before the mic was released. Once the user IS
            # mid-utterance (triggered), we let it finish so the speech isn't
            # lost — the shared loop queues it and the caller can use it.
            if stop_ev is not None and stop_ev.is_set() and not triggered:
                print("[MIC] Stop requested while idle — releasing mic.")
                return None
            try:
                chunk, overflowed = stream.read(FRAME_SIZE)
                if overflowed:
                    continue
            except sd.PortAudioError as e:
                xrun_count += 1
                print(f"[MIC] xrun #{xrun_count}: {e} — skipping frame")
                if xrun_count >= MAX_XRUNS:
                    print("[MIC] Too many xruns — aborting this listen attempt.")
                    break
                time.sleep(0.02)
                continue

            chunk = chunk.flatten()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < NOISE_GATE_RMS and not triggered:
                continue

            pcm = _to_pcm(chunk)
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
                if patient:
                    # End only after PATIENT_END_MS of CONTINUOUS quiet; any
                    # voiced frame resets the timer, so mid-sentence pauses
                    # don't cut the user off.
                    consec_silence = 0 if is_speech else consec_silence + 1
                    if consec_silence * FRAME_DURATION >= PATIENT_END_MS:
                        print("[MIC] Speech ended.")
                        break
                else:
                    ring_buffer.append((chunk.copy(), is_speech))
                    unvoiced_ratio = sum(1 for _, s in ring_buffer if not s) / len(ring_buffer)
                    if unvoiced_ratio > UNVOICED_THRESHOLD:
                        print("[MIC] Speech ended.")
                        break

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

def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset then peak-normalize. (No pre-emphasis/high-pass on
    purpose — Whisper was trained on natural audio, and filtering it can hurt.)"""
    audio = audio - float(np.mean(audio))
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio * (0.95 / peak)
    return audio.astype("float32")


def _denoise(audio: np.ndarray) -> np.ndarray:
    """Spectral-gate noise cancellation so only the speaker's voice reaches
    Whisper. No-op if noisereduce isn't installed or --no-denoise was passed."""
    if not (NR_ENABLED and _NR_AVAILABLE):
        return audio
    try:
        cleaned = _nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=False)
        return cleaned.astype("float32")
    except Exception as e:
        print(f"[NR] denoise skipped: {e}")
        return audio


_DEFAULT_HOTWORDS = ("scan rescan redo discard yes no confirm change "
                     "ready Lebanese pounds lira dollars done quit")


def transcribe(audio: np.ndarray | None, initial_prompt: str | None = None,
               hotwords: str | None = None) -> tuple[str, dict]:
    """Returns (raw_text, stats). stats carries Whisper confidence numbers.

    initial_prompt/hotwords default to the money/currency-scanner vocabulary
    (WHISPER_INITIAL_PROMPT / _DEFAULT_HOTWORDS) so existing callers (money
    recognition) are unaffected. Other features that share this same
    transcribe() — OCR, Environmental Awareness, Home Automation — need
    their OWN vocabulary passed in explicitly: the currency-only prompt
    actively biases Whisper's decoder AWAY FROM words like "load"/"add"/book
    names that never appear in it, which is the actual reason those commands
    were unreliable in features other than money recognition."""
    stats = {"avg_logprob": None, "no_speech_prob": None, "n_segments": 0}
    if audio is None:
        return "", stats

    print("[STT] Transcribing...")
    audio = _normalize_audio(audio)
    audio = _denoise(audio)            # noise cancellation: keep only the voice

    kwargs = dict(
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM,
        best_of=WHISPER_BEAM,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or WHISPER_INITIAL_PROMPT,   # ← domain biasing
        vad_filter=False,    # we already VAD-gate on the Pi — skip Silero (faster)
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    if _supports_hotwords:
        kwargs["hotwords"] = hotwords or _DEFAULT_HOTWORDS

    segments, _ = whisper_model.transcribe(audio, **kwargs)

    seg_list = list(segments)
    stats["n_segments"] = len(seg_list)
    if seg_list:
        stats["avg_logprob"]   = sum(s.avg_logprob   for s in seg_list) / len(seg_list)
        stats["no_speech_prob"] = sum(s.no_speech_prob for s in seg_list) / len(seg_list)
    text = " ".join(s.text for s in seg_list).strip().lower()
    return text, stats


def listen(retries: int = 5, initial_prompt: str | None = None,
           hotwords: str | None = None, stop_ev=None,
           quiet: bool = False, correct: bool = True,
           patient: bool = False) -> tuple[str, dict]:
    """
    Returns (corrected_text, audit). Skips the expensive Whisper call entirely
    when no speech was captured, beeps to mark each stage, and tells the user
    out loud when it heard nothing or couldn't understand.

    initial_prompt/hotwords are passed straight through to transcribe() — see
    its docstring. Pass your feature's own vocabulary here if it's not money
    recognition's currency-scanning commands.

    stop_ev: optional Event that ends the retry loop early. Without it, one
    listen() call could run its full 5 retries (~2 minutes of mic ownership
    and spoken "didn't catch that" prompts) after the caller had already
    moved on — the background voice loop would then talk over and steal
    speech from whatever prompt the feature was asking next.

    quiet: suppress the spoken "didn't catch that" / "did not understand"
    prompts. Foreground one-shot listens ask the user a question, so the
    prompts are helpful there; the background command loop just waits for
    the user to maybe say something, and nagging them every time room noise
    trips the VAD (e.g. right after a book finishes reading, when they said
    nothing at all) is noise.

    correct: apply correct_transcript()'s command-vocabulary snapping.
    Pass False when the answer is free text (a book title, a session name) —
    snapping arbitrary words toward command vocabulary corrupts names.
    """
    told = False   # avoid repeating the spoken "didn't catch that" every retry
    for attempt in range(1, retries + 1):
        if stop_ev is not None and stop_ev.is_set():
            print("[STT] Stop requested — abandoning remaining retries.")
            break
        t0 = time.time()
        audio = record_with_vad(stop_ev=stop_ev, patient=patient)   # plays the LISTEN cue inside
        t_record = time.time() - t0

        # ── Nothing to transcribe → DON'T run Whisper; tell the user ────────
        if audio is None:
            print(f"[STT] No speech captured ({attempt}/{retries}).")
            if not told and not quiet \
                    and not (stop_ev is not None and stop_ev.is_set()):
                speak("I didn't catch that. Please speak after the beep.")
                wait_speaking()
                told = True
            continue

        play_cue(_CUE_THINK)               # "transcribing now" beep
        t1 = time.time()
        raw_text, stats = transcribe(audio, initial_prompt=initial_prompt, hotwords=hotwords)
        t_transcribe = time.time() - t1

        if correct:
            corrected, corrections = correct_transcript(raw_text)
        else:
            corrected, corrections = raw_text, []

        audit = {
            "t_record": round(t_record, 3),
            "t_transcribe": round(t_transcribe, 3),
            "raw_text": raw_text,
            "corrected_text": corrected,
            "corrections": corrections,
            "avg_logprob": (round(stats["avg_logprob"], 4)
                            if stats["avg_logprob"] is not None else None),
            "no_speech_prob": (round(stats["no_speech_prob"], 4)
                               if stats["no_speech_prob"] is not None else None),
            "attempt": attempt,
        }

        # Require at least one letter or digit: Whisper hallucinates strings
        # of punctuation (". . . .") on noise-only audio, often with *high*
        # confidence — one of those got accepted and read back to a blind
        # user as a "book name" of four hundred spoken dots.
        if corrected.strip() and re.search(r"[a-z0-9]", corrected, re.IGNORECASE):
            play_cue(_CUE_ACK)   # tells the blind user: heard you, working on it
            _short = " ".join(corrected.split()[:4])
            print("\n" + "─" * 50)
            if corrections:
                fixes = ", ".join(f"{a}→{b}" for a, b in corrections[:4])
                print(f"  RAW       →  {' '.join(raw_text.split()[:4])}")
                print(f"  CORRECTED →  {_short}   [{fixes}]")
            else:
                print(f"  YOU SAID  →  {_short}")
            conf = audit["avg_logprob"]
            if conf is not None:
                print(f"  confidence (avg_logprob) = {conf}")
            print(f"  timing: record {audit['t_record']}s  transcribe {audit['t_transcribe']}s")
            print("─" * 50 + "\n")
            return corrected, audit

        # Heard sound but no words came out
        print(f"[STT] Nothing understood ({attempt}/{retries}).")
        if not told and not quiet \
                and not (stop_ev is not None and stop_ev.is_set()):
            speak("Sorry, I did not understand. Please try again.")
            wait_speaking()
            told = True

    return "", {"t_record": 0, "t_transcribe": 0, "raw_text": "",
                "corrected_text": "", "corrections": [], "attempt": retries}


def voice_listen_loop(voice_q, stop_ev, abort_ev, initial_prompt: str | None = None,
                      hotwords: str | None = None, correct: bool = True,
                      patient: bool = False) -> None:
    """Shared blocking voice worker for features (OCR, Environmental Awareness, …).

    Loops calling listen() and puts the corrected transcript into voice_q.
    Returns when stop_ev or abort_ev is set — stop_ev is passed down into
    listen() so an in-flight call abandons its retries (and its idle VAD
    wait) promptly instead of owning the mic for minutes.

    Runs listen() in quiet mode: this loop passively waits for a command,
    so the spoken "didn't catch that" retry prompts (meant for direct
    questions) would nag the user every time room noise trips the VAD.

    Pass initial_prompt/hotwords with your feature's own command vocabulary —
    these default (inside transcribe()) to money recognition's currency
    vocabulary, which actively works against recognizing unrelated commands
    like "load"/"add" in other features.
    """
    while not stop_ev.is_set() and not abort_ev.is_set():
        try:
            text, _ = listen(initial_prompt=initial_prompt, hotwords=hotwords,
                             stop_ev=stop_ev, quiet=True, correct=correct,
                             patient=patient)
        except Exception as e:
            print(f"[VOICE] listen error: {e}")
            import time as _t; _t.sleep(0.5)
            continue
        if text:
            # listen() has already finished recording (and released the mic)
            # by the time we get here, so a stop_ev set while we were
            # mid-utterance no longer protects any device from contention --
            # only the queue put is left to do. Previously this branch was
            # `if text and not stop_ev.is_set()`, which silently discarded a
            # fully-transcribed command with no trace anywhere in the logs
            # whenever the caller asked us to stop while we were still
            # listening -- e.g. right after "load the page" is recognized,
            # the LOAD handler calls _voice_off() to ask a direct follow-up
            # question, and this loop's already in-flight listen() call ate
            # the user's next utterance instead of queuing it.
            if stop_ev.is_set():
                print(f"[VOICE] stop requested mid-listen, queuing anyway: {text!r}")
            voice_q.put(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def capture_best_frame_local() -> bytes | None:
    backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_GSTREAMER, "GStreamer"),
                (cv2.CAP_ANY, "ANY")]
    cap = None
    for backend, bname in backends:
        print(f"[CAM] Trying /dev/video{CAM_DEVICE} with backend {bname}...")
        _c = cv2.VideoCapture(CAM_DEVICE, backend)
        if _c.isOpened():
            ok, _f = _c.read()
            if ok and _f is not None:
                cap = _c
                print(f"[CAM] Opened with {bname}")
                break
            _c.release()
        else:
            _c.release()

    if cap is None:
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
# NETWORK  (unchanged logic)
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
    if wav_bytes is None:
        print("[TTS] No PC audio in response — using local espeak fallback.")
    speak(tts, wav_bytes)


# ── SCAN TRIGGER WORDS ────────────────────────────────────────────────────────
# Capture a frame for scans AND redo/rescan. NOTE: confirmation words (yes/ok/no)
# are deliberately NOT here, so confirming an amount never fires the camera.
SCAN_TRIGGERS = {
    "scan", "rescan", "redo", "capture", "snap", "shoot", "photo", "picture",
    "frame", "take picture", "take photo", "scan now", "do scan", "scan it",
    "scan again", "redo scan", "do over", "yalla",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global SERVER_HOST, SERVER_PORT, WHISPER_MODEL, WHISPER_BEAM
    global ESPEAK_SPEED, ESPEAK_VOICE, ESPEAK_PITCH, ESPEAK_AMPLITUDE
    global CAM_WARMUP_FRAMES, CAM_CAPTURE_SEC, CAM_FPS, CAM_JPEG_QUALITY
    global CAM_DEVICE, AUDIO_INPUT_DEVICE, ALSA_OUTPUT_DEVICE, metrics
    global PIPER_MODEL, CUES_ENABLED, NR_ENABLED

    ap = argparse.ArgumentParser(description="OrangePi 2W Zero voice client (improved)")
    ap.add_argument("--host", default=SERVER_HOST)
    ap.add_argument("--port", default=SERVER_PORT, type=int)
    ap.add_argument("--model", default=WHISPER_MODEL, choices=["tiny", "base", "small"],
                    help="Whisper size. 'base' is the practical ceiling on a "
                         "Zero 2W; 'small' is more accurate but ~2-3x slower.")
    ap.add_argument("--beam", type=int, default=WHISPER_BEAM,
                    help="Whisper beam size (lower = faster, higher = accurate).")
    ap.add_argument("--mic", default=None, type=int)
    ap.add_argument("--alsa", default=None)
    ap.add_argument("--cam", default=None, type=int)
    ap.add_argument("--piper-model", default=PIPER_MODEL,
                    help="Path to a local Piper .onnx voice (natural TTS on the "
                         "Pi). Strongly recommended for the blind user's voice. "
                         "e.g. ~/piper/en_US-amy-medium.onnx")
    ap.add_argument("--speed", type=int, default=ESPEAK_SPEED)
    ap.add_argument("--pitch", type=int, default=ESPEAK_PITCH)
    ap.add_argument("--amplitude", type=int, default=ESPEAK_AMPLITUDE)
    ap.add_argument("--voice", type=str, default=ESPEAK_VOICE)
    ap.add_argument("--cam-warmup", type=int, default=CAM_WARMUP_FRAMES)
    ap.add_argument("--cam-sec", type=float, default=CAM_CAPTURE_SEC)
    ap.add_argument("--cam-fps", type=int, default=CAM_FPS)
    ap.add_argument("--cam-quality", type=int, default=CAM_JPEG_QUALITY)
    ap.add_argument("--metrics", default="client_metrics.jsonl",
                    help="JSONL audit log path (set '' to disable file logging).")
    ap.add_argument("--quiet-cues", action="store_true",
                    help="Disable the beep/earcon audio cues (keeps spoken cues).")
    ap.add_argument("--no-denoise", action="store_true",
                    help="Disable software noise cancellation before transcription.")
    args = ap.parse_args()

    SERVER_HOST, SERVER_PORT = args.host, args.port
    WHISPER_MODEL, WHISPER_BEAM = args.model, args.beam
    ESPEAK_SPEED, ESPEAK_PITCH = args.speed, args.pitch
    ESPEAK_AMPLITUDE, ESPEAK_VOICE = args.amplitude, args.voice
    CAM_WARMUP_FRAMES, CAM_CAPTURE_SEC = args.cam_warmup, args.cam_sec
    CAM_FPS, CAM_JPEG_QUALITY = args.cam_fps, args.cam_quality
    PIPER_MODEL = args.piper_model
    CUES_ENABLED = not args.quiet_cues
    NR_ENABLED = not args.no_denoise
    init_cues()

    if args.metrics:
        metrics = Metrics(args.metrics)

    auto_detect_all()

    if args.mic is not None:
        AUDIO_INPUT_DEVICE = args.mic
        print(f"[OVERRIDE] Mic  → device index {AUDIO_INPUT_DEVICE}")
    if args.alsa is not None:
        ALSA_OUTPUT_DEVICE = args.alsa
        print(f"[OVERRIDE] Speaker → {ALSA_OUTPUT_DEVICE}")
    if args.cam is not None:
        CAM_DEVICE = args.cam
        print(f"[OVERRIDE] Camera → /dev/video{CAM_DEVICE}")

    print("─" * 60)
    print("  DEVICES IN USE")
    sd_name = sd.query_devices(AUDIO_INPUT_DEVICE)["name"] \
              if AUDIO_INPUT_DEVICE is not None else "system default"
    print(f"    Mic      : [{AUDIO_INPUT_DEVICE}] {sd_name}")
    print(f"    Speaker  : {ALSA_OUTPUT_DEVICE}")
    print(f"    Camera   : /dev/video{CAM_DEVICE}")
    print(f"    Whisper  : {WHISPER_MODEL} (beam {WHISPER_BEAM}, {WHISPER_CPU_THREADS} threads)")
    if NR_ENABLED and _NR_AVAILABLE:
        print(f"    Denoise  : ON (noisereduce)")
    elif NR_ENABLED and not _NR_AVAILABLE:
        print(f"    Denoise  : unavailable — run 'pip install noisereduce' to enable")
    else:
        print(f"    Denoise  : OFF (--no-denoise)")
    if _piper_ready():
        print(f"    Voice    : Piper LOCAL → {os.path.basename(PIPER_MODEL)}  (natural)")
    elif PIPER_MODEL:
        print(f"    Voice    : Piper requested but unavailable "
              f"(cli={'yes' if _piper_cli else 'no'}, "
              f"model_exists={os.path.exists(PIPER_MODEL)}) → PC voice / espeak")
    else:
        print(f"    Voice    : PC-rendered WAV if sent, else espeak (robotic). "
              f"Pass --piper-model for a natural voice.")
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

    try:
        while True:
            text, audit = listen()
            if not text:
                continue

            payload: dict = {"type": "voice", "text": text}

            t_cam = 0.0
            if any(t in text for t in SCAN_TRIGGERS):
                print("[CAM] Scan trigger — capturing from Logi webcam...")
                speak("Hold the bill steady. Capturing now.")
                wait_speaking()
                tc = time.time()
                jpeg_bytes = capture_best_frame_local()
                t_cam = time.time() - tc
                if jpeg_bytes is None:
                    speak("Camera error. Could not capture a frame. Please try again.")
                    wait_speaking()
                    audit.update({"event": "utterance", "t_cam": round(t_cam, 3),
                                  "outcome": "camera_error"})
                    if metrics: metrics.log(audit)
                    continue
                play_cue(_CUE_SHOT)                  # captured
                speak("Captured. Analyzing now.")
                wait_speaking()
                payload["frame"] = jpeg_bytes

            kb = len(payload.get("frame", b"")) / 1024
            print(f"[NET] Sending → {text!r}"
                  + (f"  +{kb:.1f} KB frame" if "frame" in payload else ""))

            tn = time.time()
            resp = _send(sock, payload)
            t_network = time.time() - tn

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

            # ── audit this full round-trip ─────────────────────────────────
            audit.update({
                "event": "utterance",
                "t_cam": round(t_cam, 3),
                "t_network": round(t_network, 3),
                "t_total": round(audit["t_record"] + audit["t_transcribe"]
                                 + t_cam + t_network, 3),
                "had_frame": "frame" in payload,
                "tts_engine": LAST_TTS_ENGINE,
                "reply": resp.get("tts", "")[:160],
            })
            if metrics:
                metrics.log(audit)

            if resp.get("quit"):
                break
    finally:
        try:
            sock.close()
        except Exception:
            pass
        speak("Goodbye.")
        wait_speaking()
        if metrics:
            metrics.summary()


if __name__ == "__main__":
    main()