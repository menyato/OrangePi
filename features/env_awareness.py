"""
features/env_awareness.py — Environmental Awareness feature for the smart glove.

Records a short video clip, extracts the most informative key frames using
scene-change detection plus sharpness ranking, then sends them to Gemini 2.5
Flash with a system prompt tuned for blind users.  Supports multi-turn
conversation: the user can ask follow-up questions without re-scanning, and all
exchanges are saved as a named session so history survives a restart.

Key design points
─────────────────
• Runs entirely on the OrangePi — no laptop server required.
• Frame extraction pipeline:
    1. Record VIDEO_DURATION_S seconds at VIDEO_FPS (Pi-friendly, low CPU).
    2. Scene-change detection (HSV histogram Bhattacharyya distance) to find
       visually distinct moments — important when the user pans the camera.
    3. If fewer than 3 distinct scenes found, pad with uniform-spaced samples.
    4. Keep the MAX_KEYFRAMES sharpest (Laplacian variance) in temporal order
       so blurry frames never reach the API.
• Gemini history is injected into the system prompt each turn so the AI
  remembers prior descriptions even without native multi-turn state.
• Voice worker uses arecord subprocess (immediately killable) + faster-whisper,
  same approach as OCR reader to avoid ALSA "Device unavailable" errors.
  Raw transcript goes into the queue; the outer loop classifies commands vs.
  free-form questions.
• TTS uses local espeak-ng (fb.speak) — no server round-trip needed.

Install on OrangePi
───────────────────
    pip install google-genai
    # Then set your key in one of:
    echo "GEMINI_API_KEY=your_key" >> ~/.env
    export GEMINI_API_KEY=your_key

Gesture controls (configured via Programmable Gestures)
───────────────────────────────────────────────────────
  ENV_SCAN   → capture new view (same effect as saying "describe")
  ENV_CLOSE  → save and exit   (same effect as saying "close")
  NEXT       → also triggers a new scan (consistent with OCR reader)
"""

import contextlib
import datetime
import json
import os
import queue
import re
import threading
import time

from features.base import Feature, FeatureContext
import metrics

# ── optional imports ─────────────────────────────────────────────────────────

try:
    import cv2
    import numpy as np
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import orangepi_client as mc
    _MC_OK = True
except ImportError:
    _MC_OK = False

try:
    from google import genai
    from google.genai import types as _gtypes
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

# ── tunables ─────────────────────────────────────────────────────────────────

SESSIONS_DIR      = os.path.expanduser("~/env_sessions")
CAMERA_INDICES    = [0, 1, 2]
VIDEO_DURATION_S  = 5       # seconds to record per scan (longer = wider pan coverage)
VIDEO_FPS         = 10      # Pi-friendly frame rate (50 frames per scan)
MAX_KEYFRAMES     = 5       # maximum images sent to Gemini per turn
SCENE_THRESHOLD   = 0.30    # Bhattacharyya histogram distance for a scene cut
JPEG_QUALITY      = 75      # lower = smaller payload, faster API round-trip
MAX_IMG_WIDTH     = 1024    # resize frames wider than this before encoding
GEMINI_MODEL      = "gemini-2.5-flash"
MAX_HISTORY_TURNS = 16      # keep last N conversation turns in session context
IDLE_REMIND_S     = 60
_KEY_ENV          = "GEMINI_API_KEY"

# Domain vocabulary for every listen in this feature. Without it,
# transcribe() falls back to money recognition's currency vocabulary, which
# biases Whisper AWAY from the words this feature actually uses — and
# free-form questions to Gemini got "corrected" toward currency command
# words by correct_transcript(). Chat input must arrive verbatim.
ENV_INITIAL_PROMPT = (
    "Environmental awareness assistant for a blind user. Commands: "
    "describe, scan, look, capture, load, close, stop, exit, skip. "
    "The user also asks natural questions about their surroundings: "
    "objects, people, colors, distances, obstacles, doors, stairs, signs."
)
ENV_HOTWORDS = "describe scan look capture load close stop exit skip"

# ── Hardcoded API key (optional) ──────────────────────────────────────────────
# If you don't want to rely on GEMINI_API_KEY / ~/.env / ~/keys.env — e.g.
# because hub.py runs as a systemd service and doesn't inherit your shell's
# environment — paste your key here instead. Get one at
# https://aistudio.google.com/apikey
#
# SECURITY NOTE: a key pasted here lives in plain text in this source file.
# Do not commit/push this file to a shared or public repo with a real key in
# it; if this repo is ever made public, rotate the key immediately.
# Leave empty ("") to keep using the environment-variable / file lookup below.
GEMINI_API_KEY_HARDCODED = ""

# ── gesture → synthetic voice token ──────────────────────────────────────────

_GMAP_ENV = {
    "ENV_SCAN":  "scan",    # user-assigned capture gesture
    "ENV_CLOSE": "close",   # user-assigned close gesture
    "NEXT":      "scan",    # "next" flick = capture, consistent with OCR
}

# ── command word sets ─────────────────────────────────────────────────────────

_SCAN_W  = {"describe", "scan", "rescan", "look", "capture", "photo", "photos",
            "picture", "pictures", "take", "see", "record", "view", "camera",
            "surroundings", "around", "environment", "again", "more"}
_LOAD_W  = {"load", "recall", "open", "restore"}
_CLOSE_W = {"close", "stop", "quit", "exit", "done", "finish", "leave", "bye"}
_SKIP_W  = {"skip", "none", "no", "cancel", "default"}

# Words that, when they appear as the FIRST word, mark the utterance as a scan
# request even if it is a long sentence ("describe everything around me",
# "scan the room again"). Excludes ambiguous words like "see"/"look" that
# routinely start questions ("look, is there a door?" / "see anything red?").
_SCAN_LEAD = {"describe", "scan", "rescan", "capture", "record"}

# Multi-word phrases that trigger a (re)scan no matter where they appear or how
# long the sentence is — this is what lets the user say "let me take more
# photos" or "can you see this more" mid-conversation and get a fresh capture.
_SCAN_PHRASE = re.compile(
    r"\b(re[- ]?scan|scan (again|this|more|the)|"
    r"take (a |an |another |more |some |the )?(photo|picture|pic|shot|look)s?|"
    r"more (photo|picture|pic|shot)s?|another (photo|picture|pic|scan|shot)s?|"
    r"new (scan|photo|picture|pic)s?|look again|see (this |it )?(more|again)|"
    r"describe again|describe (my|the) surrounding|show me)\b"
)

# Explicit "I am finished" phrases — these end the session (and prompt for a
# name) even inside a longer sentence, so "okay, let's save the session" or
# "leave the session now" work without being sent to Gemini as a question.
_CLOSE_PHRASE = re.compile(
    r"\b(save (the |this )?session|leave (the |this )?session|end (the |this )?session|"
    r"close (the |this )?session|save and (close|exit|quit)|i'?m done|that'?s all)\b"
)

# ── Gemini system prompt ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an environmental awareness assistant for a blind person wearing a smart glove.
When images are provided you describe the surroundings in as much detail as possible.
When a follow-up question arrives without new images, you answer from earlier in this conversation.

This is NOT a hazard or safety alert. Do not warn about danger, obstacles, or navigation
risk unless the user explicitly asks. Your job is simply to paint a rich, vivid picture of
the scene so the user can understand where they are and what is around them.

DESCRIPTION RULES:
1. Give a complete picture: the type of place (room, street, kitchen, office...), then the \
objects, furniture, people, and features that fill it.
2. Describe spatial layout naturally: what is to the left, right, ahead, near and far, \
above and below.
3. Include rich detail: colours, materials, textures, sizes, shapes, text on signs or labels, \
and the general mood or atmosphere of the space.
4. Name people's approximate number, posture, and what they appear to be doing — never guess \
identities.
5. Note lighting, time-of-day cues, and whether the space feels indoor or outdoor, open or enclosed.
6. Be descriptive and natural, but not padded — every sentence should add real information, \
no poetry or meta-commentary.
7. Multiple frames are sequential moments from one short clip (the user may have panned the \
camera) — synthesise them into one coherent description of the whole scene, not a frame-by-frame list.
8. Never comment on image quality, blur, or the capture process.
9. Responses: a detailed paragraph (4-7 sentences) for a new scan; 1-2 sentences for a follow-up question.\
"""


# ── API key loading ───────────────────────────────────────────────────────────

def _load_api_key() -> str:
    if GEMINI_API_KEY_HARDCODED.strip():
        return GEMINI_API_KEY_HARDCODED.strip()
    key = os.environ.get(_KEY_ENV, "").strip()
    if key:
        return key
    for path in (os.path.expanduser("~/.env"),
                 os.path.expanduser("~/keys.env")):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(_KEY_ENV + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


# ── session persistence ───────────────────────────────────────────────────────

def _save_session(name: str, history: list, last_images: "list | None" = None) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", name.strip())[:60] or "session"
    path = os.path.join(SESSIONS_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "history": history,
                   "last_images": last_images or []}, f,
                  indent=2, ensure_ascii=False)
    return path


def _list_sessions() -> list:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    result = []
    for fname in sorted(os.listdir(SESSIONS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                result.append(json.load(f))
        except Exception:
            pass
    return result


def _next_auto_name() -> str:
    """Next free 'session N' label, so a session the user never names gets a
    short, speakable name ("session 3") instead of a timestamp."""
    nums = []
    for s in _list_sessions():
        m = re.fullmatch(r"session[ _]?(\d+)", s.get("name", "").strip().lower())
        if m:
            nums.append(int(m.group(1)))
    return f"session {max(nums) + 1 if nums else 1}"


def _delete_session(name: str) -> None:
    """Remove a saved session's JSON (used to drop the placeholder 'session N'
    once the user gives the same session a real name on close)."""
    safe = re.sub(r"[^\w\-]", "_", (name or "").strip())[:60] or "session"
    path = os.path.join(SESSIONS_DIR, f"{safe}.json")
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _find_session(raw: str) -> "dict | None":
    needle = raw.lower().strip()
    sessions = _list_sessions()
    for s in sessions:
        if s.get("name", "").lower() == needle:
            return s
    for s in sessions:
        if needle in s.get("name", "").lower():
            return s
    return None


def _startup_summary() -> str:
    """Brief saved-session *count* only — see ocr_reader.py's
    _startup_summary for why: reading every session's name and exchange
    count before any instruction turns into a long wall of speech once
    several are saved. Full names are spoken on demand by LOAD instead."""
    sessions = _list_sessions()
    if not sessions:
        return ""
    c = len(sessions)
    return f"You have {c} saved session{'s' if c != 1 else ''}. "


# ── image helpers ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _bgr_hist(frame) -> object:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _extract_keyframes(frames: list) -> list:
    """
    Return up to MAX_KEYFRAMES representative BGR frames in temporal order.

    Pipeline:
      1. Scene-change detection via HSV histogram Bhattacharyya distance
         to find visually distinct moments (panning, new objects in view).
      2. If fewer than 3 distinct scenes, pad with uniformly-spaced samples
         to guarantee basic coverage.
      3. If still more than MAX_KEYFRAMES candidates, keep the sharpest
         (highest Laplacian variance) so blurry frames never reach the API.
    """
    if not frames:
        return []

    # Step 1 — scene-change detection
    scene_idx: set = {0, len(frames) - 1}
    prev_hist = _bgr_hist(frames[0])
    for i in range(1, len(frames)):
        h = _bgr_hist(frames[i])
        if cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA) > SCENE_THRESHOLD:
            scene_idx.add(i)
            prev_hist = h
    scene_idx = sorted(scene_idx)

    # Step 2 — pad with uniform samples if sparse
    if len(scene_idx) < 3:
        n_pad   = min(4, len(frames))
        uniform = (np.linspace(0, len(frames) - 1, n_pad, dtype=int)
                   .tolist())
        scene_idx = sorted(set(scene_idx) | set(uniform))

    if len(scene_idx) <= MAX_KEYFRAMES:
        return [frames[i] for i in scene_idx]

    # Step 3 — keep MAX_KEYFRAMES sharpest in temporal order
    scored = sorted(
        ((cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                        cv2.CV_64F).var(), i)
         for i in scene_idx),
        reverse=True,
    )
    keep = sorted(i for _, i in scored[:MAX_KEYFRAMES])
    return [frames[i] for i in keep]


def _frame_to_jpeg(frame) -> bytes:
    """Resize to MAX_IMG_WIDTH if needed and return JPEG bytes."""
    h, w = frame.shape[:2]
    if w > MAX_IMG_WIDTH:
        frame = cv2.resize(
            frame, (MAX_IMG_WIDTH, int(h * MAX_IMG_WIDTH / w)),
            interpolation=cv2.INTER_AREA,
        )
    _, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    return buf.tobytes()


# ── key-frame persistence (local + server) ────────────────────────────────────

def _img_dir(name: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", (name or "").strip())[:60] or "session"
    return os.path.join(SESSIONS_DIR, safe + "_images")


def _save_keyframes_local(name: str, scan_idx: int, jpegs: list) -> list:
    """Write this scan's JPEG key frames to the session's image folder and
    return their absolute paths (recorded in the session so a later LOAD can
    hand them back to Gemini as visual context)."""
    d = _img_dir(name)
    os.makedirs(d, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for k, jpg in enumerate(jpegs):
        p = os.path.join(d, f"scan_{scan_idx}_{ts}_f{k}.jpg")
        try:
            with open(p, "wb") as f:
                f.write(jpg)
            paths.append(p)
        except OSError as e:
            print(f"[ENV] Keyframe save error: {e}")
    return paths


def _load_keyframes(paths: list) -> list:
    """Read saved JPEG key frames back into BGR frames (for a loaded session)."""
    frames = []
    if not _CV2_OK:
        return frames
    for p in paths or []:
        try:
            if os.path.isfile(p):
                img = cv2.imread(p)
                if img is not None and img.size > 0:
                    frames.append(img)
        except Exception:
            pass
    return frames


def _send_scan_to_server(link, session: str, user: str, reply: str,
                         jpegs: list, keyframes: int) -> None:
    """Archive the scan (key frames + Gemini reply) on the laptop server so it
    can be reviewed later. Best-effort: never blocks or breaks the feature."""
    if link is None:
        return
    import base64
    try:
        frames_b64 = [base64.b64encode(j).decode("ascii") for j in jpegs]
        link.send("env", {
            "action":     "scan",
            "session":    session,
            "user":       user,
            "reply":      reply,
            "keyframes":  keyframes,
            "frames_b64": frames_b64,
        })
    except Exception as e:
        print(f"[ENV] Server archive failed (non-fatal): {e}")


# ── camera capture ────────────────────────────────────────────────────────────

def _record_frames() -> list:
    """
    Open the first working camera, record VIDEO_DURATION_S seconds at VIDEO_FPS,
    and return the raw BGR frames.  Returns empty list on any failure.
    """
    if not _CV2_OK:
        return []
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = None
    for idx in CAMERA_INDICES:
        with _quiet_stderr():
            c = cv2.VideoCapture(idx, backend)
        if c.isOpened():
            ret, fr = c.read()
            if ret and fr is not None and fr.size > 0 and fr.max() > 0:
                cap = c
                break
            c.release()
    if cap is None:
        return []

    try:
        cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        frames: list = []
        target   = VIDEO_DURATION_S * VIDEO_FPS
        deadline = time.time() + VIDEO_DURATION_S + 2.0
        while len(frames) < target and time.time() < deadline:
            ret, fr = cap.read()
            if ret and fr is not None and fr.size > 0:
                frames.append(fr)
        print(f"[ENV] Recorded {len(frames)} frames from camera {idx}")
        return frames
    finally:
        cap.release()


# ── Gemini API ────────────────────────────────────────────────────────────────

def _build_system(history: list) -> str:
    """Append the last MAX_HISTORY_TURNS conversation turns to the system prompt."""
    if not history:
        return _SYSTEM_PROMPT
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    lines  = [_SYSTEM_PROMPT, "", "-- Conversation History --"]
    for turn in recent:
        lines.append(f"{turn['role'].upper()}: {turn['content']}")
    return "\n".join(lines)


def _is_transient_net_err(e: Exception) -> bool:
    """True for DNS/connection hiccups worth retrying (vs. a real API error)."""
    s = f"{type(e).__name__}: {e}".lower()
    return any(k in s for k in (
        "connecterror", "connecttimeout", "connect timeout", "readtimeout",
        "read timeout", "name or service not known", "temporary failure in name",
        "getaddrinfo", "errno -2", "errno -3", "connection reset",
        "connection aborted", "remoteprotocol", "timed out", "network is unreachable",
    ))


# Words that make up the Whisper priming prompt. On ~1s of silence/noise
# faster-whisper sometimes regurgitates that prompt verbatim ("the user also
# asks natural questions about the blind user") — a phantom "question" that
# then gets sent to Gemini in a loop. Any transcript that is almost entirely
# made of these words is treated as a hallucinated echo and ignored.
_PROMPT_WORDS = set(re.findall(r"[a-z]+", ENV_INITIAL_PROMPT.lower()))


def _is_prompt_echo(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 5:
        return False   # never suppress short real commands
    hits = sum(1 for w in words if w in _PROMPT_WORDS)
    return hits / len(words) >= 0.8


def _ask_gemini(client, history: list, user_text: str, frames: list) -> str:
    """
    Call Gemini with optional key frames and the latest user text.
    `frames` is a list of BGR np.ndarray; pass [] for follow-up questions
    that need no new images.
    Returns the model's reply as a plain string.
    """
    system   = _build_system(history)
    contents: list = []

    for i, frame in enumerate(frames):
        jpeg = _frame_to_jpeg(frame)
        contents.append(f"[Frame {i + 1} of {len(frames)}]")
        contents.append(
            _gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
        )

    contents.append(f"User: {user_text}")

    # thinking_budget=0 is essential: gemini-2.5-flash spends "thinking" tokens
    # out of the SAME max_output_tokens budget before it writes a single visible
    # word. With thinking on and a 512 cap, a scan came back cut off mid-sentence
    # ("...living room, with") — the model had burned the budget thinking. Turn
    # thinking off and give a generous cap so a full detailed paragraph fits.
    cfg_kwargs = dict(
        system_instruction=system,
        max_output_tokens=1200,
        temperature=0.4,
    )
    try:
        cfg_kwargs["thinking_config"] = _gtypes.ThinkingConfig(thinking_budget=0)
    except (AttributeError, TypeError):
        pass  # older google-genai without ThinkingConfig — cap alone still helps
    config = _gtypes.GenerateContentConfig(**cfg_kwargs)

    # Retry transient network/DNS failures. The Pi runs on a phone hotspot whose
    # DNS drops out intermittently ("Name or service not known" — errno -2), so
    # one query fails while the next succeeds. A couple of quick retries lets a
    # blip self-heal instead of surfacing as "Gemini did not respond".
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=config,
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            if attempt < 3 and _is_transient_net_err(e):
                print(f"[ENV] Gemini network blip (attempt {attempt}/3): {e} — retrying")
                time.sleep(1.5)
                continue
            raise
    raise last_err


# ── mc initialisation ─────────────────────────────────────────────────────────

_mc_ready = False
_mc_lock  = threading.Lock()

# ── Gemini self-test (once per process, not once per activation) ──────────────
_gemini_tested = False


def _ensure_mc(fb, link=None) -> bool:
    """Load Whisper model once per process; reuse if already loaded."""
    global _mc_ready
    if not _MC_OK:
        return False
    with _mc_lock:
        if _mc_ready:
            return True
        try:
            from features.money_recognition import MoneyRecognition  # noqa
            if MoneyRecognition._models_ready:
                _mc_ready = True
                return True
        except (ImportError, AttributeError):
            pass
        try:
            t0 = time.time()
            mc.init_cues()
            mc.auto_detect_all()
            fb.speak("Loading voice model. Please wait.")
            mc.load_models()
            _mc_ready = True
            if link is not None:
                metrics.report_load(link, "env", (time.time() - t0) * 1000)
            return True
        except Exception as e:
            print(f"[ENV] mc init error: {e}")
            return False


# (voice worker is mc.voice_listen_loop — defined in orangepi_client.py)


# ── classify ──────────────────────────────────────────────────────────────────

def _classify_env(text: str) -> "str | None":
    """
    Return a control command string ("SCAN", "LOAD", "CLOSE") or None to treat
    the text as a free-form question for Gemini.

    The earlier version only accepted commands from utterances of 3 words or
    fewer, so mis-hears like "describe, mayors, or surroundings" fell through
    to Gemini and the user's clear intent to scan was lost. Detection now works
    three ways, most specific first:

      • explicit multi-word phrases  — "save the session", "take more photos",
        "scan the room again" — match anywhere, at any length;
      • a leading command word       — "describe …", "scan …" starts a scan
        even in a long sentence (but not question-openers like "see"/"look");
      • short command utterances      — a single/short phrase whose words hit a
        command set, e.g. "load", "stop", "describe".

    Free-form questions ("what colour is the wall", "is anyone near me") match
    none of these and are sent to Gemini.
    """
    low = text.lower().strip()
    words_list = re.findall(r"[a-z']+", low)
    if not words_list:
        return None
    words = set(w.strip("'") for w in words_list)

    # 1 — explicit phrases (work at any length / position)
    if _CLOSE_PHRASE.search(low):
        return "CLOSE"
    if _SCAN_PHRASE.search(low):
        return "SCAN"

    # 2 — leading command word carries a whole sentence
    first = words_list[0].strip("'")
    if first in _SCAN_LEAD:
        return "SCAN"
    if first in _LOAD_W:
        return "LOAD"

    # 3 — short utterances: a bare command word or two
    if len(words_list) <= 3:
        if words & _CLOSE_W:
            return "CLOSE"
        if words & _LOAD_W:
            return "LOAD"
        if words & _SCAN_W:
            return "SCAN"
    return None


# ── feature class ─────────────────────────────────────────────────────────────

class EnvAwareness(Feature):
    """
    Environmental Awareness — blind-user guide to their surroundings.

    Commands
    ────────
    "describe / scan / rescan / take a photo …"
                             → record 5-second clip, extract key frames, ask
                               Gemini to describe the surroundings in detail.
                               Works as a full sentence too ("scan the room
                               again", "let me take more photos").
    "load"                   → reload a saved session — restores the whole
                               conversation AND the last scan's images so
                               Gemini keeps visual context.
    "save/leave session, close, stop, done"
                             → prompt for a session name, save, and exit.
    Any other speech         → follow-up question to Gemini, answered from the
                               conversation history plus the last scan's frames;
                               no new video captured.

    Every scan's key frames and Gemini's reply are also archived on the laptop
    server (handlers/env_scans/<session>/) for later review.
    """
    name  = "env"
    title = "Environmental awareness"

    def run(self, ctx: FeatureContext) -> None:
        fb      = ctx.feedback
        mc_ok   = _ensure_mc(fb, ctx.link)
        gq      = ctx.gesture_queue
        voice_q = queue.Queue()
        stop_ev = threading.Event()

        history: list        = []
        session_name: "str | None" = None
        # Silent working name so autosave protects the session against a crash
        # before the user has named it. The user is only asked for a real name
        # when they close/save the session (see the CLOSE branch); until then
        # nothing is announced about naming.
        auto_name = _next_auto_name()
        scan_count = 0
        # Most recent scan's key frames (BGR) kept in memory so follow-up
        # questions — and a freshly LOADed session — still have the images to
        # reason over, instead of only the text history.
        last_keyframes: list = []
        last_image_paths: list = []   # disk paths of last_keyframes (for save/load)
        idle_t = time.time() + IDLE_REMIND_S

        # ── dependency checks ─────────────────────────────────────────────────
        if not _GENAI_OK:
            fb.speak(
                "Environmental awareness requires the google-genai package. "
                "Run: pip install google-genai"
            )
            return

        api_key = _load_api_key()
        if not api_key:
            fb.speak(
                "No Gemini API key found. "
                "Either paste it into GEMINI_API_KEY_HARDCODED at the top of "
                "env_awareness dot py, or add GEMINI_API_KEY equals your key "
                "to the file dot env in your home folder."
            )
            print("[ENV] No API key: checked GEMINI_API_KEY_HARDCODED, "
                  "$GEMINI_API_KEY, ~/.env, ~/keys.env — all empty.")
            return

        t0 = time.time()
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fb.speak(f"Gemini init error. {e}")
            return
        metrics.report_load(ctx.link, "env", (time.time() - t0) * 1000, component="genai_client")

        # One-time connectivity self-test (once per process, not every
        # activation — a real Gemini call has latency and uses quota) so a
        # bad key / no internet / wrong model name is caught early.
        #
        # NON-FATAL and retried: the phone-hotspot the Pi runs on has brief DNS
        # blips ("Name or service not known"), and a single one at exactly this
        # moment used to kill the whole feature and force a full relaunch even
        # though the network recovered a second later. Now we retry a few times,
        # and if it still fails we warn but STILL open the feature — the user
        # can just say "describe" again once the connection is back, no restart.
        global _gemini_tested
        if not _gemini_tested:
            last_err = None
            for attempt in range(1, 4):
                try:
                    _probe = client.models.generate_content(
                        model=GEMINI_MODEL, contents="Reply with the single word OK.",
                    )
                    print(f"[ENV] Gemini self-test OK: {(_probe.text or '').strip()!r}")
                    _gemini_tested = True
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"[ENV] Gemini self-test attempt {attempt}/3 failed: {e}")
                    if attempt < 3 and not ctx.abort.is_set():
                        time.sleep(2.0)
            if last_err is not None:
                import traceback
                traceback.print_exception(type(last_err), last_err,
                                          last_err.__traceback__)
                fb.speak(
                    "I can't reach Gemini right now — the internet connection "
                    "seems down. I'll stay open; check the connection and say "
                    "describe to try again, or close to exit."
                )
                fb.wait(timeout=8)
                # fall through: do NOT return. _gemini_tested stays False so the
                # self-test runs again next time the feature is opened.

        # Notify server that env feature is now active
        try:
            ctx.link.send("lidar", {
                "action": "feature_state",
                "feature_name": "env",
                "state": "started",
            })
        except Exception:
            pass

        # ── voice thread helpers ──────────────────────────────────────────────

        _voice_thread = [None]

        def _voice_on() -> None:
            nonlocal stop_ev
            stop_ev = threading.Event()
            ev = stop_ev
            _drain()
            def _start():
                time.sleep(1.0)
                if not ev.is_set() and not ctx.abort.is_set():
                    if _MC_OK:
                        # drain again right before listening starts so text a
                        # dying previous worker queued late can't fire a
                        # stale command
                        _drain()
                        # correct=False: most input here is free-form chat for
                        # Gemini — vocabulary snapping corrupts questions
                        mc.voice_listen_loop(voice_q, ev, ctx.abort,
                                             initial_prompt=ENV_INITIAL_PROMPT,
                                             hotwords=ENV_HOTWORDS,
                                             correct=False)
            t = threading.Thread(target=_start, daemon=True, name="ENVVoice")
            _voice_thread[0] = t
            t.start()

        def _voice_off() -> None:
            # Join, don't just signal: the recorder and the camera share the
            # same USB webcam. Recording video while an in-flight listen still
            # owns the device produced "Device unavailable" storms (and
            # barge-in silenced announcements mid-sentence). stop_ev now ends
            # an idle listen in well under a second; a mid-utterance one runs
            # at most record + transcribe time.
            stop_ev.set()
            t = _voice_thread[0]
            if t is not None and t.is_alive():
                t.join(timeout=getattr(mc, "MAX_RECORD_SEC", 15) + 30.0)
                if t.is_alive():
                    print("[ENV] WARNING: voice worker still busy after join "
                          "timeout — mic may be contended.")

        def _drain() -> None:
            while True:
                try:
                    voice_q.get_nowait()
                except queue.Empty:
                    break

        # ── combined gesture + voice poll ─────────────────────────────────────

        def _get_input(ges_timeout: float = 0.4) -> "str | None":
            """Return the next voice text or gesture-mapped token, or None."""
            try:
                return voice_q.get_nowait()
            except queue.Empty:
                pass
            if gq is None:
                time.sleep(ges_timeout)
                return None
            deadline = time.time() + ges_timeout
            while time.time() < deadline and not ctx.abort.is_set():
                try:
                    g = gq.get(timeout=min(0.05, deadline - time.time()))
                    token = _GMAP_ENV.get(g)
                    if token:
                        return token
                except queue.Empty:
                    pass
            return None

        # ── session helpers ───────────────────────────────────────────────────

        def _autosave() -> None:
            if not history:
                return
            name = session_name or auto_name
            try:
                _save_session(name, history, last_images=last_image_paths)
                print(f"[ENV] Auto-saved '{name}' ({len(history) // 2} exchanges)")
            except Exception as e:
                print(f"[ENV] Auto-save error: {e}")

        def _append(role: str, content: str) -> None:
            history.append({"role": role, "content": content})
            # Trim to keep context manageable
            while len(history) > MAX_HISTORY_TURNS * 2:
                history.pop(0)

        # ── opening announcement ──────────────────────────────────────────────
        # _voice_on() must come AFTER the announcement finishes (or times
        # out) -- it used to fire first, and its own internal 1s head start
        # is far shorter than this announcement takes to speak, so voice
        # recognition started trying to open the mic while the announcement
        # was still playing.
        fb.speak(
            "Environmental awareness. "
            + _startup_summary()
            + "Say describe to scan your surroundings, "
            "load to open a saved session, "
            "or close to exit. "
            "You can also ask any question about what I see."
        )
        fb.wait(timeout=15)
        _voice_on()

        # ── main loop ─────────────────────────────────────────────────────────

        try:
            while not ctx.abort.is_set():
                raw = _get_input(ges_timeout=0.4)

                if ctx.abort.is_set():
                    break

                # ── idle reminder ─────────────────────────────────────────────
                if raw is None:
                    if time.time() >= idle_t:
                        _voice_off()
                        if session_name:
                            fb.speak(
                                f"Say describe to scan again, "
                                f"ask a question, or close to save and exit."
                            )
                        else:
                            fb.speak(
                                "Say describe to scan your surroundings, "
                                "load to open a session, or close to exit."
                            )
                        fb.wait(timeout=10)
                        _voice_on()
                        idle_t = time.time() + IDLE_REMIND_S
                    continue

                # Drop Whisper prompt-echo hallucinations (silence transcribed
                # as the priming prompt) before they reach the classifier and
                # get fired at Gemini as a phantom question.
                if _is_prompt_echo(raw):
                    print(f"[ENV] Ignored prompt-echo hallucination: {raw!r}")
                    continue

                cmd = _classify_env(raw)
                print(f"[ENV] voice: {raw!r} → {cmd or '(question for Gemini)'}")
                idle_t = time.time() + IDLE_REMIND_S

                # ── CLOSE (save & exit) ───────────────────────────────────────
                # Naming happens HERE, not after the first scan: the user asked
                # to keep scanning / asking freely and only be prompted for a
                # name when they actually leave. If they never name it, the
                # silent auto_name autosave already protected the session.
                if cmd == "CLOSE":
                    _voice_off()
                    _drain()
                    if history and session_name is None and mc_ok:
                        fb.speak("Say a name for this session, or say skip.")
                        fb.wait(timeout=4)
                        try:
                            vtxt, _ = mc.listen(
                                initial_prompt=ENV_INITIAL_PROMPT,
                                hotwords=ENV_HOTWORDS, correct=False)
                            print(f"[ENV] voice(direct): {vtxt!r} — "
                                  "answering 'name this session'")
                            if vtxt:
                                nwords = set(re.findall(r"[a-z']+", vtxt.lower()))
                                if not (nwords & _SKIP_W):
                                    session_name = vtxt.strip()
                        except Exception:
                            pass
                    break

                # ── LOAD ──────────────────────────────────────────────────────
                if cmd == "LOAD":
                    _voice_off()
                    saved = _list_sessions()
                    if not saved:
                        fb.speak("No saved sessions yet. Say describe to start one.")
                        fb.wait(timeout=4)
                        _voice_on()
                        continue
                    names = [s.get("name", "unknown") for s in saved]
                    names_str = ", ".join(names[:5])
                    extra = f" and {len(names) - 5} more" if len(names) > 5 else ""
                    fb.speak(f"Saved sessions: {names_str}{extra}. Say the session name.")
                    fb.wait(timeout=6)
                    name_text = ""
                    if mc_ok:
                        try:
                            name_text, _ = mc.listen(
                                initial_prompt=ENV_INITIAL_PROMPT,
                                hotwords=ENV_HOTWORDS, correct=False)
                        except Exception:
                            name_text = ""
                    print(f"[ENV] voice(direct): {name_text!r} — answering 'session name'")
                    if name_text:
                        sdata = _find_session(name_text)
                        if sdata:
                            session_name = sdata["name"]
                            history.clear()
                            history.extend(sdata.get("history", []))
                            # Restore the last scan's key frames so Gemini keeps
                            # visual context — a loaded session can answer
                            # questions about what it saw, not just recite text.
                            last_image_paths = sdata.get("last_images", []) or []
                            last_keyframes = _load_keyframes(last_image_paths)
                            scan_count = len(history) // 2
                            n = len(history) // 2
                            imgnote = ""
                            if last_keyframes:
                                imgnote = (f" I still have the {len(last_keyframes)} "
                                           "image" + ("s" if len(last_keyframes) != 1 else "")
                                           + " from the last scan.")
                            fb.speak(
                                f"Loaded {session_name}. "
                                f"{n} exchange{'s' if n != 1 else ''}.{imgnote} "
                                "Say describe to scan, or ask a question."
                            )
                        else:
                            fb.speak(f"No session found matching {name_text}.")
                    else:
                        fb.speak("No name heard.")
                    fb.wait(timeout=5)
                    _voice_on()
                    continue

                # ── SCAN (capture new view) ───────────────────────────────────
                if cmd == "SCAN":
                    _voice_off()
                    _drain()
                    fb.confirm()
                    fb.speak(
                        "Recording for five seconds. Pan slowly "
                        "to cover the whole area around you."
                    )
                    fb.wait(timeout=4)

                    frames = _record_frames()
                    if not frames:
                        fb.speak(
                            "Camera error. Check the USB camera and try again."
                        )
                        fb.wait(timeout=4)
                        _voice_on()
                        continue

                    keyframes = _extract_keyframes(frames)
                    print(
                        f"[ENV] {len(frames)} frames → "
                        f"{len(keyframes)} key frames → Gemini"
                    )
                    fb.speak("Analyzing. Please wait.")
                    fb.wait(timeout=3)

                    user_msg = ("Describe my surroundings in as much detail as "
                                "possible. Tell me what kind of place this is and "
                                "everything that is around me.")
                    try:
                        reply = _ask_gemini(client, history, user_msg, keyframes)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        fb.speak(
                            f"Could not reach Gemini. {e}"
                        )
                        fb.wait(timeout=4)
                        _voice_on()
                        continue

                    # Keep this scan's frames in memory so follow-up questions
                    # (and a later LOAD) still have the images to reason over.
                    scan_count += 1
                    last_keyframes = keyframes
                    jpegs      = [_frame_to_jpeg(f) for f in keyframes]
                    store_name = session_name or auto_name
                    try:
                        last_image_paths = _save_keyframes_local(
                            store_name, scan_count, jpegs)
                    except Exception as e:
                        print(f"[ENV] Local keyframe save error: {e}")
                        last_image_paths = []

                    _append("user", user_msg)
                    _append("assistant", reply)
                    _autosave()

                    # Archive frames + reply on the laptop server for review,
                    # off the interaction path so it never delays the user.
                    threading.Thread(
                        target=_send_scan_to_server,
                        args=(ctx.link, store_name, user_msg, reply,
                              jpegs, len(keyframes)),
                        daemon=True, name="ENVArchive",
                    ).start()
                    try:
                        ctx.link.send("lidar", {
                            "action":       "feature_state",
                            "feature_name": "env",
                            "state":        "scan_done",
                            "session":      store_name,
                            "reply_words":  len(reply.split()),
                            "keyframes":    len(keyframes),
                        })
                    except Exception:
                        pass

                    # No name prompt here — just describe and keep going. The
                    # user keeps asking questions or scanning until they close.
                    fb.speak(reply)
                    fb.wait(timeout=max(4, len(reply.split()) // 2))
                    _voice_on()
                    continue

                # ── FREE-FORM QUESTION ────────────────────────────────────────
                _voice_off()
                _drain()

                if not history:
                    fb.speak(
                        "Say describe first so I can see your surroundings."
                    )
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                fb.speak("Thinking.")
                try:
                    # Re-attach the last scan's frames so the answer is grounded
                    # in what the camera actually saw, not just the text history.
                    reply = _ask_gemini(client, history, raw, last_keyframes)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    fb.speak(f"Gemini did not respond. {e}")
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                fb.speak(reply)
                fb.wait(timeout=max(4, len(reply.split()) // 2))

                _append("user", raw)
                _append("assistant", reply)
                _autosave()
                _voice_on()

        finally:
            _voice_off()
            final_name = session_name or auto_name
            try:
                ctx.link.send("lidar", {
                    "action":       "feature_state",
                    "feature_name": "env",
                    "state":        "stopped",
                    "session":      final_name,
                    "exchanges":    len(history) // 2,
                })
            except Exception:
                pass
            if history:
                try:
                    path = _save_session(final_name, history,
                                         last_images=last_image_paths)
                    # If the user gave it a real name on close, drop the
                    # placeholder "session N" record that autosave created.
                    if session_name and session_name != auto_name:
                        _delete_session(auto_name)
                    n = len(history) // 2
                    fb.speak(
                        f"Saved as {final_name}. "
                        f"{n} exchange{'s' if n != 1 else ''}."
                    )
                    fb.wait(timeout=5)
                    print(f"[ENV] Session saved → {path}")
                except Exception as e:
                    print(f"[ENV] Save error: {e}")
