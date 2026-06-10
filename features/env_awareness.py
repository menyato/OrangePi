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
import subprocess
import tempfile
import threading
import time

from features.base import Feature, FeatureContext

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
VIDEO_DURATION_S  = 3       # seconds to record per scan
VIDEO_FPS         = 10      # Pi-friendly frame rate (30 frames per scan)
MAX_KEYFRAMES     = 5       # maximum images sent to Gemini per turn
SCENE_THRESHOLD   = 0.30    # Bhattacharyya histogram distance for a scene cut
JPEG_QUALITY      = 75      # lower = smaller payload, faster API round-trip
MAX_IMG_WIDTH     = 1024    # resize frames wider than this before encoding
GEMINI_MODEL      = "gemini-2.5-flash"
MAX_HISTORY_TURNS = 16      # keep last N conversation turns in session context
IDLE_REMIND_S     = 60
_KEY_ENV          = "GEMINI_API_KEY"

# ── gesture → synthetic voice token ──────────────────────────────────────────

_GMAP_ENV = {
    "ENV_SCAN":  "scan",    # user-assigned capture gesture
    "ENV_CLOSE": "close",   # user-assigned close gesture
    "NEXT":      "scan",    # "next" flick = capture, consistent with OCR
}

# ── command word sets ─────────────────────────────────────────────────────────

_SCAN_W  = {"describe", "scan", "look", "capture", "photo",
            "take", "see", "record", "view"}
_LOAD_W  = {"load", "recall", "open", "restore"}
_CLOSE_W = {"close", "stop", "quit", "exit", "done", "finish"}
_SKIP_W  = {"skip", "none", "no", "cancel", "default"}

# ── Gemini system prompt ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an environmental awareness assistant for a blind person wearing a smart glove.
When images are provided you describe what you see.
When a follow-up question arrives without new images, you answer from earlier in this conversation.

DESCRIPTION RULES:
1. Lead with safety first: obstacles, stairs, steps, traffic, people in close proximity.
2. Describe near-to-far: floor or ground directly ahead, then 1-3 metres, then background.
3. Use precise spatial language: "chair 1 metre ahead on your left", \
"open door at arm's reach directly ahead", "step down at your feet".
4. Name objects by their practical use: "door handle at waist height on the right", not just "door".
5. Estimate distances in metres from the viewer.
6. Mention lighting only when dim, dark, or affecting safe navigation.
7. Be concise and direct — no filler phrases, no poetry, no meta-commentary.
8. Multiple frames are sequential moments from one short clip — synthesise into one coherent description.
9. Never comment on image quality, blur, or the capture process.
10. Responses: 2-4 sentences for a new scan; 1-2 sentences for a follow-up question.\
"""


# ── API key loading ───────────────────────────────────────────────────────────

def _load_api_key() -> str:
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

def _save_session(name: str, history: list) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", name.strip())[:60] or "session"
    path = os.path.join(SESSIONS_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "history": history}, f,
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
    sessions = _list_sessions()
    if not sessions:
        return ""
    rows = []
    for s in sessions:
        name    = s.get("name", "unknown")
        n_turns = len(s.get("history", [])) // 2
        rows.append(f"{name}: {n_turns} exchange{'s' if n_turns != 1 else ''}")
    c = len(rows)
    return (f"You have {c} saved session{'s' if c != 1 else ''}. "
            + ". ".join(rows) + ". ")


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

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=_gtypes.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=512,
            temperature=0.3,
        ),
    )
    return (resp.text or "").strip()


# ── mc initialisation ─────────────────────────────────────────────────────────

_mc_ready = False
_mc_lock  = threading.Lock()


def _ensure_mc(fb) -> bool:
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
            mc.init_cues()
            mc.auto_detect_all()
            fb.speak("Loading voice model. Please wait.")
            mc.load_models()
            _mc_ready = True
            return True
        except Exception as e:
            print(f"[ENV] mc init error: {e}")
            return False


# ── voice worker ──────────────────────────────────────────────────────────────

def _env_voice_worker(abort: threading.Event,
                      voice_q: queue.Queue,
                      stop_ev: threading.Event,
                      proc_ref: list,
                      fb_ref) -> None:
    """
    Interruptible voice worker: arecord subprocess + faster-whisper.
    Puts the raw transcript string into voice_q so the outer loop can
    handle both keyword commands and free-form questions to Gemini.
    _voice_off() calls proc.terminate() + wait() to release the ALSA
    input device immediately before TTS playback starts.
    Falls back to mc.listen() when arecord is unavailable.
    """
    alsa_in = (getattr(mc, "ALSA_OUTPUT_DEVICE", None) or "default") if _MC_OK else "default"
    wm      = getattr(mc, "whisper_model", None) if _MC_OK else None

    # ── fallback: mc.listen() ────────────────────────────────────────────────
    def _mc_fallback():
        while not stop_ev.is_set() and not abort.is_set():
            try:
                if _MC_OK:
                    text, _ = mc.listen()
                    if text and not stop_ev.is_set():
                        print(f"[ENV] voice (mc): {text!r}")
                        voice_q.put(text.strip())
                else:
                    time.sleep(1.0)
            except Exception:
                time.sleep(0.5)

    if wm is None:
        _mc_fallback()
        return

    # ── main loop: arecord + Whisper ─────────────────────────────────────────
    while not stop_ev.is_set() and not abort.is_set():
        # Don't open the mic while TTS is playing — prevents ALSA conflict
        while fb_ref.is_speaking() and not stop_ev.is_set():
            time.sleep(0.05)
        if stop_ev.is_set():
            break
        time.sleep(0.1)   # brief gap so ALSA fully releases after aplay
        if stop_ev.is_set():
            break

        try:
            fd, wav_tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        except Exception:
            time.sleep(0.5)
            continue

        try:
            proc = subprocess.Popen(
                ["arecord", "-D", alsa_in, "-f", "S16_LE",
                 "-r", "16000", "-c", "1", "-d", "4", "-q", wav_tmp],
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                os.unlink(wav_tmp)
            except OSError:
                pass
            print("[ENV] arecord not found — falling back to mc.listen()")
            _mc_fallback()
            return
        except Exception as e:
            print(f"[ENV] arecord start: {e}")
            try:
                os.unlink(wav_tmp)
            except OSError:
                pass
            time.sleep(0.5)
            continue

        proc_ref[0] = proc

        # Poll every 50 ms so stop_ev kills recording immediately
        while proc.poll() is None:
            if stop_ev.is_set() or abort.is_set():
                try:
                    proc.terminate()
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
                proc_ref[0] = None
                try:
                    os.unlink(wav_tmp)
                except OSError:
                    pass
                return
            time.sleep(0.05)

        proc_ref[0] = None

        if stop_ev.is_set() or abort.is_set():
            try:
                os.unlink(wav_tmp)
            except OSError:
                pass
            break

        text = ""
        try:
            segs, _ = wm.transcribe(
                wav_tmp,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.25,
                    "min_speech_duration_ms": 100,
                    "min_silence_duration_ms": 200,
                },
                initial_prompt=(
                    "Commands: describe, scan, load, close, stop. "
                    "Or ask any question about the environment."
                ),
                condition_on_previous_text=False,
            )
            text = " ".join(s.text for s in segs).strip()
        except Exception as e:
            print(f"[ENV] transcribe: {e}")
        finally:
            try:
                os.unlink(wav_tmp)
            except OSError:
                pass

        if text and not stop_ev.is_set():
            print(f"[ENV] voice: {text!r}")
            voice_q.put(text)
            stop_ev.wait(0.15)


# ── classify ──────────────────────────────────────────────────────────────────

def _classify_env(text: str) -> "str | None":
    """
    Return a control command string ("SCAN", "LOAD", "CLOSE") if the text
    matches a command word set; return None to treat it as a Gemini question.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
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
    "describe / scan / look" → record 3-second clip, extract key frames,
                               ask Gemini to describe the environment.
    "load"                   → reload a saved session (restores history).
    "close / stop / quit"    → save session and exit.
    Any other speech         → treated as a follow-up question to Gemini;
                               no new video captured, answers from prior context.
    """
    name  = "env"
    title = "Environmental awareness"

    def run(self, ctx: FeatureContext) -> None:
        fb      = ctx.feedback
        mc_ok   = _ensure_mc(fb)
        gq      = ctx.gesture_queue
        voice_q = queue.Queue()
        stop_ev = threading.Event()
        cur_proc: list = [None]

        history: list        = []
        session_name: "str | None" = None
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
                "Add GEMINI_API_KEY equals your key to the file dot env in your home folder."
            )
            return

        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            fb.speak(f"Gemini init error. {e}")
            return

        # ── voice thread helpers ──────────────────────────────────────────────

        def _voice_on() -> None:
            nonlocal stop_ev
            _kill_proc()
            stop_ev = threading.Event()
            def _start():
                time.sleep(1.0)
                if not stop_ev.is_set() and not ctx.abort.is_set():
                    _env_voice_worker(ctx.abort, voice_q, stop_ev, cur_proc, fb)
            threading.Thread(target=_start, daemon=True).start()

        def _kill_proc() -> None:
            proc = cur_proc[0]
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
                cur_proc[0] = None

        def _voice_off() -> None:
            stop_ev.set()
            _kill_proc()

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
            if history and session_name:
                try:
                    _save_session(session_name, history)
                    print(f"[ENV] Auto-saved '{session_name}' "
                          f"({len(history) // 2} exchanges)")
                except Exception as e:
                    print(f"[ENV] Auto-save error: {e}")

        def _append(role: str, content: str) -> None:
            history.append({"role": role, "content": content})
            # Trim to keep context manageable
            while len(history) > MAX_HISTORY_TURNS * 2:
                history.pop(0)

        # ── opening announcement ──────────────────────────────────────────────

        _voice_on()
        fb.speak(
            "Environmental awareness. "
            + _startup_summary()
            + "Say describe to scan your surroundings, "
            "load to open a saved session, "
            "or close to exit. "
            "You can also ask any question about what I see."
        )
        fb.wait(timeout=15)

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

                cmd = _classify_env(raw)
                idle_t = time.time() + IDLE_REMIND_S

                # ── CLOSE ─────────────────────────────────────────────────────
                if cmd == "CLOSE":
                    break

                # ── LOAD ──────────────────────────────────────────────────────
                if cmd == "LOAD":
                    _voice_off()
                    fb.speak("Say the session name.")
                    fb.wait(timeout=4)
                    name_text = ""
                    if mc_ok:
                        try:
                            name_text, _ = mc.listen()
                        except Exception:
                            name_text = ""
                    if name_text:
                        sdata = _find_session(name_text)
                        if sdata:
                            session_name = sdata["name"]
                            history.clear()
                            history.extend(sdata.get("history", []))
                            n = len(history) // 2
                            fb.speak(
                                f"Loaded {session_name}. "
                                f"{n} exchange{'s' if n != 1 else ''}. "
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
                        "Hold the camera steady and pan slowly "
                        "if you want to cover a wider area. Recording now."
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

                    user_msg = "Please describe the environment."
                    try:
                        reply = _ask_gemini(client, history, user_msg, keyframes)
                    except Exception as e:
                        print(f"[ENV] Gemini error: {e}")
                        fb.speak(
                            "Could not reach Gemini. Check your internet connection."
                        )
                        fb.wait(timeout=4)
                        _voice_on()
                        continue

                    # First scan: name the session
                    if session_name is None:
                        fb.speak(reply)
                        fb.wait(timeout=max(4, len(reply.split()) // 2))
                        if mc_ok:
                            fb.speak("Say a name for this session, or say skip.")
                            fb.wait(timeout=4)
                            try:
                                vtxt, _ = mc.listen()
                                if vtxt:
                                    words = set(re.findall(r"[a-z]+", vtxt.lower()))
                                    if not (words & _SKIP_W):
                                        session_name = vtxt.strip()
                            except Exception:
                                pass
                        if not session_name:
                            session_name = (
                                "env_"
                                + datetime.datetime.now().strftime("%Y%m%d_%H%M")
                            )
                        fb.speak(f"Session: {session_name}.")
                        fb.wait(timeout=3)
                    else:
                        fb.speak(reply)
                        fb.wait(timeout=max(4, len(reply.split()) // 2))

                    _append("user", user_msg)
                    _append("assistant", reply)
                    _autosave()
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
                    reply = _ask_gemini(client, history, raw, [])
                except Exception as e:
                    print(f"[ENV] Gemini error: {e}")
                    fb.speak("Gemini did not respond. Please try again.")
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
            if history and session_name:
                try:
                    path = _save_session(session_name, history)
                    n = len(history) // 2
                    fb.speak(
                        f"Saved as {session_name}. "
                        f"{n} exchange{'s' if n != 1 else ''}."
                    )
                    fb.wait(timeout=5)
                    print(f"[ENV] Session saved → {path}")
                except Exception as e:
                    print(f"[ENV] Save error: {e}")
