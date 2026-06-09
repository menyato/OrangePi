"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Voice (mc.listen / Whisper, Logitech mic) handles: scan, next, close, yes, no.
Gestures handle reading controls only: pause, skip-forward, skip-back.

Flow
----
1. Open  → "Ready. Say scan to capture."
           Voice thread starts (mc.listen loop → voice_q).

2. Scan  → voice "scan" or NEXT gesture triggers capture.
           Voice thread stops so no cues during capture.
           mc.capture_best_frame_local() → sharpest of 15 frames over 4 s.
           Haptic click on success.

3. Result → server EasyOCR → announce:
           "Page N. X words. Reading now."   (if page number detected)
           "Page scanned. No number found. X words. Reading now."

4. Reading → gesture-only while text plays:
           OCR_PAUSE  (Thumb+Ring+Pinky tilt back) → pause / resume
           OCR_FWD    (Thumb flick right)           → skip 3 chunks
           OCR_BWD    (Thumb flick left)            → rewind 3 chunks
           NEXT / OCR_SCAN gesture                  → jump to next scan

5. Paused → voice thread restarts.
           Say "yes"  → break out for another page scan
           Say "no"   → resume reading
           Say "next" → next page scan immediately
           Say "close"→ exit
           OCR_PAUSE gesture also resumes.

6. After page → "Done. Say scan for next page or close to finish."
               Voice thread restarts.

7. Close → voice "close" or OCR_CLOSE gesture.
           Say a name to save session, or say "skip".
           "Saved as <name>. N pages, X words."
"""

import contextlib
import json
import os
import queue
import re
import threading
import time

try:
    import orangepi_client as mc
    _MC_OK = True
except ImportError:
    _MC_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from features.base import Feature, FeatureContext

# ── tunables ──────────────────────────────────────────────────────────────────
WORDS_PER_SEC  = 2.5
CHUNK_TARGET   = 10
SKIP_CHUNKS    = 3
SESSIONS_DIR   = os.path.expanduser("~/ocr_sessions")
CAMERA_INDICES = [0, 1, 2]
IDLE_REMIND_S  = 60   # re-prompt after this many idle seconds

# ── mc initialisation (shared with money recognition) ─────────────────────────
_mc_ready = False
_mc_lock  = threading.Lock()


def _ensure_mc(fb) -> bool:
    global _mc_ready
    if not _MC_OK:
        return False
    with _mc_lock:
        if _mc_ready:
            return True
        # Money recognition may have already loaded everything (vad + Whisper)
        try:
            from features.money_recognition import MoneyRecognition
            if MoneyRecognition._models_ready:
                _mc_ready = True
                return True
        except (ImportError, AttributeError):
            pass
        # Full init: init_cues → auto_detect_all → load_models (sets vad + Whisper)
        # load_models downloads ~150 MB on first run — warn the user
        try:
            mc.init_cues()
            mc.auto_detect_all()
            fb.speak(
                "Loading voice model. "
                "This takes about one minute on first use. Please wait."
            )
            mc.load_models()   # initialises vad (webrtcvad) + Whisper
            _mc_ready = True
            return True
        except Exception as e:
            print(f"[OCR] mc init error: {e}")
            return False


# ── cv2 fallback ──────────────────────────────────────────────────────────────

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


def _capture_jpeg_cv2() -> "bytes | None":
    if not _CV2_OK:
        return None
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = None
    for idx in CAMERA_INDICES:
        with _quiet_stderr():
            c = cv2.VideoCapture(idx, backend)
        if not c.isOpened():
            continue
        ret, fr = c.read()
        if ret and fr is not None and fr.size > 0 and fr.max() > 0:
            cap = c
            break
        c.release()
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        time.sleep(1.5)
        best, best_v = None, -1.0
        for _ in range(5):
            ret, fr = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            v    = cv2.Laplacian(gray, cv2.CV_64F).var()
            if v > best_v:
                best_v, best = v, fr
            time.sleep(0.15)
        if best is None:
            return None
        _, buf = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes()
    finally:
        cap.release()


def _capture_frame(fb) -> "bytes | None":
    if _ensure_mc(fb):
        jpeg = mc.capture_best_frame_local()
        if jpeg is not None:
            fb.confirm()   # haptic click — no audio conflict
            return jpeg
    jpeg = _capture_jpeg_cv2()
    if jpeg is not None:
        fb.confirm()
    return jpeg


# ── voice worker ──────────────────────────────────────────────────────────────

_SCAN_W  = {"scan", "capture", "take", "photo", "picture"}
_NEXT_W  = {"next", "skip", "forward", "page"}
_CLOSE_W = {"close", "stop", "quit", "done", "finish", "exit"}
_YES_W   = {"yes", "yeah", "add", "another", "more", "confirm"}
_NO_W    = {"no", "nope", "resume", "continue", "back"}


def _classify(text: str) -> "str | None":
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _SCAN_W:   return "SCAN"
    if words & _NEXT_W:   return "NEXT"
    if words & _YES_W:    return "YES"
    if words & _NO_W:     return "NO"
    if words & _CLOSE_W:  return "CLOSE"
    return None


def _voice_worker(abort: threading.Event,
                  voice_q: queue.Queue,
                  stop_ev: threading.Event) -> None:
    while not stop_ev.is_set() and not abort.is_set():
        try:
            text, _ = mc.listen()
        except Exception as e:
            print(f"[OCR] listen error: {e}")
            time.sleep(0.5)
            continue
        if not text or stop_ev.is_set():
            continue
        cmd = _classify(text)
        if cmd:
            print(f"[OCR] voice: {text!r} → {cmd}")
            voice_q.put(cmd)


# ── gesture → command map ─────────────────────────────────────────────────────

_GMAP = {
    "OCR_PAUSE":    "PAUSE",
    "OCR_FWD":      "FWD",
    "OCR_BWD":      "BWD",
    "NEXT":         "NEXT",
    "OCR_SCAN":     "SCAN",
    "OCR_CLOSE":    "CLOSE",
    "PROG_CONFIRM": "YES",
    "PROG_DISCARD": "NO",
}


# ── queue helpers ─────────────────────────────────────────────────────────────

def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _wait_gesture(ctx: FeatureContext, timeout: float) -> "str | None":
    gq = ctx.gesture_queue
    if gq is None:
        time.sleep(timeout)
        return None
    deadline = time.time() + timeout
    while time.time() < deadline and not ctx.abort.is_set():
        try:
            return gq.get(timeout=min(0.05, deadline - time.time()))
        except queue.Empty:
            pass
    return None


# ── text helpers ──────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list = []
    for s in sentences:
        words = s.split()
        if len(words) <= int(CHUNK_TARGET * 1.5):
            if s.strip():
                chunks.append(s.strip())
        else:
            parts = re.split(r'[,;]\s*', s)
            buf: list = []
            for p in parts:
                buf.extend(p.split())
                if len(buf) >= CHUNK_TARGET:
                    chunks.append(" ".join(buf))
                    buf = []
            if buf:
                chunks.append(" ".join(buf))
    return [c for c in chunks if c]


def _chunk_dur(chunk: str) -> float:
    return max(0.8, len(chunk.split()) / WORDS_PER_SEC + 0.4)


# ── session helpers ───────────────────────────────────────────────────────────

def _save_session(name: str, pages: list) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip().replace(' ', '_')
    if not safe:
        safe = f"session_{int(time.time())}"
    path = os.path.join(SESSIONS_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "pages": pages}, f, ensure_ascii=False, indent=2)
    return path


# ── feature ───────────────────────────────────────────────────────────────────

class OCRReader(Feature):
    name  = "ocr"
    title = "Book reader"

    def __init__(self):
        self._session_counter = 0

    def run(self, ctx: FeatureContext) -> None:
        gq = ctx.gesture_queue
        fb = ctx.feedback

        if gq:
            _drain(gq)

        mc_ok = _ensure_mc(fb)

        self._session_counter += 1
        pages:     list            = []
        last_page: "int | None"    = None

        # ── voice thread management ───────────────────────────────────────────
        voice_q  = queue.Queue()
        # Each voice_on() creates a fresh stop event so old threads die cleanly
        _cur_stop: list = [threading.Event()]
        _cur_stop[0].set()   # start as "stopped"

        def _voice_on() -> None:
            if not mc_ok:
                return
            _cur_stop[0].set()           # stop any previous thread
            stop = threading.Event()
            _cur_stop[0] = stop
            _drain(voice_q)
            threading.Thread(
                target=_voice_worker,
                args=(ctx.abort, voice_q, stop),
                daemon=True,
                name="OCRVoice",
            ).start()

        def _voice_off() -> None:
            _cur_stop[0].set()
            _drain(voice_q)

        def _get_cmd(ges_timeout: float = 0.1) -> "str | None":
            """Non-blocking voice check then gesture wait."""
            try:
                return voice_q.get_nowait()
            except queue.Empty:
                pass
            g = _wait_gesture(ctx, timeout=ges_timeout)
            return _GMAP.get(g) if g else None

        # ── opening ───────────────────────────────────────────────────────────
        fb.speak(
            "Book reader ready. "
            "Say scan to capture a page. "
            "Say close to exit."
        )
        _voice_on()

        try:
            idle_t = time.time() + IDLE_REMIND_S

            # ── outer loop: idle / scan wait ──────────────────────────────────
            while not ctx.abort.is_set():
                cmd = _get_cmd(ges_timeout=0.4)

                if ctx.abort.is_set():
                    break

                if cmd == "CLOSE":
                    break

                if cmd in ("SCAN", "NEXT"):
                    pass   # fall through to capture
                elif cmd is None:
                    if time.time() >= idle_t:
                        fb.speak("Still ready. Say scan to capture, or close to exit.")
                        idle_t = time.time() + IDLE_REMIND_S
                    continue
                else:
                    continue   # ignore other commands while idle

                # ── capture ───────────────────────────────────────────────────
                _voice_off()   # silence mic during capture — no cue interference
                idle_t = time.time() + IDLE_REMIND_S

                fb.speak("Hold the page flat and steady. Capturing now.")
                jpeg = _capture_frame(fb)   # ~4 s; haptic confirm on success

                if jpeg is None:
                    fb.speak("Camera error. Check USB camera and try again.")
                    _voice_on()
                    continue

                fb.speak("Processing, please wait.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable.")
                    _voice_on()
                    continue

                text_body = resp.get("text", "").strip()
                if not text_body:
                    fb.speak("No text found. Try better lighting and say scan again.")
                    _voice_on()
                    continue

                detected_page = resp.get("page")
                skipped       = resp.get("skipped_pages", [])
                word_count    = len(text_body.split())

                if skipped:
                    labels = ", ".join(str(p) for p in skipped)
                    fb.speak(
                        f"Warning: page{'s' if len(skipped) > 1 else ''} "
                        f"{labels} skipped."
                    )

                if detected_page is not None:
                    if last_page is not None and detected_page != last_page + 1:
                        missing = list(range(last_page + 1, detected_page))
                        if missing:
                            labels = ", ".join(str(p) for p in missing)
                            fb.speak(
                                f"Skipped page{'s' if len(missing) > 1 else ''} {labels}."
                            )
                    last_page = detected_page
                    fb.speak(f"Page {detected_page}. {word_count} words. Reading now.")
                else:
                    fb.speak(
                        f"Page scanned. No page number found. "
                        f"{word_count} words. Reading now."
                    )

                pages.append({
                    "page":       detected_page,
                    "text":       text_body,
                    "word_count": word_count,
                })

                # ── reading loop ──────────────────────────────────────────────
                # Voice thread OFF during reading — cues would clash with TTS.
                # Gestures only: OCR_PAUSE / OCR_FWD / OCR_BWD / NEXT.
                chunks = _chunk_text(text_body)
                if not chunks:
                    fb.speak("Page appears blank.")
                    _voice_on()
                    continue

                i               = 0
                paused          = False
                pause_announced = False

                while i < len(chunks) and not ctx.abort.is_set():

                    # ── paused state ──────────────────────────────────────────
                    if paused:
                        if not pause_announced:
                            fb.speak(
                                "Paused. "
                                "Say yes to add another page, "
                                "say no to resume, "
                                "or say next for the next page."
                            )
                            pause_announced = True
                            _voice_on()   # voice ON while paused — user expects to speak

                        cmd = _get_cmd(ges_timeout=0.4)
                        if cmd is None:
                            continue

                        _voice_off()   # command received — stop listening

                        if cmd in ("PAUSE", "NO"):
                            paused = False; pause_announced = False
                            fb.speak("Resuming.")

                        elif cmd == "YES":
                            fb.speak(
                                "Okay. Say scan or gesture next "
                                "when the page is ready."
                            )
                            i = len(chunks); paused = False; pause_announced = False

                        elif cmd in ("NEXT", "SCAN"):
                            fb.silence(); fb.speak("Next page.")
                            i = len(chunks); paused = False; pause_announced = False

                        elif cmd == "FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            paused = False; pause_announced = False
                            fb.speak("Skipped. Resuming.")

                        elif cmd == "BWD":
                            i = max(0, i - SKIP_CHUNKS)
                            paused = False; pause_announced = False
                            fb.speak("Rewound. Resuming.")

                        elif cmd == "CLOSE":
                            ctx.abort.set()

                        continue

                    # ── read one chunk ────────────────────────────────────────
                    fb.speak(chunks[i])
                    dur = _chunk_dur(chunks[i])
                    i  += 1

                    deadline = time.time() + dur
                    while (time.time() < deadline
                           and not ctx.abort.is_set()):
                        g = _wait_gesture(
                            ctx, timeout=min(0.05, deadline - time.time())
                        )
                        gcmd = _GMAP.get(g) if g else None

                        if gcmd == "PAUSE":
                            paused = True; pause_announced = False
                            fb.silence(); fb.speak("Paused.")
                            break

                        elif gcmd == "FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            fb.silence(); fb.speak(f"Skipped {skip}.")
                            break

                        elif gcmd == "BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)
                            fb.silence(); fb.speak("Rewinding.")
                            break

                        elif gcmd in ("NEXT", "SCAN"):
                            fb.silence(); fb.speak("Next page.")
                            i = len(chunks); break

                        elif gcmd == "CLOSE":
                            fb.silence(); ctx.abort.set(); break

                # Page finished
                if not ctx.abort.is_set() and i >= len(chunks):
                    label = f"Page {detected_page}" if detected_page else "Page"
                    fb.speak(
                        f"{label} done. "
                        "Say scan for the next page, "
                        "or say close to finish."
                    )
                    _voice_on()   # re-arm voice for next scan wait

        finally:
            _voice_off()

            if pages:
                # Ask user for a session name via voice
                name = f"session_{self._session_counter}"
                if mc_ok:
                    fb.speak(
                        "Say a name to save this session, "
                        "or say skip to use the default name."
                    )
                    try:
                        vtxt, _ = mc.listen()
                        if vtxt:
                            skip_words = {"skip", "done", "no", "cancel", "default"}
                            if not (set(re.findall(r"[a-z]+", vtxt.lower()))
                                    & skip_words):
                                name = vtxt.strip()
                    except Exception:
                        pass

                try:
                    path        = _save_session(name, pages)
                    total_words = sum(p["word_count"] for p in pages)
                    fb.speak(
                        f"Saved as {name}. "
                        f"{len(pages)} page{'s' if len(pages) > 1 else ''}, "
                        f"{total_words} words."
                    )
                    print(f"[OCR] Session saved → {path}")
                except Exception as e:
                    print(f"[OCR] Save error: {e}")
