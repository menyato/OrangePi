"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Flow (mirrors money recognition — voice-driven scan, gesture-controlled reading)
---------------------------------------------------------------------------------
1. Open  → "Book reader ready. Say scan or gesture next to capture."
2. Scan  → mc.capture_best_frame_local() (4 s multi-frame, picks sharpest)
            → send JPEG to server EasyOCR → receive text.
3. Feedback → "Page N. X words. Reading now." then read aloud chunk by chunk.
4. During reading, OCR gestures forwarded by the state machine:
     OCR_PAUSE          → pause; prompt "Say yes to add a page or say no to resume"
     OCR_FWD            → skip forward  ~5 s (3 chunks)
     OCR_BWD            → skip backward ~5 s (3 chunks)
     NEXT / OCR_SCAN    → jump to next page scan immediately
5. Voice while paused:
     PROG_CONFIRM ("yes") → break to scan next page
     PROG_DISCARD ("no")  → resume reading
6. After page finishes → "Page done. Say scan for next page or say close to exit."
7. Close gesture / "close" voice → save session and exit.

Voice commands (VoiceListener → feature gesture queue — no extra mic thread):
  "scan" / "capture" / "take"  → OCR_SCAN  (same as NEXT gesture)
  "close"                       → OCR_CLOSE (same as close gesture)
  "yes" / "confirm"             → PROG_CONFIRM (add page when paused)
  "no"  / "discard"             → PROG_DISCARD (resume reading when paused)
  "next" / "skip"               → NEXT       (next page when idle or reading)

Frame capture: uses orangepi_client.capture_best_frame_local() which warms up
the camera for 10 frames then picks the sharpest of the next ~6 frames over 4 s.
This eliminates the blurry / out-of-focus frames that a 3-frame capture produces.
Falls back to cv2 if orangepi_client is not available.
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

# ── tunables ─────────────────────────────────────────────────────────────────
WORDS_PER_SEC   = 2.5
CHUNK_TARGET    = 10
SKIP_CHUNKS     = 3
SESSIONS_DIR    = os.path.expanduser("~/ocr_sessions")
CAMERA_INDICES  = [0, 1, 2]
SCAN_WAIT_S     = 60
PAUSE_WAIT_S    = 60

# ── mc camera initialisation (shared with money recognition) ──────────────────
_mc_cam_ready = False
_mc_cam_lock  = threading.Lock()


def _ensure_mc_camera(fb) -> bool:
    """Initialise mc's camera detection once.  Whisper model not needed."""
    global _mc_cam_ready
    if not _MC_OK:
        return False
    with _mc_cam_lock:
        if _mc_cam_ready:
            return True
        # Money recognition may have already fully initialised mc
        try:
            from features.money_recognition import MoneyRecognition
            if MoneyRecognition._models_ready:
                _mc_cam_ready = True
                return True
        except (ImportError, AttributeError):
            pass
        try:
            mc.init_cues()
            mc.auto_detect_all()   # detects Logitech camera → sets mc.CAM_DEVICE
            _mc_cam_ready = True
            return True
        except Exception as e:
            print(f"[OCR] mc camera init error: {e}")
            return False


# ── cv2 fallback camera (used only when mc is unavailable) ───────────────────

@contextlib.contextmanager
def _suppress_cv2_stderr():
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
    """Fallback: open camera, settle, pick sharpest of 3 frames."""
    if not _CV2_OK:
        return None
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = None
    for idx in CAMERA_INDICES:
        with _suppress_cv2_stderr():
            c = cv2.VideoCapture(idx, backend)
        if not c.isOpened():
            continue
        ret, frame = c.read()
        if ret and frame is not None and frame.size > 0 and frame.max() > 0:
            cap = c
            break
        c.release()
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        time.sleep(1.5)
        best_frame, best_var = None, -1.0
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var  = cv2.Laplacian(gray, cv2.CV_64F).var()
            if var > best_var:
                best_var, best_frame = var, frame
            time.sleep(0.15)
        if best_frame is None:
            return None
        _, buf = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes()
    finally:
        cap.release()


def _capture_frame(fb) -> "bytes | None":
    """Capture the best possible frame.

    Uses mc.capture_best_frame_local() when available — it warms up 10 frames,
    captures ~6 more over 4 s, and returns the sharpest one.  Falls back to
    a quick cv2 grab when mc is not installed.
    """
    if _ensure_mc_camera(fb):
        jpeg = mc.capture_best_frame_local()
        if jpeg is not None:
            try:
                mc.play_cue(mc._CUE_SHOT)   # satisfying camera click
            except Exception:
                pass
            return jpeg
    # fallback
    return _capture_jpeg_cv2()


# ── text helpers ──────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
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


def _chunk_duration(chunk: str) -> float:
    return max(0.8, len(chunk.split()) / WORDS_PER_SEC + 0.4)


def _wait_for_gesture(ctx: FeatureContext, timeout: float) -> "str | None":
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


def _save_session(name: str, pages: list) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip().replace(' ', '_')
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

        # Drain bounce gestures from the opening gesture
        if gq is not None:
            while True:
                try:
                    gq.get_nowait()
                except queue.Empty:
                    break

        self._session_counter += 1
        session_name = f"Reading session {self._session_counter}"
        pages: list       = []
        last_page: "int | None" = None

        # Warm up the camera in the background so the first scan is instant
        def _warm():
            _ensure_mc_camera(fb)
        threading.Thread(target=_warm, daemon=True).start()

        fb.speak(
            "Book reader ready. "
            "Say scan or gesture next to capture a page. "
            "While reading: "
            "Thumb Ring Pinky back to pause. "
            "When paused, say yes to add a page or say no to resume. "
            "Thumb flick right to skip, left to rewind. "
            "Say close or use close gesture to exit."
        )
        fb.wait(timeout=16)

        try:
            # ── outer scan loop ───────────────────────────────────────────
            while not ctx.abort.is_set():

                g = _wait_for_gesture(ctx, timeout=SCAN_WAIT_S)

                if ctx.abort.is_set():
                    break

                if g is None:
                    fb.speak(
                        "Still ready. "
                        "Say scan or gesture next to capture. "
                        "Say close to exit."
                    )
                    continue

                if g == "OCR_CLOSE":
                    break

                if g == "EDIT":
                    self._session_counter += 1
                    session_name = f"Reading session {self._session_counter}"
                    fb.speak(f"Session renamed to {session_name}.")
                    continue

                if g in ("OCR_PAUSE", "OCR_FWD", "OCR_BWD",
                         "PROG_CONFIRM", "PROG_DISCARD"):
                    fb.speak("Not reading yet. Say scan or gesture next.")
                    continue

                if g not in ("NEXT", "OCR_SCAN"):
                    fb.speak("Say scan or gesture next to capture.")
                    continue

                # ── capture — like money: hold steady, multi-frame, click ──
                fb.speak("Hold the page flat and steady. Capturing now.")
                fb.wait(timeout=6)
                # mc.capture_best_frame_local() takes ~4 s internally
                jpeg = _capture_frame(fb)

                if jpeg is None:
                    fb.speak(
                        "Camera error. "
                        "Make sure the USB camera is connected and try again."
                    )
                    continue

                fb.speak("Processing, please wait.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable. Check connection.")
                    continue

                text = resp.get("text", "").strip()
                if not text:
                    fb.speak(
                        "No text found. "
                        "Hold the page flat, ensure good lighting, "
                        "and say scan again."
                    )
                    continue

                detected_page = resp.get("page")
                skipped       = resp.get("skipped_pages", [])
                word_count    = len(text.split())

                if skipped:
                    labels = ", ".join(str(p) for p in skipped)
                    fb.speak(
                        f"Warning: page{'s' if len(skipped) > 1 else ''} "
                        f"{labels} {'were' if len(skipped) > 1 else 'was'} skipped."
                    )

                if detected_page is not None:
                    if last_page is not None and detected_page != last_page + 1:
                        missing = list(range(last_page + 1, detected_page))
                        if missing:
                            labels = ", ".join(str(p) for p in missing)
                            fb.speak(
                                f"Skipped page"
                                f"{'s' if len(missing) > 1 else ''} {labels}."
                            )
                    last_page = detected_page
                    fb.speak(f"Page {detected_page}. {word_count} words. Reading now.")
                else:
                    fb.speak(f"Page scanned. {word_count} words. Reading now.")

                pages.append({
                    "page":       detected_page,
                    "text":       text,
                    "word_count": word_count,
                })

                # ── reading loop ──────────────────────────────────────────
                chunks = _chunk_text(text)
                if not chunks:
                    fb.speak("Page appears blank.")
                    continue

                fb.wait(timeout=6)

                i               = 0
                paused          = False
                pause_announced = False

                while i < len(chunks) and not ctx.abort.is_set():

                    if paused:
                        if not pause_announced:
                            fb.speak(
                                "Paused. "
                                "Say yes to add another page. "
                                "Say no or resume gesture to continue. "
                                "Say next or gesture next for next page."
                            )
                            pause_announced = True

                        g2 = _wait_for_gesture(ctx, timeout=PAUSE_WAIT_S)

                        if g2 is None:
                            pause_announced = False
                            continue

                        if g2 in ("OCR_PAUSE", "PROG_DISCARD"):
                            paused = False; pause_announced = False
                            fb.speak("Resuming.")
                            time.sleep(0.8)

                        elif g2 == "PROG_CONFIRM":
                            fb.speak(
                                "Okay. Hold the next page to the camera. "
                                "Say scan or gesture next when ready."
                            )
                            i = len(chunks); paused = False; pause_announced = False

                        elif g2 in ("NEXT", "OCR_SCAN"):
                            fb.silence()
                            fb.speak("Next page.")
                            i = len(chunks); paused = False; pause_announced = False

                        elif g2 == "OCR_BWD":
                            i = max(0, i - SKIP_CHUNKS)
                            paused = False; pause_announced = False
                            fb.speak("Rewound. Resuming.")
                            time.sleep(0.8)

                        elif g2 == "OCR_FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            paused = False; pause_announced = False
                            fb.speak("Skipped forward. Resuming.")
                            time.sleep(0.8)

                        elif g2 == "OCR_CLOSE":
                            ctx.abort.set()

                        continue

                    # Speak next chunk
                    fb.speak(chunks[i])
                    wait    = _chunk_duration(chunks[i])
                    i      += 1

                    deadline = time.time() + wait
                    handled  = False
                    while (time.time() < deadline
                           and not ctx.abort.is_set()
                           and not handled):
                        g2 = _wait_for_gesture(
                            ctx, timeout=min(0.05, deadline - time.time())
                        )
                        if g2 == "OCR_PAUSE":
                            paused = True; pause_announced = False
                            fb.silence()
                            fb.speak("Paused.")
                            handled = True
                        elif g2 == "OCR_FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            fb.silence(); fb.speak(f"Skipped {skip}.")
                            handled = True
                        elif g2 == "OCR_BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)
                            fb.silence(); fb.speak("Rewinding.")
                            handled = True
                        elif g2 in ("NEXT", "OCR_SCAN"):
                            fb.silence(); fb.speak("Next page.")
                            i = len(chunks); handled = True
                        elif g2 == "OCR_CLOSE":
                            fb.silence(); ctx.abort.set(); handled = True

                if not ctx.abort.is_set() and i >= len(chunks):
                    page_label = (f"Page {detected_page}" if detected_page
                                  else "Page")
                    fb.speak(
                        f"{page_label} done. "
                        "Say scan or gesture next for the next page. "
                        "Say close or use close gesture to finish."
                    )

        finally:
            if pages:
                try:
                    path = _save_session(session_name, pages)
                    total_words = sum(p["word_count"] for p in pages)
                    fb.speak(
                        f"Session saved. "
                        f"{len(pages)} page{'s' if len(pages) > 1 else ''}, "
                        f"{total_words} words total."
                    )
                    print(f"[OCR] Session saved to {path}")
                except Exception as e:
                    print(f"[OCR] Save error: {e}")
