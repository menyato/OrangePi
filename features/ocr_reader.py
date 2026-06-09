"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Flow (mirrors money recognition — voice-driven, gestures as backup)
--------------------------------------------------------------------
1. Open  → "Book reader ready. Say scan or gesture next to capture."
2. Scan  → capture frame → server EasyOCR → receive text.
3. Feedback → "Page N. X words." then read aloud sentence by sentence.
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

Voice commands (always active via VoiceListener → feature gesture queue):
  "scan" / "capture" / "take"  → OCR_SCAN  (same as NEXT gesture)
  "close"                       → OCR_CLOSE (same as close gesture)
  "yes" / "confirm"             → PROG_CONFIRM (confirm add page when paused)
  "no"  / "discard"             → PROG_DISCARD (resume reading when paused)
  "next" / "skip"               → NEXT       (next page when idle or reading)

Hardcoded OCR gesture meanings (only active while Book Reader is running):
  OCR_PAUSE   — Thumb + Ring + Pinky, tilt back  — pause / resume
  OCR_FWD     — Thumb only, flick right           — skip forward
  OCR_BWD     — Thumb only, flick left            — skip backward / rewind
"""

import contextlib
import json
import os
import queue
import re
import time

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from features.base import Feature, FeatureContext

# ── tunables ─────────────────────────────────────────────────────────────────
WORDS_PER_SEC   = 2.5           # espeak-ng at 150 wpm ≈ 2.5 w/s
CHUNK_TARGET    = 10            # target words per TTS chunk
SKIP_CHUNKS     = 3             # chunks to skip per OCR_FWD/BWD
SESSIONS_DIR    = os.path.expanduser("~/ocr_sessions")
CAMERA_INDICES  = [0, 1, 2]    # try these in order until one opens
SCAN_WAIT_S     = 60            # seconds to wait for scan command before reminding
PAUSE_WAIT_S    = 60            # seconds to wait in pause before re-prompting


# ── camera helpers ────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_cv2_stderr():
    """Redirect fd 2 to /dev/null while OpenCV probes V4L2 nodes."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _open_camera():
    """Return a working cv2.VideoCapture or None.

    Probes V4L2 only; validates with a test read so codec/metadata nodes
    (which report isOpened()=True but return all-zero frames) are rejected.
    """
    if not _CV2_OK:
        return None
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    for idx in CAMERA_INDICES:
        with _suppress_cv2_stderr():
            cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0 and frame.max() > 0:
            return cap
        cap.release()
    return None


def _capture_jpeg(cap) -> "bytes | None":
    """Settle autofocus then grab the sharpest of 3 frames."""
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        time.sleep(0.6)
        best_frame, best_var = None, -1.0
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var  = cv2.Laplacian(gray, cv2.CV_64F).var()
            if var > best_var:
                best_var, best_frame = var, frame
            time.sleep(0.1)
        if best_frame is None:
            return None
        _, buf = cv2.imencode(".jpg", best_frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes()
    except Exception:
        return None


# ── text helpers ──────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list:
    """Split text into natural reading chunks of ~CHUNK_TARGET words."""
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

        cap = _open_camera()
        if cap is None:
            fb.speak("No camera found. Book reader needs a USB camera.")
            return

        # Opening announcement — voice-first, then gestures as backup
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

                # Voice or gesture close
                if g == "OCR_CLOSE":
                    break

                # Rename session
                if g == "EDIT":
                    self._session_counter += 1
                    session_name = f"Reading session {self._session_counter}"
                    fb.speak(f"Session renamed to {session_name}.")
                    continue

                # Stray OCR controls before any reading has started
                if g in ("OCR_PAUSE", "OCR_FWD", "OCR_BWD", "PROG_CONFIRM",
                         "PROG_DISCARD"):
                    fb.speak("Not reading yet. Say scan or gesture next to capture.")
                    continue

                # Anything other than scan trigger → prompt
                if g not in ("NEXT", "OCR_SCAN"):
                    fb.speak("Say scan or gesture next to capture.")
                    continue

                # ── capture ───────────────────────────────────────────────
                fb.confirm()
                fb.speak("Scanning.")
                jpeg = _capture_jpeg(cap)
                if jpeg is None:
                    fb.speak("Camera error. Try again.")
                    continue

                fb.speak("Processing, please wait.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable. Check connection.")
                    continue

                text = resp.get("text", "").strip()
                if not text:
                    fb.speak(
                        "No text found on this page. "
                        "Hold the page flat and closer, then say scan again."
                    )
                    continue

                detected_page = resp.get("page")
                skipped       = resp.get("skipped_pages", [])
                word_count    = len(text.split())

                # Page number feedback
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
                                f"Skipped page{'s' if len(missing) > 1 else ''} {labels}."
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

                fb.wait(timeout=5)   # let the page announcement finish

                i      = 0
                paused = False
                pause_announced = False  # speak pause prompt only once per pause

                while i < len(chunks) and not ctx.abort.is_set():

                    if paused:
                        if not pause_announced:
                            fb.speak(
                                "Paused. "
                                "Say yes to add another page. "
                                "Say no or resume gesture to continue reading. "
                                "Say next or gesture next for next page."
                            )
                            pause_announced = True

                        g2 = _wait_for_gesture(ctx, timeout=PAUSE_WAIT_S)

                        if g2 is None:
                            # Re-prompt after timeout
                            pause_announced = False
                            continue

                        if g2 in ("OCR_PAUSE", "PROG_DISCARD"):
                            # Resume reading
                            paused          = False
                            pause_announced = False
                            fb.speak("Resuming.")
                            time.sleep(0.8)

                        elif g2 == "PROG_CONFIRM":
                            # User said "yes" — go scan next page
                            fb.speak(
                                "Okay. Hold the next page to the camera. "
                                "Say scan or gesture next when ready."
                            )
                            i = len(chunks)   # exit reading loop → back to scan loop
                            paused          = False
                            pause_announced = False

                        elif g2 in ("NEXT", "OCR_SCAN"):
                            fb.silence()
                            fb.speak("Next page.")
                            i = len(chunks)   # exit reading loop
                            paused          = False
                            pause_announced = False

                        elif g2 == "OCR_BWD":
                            i               = max(0, i - SKIP_CHUNKS)
                            paused          = False
                            pause_announced = False
                            fb.speak(f"Rewound. Resuming.")
                            time.sleep(0.8)

                        elif g2 == "OCR_FWD":
                            skip            = min(SKIP_CHUNKS, len(chunks) - i)
                            i               = min(i + skip, len(chunks))
                            paused          = False
                            pause_announced = False
                            fb.speak(f"Skipped forward. Resuming.")
                            time.sleep(0.8)

                        elif g2 == "OCR_CLOSE":
                            ctx.abort.set()

                        continue

                    # Speak next chunk
                    fb.speak(chunks[i])
                    wait    = _chunk_duration(chunks[i])
                    i      += 1

                    # Poll gestures while chunk plays
                    deadline = time.time() + wait
                    handled  = False
                    while (time.time() < deadline
                           and not ctx.abort.is_set()
                           and not handled):
                        g2 = _wait_for_gesture(
                            ctx, timeout=min(0.05, deadline - time.time())
                        )
                        if g2 == "OCR_PAUSE":
                            paused          = True
                            pause_announced = False
                            fb.silence()
                            fb.speak("Paused.")
                            handled = True
                        elif g2 == "OCR_FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            fb.silence()
                            fb.speak(f"Skipped {skip}.")
                            handled = True
                        elif g2 == "OCR_BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)
                            fb.silence()
                            fb.speak("Rewinding.")
                            handled = True
                        elif g2 in ("NEXT", "OCR_SCAN"):
                            fb.silence()
                            fb.speak("Next page.")
                            i       = len(chunks)   # exit reading loop
                            handled = True
                        elif g2 == "OCR_CLOSE":
                            fb.silence()
                            ctx.abort.set()
                            handled = True

                # Reading finished (natural end)
                if not ctx.abort.is_set() and i >= len(chunks):
                    page_label = (f"Page {detected_page}" if detected_page
                                  else "Page")
                    fb.speak(
                        f"{page_label} done. "
                        "Say scan or gesture next for the next page. "
                        "Say close or use close gesture to finish."
                    )

        finally:
            cap.release()
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
