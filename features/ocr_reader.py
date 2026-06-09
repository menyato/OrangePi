"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Flow
----
1. Open → guide blind user to point camera at page.
2. NEXT  → capture frame → send to server for PaddleOCR → receive text.
3. Read text aloud sentence by sentence via local TTS (espeak-ng).
4. During reading, OCR-only gestures (forwarded by the state machine):
     EDIT  → pause / resume
     NEXT  → skip forward  ~5 s of text (3 chunks)
     BACK  → skip backward ~5 s of text (3 chunks)
5. After the page finishes → NEXT scans the next page.
6. Page numbers are detected; hub warns if a page was skipped.
7. Session is saved to ~/ocr_sessions/<name>.json on the OrangePi.

Hardcoded OCR gesture meanings (only active while Book Reader is RUNNING):
  EDIT → pause / resume reading
  NEXT → skip 3 chunks forward  OR  scan next page (when idle)
  BACK → skip 3 chunks backward  (Thumb + Index, flick left — new system gesture)
"""

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
SKIP_CHUNKS     = 3             # how many chunks to skip per NEXT/BACK
SESSIONS_DIR    = os.path.expanduser("~/ocr_sessions")
CAMERA_INDICES  = [0, 1, 2]    # try these in order until one opens


# ── helpers ───────────────────────────────────────────────────────────────────

def _open_camera():
    """Return an opened cv2.VideoCapture or None."""
    if not _CV2_OK:
        return None
    for idx in CAMERA_INDICES:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
    return None


def _capture_jpeg(cap) -> "bytes | None":
    """Settle autofocus then grab the sharpest of 3 frames."""
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        time.sleep(0.6)               # autofocus settle
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


def _chunk_text(text: str) -> list:
    """Split text into natural reading chunks of ~CHUNK_TARGET words."""
    # First split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for s in sentences:
        words = s.split()
        if len(words) <= int(CHUNK_TARGET * 1.5):
            if s.strip():
                chunks.append(s.strip())
        else:
            # Long sentence: split on commas / semicolons too
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
    """Estimated seconds for espeak-ng to speak this chunk."""
    return max(0.8, len(chunk.split()) / WORDS_PER_SEC + 0.4)


def _wait_for_gesture(ctx: FeatureContext, timeout: float) -> "str | None":
    """Wait up to `timeout` seconds for a gesture; return name or None."""
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
    """Save session pages to ~/ocr_sessions/<name>.json. Returns path."""
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
        gq  = ctx.gesture_queue
        fb  = ctx.feedback

        # Auto-name the session
        self._session_counter += 1
        session_name = f"Reading session {self._session_counter}"

        pages: list        = []
        last_page: "int | None" = None

        cap = _open_camera()
        if cap is None:
            fb.speak("No camera found. Book reader needs a USB camera.")
            return

        fb.speak(
            "Book reader. "
            "Hold page flat to camera. "
            "Next to capture. "
            "Edit to rename session. "
            "While reading: Thumb Ring Pinky tilt back to pause. "
            "Thumb flick right to skip forward. "
            "Thumb flick left to skip back. "
            "Next for next page."
        )

        try:
            # ── outer scan loop ───────────────────────────────────────────
            while not ctx.abort.is_set():

                # Wait for NEXT (capture) or EDIT (rename session) at top level
                g = _wait_for_gesture(ctx, timeout=60)

                if ctx.abort.is_set():
                    break

                if g is None:
                    # Timeout — remind the user
                    fb.speak("Still waiting. Next to capture, or feature gesture to close.")
                    continue

                if g == "EDIT":
                    self._session_counter += 1
                    session_name = f"Reading session {self._session_counter}"
                    fb.speak(f"Session: {session_name}. Next to capture.")
                    continue

                if g in ("OCR_PAUSE", "OCR_FWD", "OCR_BWD"):
                    fb.speak("Not reading yet. Next to capture first.")
                    continue

                if g != "NEXT":
                    fb.speak("Next to capture page.")
                    continue

                # ── capture ───────────────────────────────────────────────
                fb.speak("Capturing.")
                jpeg = _capture_jpeg(cap)
                if jpeg is None:
                    fb.speak("Camera error. Try again.")
                    continue

                fb.speak("Processing. Please wait.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable. Check connection.")
                    continue

                text = resp.get("text", "").strip()
                if not text:
                    fb.speak("No text found. Move camera closer and try again.")
                    continue

                detected_page = resp.get("page")   # int or None
                skipped       = resp.get("skipped_pages", [])

                # Page-skip warning
                if skipped:
                    labels = ", ".join(str(p) for p in skipped)
                    fb.speak(
                        f"Warning: looks like page{'s' if len(skipped)>1 else ''} "
                        f"{labels} {'were' if len(skipped)>1 else 'was'} skipped."
                    )

                if detected_page is not None:
                    if last_page is not None and detected_page != last_page + 1:
                        # Server didn't catch the skip — hub does a second check
                        missing = list(range(last_page + 1, detected_page))
                        if missing:
                            labels = ", ".join(str(p) for p in missing)
                            fb.speak(f"Skipped page{'s' if len(missing)>1 else ''} {labels}.")
                    last_page = detected_page
                    fb.speak(f"Page {detected_page}.")

                pages.append({"page": detected_page, "text": text,
                               "word_count": len(text.split())})

                # ── reading loop ──────────────────────────────────────────
                chunks = _chunk_text(text)
                if not chunks:
                    fb.speak("Page seems blank.")
                    continue

                fb.speak(
                    f"Reading page. "
                    "Thumb Ring Pinky back to pause. "
                    "Thumb right to skip, Thumb left to rewind. "
                    "Next for next page."
                )
                time.sleep(2.5)

                i       = 0
                paused  = False

                while i < len(chunks) and not ctx.abort.is_set():

                    if paused:
                        fb.silence()
                        g2 = _wait_for_gesture(ctx, timeout=30)
                        if g2 == "OCR_PAUSE":
                            paused = False
                            fb.speak("Resuming.")
                            time.sleep(0.8)
                        elif g2 == "NEXT":
                            fb.silence()
                            fb.speak("Next page. Hold page to camera. Next to capture.")
                            break
                        elif g2 == "OCR_BWD":
                            i = max(0, i - SKIP_CHUNKS)
                            paused = False
                            fb.speak(f"Rewound to chunk {i + 1}. Resuming.")
                            time.sleep(0.8)
                        elif g2 == "OCR_FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i = min(i + skip, len(chunks))
                            paused = False
                            fb.speak(f"Skipped {skip}. Resuming.")
                            time.sleep(0.8)
                        continue

                    # Speak chunk
                    fb.speak(chunks[i])
                    wait = _chunk_duration(chunks[i])
                    i += 1

                    # Poll for gestures while chunk plays
                    deadline = time.time() + wait
                    handled  = False
                    while time.time() < deadline and not ctx.abort.is_set() and not handled:
                        g2 = _wait_for_gesture(ctx,
                                               timeout=min(0.05, deadline - time.time()))
                        if g2 == "OCR_PAUSE":
                            paused = True
                            fb.silence()
                            fb.speak("Paused. Same gesture to resume.")
                            handled = True
                        elif g2 == "OCR_FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i = min(i + skip, len(chunks))
                            fb.silence()
                            fb.speak(f"Skipped {skip}.")
                            handled = True
                        elif g2 == "OCR_BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)
                            fb.silence()
                            fb.speak("Rewinding.")
                            handled = True
                        elif g2 == "NEXT":
                            fb.silence()
                            fb.speak("Next page. Hold page to camera. Next to capture.")
                            i = len(chunks)  # break out of reading loop
                            handled = True

                # Reading finished (or user broke out)
                if not ctx.abort.is_set() and i >= len(chunks):
                    fb.speak(
                        "Page finished. "
                        "Next to scan another page, or close Book reader to exit."
                    )

        finally:
            cap.release()
            if pages:
                try:
                    path = _save_session(session_name, pages)
                    fb.speak(f"Session {session_name} saved.")
                    print(f"[OCR] Session saved to {path}")
                except Exception as e:
                    print(f"[OCR] Save error: {e}")
