"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Flow
----
1. Open  → "Say scan to capture."  Voice thread starts (mc.listen).

2. Scan  → voice "scan" or NEXT gesture:
           "Hold steady. Capturing." → mc.capture_best_frame_local() (4 s)
           → server EasyOCR → detect page number.
           Announce: "Processed. Page 3. 45 words."
                  or "Processed. No page number found. 45 words."

3. First scan only → "Say a name for this session." → mc.listen() → save name.
   "Session: my history book. Reading now."

4. Read  → chunks spoken via TTS.  Voice thread OFF during reading.
           Gestures only:
             OCR_PAUSE  (Thumb+Ring+Pinky tilt-back) → pause
             OCR_FWD    (Thumb flick right)           → skip 3 chunks
             OCR_BWD    (Thumb flick left)            → rewind 3 chunks
             NEXT / OCR_SCAN gesture                  → skip to next page

5. Paused → voice thread ON.
           Say "scan"  → go capture another page (add to session)
           Say "yes"   → resume reading
           Say "no"    → resume reading
           Say "next"  → jump to next already-scanned page
           Say "close" → save and exit
           OCR_PAUSE gesture also resumes.

6. All pages of current session read:
           "All pages read. Say scan to add more, or close to finish."

7. Close → sort pages by page number → warn about gaps → save session.

ALSA note
---------
The Logitech headset uses the same ALSA device for output (aplay/TTS) and input
(sounddevice/mic).  Opening the mic while aplay is still active causes:
  "Error opening InputStream: Device unavailable [PaErrorCode -9985]"
Fix: call fb.wait() to block until TTS finishes, then sleep 1 s inside _voice_on()
before sounddevice opens the stream — by then ALSA is free.
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
IDLE_REMIND_S  = 60

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
        try:
            from features.money_recognition import MoneyRecognition
            if MoneyRecognition._models_ready:
                _mc_ready = True
                return True
        except (ImportError, AttributeError):
            pass
        try:
            mc.init_cues()
            mc.auto_detect_all()
            fb.speak("Loading voice model. Please wait about one minute.")
            mc.load_models()   # initialises vad + Whisper — must be called before mc.listen()
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
            fb.confirm()
            return jpeg
    jpeg = _capture_jpeg_cv2()
    if jpeg is not None:
        fb.confirm()
    return jpeg


# ── voice worker ──────────────────────────────────────────────────────────────

_SCAN_W  = {"scan", "capture", "take", "photo"}
_NEXT_W  = {"next", "skip", "forward"}
_CLOSE_W = {"close", "stop", "quit", "finish", "exit"}
_YES_W   = {"yes", "yeah", "resume", "continue"}
_NO_W    = {"no", "nope", "back"}


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


# ── gesture → command ─────────────────────────────────────────────────────────

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


# ── queue / timing helpers ────────────────────────────────────────────────────

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


# ── page ordering / session helpers ──────────────────────────────────────────

def _reading_order(pages: dict) -> list:
    """Numbered pages first (ascending), then unnumbered in capture order."""
    numbered = sorted([k for k in pages if k[0] == "num"], key=lambda k: k[1])
    indexed  = sorted([k for k in pages if k[0] == "idx"], key=lambda k: k[1])
    return numbered + indexed


def _next_key(pages: dict, current: tuple) -> "tuple | None":
    order = _reading_order(pages)
    try:
        pos = order.index(current)
        return order[pos + 1] if pos + 1 < len(order) else None
    except ValueError:
        return None


def _page_label(page_data: dict) -> str:
    p = page_data.get("page")
    return f"page {p}" if p is not None else "unlabeled page"


def _missing_pages(pages: dict) -> list:
    nums = sorted(k[1] for k in pages if k[0] == "num")
    if len(nums) < 2:
        return []
    missing = []
    for a, b in zip(nums, nums[1:]):
        missing.extend(range(a + 1, b))
    return missing


def _save_session(name: str, pages_list: list) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip().replace(' ', '_')
    if not safe:
        safe = f"session_{int(time.time())}"
    path = os.path.join(SESSIONS_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "pages": pages_list}, f, ensure_ascii=False, indent=2)
    return path


# ── feature ───────────────────────────────────────────────────────────────────

class OCRReader(Feature):
    name  = "ocr"
    title = "Book reader"

    def __init__(self):
        self._session_counter = 0

    def run(self, ctx: FeatureContext) -> None:  # noqa: C901
        gq = ctx.gesture_queue
        fb = ctx.feedback

        if gq:
            _drain(gq)

        mc_ok = _ensure_mc(fb)

        self._session_counter += 1
        session_name: "str | None" = None
        pages:        dict         = {}   # ("num", N) or ("idx", N) → page_data
        page_idx      = 0                 # counter for unnumbered pages

        # ── voice thread management ───────────────────────────────────────────
        voice_q   = queue.Queue()
        _cur_stop = [threading.Event()]
        _cur_stop[0].set()

        def _voice_on() -> None:
            """Start a new listen thread.  Delays 1 s so aplay releases ALSA first."""
            if not mc_ok:
                return
            _cur_stop[0].set()           # stop previous thread
            stop = threading.Event()
            _cur_stop[0] = stop
            _drain(voice_q)

            def _delayed():
                time.sleep(1.0)          # wait for aplay to free the ALSA device
                if not stop.is_set():
                    _voice_worker(ctx.abort, voice_q, stop)

            threading.Thread(target=_delayed, daemon=True, name="OCRVoice").start()

        def _voice_off() -> None:
            _cur_stop[0].set()
            _drain(voice_q)

        def _get_cmd(ges_timeout: float = 0.1) -> "str | None":
            try:
                return voice_q.get_nowait()
            except queue.Empty:
                pass
            g = _wait_gesture(ctx, timeout=ges_timeout)
            return _GMAP.get(g) if g else None

        # ── opening ───────────────────────────────────────────────────────────
        fb.speak("Book reader. Say scan to capture, or close to exit.")
        _voice_on()

        try:
            idle_t = time.time() + IDLE_REMIND_S

            # ── outer scan-wait loop ──────────────────────────────────────────
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
                        fb.speak("Say scan to capture a page, or close to exit.")
                        idle_t = time.time() + IDLE_REMIND_S
                    continue
                else:
                    continue

                # ── CAPTURE ───────────────────────────────────────────────────
                _voice_off()
                idle_t = time.time() + IDLE_REMIND_S

                fb.speak("Hold the page flat and steady. Capturing now.")
                jpeg = _capture_frame(fb)

                if jpeg is None:
                    fb.speak("Camera error. Check USB camera.")
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                fb.speak("Processing, please wait.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable.")
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                text_body = resp.get("text", "").strip()
                if not text_body:
                    fb.speak("No text found. Try better lighting and say scan again.")
                    fb.wait(timeout=6)
                    _voice_on()
                    continue

                detected_page = resp.get("page")
                word_count    = len(text_body.split())

                if detected_page is not None:
                    key = ("num", detected_page)
                    if key in pages:
                        fb.speak(f"Page {detected_page} rescanned. Replacing.")
                else:
                    key = ("idx", page_idx)
                    page_idx += 1

                pages[key] = {
                    "page":       detected_page,
                    "text":       text_body,
                    "word_count": word_count,
                }

                if detected_page is not None:
                    fb.speak(f"Processed. Page {detected_page}. {word_count} words.")
                else:
                    fb.speak(f"Processed. No page number found. {word_count} words.")

                # ── NAME SESSION (first scan only) ────────────────────────────
                if session_name is None:
                    fb.wait(timeout=6)   # wait for "Processed..." to finish
                    if mc_ok:
                        fb.speak("Say a name for this reading session.")
                        fb.wait(timeout=5)
                        try:
                            vtxt, _ = mc.listen()
                            if vtxt:
                                skip_w = {"skip", "done", "no", "cancel", "default"}
                                clean  = vtxt.strip()
                                if not (set(re.findall(r"[a-z]+", clean.lower())) & skip_w):
                                    session_name = clean
                        except Exception:
                            pass
                    if not session_name:
                        session_name = f"reading_{self._session_counter}"
                    fb.speak(f"Session: {session_name}. Reading now.")
                    fb.wait(timeout=5)
                else:
                    fb.wait(timeout=5)
                    fb.speak("Reading now.")

                # ── READING LOOP ──────────────────────────────────────────────
                # Start with the freshly-scanned page; continue into later pages
                # in reading order if they were already scanned.
                current_key = key
                go_scan     = False   # set True when user says "scan" while paused

                while current_key is not None and not ctx.abort.is_set() and not go_scan:
                    pdata  = pages[current_key]
                    chunks = _chunk_text(pdata["text"])

                    if not chunks:
                        fb.speak("Page appears blank.")
                        break

                    i               = 0
                    paused          = False
                    pause_announced = False

                    while i < len(chunks) and not ctx.abort.is_set():

                        # ── PAUSED ────────────────────────────────────────────
                        if paused:
                            if not pause_announced:
                                fb.speak(
                                    "Paused. "
                                    "Say scan to add a page. "
                                    "Say yes to resume. "
                                    "Say next for the next page. "
                                    "Say close to save and exit."
                                )
                                pause_announced = True
                                fb.wait(timeout=8)   # wait for TTS before opening mic
                                _voice_on()

                            cmd = _get_cmd(ges_timeout=0.4)
                            if cmd is None:
                                continue
                            _voice_off()

                            if cmd in ("PAUSE", "YES", "NO"):
                                paused = False; pause_announced = False
                                fb.speak("Resuming.")

                            elif cmd == "SCAN":
                                paused = False; pause_announced = False
                                i       = len(chunks)   # stop reading current page
                                go_scan = True          # signal outer loop to re-scan

                            elif cmd == "NEXT":
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

                        # ── READ CHUNK ────────────────────────────────────────
                        fb.speak(chunks[i])
                        dur = _chunk_dur(chunks[i])
                        i  += 1

                        deadline = time.time() + dur
                        while time.time() < deadline and not ctx.abort.is_set():
                            g    = _wait_gesture(ctx, timeout=min(0.05,
                                                                   deadline - time.time()))
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

                    if go_scan:
                        break

                    # ── ADVANCE TO NEXT PAGE ──────────────────────────────────
                    if i >= len(chunks) and not ctx.abort.is_set():
                        nk = _next_key(pages, current_key)
                        if nk is not None:
                            label = _page_label(pages[nk])
                            fb.speak(f"Page done. Next is {label}. Reading.")
                            current_key = nk
                        else:
                            fb.speak(
                                "All pages read. "
                                "Say scan to add more, or close to save and exit."
                            )
                            fb.wait(timeout=8)
                            current_key = None

                # Back to scan-wait (also reached when go_scan=True)
                if not ctx.abort.is_set():
                    _voice_on()

        finally:
            _voice_off()

            if pages and session_name:
                order      = _reading_order(pages)
                pages_list = [pages[k] for k in order]
                missing    = _missing_pages(pages)

                if missing:
                    labels = ", ".join(str(p) for p in missing[:5])
                    more   = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
                    fb.speak(f"Warning: pages {labels}{more} were not scanned.")

                try:
                    path        = _save_session(session_name, pages_list)
                    total_words = sum(p["word_count"] for p in pages_list)
                    fb.speak(
                        f"Saved as {session_name}. "
                        f"{len(pages_list)} page{'s' if len(pages_list) > 1 else ''}, "
                        f"{total_words} words."
                    )
                    print(f"[OCR] Session saved → {path}")
                except Exception as e:
                    print(f"[OCR] Save error: {e}")
