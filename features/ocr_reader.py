"""
features/ocr_reader.py — Book Reader feature for the smart glove.

State transitions (announced via TTS)
--------------------------------------
  IDLE     → "Book reader. Say scan, load <name>, or close."
  CAPTURE  → "Hold steady. Capturing now."  haptic on success
  PROCESS  → "Scanning complete. Processing text."
  RESULT   → "Page N. X words."
           or "No page number. X words. Say the number or skip."
  NAME     → "Say a name for this session."  (first scan only)
  READ     → "Session <name>. Reading now."
  PAUSE    → "Paused. Scan, yes=resume, next, close."
  DONE     → "All pages read. Scan for more or close."
  CLOSE    → gap-warning → "Saved as <name>. N pages, X words."

Gesture controls (during reading only — voice OFF, no cue sounds)
-----------------------------------------------------------------
  OCR_PAUSE  (Thumb+Ring+Pinky tilt-back)  → pause / resume
  OCR_FWD    (Thumb flick right)            → skip 3 chunks forward
  OCR_BWD    (Thumb flick left)             → rewind 3 chunks
  NEXT / OCR_SCAN gesture                   → jump to next page scan

Voice controls (mc.listen / Whisper — ON during idle, paused, and after page)
-----------------------------------------------------------------
  "scan"       → capture a new page
  "load <name>"→ load a previously saved session and re-read it
  "next"       → jump to next already-scanned page
  "close"      → save and exit
  (paused) "yes" / "resume"  → resume reading
  (paused) "scan"            → break out to capture a new page
  (paused) "next"            → skip to next page
  (paused) "close"           → save and exit
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

# ── mc initialisation ─────────────────────────────────────────────────────────
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
            mc.load_models()
            _mc_ready = True
            return True
        except Exception as e:
            print(f"[OCR] mc init error: {e}")
            return False


# ── cv2 fallback camera ───────────────────────────────────────────────────────

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


# ── voice helpers ─────────────────────────────────────────────────────────────

_SCAN_W  = {"scan", "capture", "take", "photo"}
_LOAD_W  = {"load", "recall"}
_NEXT_W  = {"next", "skip", "forward"}
_CLOSE_W = {"close", "stop", "quit", "finish", "exit"}
_YES_W   = {"yes", "yeah", "resume", "continue", "ok"}
_NO_W    = {"no", "nope", "back"}
_SKIP_W  = {"skip", "none", "no", "cancel", "default"}

# Words used to say page numbers
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _parse_number(text: str) -> "int | None":
    """Extract first integer from spoken text (digits or words)."""
    digits = re.findall(r'\b(\d+)\b', text)
    if digits:
        n = int(digits[0])
        if 1 <= n <= 9999:
            return n
    for w in re.findall(r'[a-z]+', text.lower()):
        n = _WORD_NUMS.get(w)
        if n and n > 0:
            return n
    return None


def _classify(text: str) -> "str | None":
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _LOAD_W:   return "LOAD"
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
        else:
            print(f"[OCR] voice: {text!r} → (unrecognised)")


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

def _reading_order(pages: dict) -> list:
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


def _find_session(query: str) -> "dict | None":
    """Find a saved session by name (exact then partial match)."""
    if not os.path.isdir(SESSIONS_DIR):
        return None
    q = query.lower().strip()
    candidates = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            candidates.append(data)
        except Exception:
            pass
    # Exact match first
    for d in candidates:
        if d.get("name", "").lower() == q:
            return d
    # Partial match
    for d in candidates:
        name = d.get("name", "").lower()
        if q in name or name in q:
            return d
    return None


def _pages_from_session(data: dict) -> "tuple[dict, int]":
    """Convert saved session data to pages dict. Returns (pages, next_idx)."""
    pages: dict = {}
    idx = 0
    for p in data.get("pages", []):
        pnum = p.get("page")
        if pnum is not None:
            pages[("num", pnum)] = p
        else:
            pages[("idx", idx)] = p
            idx += 1
    return pages, idx


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
        pages:        dict         = {}
        page_idx      = 0

        # ── voice thread ──────────────────────────────────────────────────────
        voice_q   = queue.Queue()
        _cur_stop = [threading.Event()]
        _cur_stop[0].set()

        def _voice_on() -> None:
            if not mc_ok:
                return
            _cur_stop[0].set()
            stop = threading.Event()
            _cur_stop[0] = stop
            _drain(voice_q)

            def _delayed():
                # Wait for aplay to release ALSA before sounddevice opens the mic
                time.sleep(1.0)
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
        fb.speak(
            "Book reader. "
            "Say scan to capture a page, "
            "load to open a saved session, "
            "or close to exit."
        )
        _voice_on()

        # ── nested reading loop ───────────────────────────────────────────────
        def _read_pages(start_key: tuple) -> str:
            """Read pages starting at start_key.

            Returns:
              "scan"  — user asked to scan a new page (from pause)
              "close" — user asked to close
              "done"  — finished reading all pages
            """
            current_key: "tuple | None" = start_key

            while current_key is not None and not ctx.abort.is_set():
                pdata  = pages[current_key]
                chunks = _chunk_text(pdata["text"])

                if not chunks:
                    fb.speak("Page appears blank.")
                    nk = _next_key(pages, current_key)
                    current_key = nk
                    continue

                i               = 0
                paused          = False
                pause_announced = False

                while i < len(chunks) and not ctx.abort.is_set():

                    # ── PAUSED ────────────────────────────────────────────────
                    if paused:
                        if not pause_announced:
                            fb.speak(
                                "Paused. "
                                "Say scan to add a page. "
                                "Say yes to resume. "
                                "Say next for the next page. "
                                "Say close to finish. "
                                "Or use pause gesture to resume."
                            )
                            pause_announced = True
                            fb.wait(timeout=10)
                            _voice_on()

                        cmd = _get_cmd(ges_timeout=0.4)
                        if cmd is None:
                            continue

                        if cmd.startswith("TEXT:"):
                            # Raw text in pause — ignore (not a command)
                            continue

                        _voice_off()

                        if cmd in ("PAUSE", "YES", "NO"):
                            paused = False; pause_announced = False
                            fb.speak("Resuming.")

                        elif cmd == "SCAN":
                            paused = False; pause_announced = False
                            return "scan"

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
                            return "close"

                        continue

                    # ── READ CHUNK ────────────────────────────────────────────
                    fb.speak(chunks[i])
                    dur = _chunk_dur(chunks[i])
                    i  += 1

                    deadline = time.time() + dur
                    while time.time() < deadline and not ctx.abort.is_set():
                        g    = _wait_gesture(ctx,
                                             timeout=min(0.05, deadline - time.time()))
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
                            fb.silence(); ctx.abort.set(); return "close"

                # Advance to next page
                if not ctx.abort.is_set() and i >= len(chunks):
                    nk = _next_key(pages, current_key)
                    if nk is not None:
                        label = _page_label(pages[nk])
                        fb.speak(f"Page done. Next is {label}. Reading.")
                        current_key = nk
                    else:
                        fb.speak(
                            "All pages read. "
                            "Say scan to add more pages, "
                            "or say close to save and exit."
                        )
                        fb.wait(timeout=8)
                        current_key = None

            return "done"

        # ── outer scan-wait loop ──────────────────────────────────────────────
        try:
            idle_t = time.time() + IDLE_REMIND_S

            while not ctx.abort.is_set():
                cmd = _get_cmd(ges_timeout=0.4)

                if ctx.abort.is_set():
                    break
                if cmd == "CLOSE":
                    break
                if cmd is None:
                    if time.time() >= idle_t:
                        fb.speak("Say scan to capture, load for a saved session, or close.")
                        idle_t = time.time() + IDLE_REMIND_S
                    continue

                # ── LOAD saved session ─────────────────────────────────────
                if cmd == "LOAD":
                    _voice_off()
                    raw_text = ""

                    if not raw_text:
                        fb.speak("Say the session name.")
                        fb.wait(timeout=5)
                        if mc_ok:
                            try:
                                raw_text, _ = mc.listen()
                            except Exception:
                                raw_text = ""

                    if raw_text:
                        sdata = _find_session(raw_text)
                        if sdata:
                            loaded_name  = sdata["name"]
                            loaded_pages, loaded_idx = _pages_from_session(sdata)
                            if loaded_pages:
                                session_name = session_name or loaded_name
                                pages.update(loaded_pages)
                                page_idx = max(page_idx, loaded_idx)
                                fb.speak(
                                    f"Loaded {loaded_name}. "
                                    f"{len(loaded_pages)} pages. Reading now."
                                )
                                fb.wait(timeout=5)
                                order      = _reading_order(pages)
                                start_key  = order[0] if order else None
                                if start_key:
                                    result = _read_pages(start_key)
                                    if result == "close":
                                        break
                                    _voice_on()
                                    idle_t = time.time() + IDLE_REMIND_S
                                continue
                            else:
                                fb.speak(f"Session {loaded_name} has no pages.")
                        else:
                            fb.speak(f"No session found matching {raw_text!r}.")
                    _voice_on()
                    continue

                if cmd not in ("SCAN", "NEXT"):
                    continue

                # ── CAPTURE ────────────────────────────────────────────────────
                _voice_off()
                idle_t = time.time() + IDLE_REMIND_S

                fb.speak("Hold the page flat and steady. Capturing now.")
                jpeg = _capture_frame(fb)

                if jpeg is None:
                    fb.speak("Camera error. Check USB camera.")
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                fb.speak("Scanning complete. Processing text.")
                resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

                if resp is None:
                    fb.speak("Server not reachable.")
                    fb.wait(timeout=4)
                    _voice_on()
                    continue

                text_body = resp.get("text", "").strip()
                if not text_body:
                    fb.speak(
                        "No text found. "
                        "Hold page 30 to 40 centimetres from camera "
                        "with good lighting, then scan again."
                    )
                    fb.wait(timeout=8)
                    _voice_on()
                    continue

                detected_page = resp.get("page")
                word_count    = len(text_body.split())

                # ── ANNOUNCE RESULT ───────────────────────────────────────────
                if detected_page is not None:
                    fb.speak(f"Page {detected_page}. {word_count} words.")
                else:
                    fb.speak(f"No page number. {word_count} words.")
                    fb.wait(timeout=5)
                    # Ask user to provide the page number
                    if mc_ok:
                        fb.speak("Say the page number, or say skip.")
                        fb.wait(timeout=5)
                        try:
                            vtxt, _ = mc.listen()
                            if vtxt:
                                skip = set(re.findall(r"[a-z]+", vtxt.lower())) & _SKIP_W
                                if not skip:
                                    num = _parse_number(vtxt)
                                    if num is not None:
                                        detected_page = num
                                        fb.speak(f"Page {detected_page}.")
                                        fb.wait(timeout=3)
                        except Exception:
                            pass

                # ── STORE PAGE ────────────────────────────────────────────────
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

                # ── NAME SESSION (first scan) ──────────────────────────────────
                if session_name is None:
                    fb.wait(timeout=4)
                    if mc_ok:
                        fb.speak("Say a name for this reading session.")
                        fb.wait(timeout=5)
                        try:
                            vtxt, _ = mc.listen()
                            if vtxt:
                                skip = set(re.findall(r"[a-z]+", vtxt.lower())) & _SKIP_W
                                if not skip:
                                    session_name = vtxt.strip()
                        except Exception:
                            pass
                    if not session_name:
                        session_name = f"reading_{self._session_counter}"
                    fb.speak(f"Session: {session_name}. Reading now.")
                    fb.wait(timeout=5)
                else:
                    fb.wait(timeout=4)
                    fb.speak("Reading now.")
                    fb.wait(timeout=3)

                # ── READ ──────────────────────────────────────────────────────
                result = _read_pages(key)
                if result == "close":
                    break
                # result == "scan" or "done" → loop back to scan-wait
                _voice_on()
                idle_t = time.time() + IDLE_REMIND_S

        finally:
            _voice_off()

            if pages and session_name:
                order      = _reading_order(pages)
                pages_list = [pages[k] for k in order]
                missing    = _missing_pages(pages)

                if missing:
                    labels = ", ".join(str(p) for p in missing[:5])
                    more   = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
                    fb.speak(f"Warning: pages {labels}{more} not scanned.")

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
