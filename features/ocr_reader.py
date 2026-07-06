"""
features/ocr_reader.py — Book Reader feature for the smart glove.

Flow (scan and read are separate — reading never interrupts scanning)
----------------------------------------------------------------------
  IDLE     → "Book reader. Say scan, load <name>, or close."
  SCAN     → batch: capture page → page number → "scan for next page,
             or done to finish" → repeat until done/save
  NAME     → "Say a name for this book."  (first save only)
  SAVED    → "Saved as <name>. Say read to listen now, scan to add
             more pages, or close to exit."
  LOAD     → "Loaded <name>. Say scan to add new pages first, or read
             to begin."  then reads all pages in page-number order
  READ     → continuous, gesture-controlled; pause menu offers scan /
             new book / load / add / next / close
  DONE     → "All pages read."
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
  "add <name>" → load a saved session to add pages to it
  "new book"   → save the current book and start a fresh one
  "next"       → jump to next already-scanned page
  "close"      → save and exit
  (paused) "yes" / "resume"  → resume reading
  (paused) "scan"            → break out to capture a new page
  (paused) "new book"        → save this book, start a fresh one
  (paused) "load" / "add"    → switch to a different saved book
  (paused) "next"            → skip to next page
  (paused) "close"           → save and exit
"""

import base64
import contextlib
import difflib
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
# Only the newest MAX_SAVED_SESSIONS books are kept on disk (older ones are
# deleted at feature start) and all of them are announced newest-first —
# reading out a long history of book names is impossible to track by ear.
MAX_SAVED_SESSIONS   = 3
NAMES_ANNOUNCE_LIMIT = 3
IDLE_REMIND_S  = 60

# Domain vocabulary passed to every mc.listen()/voice_listen_loop() call in
# this file — transcribe()'s default initial_prompt/hotwords are tuned for
# money recognition's currency-scanning commands and actively bias Whisper's
# decoder AWAY from words that never appear in it, like "load"/"add"/book
# names/page numbers. This is the actual reason "load" was unreliable here.
OCR_INITIAL_PROMPT = (
    "Book reader voice commands. Words used: scan, load, add, close, next, "
    "skip, cancel, yes, no, page, book, session, name. "
    "Numbers: one, two, three, four, five, six, seven, eight, nine, ten, "
    "twenty, fifty, one hundred."
)
OCR_HOTWORDS = "scan load add close next skip cancel yes no page book session"

# ── server TTS helper ─────────────────────────────────────────────────────────

def _srvspeak(ctx, fb, text: str) -> None:
    """Speak text using server Windows voice (SAPI/Piper).

    Sends {"feature": "tts", "text": text} to the server, receives
    {"audio": base64_wav} back, and plays the WAV via fb.play_raw().
    Falls back to local espeak-ng (fb.speak) if:
      - server is unreachable
      - server returns no audio
      - server returns an error response (resp["ok"] is not True)
    """
    try:
        resp = ctx.link.send("tts", {"text": text})
        if resp and resp.get("ok") and resp.get("audio"):
            fb.play_raw(base64.b64decode(resp["audio"]))
            return
    except Exception as e:
        print(f"[OCR] server TTS failed: {e}")
    fb.speak(text)


def _play_wav_b64(fb, b64: str) -> float:
    """Decode a base64 WAV string and play it; returns duration in seconds."""
    try:
        return fb.play_raw(base64.b64decode(b64))
    except Exception as e:
        print(f"[OCR] play_wav_b64: {e}")
        return 0.0


# ── mc initialisation ─────────────────────────────────────────────────────────
_mc_ready = False
_mc_lock  = threading.Lock()


def _ensure_mc(fb, link=None) -> bool:
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
            t0 = time.time()
            mc.init_cues()
            mc.auto_detect_all()
            fb.speak("Loading voice model. Please wait about one minute.")
            mc.load_models()
            _mc_ready = True
            if link is not None:
                import metrics
                metrics.report_load(link, "ocr", (time.time() - t0) * 1000)
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

_SCAN_W  = {"scan", "capture", "take", "photo", "scam", "skan", "skun", "scanned"}
_LOAD_W  = {"load", "recall", "open", "read"}
_ADD_W   = {"add", "append", "insert"}
_NEW_W   = {"new", "fresh"}
_NEXT_W  = {"next", "skip", "forward"}
_CLOSE_W = {"close", "stop", "quit", "finish", "exit", "done", "save", "saved"}
_YES_W   = {"yes", "yeah", "resume", "continue", "ok", "yep", "sure"}
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


# Priority order for both exact and fuzzy matching -- first hit wins, same
# order a human would disambiguate a multi-intent utterance in. NEW comes
# before LOAD/ADD so "load a new book" / "add a new book" mean "start a
# fresh book", not "open a saved one".
_CMD_VOCAB = [
    ("NEW",   _NEW_W),
    ("LOAD",  _LOAD_W),
    ("ADD",   _ADD_W),
    ("SCAN",  _SCAN_W),
    ("NEXT",  _NEXT_W),
    ("YES",   _YES_W),
    ("NO",    _NO_W),
    ("CLOSE", _CLOSE_W),
]


def _fuzzy_hit(word: str, vocab: set) -> bool:
    """True if `word` is a close spelling of some vocabulary word.

    Both sides must be at least 4 letters: short strings score deceptively
    high on difflib's ratio — "one" vs "no" is 0.8 and "one" vs "none" is
    0.86, which live-classified a user's answer of "one" as NO/skip (their
    session got silently renamed to the reading_N fallback). Short command
    words ("no", "ok", "yes", "add") therefore match exactly only.
    """
    if len(word) < 4:
        return False
    candidates = [v for v in vocab if len(v) >= 4]
    return bool(difflib.get_close_matches(word, candidates, n=1, cutoff=0.75))


def _classify(text: str) -> "str | None":
    words = set(re.findall(r"[a-z]+", text.lower()))
    for cmd, vocab in _CMD_VOCAB:
        if words & vocab:
            return cmd

    # ── fuzzy fallback ───────────────────────────────────────────────────
    # A blind user gets no visual confirmation of what Whisper actually
    # heard, so a mispronounced or slightly-misheard command word (e.g.
    # "lowd" for "load", "closs" for "close", "sken" for "scan") would
    # otherwise be reported as "(unrecognised)" with no way for the user
    # to know why -- they'd just have to guess and try again. Accept a
    # close spelling match per word instead of requiring an exact one.
    for cmd, vocab in _CMD_VOCAB:
        for w in words:
            if _fuzzy_hit(w, vocab):
                return cmd
    return None


def _is_skip(text: str) -> bool:
    """Same exact-then-fuzzy leniency as _classify(), for the separate
    'say skip' vocabulary used by the page-number/session-name prompts."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _SKIP_W:
        return True
    return any(_fuzzy_hit(w, _SKIP_W) for w in words)


def _voice_worker(abort: threading.Event,
                  voice_q: queue.Queue,
                  stop_ev: threading.Event) -> None:
    """Voice worker — delegates to the shared mc.voice_listen_loop()."""
    if _MC_OK:
        mc.voice_listen_loop(voice_q, stop_ev, abort,
                             initial_prompt=OCR_INITIAL_PROMPT, hotwords=OCR_HOTWORDS)
    else:
        while not stop_ev.is_set() and not abort.is_set():
            time.sleep(1.0)


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


def _session_files_by_mtime() -> list:
    """Full paths of saved session files, newest first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    paths = [os.path.join(SESSIONS_DIR, f)
             for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def _prune_sessions(keep: int = MAX_SAVED_SESSIONS) -> None:
    """Delete all but the `keep` most recently saved books."""
    for path in _session_files_by_mtime()[keep:]:
        try:
            os.remove(path)
            print(f"[OCR] Pruned old session → {path}")
        except Exception as e:
            print(f"[OCR] Prune error for {path}: {e}")


def _saved_session_names() -> list:
    """Names of saved sessions, newest first — used to announce choices
    before asking the user to pick one by voice (LOAD and ADD)."""
    names = []
    for path in _session_files_by_mtime():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            names.append(data.get("name", os.path.basename(path)[:-5]))
        except Exception:
            pass
    return names


# Words that carry no naming information when they show up in a spoken
# answer to "say the book name" -- e.g. the user says "load a book" or
# "the harry potter book" instead of just the title.
_NAME_FILLER_W = _LOAD_W | _ADD_W | {"a", "the", "book", "please", "session"}


def _strip_filler(query: str) -> str:
    words = [w for w in re.findall(r"[a-z']+", query.lower()) if w not in _NAME_FILLER_W]
    return " ".join(words)


def _find_session(query: str) -> "dict | None":
    """Find a saved session by name.

    Tries, in order: exact match, substring match (either direction), then
    the same two again with filler words ("load"/"a"/"book"/...) stripped
    out, then a fuzzy spelling match. The filler-stripping and fuzzy pass
    give a blind user real room for error -- they get no visual feedback on
    what was transcribed, so "load a book" or a slightly misheard title
    ("herry potter") should still resolve instead of failing outright.
    """
    if not os.path.isdir(SESSIONS_DIR):
        return None
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
    if not candidates:
        return None

    q       = query.lower().strip()
    cleaned = _strip_filler(query)

    for probe in (q, cleaned):
        if not probe:
            continue
        for d in candidates:
            if d.get("name", "").lower() == probe:
                return d
        for d in candidates:
            name = d.get("name", "").lower()
            if probe in name or name in probe:
                return d

    names_lower = [d.get("name", "").lower() for d in candidates]
    close = difflib.get_close_matches(cleaned or q, names_lower, n=1, cutoff=0.6)
    if close:
        return candidates[names_lower.index(close[0])]
    return None


def _coerce_page_num(value) -> "int | None":
    """Normalize a page number to int (or None). Keys must be uniformly
    typed: a "2" (str) next to a 3 (int) would crash _reading_order()'s
    sort — and ordering is what guarantees pages are read in numeric
    order no matter what order they were scanned or saved in."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pages_from_session(data: dict) -> "tuple[dict, int]":
    """Convert saved session data to pages dict. Returns (pages, next_idx)."""
    pages: dict = {}
    idx = 0
    for p in data.get("pages", []):
        pnum = _coerce_page_num(p.get("page"))
        p["page"] = pnum
        if pnum is not None:
            pages[("num", pnum)] = p
        else:
            pages[("idx", idx)] = p
            idx += 1
    return pages, idx


# ── built-in sample books ─────────────────────────────────────────────────────
# Installed on first run only (no saved books yet) so reading, pause/resume,
# and skip gestures can be tested immediately without scanning anything.
# Names are single common words that Whisper hears reliably. Each page is
# ~120 words (~12 chunks ≈ a minute of speech) — long enough to practise
# pausing and skipping mid-page. They are saved oldest-first, so as the
# user saves real books the samples are pruned away first.

_SAMPLE_BOOKS = [
    ("garden", [
        "The garden woke up slowly in the morning light. Dew covered every "
        "leaf and every petal, and the air smelled of wet earth. A small "
        "bird landed on the fence and looked around before singing its "
        "first song of the day. The tomatoes were still green, but the "
        "beans had grown taller than the sticks that held them. Near the "
        "wall, the old rose bush opened one new flower, deep red and "
        "wide. An ant climbed along the handle of a forgotten spade. "
        "Somewhere behind the hedge, a cat moved without a sound. The "
        "gardener arrived with a cup of tea in one hand, stood by the "
        "gate, and smiled at all of it before starting the day's work.",
        "By noon the sun stood high over the garden and the shadows had "
        "pulled back under the trees. Bees moved from flower to flower "
        "with a steady hum, never resting long in one place. The gardener "
        "pulled weeds from between the carrots and threw them into a "
        "basket. Water from the green hose made small rainbows above the "
        "lettuce. A butterfly with orange wings settled on the edge of "
        "the water barrel and drank. The cat from the morning now slept "
        "openly on the warm stone path, one ear turning at every sound. "
        "When the church bell rang far away, the gardener sat in the "
        "shade, ate bread and cheese, and watched the beans climb.",
    ]),
    ("ocean", [
        "The ocean began where the white sand ended, and it went on until "
        "it touched the sky. Waves arrived one after another, each with "
        "its own quiet thunder, each leaving a line of foam that sank "
        "into the sand. A fishing boat with a blue hull rocked near the "
        "pier, ropes creaking softly. Seagulls hung in the wind above it, "
        "hardly moving their wings. A child ran along the waterline with "
        "a red bucket, chasing the foam and then running from it. Far "
        "out, where the water turned dark, something silver jumped and "
        "vanished. The old man who sold shells from a wooden table said "
        "it was a good sign, and nobody argued with him.",
        "In the evening the ocean changed its voice. The waves came "
        "slower and lower, as if they were tired from the long day. The "
        "sun dropped toward the water and laid a road of gold across it, "
        "from the beach to the very edge of the world. The fishing boat "
        "came home with its lights on, and the men tied it to the pier "
        "with hands that knew every knot in the dark. Someone lit a small "
        "fire on the sand, and the smell of grilled fish drifted along "
        "the beach. The child with the red bucket slept on a towel, "
        "holding one perfect shell. The ocean kept talking quietly, the "
        "way it always does, to anyone who stays and listens.",
    ]),
    ("mountain", [
        "The mountain stood at the end of the valley like a wall built "
        "before time began. Its lower slopes were covered with pine "
        "forest, dark green and thick, and above the trees the bare rock "
        "climbed into the clouds. A narrow path started behind the last "
        "farmhouse and went up in long, patient turns. Two hikers set "
        "out at dawn with heavy packs and light hearts. They crossed a "
        "wooden bridge over a stream so cold it made their fingers ache "
        "when they filled their bottles. Higher up, the trees grew "
        "shorter and the wind grew stronger. A hawk circled once above "
        "them and slid away along the ridge without a single wingbeat.",
        "By afternoon the hikers reached the stone shelter below the "
        "summit. Clouds moved past them now instead of above them, wet "
        "and fast and silent. They ate chocolate and dried fruit sitting "
        "with their backs against the warm south wall. Far below, the "
        "valley looked like a map of itself, with tiny fields and a road "
        "as thin as a thread. The last part of the climb was steep, hand "
        "over hand beside an old steel cable. Then, all at once, there "
        "was no more up. The summit was a small flat place with a wooden "
        "cross and a metal box holding a notebook. They wrote their "
        "names, sat down in the enormous quiet, and let the mountain "
        "hold them above the world for a while.",
    ]),
]


def _seed_sample_books() -> None:
    """Install the built-in sample books if no books are saved yet."""
    if _session_files_by_mtime():
        return
    for name, texts in _SAMPLE_BOOKS:
        pages_list = [{"page": i + 1, "text": t,
                       "word_count": len(t.split()),
                       "chunks": None, "chunk_wavs": None}
                      for i, t in enumerate(texts)]
        try:
            _save_session(name, pages_list)
            # keep mtimes strictly ordered so pruning removes samples
            # in a predictable oldest-first order
            time.sleep(0.05)
        except Exception as e:
            print(f"[OCR] sample book seed error: {e}")
    print("[OCR] No saved books found — installed sample books: "
          + ", ".join(n for n, _ in _SAMPLE_BOOKS))


# ── startup session summary ───────────────────────────────────────────────────

def _startup_summary() -> str:
    """Return a brief saved-book *count* for the opening announcement — just
    enough for the user to know books exist, without reading every single
    one's name/pages/word-count up front (that used to make the very first
    thing a blind user heard, before any instructions, a multi-minute wall of
    speech once more than a couple of books were saved — with a long enough
    announcement it could even outrun fb.wait()'s timeout, so the mic started
    listening for a command while the announcement was still playing). Full
    names are still spoken on demand: LOAD and ADD both call
    _saved_session_names() and read them out before asking which one."""
    if not os.path.isdir(SESSIONS_DIR):
        return ""
    count = 0
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            if data.get("pages"):
                count += 1
        except Exception:
            pass
    if count == 0:
        return ""
    return f"You have {count} saved book{'s' if count != 1 else ''}. "


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

        mc_ok = _ensure_mc(fb, ctx.link)

        _seed_sample_books()   # first run: install test books to read
        _prune_sessions()      # keep only the newest MAX_SAVED_SESSIONS books

        self._session_counter += 1
        session_name: "str | None" = None
        pages:        dict         = {}
        page_idx      = 0

        # ── voice thread ──────────────────────────────────────────────────────
        voice_q     = queue.Queue()
        _cur_stop   = [threading.Event()]
        _cur_stop[0].set()
        _cur_thread = [None]   # the OCRVoice Thread currently owning the mic, if any
        _last_raw   = [""]     # full text of the utterance behind the last voice cmd

        def _voice_on() -> None:
            if not mc_ok:
                return
            _cur_stop[0].set()
            stop = threading.Event()
            _cur_stop[0] = stop
            _drain(voice_q)
            def _delayed():
                time.sleep(1.0)
                if not stop.is_set():
                    # Drain again right before listening starts: a previous
                    # worker that was mid-transcription when it was stopped
                    # can queue its (now stale) text after the drain above —
                    # acting on it here would fire a command the user said
                    # a whole flow ago (e.g. re-triggering a scan they had
                    # already completed).
                    _drain(voice_q)
                    _voice_worker(ctx.abort, voice_q, stop)
            t = threading.Thread(target=_delayed, daemon=True, name="OCRVoice")
            _cur_thread[0] = t
            t.start()

        def _voice_off() -> "str | None":
            # Setting the stop Event only tells the loop not to start another
            # listen() call — the one already in flight (record_with_vad(),
            # now correctly capped at MAX_RECORD_SEC wall-clock time) keeps
            # running to completion. Joining here means capture/TTS that
            # follows never overlaps a still-open microphone stream — without
            # this, capture and mc.listen() raced for the same USB webcam's
            # combined mic+camera interface, producing "audio open error:
            # Device or resource busy" / "Device unavailable" and silently
            # dropped instructions (aplay failing to open mid-race, with
            # nothing surfacing that failure back to the caller).
            #
            # If the user was already mid-utterance when this was called
            # (e.g. said "load the page" and, without waiting for a reply,
            # immediately kept talking), the in-flight listen() above can
            # still finish and land in voice_q during the join. That text
            # used to be silently discarded by the drain below — the user
            # would hear no acknowledgement at all that anything was heard.
            # Return it instead so callers that are about to ask a direct
            # follow-up question can use it as the answer.
            _cur_stop[0].set()
            t = _cur_thread[0]
            if t is not None and t.is_alive():
                # Worst case the loop is mid-utterance: full record window
                # plus a slow Whisper pass (23s observed live on the Pi).
                # An idle listener releases the mic in well under a second
                # now that record_with_vad() checks stop_ev while waiting.
                t.join(timeout=getattr(mc, "MAX_RECORD_SEC", 15) + 30.0)
                if t.is_alive():
                    print("[OCR] WARNING: voice worker still busy after join "
                          "timeout — mic may be contended.")
            leftover = None
            try:
                leftover = voice_q.get_nowait()
            except queue.Empty:
                pass
            _drain(voice_q)
            return leftover

        def _get_cmd(ges_timeout: float = 0.1) -> "str | None":
            _last_raw[0] = ""    # cleared so a gesture cmd can't reuse stale text
            try:
                raw = voice_q.get_nowait()
                _last_raw[0] = raw
                cmd = _classify(raw)
                if cmd:
                    print(f"[OCR] voice: {raw!r} → {cmd}")
                else:
                    print(f"[OCR] voice: {raw!r} → (unrecognised)")
                return cmd
            except queue.Empty:
                pass
            g = _wait_gesture(ctx, timeout=ges_timeout)
            return _GMAP.get(g) if g else None

        def _session_stats() -> "tuple[int, int]":
            """Return (n_pages, total_words) for the current pages dict."""
            order = _reading_order(pages)
            return (
                len(order),
                sum(pages[k].get("word_count", 0) for k in order),
            )

        def _autosave() -> None:
            if pages and session_name:
                try:
                    order = _reading_order(pages)
                    _save_session(session_name, [pages[k] for k in order])
                    print(f"[OCR] Auto-saved '{session_name}' ({len(order)} pages)")
                except Exception as e:
                    print(f"[OCR] Auto-save error: {e}")

        def _ask_book_name(leftover: "str | None", first_prompt: str,
                           retry_prompt: str) -> "tuple[str, dict | None]":
            """Get a book name from the user and resolve it to a saved session.

            Tries `leftover` first (speech already captured by _voice_off()
            while it was joining the background listener — previously this
            was silently dropped, which is exactly what ate "load a book" the
            first time this bug was seen live). Otherwise does a direct
            mc.listen() call. Gives one retry if the name doesn't match a
            saved session. Every attempt is logged through the same
            "[OCR] voice(...)" convention _get_cmd() uses, so these follow-up
            prompts are no longer a blind spot in the logs.

            Returns (last_raw_text_heard, matched_session_or_None).
            """
            # A leftover that strips to nothing ("load", "read", "a book") is
            # the user repeating the command, not naming a book — using it as
            # a name burned one of the two attempts on a guaranteed miss.
            if leftover and not _strip_filler(leftover):
                print(f"[OCR] voice(leftover): {leftover!r} — just a command "
                      "word, not a name; ignoring")
                leftover = None

            raw_text = ""
            for attempt in range(2):
                if leftover:
                    raw_text = leftover
                    print(f"[OCR] voice(leftover): {raw_text!r} — used as book name")
                    leftover = None
                else:
                    _srvspeak(ctx, fb, first_prompt if attempt == 0 else retry_prompt)
                    fb.wait(timeout=6)
                    raw_text = ""
                    if mc_ok:
                        try:
                            # correct=False: this is a free-text title, and
                            # the command-vocab corrector mangles names
                            # ("that's the one" → "that's three one")
                            raw_text, _ = mc.listen(initial_prompt=OCR_INITIAL_PROMPT,
                                                    hotwords=OCR_HOTWORDS,
                                                    correct=False)
                        except Exception:
                            raw_text = ""
                    if not raw_text:
                        # The old background listener may have finished its
                        # in-flight transcription DURING our direct listen and
                        # queued the user's answer there instead ("read page
                        # one" was observed landing in voice_q while the
                        # direct listen heard nothing).
                        try:
                            raw_text = voice_q.get_nowait()
                            print(f"[OCR] voice(late): {raw_text!r} — "
                                  "recovered from background queue")
                        except queue.Empty:
                            pass
                    print(f"[OCR] voice(direct): {raw_text!r} — answering 'book name'")

                if not raw_text:
                    continue
                sdata = _find_session(raw_text)
                if sdata:
                    return raw_text, sdata

            return raw_text, None

        def _ask_direct(prompt: str, timeout: int = 6, correct: bool = True,
                        label: str = "") -> str:
            """Speak a question, take ONE direct listen, return the transcript
            ('' if nothing heard). Background voice loop must be off."""
            _srvspeak(ctx, fb, prompt)
            fb.wait(timeout=timeout)
            txt = ""
            if mc_ok:
                try:
                    txt, _ = mc.listen(initial_prompt=OCR_INITIAL_PROMPT,
                                       hotwords=OCR_HOTWORDS, correct=correct)
                except Exception:
                    txt = ""
            print(f"[OCR] voice(direct): {txt!r} — answering {label!r}")
            return txt or ""

        def _capture_one_page() -> bool:
            """Capture and OCR one page into `pages` (with page number ask).
            Returns True if a page was stored."""
            nonlocal page_idx

            # Speak the hold instruction and let it actually finish before
            # capturing — _srvspeak() is non-blocking, so without the waits
            # the photo was taken while the instruction was still being read.
            fb.beep()   # non-verbal "getting ready to scan" heads-up
            _srvspeak(ctx, fb, "Hold the page flat and steady.")
            fb.wait(timeout=6)
            fb.speak("Capturing now.")
            fb.wait(timeout=3)
            jpeg = _capture_frame(fb)

            if jpeg is None:
                _srvspeak(ctx, fb, "Camera error. Check USB camera.")
                fb.wait(timeout=4)
                return False

            _srvspeak(ctx, fb, "Scanning complete. Processing text.")
            resp = ctx.link.send("ocr", {"type": "scan", "frame": jpeg})

            if resp is None:
                fb.speak("Server not reachable.")   # local only — server is down
                fb.wait(timeout=4)
                return False

            text_body = resp.get("text", "").strip()
            if not text_body:
                _srvspeak(
                    ctx, fb,
                    "No text found. "
                    "Hold page 30 to 40 centimetres from camera "
                    "with good lighting, then scan again."
                )
                fb.wait(timeout=8)
                return False

            detected_page  = _coerce_page_num(resp.get("page"))
            word_count     = len(text_body.split())
            server_chunks  = resp.get("chunks")          # server-split text chunks
            chunk_wavs_b64 = resp.get("chunk_wavs")      # parallel list of base64 WAVs

            # ── announce result (server WAV preferred) ────────────────────
            ann_b64 = resp.get("announcement_wav")
            if ann_b64:
                _play_wav_b64(fb, ann_b64)
            elif detected_page is not None:
                fb.speak(f"Page {detected_page}. {word_count} words.")
            else:
                fb.speak(f"No page number. {word_count} words.")

            if detected_page is None:
                fb.wait(timeout=5)
                if mc_ok:
                    vtxt = _ask_direct("Say the page number, or say skip.",
                                       timeout=5, label="page number")
                    if vtxt and not _is_skip(vtxt):
                        num = _parse_number(vtxt)
                        if num is not None:
                            detected_page = num
                            _srvspeak(ctx, fb, f"Page {detected_page}.")
                            fb.wait(timeout=3)

            if detected_page is not None:
                key = ("num", detected_page)
                if key in pages:
                    _srvspeak(ctx, fb, f"Page {detected_page} rescanned. Replacing.")
                    fb.wait(timeout=3)
            else:
                key = ("idx", page_idx)
                page_idx += 1

            pages[key] = {
                "page":       detected_page,
                "text":       text_body,
                "word_count": word_count,
                "chunks":     server_chunks,     # may be None for loaded sessions
                "chunk_wavs": chunk_wavs_b64,    # may be None
            }
            return True

        def _scan_loop() -> int:
            """Batch-scan pages until the user says done/save. After every
            page the user is asked "scan or done", so a whole book can be
            captured in one sitting without reading starting in between.
            Returns the number of pages added."""
            added = 0
            while not ctx.abort.is_set():
                if _capture_one_page():
                    added += 1
                    ans = _ask_direct(
                        "Page saved. Say scan for the next page, "
                        "or done to finish.", label="scan next or done")
                else:
                    ans = _ask_direct(
                        "Say scan to try again, or done to stop.",
                        label="retry scan or done")
                c = _classify(ans) if ans else None
                if c in ("SCAN", "NEXT", "YES"):
                    continue
                break   # done / save / close / silence → finish scanning
            return added

        def _ask_session_name() -> None:
            """Ask the user to name the current book (first save only)."""
            nonlocal session_name
            if session_name is None and mc_ok:
                vtxt = _ask_direct("Say a name for this book.", timeout=5,
                                   correct=False, label="session name")
                if vtxt and not _is_skip(vtxt):
                    session_name = vtxt.strip()
            if not session_name:
                session_name = f"reading_{self._session_counter}"

        def _read_current(start_key: "tuple | None" = None) -> bool:
            """Read the in-memory book from its first page (or start_key),
            in page-number order. Returns True if the outer loop should
            break (user closed during reading)."""
            order = _reading_order(pages)
            if not order:
                _srvspeak(ctx, fb, "No pages to read.")
                fb.wait(timeout=3)
                _voice_on()
                return False
            if gq: _drain(gq)
            result = _read_pages(start_key or order[0])
            return _after_read(result)

        def _after_read(result: str) -> bool:
            """Handle the transition after _read_pages() returns.

            Auto-saves progress, announces session status with next-step options,
            re-enables voice, and for a 'scan' return injects the SCAN command so
            capture starts immediately without requiring a second voice command.
            'load'/'add' returns (the user interrupted a pause to switch books)
            are injected the same way, so they land back in the outer loop's
            LOAD/ADD handling instead of being silently dropped.

            Returns True if the outer loop should break (session close).
            """
            if result == "close":
                return True
            _autosave()
            if result == "done":
                n_pg, total_w = _session_stats()
                _srvspeak(
                    ctx, fb,
                    f"Done reading {session_name}. "
                    f"{n_pg} page{'s' if n_pg != 1 else ''}, {total_w} words. "
                    "Say scan to add a page, "
                    "new book to start a brand new book, "
                    "load to open a different session, "
                    "or close to save and exit."
                )
                fb.wait(timeout=10)
            _voice_on()
            if result == "scan":
                # User said "scan" while paused — go straight to capture
                voice_q.put("SCAN")
            elif result == "load":
                # User said "load" while paused — go straight to book selection
                voice_q.put("LOAD")
            elif result == "add":
                # User said "add" while paused — go straight to add-to-book
                voice_q.put("ADD")
            elif result == "new":
                # User said "new book" while paused — start a fresh book
                voice_q.put("NEW")
            return False

        # ── opening ───────────────────────────────────────────────────────────
        # Spoken in two short, sequenced chunks — menu first, gesture controls
        # second — rather than one long paragraph. Book *names* are announced
        # on demand by LOAD/ADD (_saved_session_names()), not dumped here: with
        # several saved books, reading every one's name/pages/word-count before
        # any instruction meant a blind user had to sit through a multi-minute
        # wall of speech just to learn what commands exist — and a long enough
        # announcement could outrun fb.wait()'s timeout, so the mic started
        # listening for a command while the announcement was still playing.
        _voice_on()   # start early so 1-second init overlaps with TTS
        _srvspeak(
            ctx, fb,
            "Book reader. "
            + _startup_summary()
            + "Say scan to capture a new page. "
            "Say load to read a saved book — I'll tell you their names first. "
            "Say add to add pages to a saved book. "
            "Or say close to exit."
        )
        fb.wait(timeout=20)
        _srvspeak(
            ctx, fb,
            "During reading: close thumb, ring, and pinky, and tilt your hand "
            "back, to pause or resume. Flick thumb right to skip forward. "
            "Flick thumb left to rewind. Flick pinky right to jump to the "
            "next page."
        )
        fb.wait(timeout=16)
        if gq:
            _drain(gq)        # discard any gesture that triggered the feature open

        # ── nested reading loop ───────────────────────────────────────────────
        def _read_pages(start_key: tuple) -> str:
            """Read pages starting at start_key.

            Returns:
              "scan"  — user asked to scan a new page (from pause)
              "load"  — user asked to open a different saved book (from pause)
              "add"   — user asked to add pages to a different book (from pause)
              "new"   — user asked to start a brand-new book (from pause)
              "close" — user asked to close
              "done"  — finished reading all pages
            """
            current_key: "tuple | None" = start_key

            while current_key is not None and not ctx.abort.is_set():
                pdata      = pages[current_key]
                # Server may have pre-split the text into chunks; fall back to local split
                chunks     = pdata.get("chunks") or _chunk_text(pdata["text"])
                wavs_b64   = pdata.get("chunk_wavs")  # list[str|None] or None

                if not chunks:
                    _srvspeak(ctx, fb, "Page appears blank.")
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
                            _srvspeak(
                                ctx, fb,
                                "Paused. "
                                "Say scan to add a page. "
                                "Say new book to start a brand new book. "
                                "Say load to open a different book. "
                                "Say add to add pages to a different book. "
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

                        _voice_off()

                        if cmd in ("PAUSE", "YES", "NO"):
                            paused = False; pause_announced = False
                            _srvspeak(ctx, fb, "Resuming.")
                            fb.wait(timeout=4)      # let message finish before chunk plays

                        elif cmd == "SCAN":
                            paused = False; pause_announced = False
                            return "scan"

                        elif cmd == "LOAD":
                            paused = False; pause_announced = False
                            return "load"

                        elif cmd == "ADD":
                            paused = False; pause_announced = False
                            return "add"

                        elif cmd == "NEW":
                            paused = False; pause_announced = False
                            return "new"

                        elif cmd == "NEXT":
                            i = len(chunks); paused = False; pause_announced = False
                            # advance-logic block announces the next page — no extra TTS

                        elif cmd == "FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            paused = False; pause_announced = False
                            _srvspeak(ctx, fb, "Skipped. Resuming.")
                            fb.wait(timeout=4)

                        elif cmd == "BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)   # -1: i already incremented
                            paused = False; pause_announced = False
                            _srvspeak(ctx, fb, "Rewound. Resuming.")
                            fb.wait(timeout=4)

                        elif cmd == "CLOSE":
                            ctx.abort.set()
                            return "close"

                        continue

                    # ── READ CHUNK (server WAV preferred, espeak fallback) ─────
                    b64 = wavs_b64[i] if (wavs_b64 and i < len(wavs_b64)) else None
                    if b64:
                        dur = _play_wav_b64(fb, b64)
                        if dur < 0.5:        # play_wav_b64 failed → fallback
                            fb.speak(chunks[i])
                            dur = _chunk_dur(chunks[i])
                    else:
                        fb.speak(chunks[i])
                        dur = _chunk_dur(chunks[i])
                    i += 1

                    deadline = time.time() + dur
                    while time.time() < deadline and not ctx.abort.is_set():
                        g    = _wait_gesture(ctx,
                                             timeout=min(0.05, deadline - time.time()))
                        gcmd = _GMAP.get(g) if g else None

                        if gcmd == "PAUSE":
                            paused = True; pause_announced = False
                            fb.silence()        # pause-state loop announces itself
                            break
                        elif gcmd == "FWD":
                            skip = min(SKIP_CHUNKS, len(chunks) - i)
                            i    = min(i + skip, len(chunks))
                            fb.silence()        # new chunk starting is the confirmation
                            break
                        elif gcmd == "BWD":
                            i = max(0, i - SKIP_CHUNKS - 1)
                            fb.silence()
                            break
                        elif gcmd in ("NEXT", "SCAN"):
                            fb.silence()
                            i = len(chunks); break   # advance logic announces next page
                        elif gcmd == "CLOSE":
                            fb.silence(); ctx.abort.set(); return "close"

                # Advance to next page
                if not ctx.abort.is_set() and i >= len(chunks):
                    nk = _next_key(pages, current_key)
                    if nk is not None:
                        label = _page_label(pages[nk])
                        _srvspeak(ctx, fb, f"Page done. Next is {label}. Reading.")
                        current_key = nk
                    else:
                        _srvspeak(ctx, fb, "All pages read.")
                        fb.wait(timeout=3)
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
                        _voice_off()
                        if session_name:
                            _srvspeak(ctx, fb,
                                f"Say scan to add a page to {session_name}, "
                                "load to open a session, or close to save and exit.")
                        else:
                            _srvspeak(ctx, fb,
                                "Say scan to capture a page, "
                                "load to open a session, or close to exit.")
                        fb.wait(timeout=10)
                        _voice_on()
                        idle_t = time.time() + IDLE_REMIND_S
                    continue

                # ── LOAD saved session ─────────────────────────────────────
                if cmd == "LOAD":
                    leftover = _voice_off()
                    # Persist whatever's already in memory (e.g. the book we
                    # were mid-read on, interrupted via a pause to switch)
                    # before it gets replaced below.
                    _autosave()

                    # The command itself may already name the book — "load
                    # page", "read harry potter" — so try that before asking
                    # a follow-up question at all.
                    raw_text, sdata = "", None
                    hint = _strip_filler(_last_raw[0])
                    cmd_words = set(re.findall(r"[a-z]+", _last_raw[0].lower()))

                    # Bare "read" with a book already in memory (just scanned
                    # or just finished) → read it, no selection dialog.
                    if not hint and pages and "read" in cmd_words:
                        if _read_current():
                            break
                        idle_t = time.time() + IDLE_REMIND_S
                        continue

                    saved_names = _saved_session_names()
                    if not saved_names:
                        _srvspeak(ctx, fb,
                            "No saved books found yet. Say scan to start one.")
                        fb.wait(timeout=4)
                        _voice_on()
                        continue

                    if hint:
                        sdata = _find_session(hint)
                        if sdata:
                            raw_text = hint
                            print(f"[OCR] book name taken from command: {hint!r}")

                    # Only one book saved → nothing to disambiguate, load it.
                    if sdata is None and len(saved_names) == 1:
                        sdata = _find_session(saved_names[0])
                        if sdata:
                            raw_text = saved_names[0]
                            print(f"[OCR] only one saved book — loading "
                                  f"{saved_names[0]!r} without asking")

                    if sdata is None:
                        names_str = ", ".join(saved_names[:NAMES_ANNOUNCE_LIMIT])
                        extra = (f" and {len(saved_names) - NAMES_ANNOUNCE_LIMIT} more"
                                 if len(saved_names) > NAMES_ANNOUNCE_LIMIT else "")
                        raw_text, sdata = _ask_book_name(
                            leftover,
                            f"Saved books: {names_str}{extra}. Say the book name to read.",
                            "I couldn't find that book. Say the name again.",
                        )

                    if sdata:
                        loaded_name  = sdata["name"]
                        loaded_pages, loaded_idx = _pages_from_session(sdata)
                        if loaded_pages:
                            # Replace, don't merge: this book's pages must not
                            # mix with whatever a previous book (just
                            # autosaved above) left in memory, or reading and
                            # saving afterward would corrupt both sessions.
                            session_name = loaded_name
                            pages        = dict(loaded_pages)
                            page_idx     = loaded_idx

                            # Offer to add pages BEFORE reading starts, so new
                            # pages (with their numbers) slot into order and
                            # the whole book then reads on its own.
                            ans = _ask_direct(
                                f"Loaded {loaded_name}. "
                                f"{len(loaded_pages)} pages. "
                                "Say scan to add new pages first, "
                                "or read to begin.",
                                timeout=8, label="scan first or read")
                            c = _classify(ans) if ans else None
                            if c in ("SCAN", "NEXT"):
                                if _scan_loop():
                                    _autosave()
                            elif c == "CLOSE":
                                break
                            _srvspeak(ctx, fb, "Reading now.")
                            fb.wait(timeout=3)
                            if _read_current():
                                break
                            idle_t = time.time() + IDLE_REMIND_S
                            continue
                        else:
                            _srvspeak(ctx, fb, f"Session {loaded_name} has no pages.")
                    elif raw_text:
                        # first few words only — echoing a long mis-transcription
                        # back at a blind user is just noise
                        _srvspeak(ctx, fb, "No session found matching "
                                  f"{' '.join(raw_text.split()[:6])}.")
                    _voice_on()
                    continue

                # ── ADD (load a session for editing, not for reading) ─────────
                if cmd == "ADD":
                    leftover = _voice_off()
                    # Persist whatever's already in memory (e.g. the book we
                    # were mid-read on, interrupted via a pause to switch)
                    # before it gets replaced below.
                    _autosave()
                    saved_names = _saved_session_names()
                    if not saved_names:
                        _srvspeak(ctx, fb,
                            "No saved books found. Say scan to start a new one.")
                        fb.wait(timeout=4)
                        _voice_on()
                        continue

                    # Same shortcuts as LOAD: name embedded in the command
                    # ("add to harry potter"), then single-saved-book.
                    raw_text, sdata = "", None
                    hint = _strip_filler(_last_raw[0])
                    if hint:
                        sdata = _find_session(hint)
                        if sdata:
                            raw_text = hint
                            print(f"[OCR] book name taken from command: {hint!r}")

                    if sdata is None and len(saved_names) == 1:
                        sdata = _find_session(saved_names[0])
                        if sdata:
                            raw_text = saved_names[0]
                            print(f"[OCR] only one saved book — adding to "
                                  f"{saved_names[0]!r} without asking")

                    if sdata is None:
                        names_str = ", ".join(saved_names[:NAMES_ANNOUNCE_LIMIT])
                        extra = (f" and {len(saved_names) - NAMES_ANNOUNCE_LIMIT} more"
                                 if len(saved_names) > NAMES_ANNOUNCE_LIMIT else "")
                        raw_text, sdata = _ask_book_name(
                            leftover,
                            f"Saved books: {names_str}{extra}. Say the book name.",
                            "I couldn't find that book. Say the name again.",
                        )

                    if sdata:
                        loaded_name  = sdata["name"]
                        loaded_pages, loaded_idx = _pages_from_session(sdata)
                        # Replace, don't merge -- see the matching comment in
                        # the LOAD handler above.
                        session_name = loaded_name
                        pages        = dict(loaded_pages)
                        page_idx     = loaded_idx
                        n_loaded     = len(loaded_pages)
                        _srvspeak(ctx, fb,
                            f"Ready to add to {loaded_name}. "
                            f"{n_loaded} page{'s' if n_loaded != 1 else ''} "
                            "already saved. Say scan to capture a new page.")
                    elif raw_text:
                        _srvspeak(ctx, fb, "No book found matching "
                                  f"{' '.join(raw_text.split()[:6])}.")
                    else:
                        _srvspeak(ctx, fb, "No name heard.")
                    fb.wait(timeout=5)
                    _voice_on()
                    continue

                # ── NEW (save current book, start a fresh one) ───────────────
                if cmd == "NEW":
                    _autosave()
                    if pages and session_name:
                        _srvspeak(ctx, fb,
                            f"Saved {session_name}. Starting a new book.")
                    else:
                        _srvspeak(ctx, fb, "Starting a new book.")
                    fb.wait(timeout=5)
                    session_name = None
                    pages        = {}
                    page_idx     = 0
                    # Fall through to capture: first scan of the fresh book
                    # asks for its name, same as a brand-new session.
                    cmd = "SCAN"

                if cmd not in ("SCAN", "NEXT"):
                    continue

                # ── SCAN FLOW: batch-scan pages, then name and save ───────────
                # Reading does NOT start here: the user scans as many pages as
                # they want ("scan" after each page), then says done — the book
                # is named (first save only) and saved. Reading happens when
                # they ask for it ("read"), always in page-number order.
                _voice_off()
                idle_t = time.time() + IDLE_REMIND_S

                n_added = _scan_loop()
                if n_added == 0:
                    _voice_on()
                    continue

                _ask_session_name()
                _autosave()
                n_pg, total_w = _session_stats()
                ans = _ask_direct(
                    f"Saved as {session_name}. "
                    f"{n_pg} page{'s' if n_pg != 1 else ''}, {total_w} words. "
                    "Say read to listen now, scan to add more pages, "
                    "or close to exit.", timeout=8, label="after save")
                c = _classify(ans) if ans else None
                if c == "LOAD":                 # "read" is in the LOAD word set
                    if _read_current():
                        break
                elif c in ("SCAN", "NEXT"):
                    _voice_on()
                    voice_q.put("SCAN")
                    continue
                elif c == "CLOSE":
                    break
                else:
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
                    _srvspeak(ctx, fb, f"Warning: pages {labels}{more} not scanned.")

                try:
                    path        = _save_session(session_name, pages_list)
                    total_words = sum(p["word_count"] for p in pages_list)
                    _srvspeak(
                        ctx, fb,
                        f"Saved as {session_name}. "
                        f"{len(pages_list)} page{'s' if len(pages_list) > 1 else ''}, "
                        f"{total_words} words."
                    )
                    print(f"[OCR] Session saved → {path}")
                    _prune_sessions()   # just-saved book is newest, survives
                except Exception as e:
                    print(f"[OCR] Save error: {e}")
