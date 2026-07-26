"""
net/monitor.py — streams the OrangePi's session to the laptop's live dashboard.

Sends three things to the server's monitor (handlers/monitor.py, port 8100):
  - every terminal line (stdout + stderr are tee'd)      -> kind "log"
  - what the glove SAID (call monitor.tts(text))         -> kind "tts"
  - what the USER SAID (call monitor.stt(text))          -> kind "stt"
  - active feature / results (feature()/result())        -> kinds feature/result
  - camera frames (frame(jpeg_bytes))                    -> POST /frame

All sending is done on a background daemon thread with short timeouts, so it
never blocks or crashes the feature if the laptop is slow/absent. install() is
idempotent and safe to call once at hub startup.
"""

import base64
import json
import queue
import sys
import threading
import time
import urllib.request

_host: "str | None" = None
_port = 8100
_evq: "queue.Queue" = queue.Queue(maxsize=5000)
_frame_lock = threading.Lock()
_pending_frame: "tuple[bytes, str] | None" = None
_started = False
_real_stdout = None


# ── public API ─────────────────────────────────────────────────────────────────

def event(kind: str, text: str, feature: str = "") -> None:
    if _host is None or not text:
        return
    try:
        _evq.put_nowait({"t": time.time(), "kind": kind,
                         "text": str(text)[:2000], "feature": feature})
    except queue.Full:
        pass


def tts(text: str) -> None:      event("tts", text)
def stt(text: str) -> None:      event("stt", text)
def result(text: str) -> None:   event("result", text)
def feature(name: str) -> None:  event("feature", name)


def frame(jpeg: bytes, feature: str = "") -> None:
    """Push the latest camera/keyframe JPEG (only the newest is kept)."""
    global _pending_frame
    if _host is None or not jpeg:
        return
    with _frame_lock:
        _pending_frame = (jpeg, feature)


# ── stdout/stderr tee ──────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, real, kind_stream):
        self._real = real
        self._buf = ""
        self._stream = kind_stream

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        try:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    event("log", line)
        except Exception:
            pass

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def __getattr__(self, k):
        return getattr(self._real, k)


# ── sender thread ──────────────────────────────────────────────────────────────

def _post(path: str, data: bytes, ctype: str, extra_headers=None) -> None:
    url = f"http://{_host}:{_port}{path}"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", ctype)
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=3) as r:
        r.read()


def _sender():
    global _pending_frame
    while True:
        time.sleep(0.6)
        # ── events ────────────────────────────────────────────────────────
        batch = []
        try:
            while len(batch) < 200:
                batch.append(_evq.get_nowait())
        except queue.Empty:
            pass
        if batch:
            try:
                _post("/ingest", json.dumps({"events": batch}).encode("utf-8"),
                      "application/json")
            except Exception:
                pass   # laptop down / slow — drop this batch, keep running
        # ── frame ─────────────────────────────────────────────────────────
        fr = None
        with _frame_lock:
            if _pending_frame is not None:
                fr = _pending_frame
                _pending_frame = None
        if fr is not None:
            try:
                _post("/frame", fr[0], "image/jpeg", {"X-Feature": fr[1]})
            except Exception:
                pass


def install(host: str, port: int = 8100) -> None:
    """Point the monitor at the server and start tee'ing stdout/stderr. Safe to
    call once; no-op on repeat."""
    global _host, _port, _started, _real_stdout
    if _started:
        return
    _host, _port = host, port
    _real_stdout = sys.stdout
    sys.stdout = _Tee(sys.stdout, "out")
    sys.stderr = _Tee(sys.stderr, "err")
    threading.Thread(target=_sender, daemon=True, name="monitor-sender").start()
    _started = True
    print(f"[MONITOR] streaming session to http://{host}:{port}/")
