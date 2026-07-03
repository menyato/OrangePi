"""
features/lidar_nav.py — LiDAR SLAM: mapping, saving, navigation, obstacle avoidance.

Motor vibration
  MT 1 (right)  — obstacle RIGHT  / turn-right navigation cue
  MT 2 (left)   — obstacle LEFT   / turn-left  navigation cue
  MT 3 (bottom) — obstacle FRONT  / proximity danger

Obstacle sectors (each ±45°, always active)
  Front  0°  → MT3    Left  90°  → MT2    Right 270° → MT1

Pulse strength ← distance:
  > 1.5 m  → silent   1.0–1.5 → 70 ms   0.6–1.0 → 160 ms
  0.3–0.6  → 300 ms   < 0.3 m → 420 ms

────────────────────────────────────────────────────────────────────────────────
Feature classes (all launched via hub.py bypass args):

  LidarNavigation    --lidar            Full flow: map + voice save + navigate
  LidarObstacleTest  --obstacles        Obstacle detection only (no SLAM)
  LidarMappingTest   --mapping          Mapping only + live server map
  LidarNavigateTest  --navigate         Navigate to a saved room

Voice commands (all modes)
  "save [name]"             save map as [name]
  "take me to [name]"       navigate to saved room (fuzzy match)
  "list rooms"              speak all saved room names
  "stop"                    exit

Gesture backups
  NEXT → save (mapping) / cycle rooms (navigation)
  EDIT → return to mapping

Turn-in-place fix: ICP_MAX_ROT_DEG set to 20° so body turns don't get clamped.
────────────────────────────────────────────────────────────────────────────────
"""

import difflib
import io
import math
import queue
import re
import threading
import time
from typing import Optional

from features.base import Feature, FeatureContext
import metrics

try:
    from lidar_adapter import MS200Adapter
    from slam_engine   import SLAMEngine, ROOMS_DIR
    _LIDAR_OK = True
except ImportError:
    _LIDAR_OK = False


# ── tunables ──────────────────────────────────────────────────────────────────
DEFAULT_PORT      = "auto"
LIDAR_BAUD        = 230400
SLAM_RES          = 0.05      # metres per grid cell
SLAM_SIZE_M       = 30.0

SECTOR_HALF_DEG   = 45.0
_DIST_LEVELS      = [(0.30, 420), (0.60, 300), (1.00, 160), (1.50, 70)]
HAPTIC_INTERVAL   = 0.22
# LiDAR is inside a stabilizer — ignore returns closer than this (own housing)
MIN_OBSTACLE_M    = 0.10      # 10 cm; raise if motors still fire on housing
NAV_SPEAK_S       = 6.0
MIN_KF_TO_SAVE    = 5
POSE_UPDATE_S     = 2.0
MAP_LIVE_UPDATE_S = 15.0       # mapping test: live map upload interval

# Allow 20°/frame ICP correction — override slam_engine default (6°).
# At 10 Hz LiDAR a 90°/s body turn is 9°/frame; 20° covers comfortable turns.
ICP_ROT_LIMIT     = 20.0


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_port(port: str, feedback) -> Optional[str]:
    if port == "auto":
        feedback.speak("Searching for lidar.")
        port = MS200Adapter.find_port(baud=LIDAR_BAUD, timeout=2.0)
        if port is None:
            feedback.speak("Lidar not found. Check the USB cable.")
            return None
        feedback.speak("Lidar found.")
    return port


def _new_slam() -> "SLAMEngine":
    slam = SLAMEngine(map_resolution=SLAM_RES, map_size_m=SLAM_SIZE_M,
                      db_dir=ROOMS_DIR, debug=False)
    slam.ICP_MAX_ROT_DEG = ICP_ROT_LIMIT   # instance override, no import change
    return slam


def _new_adapter(port: str) -> "MS200Adapter":
    a = MS200Adapter(port=port, baud=LIDAR_BAUD, median_kernel=5, max_jump_m=0.5)
    a.start()
    return a


def _scan_worker(adapter, scan_q: queue.Queue, abort):
    while not abort.is_set():
        s = adapter.get_scan(timeout=1.0)
        if s is not None and not scan_q.full():
            scan_q.put_nowait(s)


# ─── geometry ─────────────────────────────────────────────────────────────────

def _sector_min(scan, center_deg: float,
                half_deg: float = SECTOR_HALF_DEG,
                min_m: float = 0.0) -> float:
    lo = math.radians((center_deg - half_deg) % 360)
    hi = math.radians((center_deg + half_deg) % 360)
    a, r = scan.angles_rad, scan.ranges_m
    mask = ((a >= lo) & (a <= hi) if lo <= hi else (a >= lo) | (a <= hi)) & (r > min_m)
    return float(r[mask].min()) if mask.any() else float("inf")


def _dist_ms(d: float) -> int:
    for thresh, ms in _DIST_LEVELS:
        if d < thresh:
            return ms
    return 0


def _bearing_str(deg: float) -> str:
    if   deg >  15: return "turn left"
    if   deg < -15: return "turn right"
    return "straight ahead"


def _bearing_ms(abs_deg: float) -> int:
    if abs_deg >= 60: return 220
    if abs_deg >= 30: return 130
    return 70


# ─── haptics ──────────────────────────────────────────────────────────────────

def _obstacle_haptics(scan, ctx, last_haptic: float,
                      nav_target=None, slam=None) -> float:
    """
    Fire haptic pulses for obstacles and (if navigating) direction cues.
    Returns updated last_haptic timestamp.
    """
    now = time.time()
    if now - last_haptic < HAPTIC_INTERVAL:
        return last_haptic

    front_ms = _dist_ms(_sector_min(scan,   0, min_m=MIN_OBSTACLE_M))
    left_ms  = _dist_ms(_sector_min(scan,  90, min_m=MIN_OBSTACLE_M))
    right_ms = _dist_ms(_sector_min(scan, 270, min_m=MIN_OBSTACLE_M))

    if front_ms:
        ctx.feedback._pulse(3, front_ms)
        return now
    if left_ms:
        ctx.feedback._pulse(2, left_ms)
        return now
    if right_ms:
        ctx.feedback._pulse(1, right_ms)
        return now

    if nav_target and slam:
        d = slam.direction_to_room(nav_target)
        if d:
            _, _, _, bearing = d
            if   bearing >  15:
                ctx.feedback._pulse(2, _bearing_ms(abs(bearing)))
                return now
            elif bearing < -15:
                ctx.feedback._pulse(1, _bearing_ms(abs(bearing)))
                return now

    return last_haptic


# ─── map PNG ──────────────────────────────────────────────────────────────────

def _map_to_png(slam) -> Optional[bytes]:
    img = slam.occ_map.to_image()
    try:
        import cv2
        ok, buf = cv2.imencode(".png", img)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    try:
        from PIL import Image
        out = io.BytesIO()
        Image.fromarray(img, mode="L").save(out, format="PNG")
        return out.getvalue()
    except Exception:
        pass
    h, w = img.shape
    return f"P5\n{w} {h}\n255\n".encode() + img.tobytes()


def _send_map(ctx, room_name: str, slam, action: str = "map_save",
              connected_to: "str | None" = None) -> None:
    """Upload current occupancy grid PNG to server."""
    if not ctx.link:
        return
    png = _map_to_png(slam)
    if png:
        msg = {"action": action, "room_name": room_name, "frame": png}
        if connected_to:
            msg["connected_to"] = connected_to
        ctx.link.send("lidar", msg)


# ─── voice parsing ────────────────────────────────────────────────────────────

def _parse_voice(text: str):
    t     = text.lower().strip()
    words = set(t.split())

    if words & {"stop", "abort", "quit", "exit"}:
        return ("abort",)

    if "list" in words or ("what" in words and "room" in words):
        return ("list",)

    m = re.match(r"save(?:\s+room)?\s+(.*)", t)
    if m:
        name = re.sub(r"\s+", "_", m.group(1).strip()) or None
        return ("save", name)
    if t in ("save", "save room"):
        return ("save", None)

    for pat in [r"take me to\s+(.*)", r"bring me to\s+(.*)",
                r"navigate to\s+(.*)", r"go to\s+(.*)", r"head to\s+(.*)"]:
        m = re.match(pat, t)
        if m:
            name = re.sub(r"\s+", "_",
                          re.sub(r"^(the|a|an)\s+", "", m.group(1).strip()))
            return ("navigate", name) if name else None

    return None


def _match_room(spoken: str, rooms: list) -> Optional[str]:
    spoken_flat = spoken.lower().replace("_", " ")
    flat        = [r.lower().replace("_", " ") for r in rooms]
    matches     = difflib.get_close_matches(spoken_flat, flat, n=1, cutoff=0.4)
    return rooms[flat.index(matches[0])] if matches else None


# ─── internal voice listener (sounddevice + faster-whisper, no PyAudio) ───────

class _LidarVoice:
    """
    Always-on background voice listener.
    Uses sounddevice (mic), webrtcvad (VAD), faster-whisper/tiny (STT).
    All in requirements_orangepi.txt — no PyAudio needed.

    Architecture: audio callback is non-blocking — it only accumulates
    PCM frames and posts completed utterances to a queue.  A separate
    worker thread runs Whisper (0.3–1 s) so the mic never misses audio
    while transcribing.
    """
    SAMPLERATE   = 16000
    FRAME_MS     = 30          # webrtcvad supports 10/20/30 ms
    SPEECH_LEAD  = 4           # speech frames needed to start recording
    SILENCE_TAIL = 25          # silent frames to end utterance  (~0.75 s)
    MAX_FRAMES   = 333         # hard cap ~10 s

    def __init__(self, cmd_q: queue.Queue, link=None):
        self._q      = cmd_q
        self._link   = link                # ServerLink, for load-time metrics only
        self._stop   = threading.Event()
        self._utt_q  = queue.Queue()   # raw PCM bytes → ASR worker
        self._t      = None
        self._ready  = threading.Event()   # set when voice is up (or has failed)
        self.failed  = False               # True if init error

    def start(self) -> None:
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True, name="lidar-voice")
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    # ── ASR worker (runs Whisper in its own thread) ───────────────────────────
    def _asr_worker(self, whisper) -> None:
        import numpy as np
        while not self._stop.is_set():
            try:
                pcm = self._utt_q.get(timeout=0.3)
            except queue.Empty:
                continue
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                segs, _ = whisper.transcribe(arr, language="en", beam_size=1)
                text = " ".join(s.text for s in segs).strip()
                if text:
                    print(f"[LIDAR VOICE] heard: {text!r}")
                    cmd = _parse_voice(text)
                    if cmd:
                        self._q.put(cmd)
            except Exception as exc:
                print(f"[LIDAR VOICE] ASR error: {exc}")

    # ── main thread: open mic stream, feed VAD ────────────────────────────────
    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            print(f"[LIDAR VOICE] sounddevice missing ({e}) — voice disabled.")
            self.failed = True; self._ready.set()
            return

        try:
            t0 = time.time()
            from faster_whisper import WhisperModel
            whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            if self._link is not None:
                metrics.report_load(self._link, "lidar", (time.time() - t0) * 1000,
                                    component="whisper_tiny")
        except Exception as e:
            print(f"[LIDAR VOICE] faster-whisper unavailable ({e}) — voice disabled.")
            self.failed = True; self._ready.set()
            return

        try:
            import webrtcvad
            vad = webrtcvad.Vad(2)
        except ImportError:
            vad = None

        # Start ASR worker before opening mic
        asr_t = threading.Thread(target=self._asr_worker, args=(whisper,),
                                 daemon=True, name="lidar-voice-asr")
        asr_t.start()

        FRAME_SAMP = self.SAMPLERATE * self.FRAME_MS // 1000   # 480 samples

        ring         : list = []   # rolling pre-roll buffer (never cleared between utterances)
        speech       : list = []
        silent_count = 0
        speech_count = 0
        in_speech    = False

        def _cb(data, frames, _t, _s):
            nonlocal silent_count, speech_count, in_speech
            if self._stop.is_set():
                raise sd.CallbackStop()

            import numpy as _np
            frame = (_np.squeeze(data) * 32767).astype(_np.int16).tobytes()
            is_speech = vad.is_speech(frame, self.SAMPLERATE) if vad else True

            # ── waiting for speech start ──────────────────────────────────────
            if not in_speech:
                ring.append(frame)
                if len(ring) > self.SPEECH_LEAD * 4:
                    ring.pop(0)   # keep ring rolling — don't clear between utterances
                if is_speech:
                    speech_count += 1
                    if speech_count >= self.SPEECH_LEAD:
                        in_speech = True
                        speech[:] = list(ring)   # include pre-roll
                        silent_count = 0
                else:
                    speech_count = 0

            # ── inside utterance ──────────────────────────────────────────────
            else:
                speech.append(frame)
                if is_speech:
                    silent_count = 0
                else:
                    silent_count += 1

                if silent_count >= self.SILENCE_TAIL or len(speech) >= self.MAX_FRAMES:
                    try:
                        self._utt_q.put_nowait(b"".join(speech))
                    except queue.Full:
                        pass   # ASR busy; drop this utterance rather than block
                    speech.clear()
                    # keep ring rolling — do NOT clear it here
                    in_speech    = False
                    speech_count = 0
                    silent_count = 0

        print("[LIDAR VOICE] Ready — say 'save kitchen', 'take me to bathroom', 'stop'…")
        try:
            with sd.InputStream(samplerate=self.SAMPLERATE, channels=1,
                                dtype="float32", blocksize=FRAME_SAMP,
                                callback=_cb):
                self._ready.set()   # mic is open, voice is fully operational
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception as e:
            print(f"[LIDAR VOICE] stream error: {e}")
            self.failed = True
            self._ready.set()


# ─── radar PNG (obstacle live view) ──────────────────────────────────────────

def _make_radar_png(scan) -> Optional[bytes]:
    """
    Full 360° polar radar from a raw LaserScan.

    Physical direction mapping (sensor is 90° rotated on the glove):
      sensor  0° = physical LEFT   → appears LEFT  on screen
      sensor 90° = physical FRONT  → appears TOP   on screen
      sensor180° = physical RIGHT  → appears RIGHT on screen
      sensor270° = physical BACK   → appears BOTTOM on screen

    Colour = urgency: red <0.3m · orange <0.6m · yellow <1.0m · green <1.5m · grey beyond
    """
    try:
        import cv2, numpy as np
    except ImportError:
        return None

    SZ = 400; CX = CY = SZ // 2
    DISP_MAX_M = 1.5
    SCL = (SZ // 2 - 28) / DISP_MAX_M   # ~115 px/m, 28 px border margin

    # subtract 90° so sensor 90° (physical FRONT) appears at the top
    ROT = -math.pi / 2

    img = np.full((SZ, SZ, 3), 18, dtype=np.uint8)

    # faint compass spokes every 30°
    for a_deg in range(0, 360, 30):
        a = math.radians(a_deg)
        ex = CX + int((SZ // 2 - 6) * math.sin(a))
        ey = CY - int((SZ // 2 - 6) * math.cos(a))
        cv2.line(img, (CX, CY), (ex, ey), (35, 35, 35), 1)

    # distance rings
    for d_m, col in [(0.30, (80, 0, 0)), (0.60, (0, 60, 110)),
                     (1.00, (0, 100, 100)), (1.50, (0, 90, 0))]:
        r = int(d_m * SCL)
        cv2.circle(img, (CX, CY), r, col, 1)
        cv2.putText(img, f"{d_m}m", (CX + r + 2, CY - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (75, 75, 75), 1)

    # plot every scan point (skip stabilizer housing returns)
    for ang_rad, dist_m in zip(scan.angles_rad, scan.ranges_m):
        if dist_m <= MIN_OBSTACLE_M:
            continue
        display_d = min(dist_m, DISP_MAX_M)
        pa = ang_rad + ROT
        px = CX + int(math.sin(pa) * display_d * SCL)
        py = CY - int(math.cos(pa) * display_d * SCL)

        if   dist_m < 0.30: col, r = (0,   0, 255), 4
        elif dist_m < 0.60: col, r = (0,  80, 255), 3
        elif dist_m < 1.00: col, r = (0, 200, 255), 2
        elif dist_m < 1.50: col, r = (0, 255, 120), 2
        else:                col, r = (50,  50,  50), 1   # beyond display range

        cv2.circle(img, (px, py), r, col, -1)

    # sector minimums for text (corrected physical mapping, housing filtered)
    def _d(v): return f"{v:.2f}m" if v < 9.9 else u"—"
    front_m = _sector_min(scan,  90, min_m=MIN_OBSTACLE_M)
    back_m  = _sector_min(scan, 270, min_m=MIN_OBSTACLE_M)
    left_m  = _sector_min(scan,   0, min_m=MIN_OBSTACLE_M)
    right_m = _sector_min(scan, 180, min_m=MIN_OBSTACLE_M)

    cv2.putText(img, "FRONT",     (CX - 30,  14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(img, _d(front_m), (CX - 24,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,100), 1)
    cv2.putText(img, "BACK",      (CX - 22, SZ - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(img, _d(back_m),  (CX - 22, SZ - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,100), 1)
    cv2.putText(img, "LEFT",      (4,  CY -  8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(img, _d(left_m),  (4,  CY +  9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,100), 1)
    cv2.putText(img, "RIGHT",     (SZ - 60, CY -  8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(img, _d(right_m), (SZ - 54, CY +  9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,100), 1)

    # sensor origin dot
    cv2.circle(img, (CX, CY), 5, (255, 255, 255), -1)
    cv2.circle(img, (CX, CY), 9, (150, 150, 150), 1)

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


# ─── report helpers ───────────────────────────────────────────────────────────

def _path_length(log: list, x="x", y="y") -> float:
    total = 0.0
    for i in range(1, len(log)):
        total += math.hypot(log[i][x] - log[i-1][x], log[i][y] - log[i-1][y])
    return total


def _send_report(ctx, mode: str, session_id: str, data: dict) -> None:
    if ctx.link:
        ctx.link.send("lidar", {"action": "report", "mode": mode,
                                "session_id": session_id, "data": data})


# ─── save helper ─────────────────────────────────────────────────────────────

def _do_save(slam, ctx: "FeatureContext", room_name: str,
             prev_room: "str | None" = None) -> bool:
    if len(slam.keyframes) < MIN_KF_TO_SAVE:
        ctx.feedback.speak(
            f"Not enough data — {len(slam.keyframes)} keyframes. Walk more."
        )
        return False
    slam.save_room(room_name)
    link_msg = f" Connected to {prev_room.replace('_',' ')}." if prev_room else ""
    ctx.feedback.speak(f"{room_name.replace('_', ' ')} saved.{link_msg}")
    _send_map(ctx, room_name, slam, action="map_save", connected_to=prev_room)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1 — Full lidar (mapping + navigation + voice)       --lidar
# ═══════════════════════════════════════════════════════════════════════════════

class LidarNavigation(Feature):
    name  = "lidar"
    title = "Lidar Navigation"
    port: str = DEFAULT_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak("Lidar libraries not installed.")
            return

        port = _resolve_port(self.port, ctx.feedback)
        if not port:
            return

        try:
            adapter = _new_adapter(port)
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        slam = _new_slam()

        mode, nav_target = "mapping", None
        last_haptic = last_nav_spk = last_pose_send = 0.0
        t0 = time.time()

        scan_q     : queue.Queue = queue.Queue(maxsize=2)
        voice_cmd_q: queue.Queue = queue.Queue()

        scan_t = threading.Thread(target=_scan_worker,
                                  args=(adapter, scan_q, ctx.abort), daemon=True)
        scan_t.start()
        voice = _LidarVoice(voice_cmd_q, ctx.link)
        voice.start()

        # Wait up to 10 s for voice model to load then report status
        voice._ready.wait(timeout=10.0)
        voice_status = "Voice ready." if not voice.failed else "Voice unavailable. Use gestures."

        saved = slam.list_rooms()
        _session_rooms: list = []   # rooms saved this session — for room connections
        ctx.feedback.speak(
            f"Lidar mapping. {len(saved)} rooms saved. "
            "Say save kitchen to save. Say take me to kitchen to navigate. "
            + voice_status
        )

        try:
            while not ctx.abort.is_set():

                # ── voice ─────────────────────────────────────────────────────
                try:
                    vcmd = voice_cmd_q.get_nowait()
                except queue.Empty:
                    vcmd = None

                if vcmd:
                    tag = vcmd[0]
                    if tag == "abort":
                        ctx.feedback.speak("Stopping lidar.")
                        ctx.abort.set(); break

                    elif tag == "list":
                        rooms = slam.list_rooms()
                        ctx.feedback.speak(
                            ("Saved rooms: " + ", ".join(r.replace("_"," ") for r in rooms) + ".")
                            if rooms else "No rooms saved yet."
                        )

                    elif tag == "save":
                        name = vcmd[1] if len(vcmd)>1 and vcmd[1] else f"room_{len(slam.list_rooms())+1}"
                        prev = _session_rooms[-1] if _session_rooms else None
                        if _do_save(slam, ctx, name, prev_room=prev):
                            _session_rooms.append(name)

                    elif tag == "navigate":
                        spoken = vcmd[1] if len(vcmd)>1 else ""
                        rooms  = slam.list_rooms()
                        if not rooms:
                            ctx.feedback.speak("No rooms saved. Save a room first.")
                        else:
                            matched = _match_room(spoken, rooms)
                            if matched:
                                mode = "navigation"; nav_target = matched
                                last_nav_spk = 0.0
                                ctx.feedback.speak(f"Navigating to {nav_target.replace('_',' ')}.")
                            else:
                                ctx.feedback.speak(
                                    "Room not found. Saved: "
                                    + ", ".join(r.replace("_"," ") for r in rooms) + "."
                                )

                # ── gesture backups ───────────────────────────────────────────
                if ctx.gesture_queue:
                    try:
                        g = ctx.gesture_queue.get_nowait()
                        if g == "NEXT":
                            if mode == "mapping":
                                name = f"room_{len(slam.list_rooms())+1}"
                                prev = _session_rooms[-1] if _session_rooms else None
                                if _do_save(slam, ctx, name, prev_room=prev):
                                    _session_rooms.append(name)
                            else:
                                rooms = slam.list_rooms()
                                if rooms:
                                    idx = (rooms.index(nav_target)+1)%len(rooms) if nav_target in rooms else 0
                                    nav_target = rooms[idx]; last_nav_spk = 0.0
                                    ctx.feedback.speak(f"Navigating to {nav_target.replace('_',' ')}.")
                        elif g == "EDIT" and mode == "navigation":
                            mode = "mapping"; nav_target = None
                            ctx.feedback.speak("Mapping mode.")
                    except queue.Empty:
                        pass

                # ── scan ──────────────────────────────────────────────────────
                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()

                last_haptic = _obstacle_haptics(
                    scan, ctx, last_haptic,
                    nav_target=(nav_target if mode=="navigation" else None),
                    slam=slam,
                )

                if mode == "navigation" and nav_target and ctx.link and now - last_pose_send >= POSE_UPDATE_S:
                    ctx.link.send("lidar", {"action": "pose_update", "room_name": nav_target,
                                            "x": float(result.pose.x), "y": float(result.pose.y)})
                    last_pose_send = now

                if mode == "navigation" and nav_target and now - last_nav_spk >= NAV_SPEAK_S:
                    d = slam.direction_to_room(nav_target)
                    if d:
                        _, _, dist_m, bearing = d
                        ctx.feedback.speak(
                            f"{nav_target.replace('_',' ')}: {dist_m:.1f} m, {_bearing_str(bearing)}."
                        )
                    last_nav_spk = now

                if mode == "mapping" and result.room_match:
                    ctx.feedback.speak(f"Recognised: {result.room_match.replace('_',' ')}.")

        except KeyboardInterrupt:
            ctx.abort.set()
        finally:
            ctx.abort.set()
            voice.stop()
            scan_t.join(timeout=2.0)
            adapter.stop()
            # Send a session summary so the server always gets a report
            if ctx.link and len(slam.keyframes) >= MIN_KF_TO_SAVE:
                sid = time.strftime("%Y%m%d_%H%M%S")
                kf_log = [{"kf_id": kf.id, "t": round(kf.timestamp, 2),
                           "x": round(kf.pose.x, 3), "y": round(kf.pose.y, 3)}
                          for kf in slam.keyframes]
                report_mode = "mapping" if mode == "mapping" else "navigate"
                _send_report(ctx, report_mode, sid, {
                    "room_name":     nav_target or "unsaved",
                    "keyframes":     len(slam.keyframes),
                    "distance_m":    round(_path_length(kf_log, "x", "y"), 2),
                    "keyframe_log":  kf_log,
                    "duration_s":    round(time.time() - t0, 1),
                    "session_rooms": _session_rooms,
                })


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 2 — Obstacle-detection test only (no SLAM)          --obstacles
# ═══════════════════════════════════════════════════════════════════════════════

class LidarObstacleTest(Feature):
    """
    Reads LiDAR, fires haptics for obstacles, logs every sample.
    No SLAM — instant start, pure sensor test.
    Report sent to server on stop, saved as JSON + viewable HTML.
    """
    name  = "obstacles"
    title = "Lidar Obstacle Test"
    port: str = DEFAULT_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak("Lidar not installed.")
            return

        port = _resolve_port(self.port, ctx.feedback)
        if not port:
            return

        try:
            adapter = _new_adapter(port)
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        scan_q  = queue.Queue(maxsize=2)
        voice_q = queue.Queue()
        scan_t  = threading.Thread(target=_scan_worker,
                                   args=(adapter, scan_q, ctx.abort), daemon=True)
        scan_t.start()
        voice = _LidarVoice(voice_q, ctx.link)
        voice.start()

        ctx.feedback.speak("Obstacle test. Walk around. Say stop to finish.")
        session_id  = time.strftime("%Y%m%d_%H%M%S")
        t0          = time.time()
        last_haptic = last_radar_up = 0.0
        events: list = []

        try:
            while not ctx.abort.is_set():
                try:
                    vcmd = voice_q.get_nowait()
                    if vcmd and vcmd[0] == "abort":
                        ctx.feedback.speak("Stopping obstacle test.")
                        ctx.abort.set(); break
                except queue.Empty:
                    pass

                try:
                    scan = scan_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                now     = time.time()
                front_m = _sector_min(scan,   0, min_m=MIN_OBSTACLE_M)
                left_m  = _sector_min(scan,  90, min_m=MIN_OBSTACLE_M)
                right_m = _sector_min(scan, 270, min_m=MIN_OBSTACLE_M)
                front_ms = _dist_ms(front_m)
                left_ms  = _dist_ms(left_m)
                right_ms = _dist_ms(right_m)

                motor = None
                if now - last_haptic >= HAPTIC_INTERVAL:
                    if front_ms:
                        ctx.feedback._pulse(3, front_ms); motor = ("MT3", front_ms); last_haptic = now
                    elif left_ms:
                        ctx.feedback._pulse(2, left_ms);  motor = ("MT2", left_ms);  last_haptic = now
                    elif right_ms:
                        ctx.feedback._pulse(1, right_ms); motor = ("MT1", right_ms); last_haptic = now

                events.append({
                    "t"        : round(now - t0, 2),
                    "front_m"  : round(front_m, 3) if front_m < 99 else None,
                    "left_m"   : round(left_m,  3) if left_m  < 99 else None,
                    "right_m"  : round(right_m, 3) if right_m < 99 else None,
                    "motor"    : motor[0] if motor else None,
                    "motor_ms" : motor[1] if motor else None,
                })

                # live radar → server every 1 s
                if ctx.link and now - last_radar_up >= 1.0:
                    radar = _make_radar_png(scan)
                    if radar:
                        ctx.link.send("lidar", {
                            "action": "map_update", "room_name": "live_obstacles",
                            "frame": radar,
                        })
                    last_radar_up = now

        except KeyboardInterrupt:
            ctx.abort.set()
        finally:
            voice.stop()
            scan_t.join(timeout=2.0)
            adapter.stop()

        duration = round(time.time() - t0, 1)
        ctx.feedback.speak(
            f"Obstacle test complete. {len(events)} samples, {int(duration)} seconds."
        )
        report = _build_obstacle_report(session_id, duration, events)
        _send_report(ctx, "obstacles", session_id, report)


def _build_obstacle_report(session_id: str, duration: float, events: list) -> dict:
    def finite(vals): return [v for v in vals if v is not None]
    def stats(vals):
        if not vals: return {"samples": 0}
        return {"samples": len(vals), "min_m": round(min(vals),3),
                "max_m": round(max(vals),3), "avg_m": round(sum(vals)/len(vals),3)}

    motors = [e["motor"] for e in events if e["motor"]]
    return {
        "session_id" : session_id,
        "duration_s" : duration,
        "total_samples": len(events),
        "front"      : stats(finite([e["front_m"] for e in events])),
        "left"       : stats(finite([e["left_m"]  for e in events])),
        "right"      : stats(finite([e["right_m"] for e in events])),
        "motor_fires": {"MT1": motors.count("MT1"),
                        "MT2": motors.count("MT2"),
                        "MT3": motors.count("MT3")},
        "events"     : events,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3 — Mapping test                                     --mapping
# ═══════════════════════════════════════════════════════════════════════════════

class LidarMappingTest(Feature):
    """
    SLAM mapping only — no navigation.
    Live map uploaded to server every MAP_LIVE_UPDATE_S seconds (room 'live_map').
    Voice 'save [name]' saves the final room and sends a mapping report.
    Voice 'stop' quits without saving.
    """
    name  = "mapping"
    title = "Lidar Mapping Test"
    port: str = DEFAULT_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak("Lidar not installed.")
            return

        port = _resolve_port(self.port, ctx.feedback)
        if not port:
            return

        try:
            adapter = _new_adapter(port)
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        slam = _new_slam()
        scan_q  = queue.Queue(maxsize=2)
        voice_q = queue.Queue()
        scan_t  = threading.Thread(target=_scan_worker,
                                   args=(adapter, scan_q, ctx.abort), daemon=True)
        scan_t.start()
        voice = _LidarVoice(voice_q, ctx.link)
        voice.start()

        session_id  = time.strftime("%Y%m%d_%H%M%S")
        t0          = time.time()
        last_haptic = last_map_up = 0.0
        loop_closures = 0
        _report_sent = False

        ctx.feedback.speak(
            "Mapping test. Walk to build the map. "
            "Say save kitchen to save. Say stop to quit."
        )

        try:
            while not ctx.abort.is_set():
                # voice
                try:
                    vcmd = voice_q.get_nowait()
                    if vcmd:
                        tag = vcmd[0]
                        if tag == "abort":
                            ctx.feedback.speak("Mapping stopped.")
                            ctx.abort.set(); break
                        elif tag == "save":
                            name = vcmd[1] if len(vcmd)>1 and vcmd[1] \
                                   else f"room_{len(slam.list_rooms())+1}"
                            ok = _do_save(slam, ctx, name)
                            if ok:
                                _send_mapping_report(slam, ctx, session_id, t0,
                                                     loop_closures, name)
                                _report_sent = True
                        elif tag == "list":
                            rooms = slam.list_rooms()
                            ctx.feedback.speak(
                                ("Saved rooms: " + ", ".join(rooms) + ".")
                                if rooms else "No rooms saved yet."
                            )
                except queue.Empty:
                    pass

                # gesture backup
                if ctx.gesture_queue:
                    try:
                        g = ctx.gesture_queue.get_nowait()
                        if g == "NEXT":
                            name = f"room_{len(slam.list_rooms())+1}"
                            ok   = _do_save(slam, ctx, name)
                            if ok:
                                _send_mapping_report(slam, ctx, session_id, t0,
                                                     loop_closures, name)
                                _report_sent = True
                    except queue.Empty:
                        pass

                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()

                if result.loop_closed:
                    loop_closures += 1

                # live map to server
                if now - last_map_up >= MAP_LIVE_UPDATE_S:
                    _send_map(ctx, "live_map", slam, action="map_update")
                    last_map_up = now

                # obstacle detection always on
                last_haptic = _obstacle_haptics(scan, ctx, last_haptic)

        except KeyboardInterrupt:
            ctx.abort.set()
        finally:
            voice.stop()
            scan_t.join(timeout=2.0)
            adapter.stop()
            _send_map(ctx, "live_map", slam, action="map_update")
            if not _report_sent and len(slam.keyframes) >= MIN_KF_TO_SAVE:
                _send_mapping_report(slam, ctx, session_id, t0,
                                     loop_closures, "unsaved_session")


def _send_mapping_report(slam, ctx, session_id, t0, loop_closures, room_name):
    kfs   = slam.keyframes
    t_end = time.time()
    kf_log = [
        {"kf_id": kf.id, "t": round(kf.timestamp - t0, 1),
         "x": round(kf.pose.x, 3), "y": round(kf.pose.y, 3),
         "yaw_deg": round(math.degrees(kf.pose.yaw), 1)}
        for kf in kfs
    ]
    occ_cells = int((slam.occ_map.grid > 0.5).sum())
    dist_m    = _path_length(kf_log)
    _send_report(ctx, "mapping", session_id, {
        "session_id"     : session_id,
        "room_name"      : room_name,
        "duration_s"     : round(t_end - t0, 1),
        "keyframes"      : len(kfs),
        "distance_m"     : round(dist_m, 2),
        "loop_closures"  : loop_closures,
        "occupied_cells" : occ_cells,
        "keyframe_log"   : kf_log,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 4 — Navigate test                                    --navigate
# ═══════════════════════════════════════════════════════════════════════════════

class LidarNavigateTest(Feature):
    """
    Navigate to a pre-saved room.
    The map is re-sent to the server so the live viewer shows it immediately.
    Pose dot updates every 2 s.  Final report saved on stop or arrival (<0.5 m).

    Start at the SAME physical position as when the map was originally built
    (SLAM origin = (0, 0)).
    """
    name      = "navigate"
    title     = "Lidar Navigate Test"
    port: str = DEFAULT_PORT
    room_name : str = ""       # set from --navigate-room

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak("Lidar not installed.")
            return

        slam  = _new_slam()
        rooms = slam.list_rooms()

        if not self.room_name:
            ctx.feedback.speak(
                "No room specified. Use --navigate-room."
                + (f" Saved rooms: {', '.join(rooms)}." if rooms else "")
            )
            return

        matched = _match_room(self.room_name, rooms) if rooms else None
        if not matched:
            ctx.feedback.speak(
                f"Room '{self.room_name}' not found."
                + (f" Saved: {', '.join(rooms)}." if rooms else " No rooms saved.")
            )
            return

        port = _resolve_port(self.port, ctx.feedback)
        if not port:
            return

        try:
            adapter = _new_adapter(port)
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        slam.load_room(matched)

        # Upload the saved room map so the browser shows it before navigation starts
        _send_map(ctx, matched, slam, action="map_save")

        ctx.feedback.speak(
            f"Navigating to {matched.replace('_',' ')}. "
            "Stand at your starting position. Obstacle detection active."
        )

        scan_q  = queue.Queue(maxsize=2)
        voice_q = queue.Queue()
        scan_t  = threading.Thread(target=_scan_worker,
                                   args=(adapter, scan_q, ctx.abort), daemon=True)
        scan_t.start()
        voice = _LidarVoice(voice_q, ctx.link)
        voice.start()

        session_id  = time.strftime("%Y%m%d_%H%M%S")
        t0          = time.time()
        last_haptic = last_nav_spk = last_pose_send = 0.0
        pose_log: list = []
        arrived = False

        try:
            while not ctx.abort.is_set():
                try:
                    vcmd = voice_q.get_nowait()
                    if vcmd and vcmd[0] == "abort":
                        ctx.feedback.speak("Navigation stopped.")
                        ctx.abort.set(); break
                except queue.Empty:
                    pass

                if ctx.gesture_queue:
                    try:
                        g = ctx.gesture_queue.get_nowait()
                        if g in ("EDIT", "NEXT"):
                            ctx.abort.set(); break
                    except queue.Empty:
                        pass

                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()
                elapsed = round(now - t0, 1)

                d       = slam.direction_to_room(matched)
                dist_m  = d[2] if d else None
                bearing = d[3] if d else None

                # pose update → server
                if ctx.link and now - last_pose_send >= POSE_UPDATE_S:
                    ctx.link.send("lidar", {"action": "pose_update", "room_name": matched,
                                            "x": float(result.pose.x), "y": float(result.pose.y)})
                    last_pose_send = now
                    if dist_m is not None:
                        pose_log.append({
                            "t": elapsed, "x": round(result.pose.x, 3), "y": round(result.pose.y, 3),
                            "dist_m": round(dist_m, 3),
                            "bearing_deg": round(bearing, 1) if bearing is not None else None,
                        })

                # arrived?
                if not arrived and dist_m is not None and dist_m < 0.5:
                    arrived = True
                    ctx.feedback.speak(f"Arrived at {matched.replace('_',' ')}!")
                    ctx.abort.set(); break

                # haptics
                last_haptic = _obstacle_haptics(
                    scan, ctx, last_haptic, nav_target=matched, slam=slam
                )

                # TTS direction
                if d and now - last_nav_spk >= NAV_SPEAK_S:
                    ctx.feedback.speak(
                        f"{matched.replace('_',' ')}: {dist_m:.1f} m, {_bearing_str(bearing)}."
                    )
                    last_nav_spk = now

        except KeyboardInterrupt:
            ctx.abort.set()
        finally:
            voice.stop()
            scan_t.join(timeout=2.0)
            adapter.stop()

        # report
        duration = round(time.time() - t0, 1)
        ctx.feedback.speak("Sending navigation report.")
        _send_report(ctx, "navigate", session_id, {
            "session_id"    : session_id,
            "room"          : matched,
            "duration_s"    : duration,
            "arrived"       : arrived,
            "start_dist_m"  : pose_log[0]["dist_m"]  if pose_log else None,
            "end_dist_m"    : pose_log[-1]["dist_m"] if pose_log else None,
            "path_length_m" : round(_path_length(pose_log), 2),
            "pose_log"      : pose_log,
        })
