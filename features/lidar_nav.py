"""
features/lidar_nav.py — LiDAR SLAM: mapping, saving, navigation, obstacle avoidance.

Motor vibration
  MT 1 (right)  — obstacle RIGHT  / turn-right navigation cue
  MT 2 (left)   — obstacle LEFT   / turn-left  navigation cue
  MT 3 (bottom) — obstacle FRONT  / proximity danger

Obstacle sectors (each ±45°, always active in both modes)
  Front  0°  → MT3
  Left  90°  → MT2
  Right 270° → MT1

Pulse strength ← distance:
  > 1.5 m  → silent
  1.0–1.5  → 70 ms
  0.6–1.0  → 160 ms
  0.3–0.6  → 300 ms
  < 0.3 m  → 420 ms

Modes
  MAPPING    : walk + turn in place to build occupancy map.
               The LiDAR sensor is used exclusively — no camera.
               ICP_MAX_ROT_DEG raised to 20° so fast body turns
               do not produce duplicate/overlapping scan artefacts.

  NAVIGATION : haptic direction cues + periodic TTS toward target.
               Pose sent to server every 2 s so the browser map
               shows a live red-dot at your position.

Voice commands (handled by internal listener, always active)
  "save [name]"             — save map as [name]  (e.g. "save kitchen")
  "take me to [name]"       — navigate to saved room (fuzzy match)
  "go to [name]"            — same
  "navigate to [name]"      — same
  "list rooms"              — speak all saved room names
  "stop"                    — exit lidar

Gesture backups (glove buttons, when gesture_queue is connected)
  NEXT  (mapping)    — save as auto-named room_N
  NEXT  (navigation) — cycle to next saved room
  EDIT  (navigation) — return to mapping
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

try:
    from lidar_adapter import MS200Adapter
    from slam_engine   import SLAMEngine, ROOMS_DIR
    _LIDAR_OK = True
except ImportError:
    _LIDAR_OK = False

try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False

# ── tunables ──────────────────────────────────────────────────────────────────
DEFAULT_PORT    = "auto"
LIDAR_BAUD      = 230400
SLAM_RES        = 0.05     # metres per grid cell
SLAM_SIZE_M     = 30.0

SECTOR_HALF_DEG = 45.0
_DIST_LEVELS    = [(0.30, 420), (0.60, 300), (1.00, 160), (1.50, 70)]
HAPTIC_INTERVAL = 0.22     # seconds between haptic pulses
NAV_SPEAK_S     = 6.0      # seconds between TTS navigation updates
MIN_KF_TO_SAVE  = 5        # minimum keyframes required to save
POSE_UPDATE_S   = 2.0      # seconds between server pose updates

# Override slam_engine default (6°) so body turns > 6°/scan don't get clamped.
# At 10 Hz LiDAR and a 90°/s turn: 9°/frame — needs ≥ 9° limit.
ICP_ROT_LIMIT   = 20.0     # degrees


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _sector_min(scan, center_deg: float, half_deg: float = SECTOR_HALF_DEG) -> float:
    """Minimum valid range (m) inside [center ± half_deg] sector."""
    lo = math.radians((center_deg - half_deg) % 360)
    hi = math.radians((center_deg + half_deg) % 360)
    a, r = scan.angles_rad, scan.ranges_m
    if lo <= hi:
        mask = (a >= lo) & (a <= hi) & (r > 0)
    else:
        mask = ((a >= lo) | (a <= hi)) & (r > 0)
    return float(r[mask].min()) if mask.any() else float("inf")


def _dist_ms(dist_m: float) -> int:
    for threshold, ms in _DIST_LEVELS:
        if dist_m < threshold:
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


# ─── map → PNG ────────────────────────────────────────────────────────────────

def _map_to_png(slam) -> Optional[bytes]:
    img = slam.occ_map.to_image()      # (N, N) uint8 grayscale
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


# ─── voice parsing ────────────────────────────────────────────────────────────

def _parse_voice(text: str):
    """
    Parse raw transcribed text into a lidar command tuple:
      ("abort",)
      ("list",)
      ("save",  name_or_None)
      ("navigate", room_name)
    Returns None if not a lidar command.
    """
    t     = text.lower().strip()
    words = set(t.split())

    if words & {"stop", "abort", "quit", "exit"}:
        return ("abort",)

    if "list" in words or ("what" in words and "room" in words):
        return ("list",)

    # "save [name]" / "save room [name]"
    m = re.match(r"save(?:\s+room)?\s+(.*)", t)
    if m:
        name = m.group(1).strip()
        name = re.sub(r"\s+", "_", name) or None
        return ("save", name)
    if t == "save" or t == "save room":
        return ("save", None)

    # navigate patterns
    for pat in [
        r"take me to\s+(.*)",
        r"bring me to\s+(.*)",
        r"navigate to\s+(.*)",
        r"go to\s+(.*)",
        r"head to\s+(.*)",
        r"to\s+the\s+(.*)",
    ]:
        m = re.match(pat, t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"^(the|a|an)\s+", "", name)
            name = re.sub(r"\s+", "_", name)
            return ("navigate", name) if name else None

    return None


def _match_room(spoken: str, rooms: list) -> Optional[str]:
    """Fuzzy-match a spoken room name to the saved rooms list."""
    spoken_flat  = spoken.lower().replace("_", " ")
    flat_rooms   = [r.lower().replace("_", " ") for r in rooms]
    matches = difflib.get_close_matches(spoken_flat, flat_rooms, n=1, cutoff=0.4)
    if matches:
        return rooms[flat_rooms.index(matches[0])]
    return None


# ─── internal voice listener ──────────────────────────────────────────────────

class _LidarVoice:
    """
    Background speech → command thread, self-contained inside LidarNavigation.
    Runs its own Microphone so it does NOT share the hub's VoiceListener.
    hub.py skips its global VoiceListener when --lidar bypass is active.
    """

    def __init__(self, cmd_q: queue.Queue):
        self._q    = cmd_q
        self._stop = threading.Event()
        self._t    = None

    def start(self) -> None:
        if not _SR_OK:
            print("[LIDAR VOICE] SpeechRecognition not installed — voice commands disabled.")
            print("[LIDAR VOICE]   pip install SpeechRecognition")
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True, name="lidar-voice")
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        r = sr.Recognizer()
        r.energy_threshold         = 3000
        r.dynamic_energy_threshold = True
        r.pause_threshold          = 0.6
        try:
            mic = sr.Microphone()
        except OSError as e:
            print(f"[LIDAR VOICE] cannot open mic: {e}")
            return
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("[LIDAR VOICE] Ready — 'save kitchen', 'take me to bathroom', 'stop'.")
            while not self._stop.is_set():
                try:
                    audio = r.listen(source, timeout=5, phrase_time_limit=8)
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    continue
                try:
                    text = r.recognize_google(audio).lower().strip()
                    print(f"[LIDAR VOICE] heard: {text!r}")
                    cmd = _parse_voice(text)
                    if cmd:
                        self._q.put(cmd)
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"[LIDAR VOICE] error: {e}")


# ─── save helper ─────────────────────────────────────────────────────────────

def _do_save(slam, ctx: "FeatureContext", room_name: str) -> bool:
    """Save current SLAM map as room_name, upload PNG to server. Returns True on success."""
    if len(slam.keyframes) < MIN_KF_TO_SAVE:
        ctx.feedback.speak(
            f"Not enough data yet — {len(slam.keyframes)} keyframes. "
            "Walk more to build the map."
        )
        return False

    slam.save_room(room_name)
    ctx.feedback.speak(f"{room_name.replace('_', ' ')} saved.")

    png = _map_to_png(slam)
    if png and ctx.link:
        reply = ctx.link.send("lidar", {
            "action"    : "map_save",
            "room_name" : room_name,
            "frame"     : png,
        })
        if reply and reply.get("tts"):
            print(f"[LIDAR] server: {reply['tts']}")
    return True


# ─────────────────────────────────────────────────────────────────────────────

class LidarNavigation(Feature):
    name  = "lidar"
    title = "Lidar Navigation"
    port: str = DEFAULT_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak(
                "Lidar libraries not installed. Run: pip install pyserial scipy"
            )
            return

        # ── Port resolution ───────────────────────────────────────────────────
        port = self.port
        if port == "auto":
            ctx.feedback.speak("Searching for lidar.")
            port = MS200Adapter.find_port(baud=LIDAR_BAUD, timeout=2.0)
            if port is None:
                ctx.feedback.speak("Lidar not found. Check the USB cable.")
                return
            ctx.feedback.speak("Lidar found.")

        # ── Adapter ───────────────────────────────────────────────────────────
        try:
            adapter = MS200Adapter(
                port=port, baud=LIDAR_BAUD,
                median_kernel=5, max_jump_m=0.5,
            )
            adapter.start()
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        # ── SLAM ──────────────────────────────────────────────────────────────
        slam = SLAMEngine(
            map_resolution=SLAM_RES,
            map_size_m=SLAM_SIZE_M,
            db_dir=ROOMS_DIR,
            debug=False,
        )
        # Allow up to 20° rotation per scan so body turns don't get clamped.
        # The default (6°) rejects fast turns, causing overlapping scan artefacts.
        slam.ICP_MAX_ROT_DEG = ICP_ROT_LIMIT

        # ── State ─────────────────────────────────────────────────────────────
        mode           = "mapping"
        nav_target     = None      # current navigation target room name
        last_haptic    = 0.0
        last_nav_spk   = 0.0
        last_pose_send = 0.0

        # ── Queues ────────────────────────────────────────────────────────────
        scan_q     : queue.Queue = queue.Queue(maxsize=2)
        voice_cmd_q: queue.Queue = queue.Queue()

        def _scan_thread():
            while not ctx.abort.is_set():
                s = adapter.get_scan(timeout=1.0)
                if s is not None and not scan_q.full():
                    scan_q.put_nowait(s)

        scan_t = threading.Thread(target=_scan_thread, daemon=True)
        scan_t.start()

        voice = _LidarVoice(voice_cmd_q)
        voice.start()

        saved = slam.list_rooms()
        ctx.feedback.speak(
            f"Lidar mapping. {len(saved)} rooms saved. "
            "Say save kitchen to save a room. "
            "Say take me to kitchen to navigate."
        )

        try:
            while not ctx.abort.is_set():

                # ── Voice commands ────────────────────────────────────────────
                try:
                    vcmd = voice_cmd_q.get_nowait()
                except queue.Empty:
                    vcmd = None

                if vcmd:
                    tag = vcmd[0]

                    if tag == "abort":
                        ctx.feedback.speak("Stopping lidar.")
                        ctx.abort.set()
                        break

                    elif tag == "list":
                        rooms = slam.list_rooms()
                        if rooms:
                            ctx.feedback.speak(
                                "Saved rooms: "
                                + ", ".join(r.replace("_", " ") for r in rooms) + "."
                            )
                        else:
                            ctx.feedback.speak("No rooms saved yet.")

                    elif tag == "save":
                        spoken_name = vcmd[1] if len(vcmd) > 1 and vcmd[1] else None
                        n = len(slam.list_rooms()) + 1
                        room_name = spoken_name or f"room_{n}"
                        _do_save(slam, ctx, room_name)

                    elif tag == "navigate":
                        spoken = vcmd[1] if len(vcmd) > 1 else ""
                        rooms  = slam.list_rooms()
                        if not rooms:
                            ctx.feedback.speak("No rooms saved. Save a room first.")
                        else:
                            matched = _match_room(spoken, rooms)
                            if matched:
                                mode         = "navigation"
                                nav_target   = matched
                                last_nav_spk = 0.0
                                slam.load_room(nav_target)
                                ctx.feedback.speak(
                                    f"Navigating to {nav_target.replace('_', ' ')}."
                                )
                            else:
                                ctx.feedback.speak(
                                    "Room not found. "
                                    "Saved rooms: "
                                    + ", ".join(r.replace("_", " ") for r in rooms) + "."
                                )

                # ── Gesture backups (glove) ───────────────────────────────────
                gesture = None
                if ctx.gesture_queue:
                    try:
                        gesture = ctx.gesture_queue.get_nowait()
                    except queue.Empty:
                        pass

                if gesture == "NEXT":
                    if mode == "mapping":
                        n = len(slam.list_rooms()) + 1
                        _do_save(slam, ctx, f"room_{n}")
                    else:
                        rooms = slam.list_rooms()
                        if rooms:
                            idx = (rooms.index(nav_target) + 1) % len(rooms) \
                                  if nav_target in rooms else 0
                            nav_target   = rooms[idx]
                            last_nav_spk = 0.0
                            slam.load_room(nav_target)
                            ctx.feedback.speak(
                                f"Navigating to {nav_target.replace('_', ' ')}."
                            )
                elif gesture == "EDIT":
                    if mode == "navigation":
                        mode       = "mapping"
                        nav_target = None
                        ctx.feedback.speak("Mapping mode. Say save name to save.")

                # ── SLAM update ───────────────────────────────────────────────
                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()

                # ── Obstacle detection — active in ALL modes ──────────────────
                if now - last_haptic >= HAPTIC_INTERVAL:
                    front_m = _sector_min(scan,   0)
                    left_m  = _sector_min(scan,  90)
                    right_m = _sector_min(scan, 270)

                    front_ms = _dist_ms(front_m)
                    left_ms  = _dist_ms(left_m)
                    right_ms = _dist_ms(right_m)

                    if front_ms:
                        ctx.feedback._pulse(3, front_ms)      # MT3 — front
                        last_haptic = now
                    elif left_ms:
                        ctx.feedback._pulse(2, left_ms)       # MT2 — left
                        last_haptic = now
                    elif right_ms:
                        ctx.feedback._pulse(1, right_ms)      # MT1 — right
                        last_haptic = now
                    elif mode == "navigation" and nav_target:
                        # No obstacle — fire direction cue toward target
                        d = slam.direction_to_room(nav_target)
                        if d:
                            _, _, _, bearing = d
                            if bearing > 15:
                                ctx.feedback._pulse(2, _bearing_ms(abs(bearing)))
                                last_haptic = now
                            elif bearing < -15:
                                ctx.feedback._pulse(1, _bearing_ms(abs(bearing)))
                                last_haptic = now

                # ── Pose update → server (navigation only) ────────────────────
                if (mode == "navigation" and nav_target and ctx.link
                        and now - last_pose_send >= POSE_UPDATE_S):
                    ctx.link.send("lidar", {
                        "action"    : "pose_update",
                        "room_name" : nav_target,
                        "x"         : float(result.pose.x),
                        "y"         : float(result.pose.y),
                    })
                    last_pose_send = now

                # ── Navigation TTS (periodic) ─────────────────────────────────
                if (mode == "navigation" and nav_target
                        and now - last_nav_spk >= NAV_SPEAK_S):
                    d = slam.direction_to_room(nav_target)
                    if d:
                        _, _, dist_m, bearing = d
                        ctx.feedback.speak(
                            f"{nav_target.replace('_', ' ')}: "
                            f"{dist_m:.1f} meters, {_bearing_str(bearing)}."
                        )
                    last_nav_spk = now

                # ── Room auto-recognition (mapping only) ──────────────────────
                if mode == "mapping" and result.room_match:
                    ctx.feedback.speak(
                        f"Recognised: {result.room_match.replace('_', ' ')}."
                    )

        finally:
            ctx.abort.set()
            voice.stop()
            scan_t.join(timeout=2.0)
            adapter.stop()
