"""
features/lidar_nav.py — LiDAR SLAM: mapping, saving, navigation, obstacle avoidance.

Motor vibration
  MT 1 (right)  — obstacle on the RIGHT side  /  turn-right navigation cue
  MT 2 (left)   — obstacle on the LEFT side   /  turn-left  navigation cue
  MT 3 (bottom) — obstacle straight AHEAD     /  proximity danger

Obstacle sectors (each ±45° wide)
  Front  0°   → MT3
  Left   90°  → MT2
  Right  270° → MT1
  Back   180° → ignored (no motor)

Pulse strength scales with distance:
  > 1.5 m  → silent
  1.0–1.5  → 70 ms  (light)
  0.6–1.0  → 160 ms (medium)
  0.3–0.6  → 300 ms (strong)
  < 0.3 m  → 420 ms (danger)

Modes
  MAPPING    : SLAM builds occupancy map.
               NEXT  → save room + send map image to server.
               After save → auto-transition to NAVIGATION.
  NAVIGATION : directional haptics + periodic TTS guide to target.
               NEXT  → cycle saved rooms as target.
               EDIT  → return to MAPPING.

Gesture controls (ctx.gesture_queue)
  NEXT  → (mapping)    save current map
        → (navigation) next target room
  EDIT  → (navigation) back to mapping mode

Port defaults to "auto" — scans /dev/ttyUSB* and /dev/ttyACM* and uses the
first that streams valid MS200 frames (same strategy as audio/camera probing).
Override with --lidar-port /dev/ttyACM0.
"""

import io
import math
import queue
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

# ── tunables ──────────────────────────────────────────────────────────────────
DEFAULT_PORT    = "auto"
LIDAR_BAUD      = 230400
SLAM_RES        = 0.05    # m per grid cell
SLAM_SIZE_M     = 30.0

SECTOR_HALF_DEG = 45.0    # half-angle of each obstacle sector

# Distance → motor pulse duration (ms).  0 = silent.
_DIST_LEVELS = [
    (0.30, 420),
    (0.60, 300),
    (1.00, 160),
    (1.50,  70),
]

# Minimum time between haptic pulses (seconds)
HAPTIC_INTERVAL = 0.22

# Seconds between spoken navigation updates
NAV_SPEAK_S = 6.0

# Min keyframes before a save is meaningful
MIN_KF_TO_SAVE = 5


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _sector_min(scan, center_deg: float, half_deg: float = SECTOR_HALF_DEG) -> float:
    """Minimum valid range (m) inside [center ± half_deg] sector."""
    lo = math.radians((center_deg - half_deg) % 360)
    hi = math.radians((center_deg + half_deg) % 360)
    a  = scan.angles_rad
    r  = scan.ranges_m
    if lo <= hi:
        mask = (a >= lo) & (a <= hi) & (r > 0)
    else:                            # sector wraps around 0°
        mask = ((a >= lo) | (a <= hi)) & (r > 0)
    return float(r[mask].min()) if mask.any() else float("inf")


def _dist_ms(dist_m: float) -> int:
    """Convert obstacle distance to motor pulse length (ms). 0 = no pulse."""
    for threshold, ms in _DIST_LEVELS:
        if dist_m < threshold:
            return ms
    return 0


def _bearing_str(deg: float) -> str:
    if deg > 15:  return "turn left"
    if deg < -15: return "turn right"
    return "straight ahead"


def _bearing_ms(abs_deg: float) -> int:
    if abs_deg >= 60: return 220
    if abs_deg >= 30: return 130
    return 70


# ─── map → PNG bytes ──────────────────────────────────────────────────────────

def _map_to_png(slam: "SLAMEngine") -> Optional[bytes]:
    """Convert the occupancy grid to PNG bytes for server upload."""
    img = slam.occ_map.to_image()          # (N, N) uint8
    # try cv2 first (already in venv as symlink), fall back to PIL
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
    # last resort: raw PGM bytes (server can still handle as image)
    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode()
    return header + img.tobytes()


# ─────────────────────────────────────────────────────────────────────────────

class LidarNavigation(Feature):
    name  = "lidar"
    title = "Lidar Navigation"

    # Overridden by hub.py --lidar-port
    port: str = DEFAULT_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak(
                "Lidar libraries not installed. "
                "Run: pip install pyserial numpy scipy"
            )
            return

        # ── Port resolution ───────────────────────────────────────────────────
        port = self.port
        if port == "auto":
            ctx.feedback.speak("Searching for lidar.")
            port = MS200Adapter.find_port(baud=LIDAR_BAUD, timeout=2.0)
            if port is None:
                ctx.feedback.speak(
                    "Lidar not found. Check the USB cable and try again."
                )
                return
            ctx.feedback.speak("Lidar found.")

        # ── Open adapter ──────────────────────────────────────────────────────
        try:
            adapter = MS200Adapter(
                port=port, baud=LIDAR_BAUD,
                median_kernel=5, max_jump_m=0.5,
            )
            adapter.start()
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        # ── SLAM engine ───────────────────────────────────────────────────────
        slam = SLAMEngine(
            map_resolution=SLAM_RES,
            map_size_m=SLAM_SIZE_M,
            db_dir=ROOMS_DIR,
            debug=False,
        )

        # ── State ─────────────────────────────────────────────────────────────
        mode         = "mapping"
        nav_target   = None
        last_haptic  = 0.0
        last_nav_spk = 0.0

        # Background scan thread
        scan_q: queue.Queue = queue.Queue(maxsize=2)

        def _scan_thread():
            while not ctx.abort.is_set():
                s = adapter.get_scan(timeout=1.0)
                if s is not None and not scan_q.full():
                    scan_q.put_nowait(s)

        t = threading.Thread(target=_scan_thread, daemon=True)
        t.start()

        saved = slam.list_rooms()
        ctx.feedback.speak(
            f"Lidar mapping. {len(saved)} rooms saved. "
            "Next to save this room. Edit to navigate."
        )

        try:
            while not ctx.abort.is_set():

                # ── Gestures ──────────────────────────────────────────────────
                gesture = None
                if ctx.gesture_queue:
                    try:
                        gesture = ctx.gesture_queue.get_nowait()
                    except queue.Empty:
                        pass

                if gesture == "NEXT":
                    if mode == "mapping":
                        _do_save(slam, ctx, port)
                        # transition to navigation after first save
                        rooms = slam.list_rooms()
                        if rooms:
                            mode       = "navigation"
                            nav_target = rooms[-1]   # just-saved room
                            slam.load_room(nav_target)
                            last_nav_spk = 0.0
                            ctx.feedback.speak(
                                f"Navigating to {nav_target}. "
                                "Next to cycle targets. Edit to return to mapping."
                            )
                    else:
                        rooms = slam.list_rooms()
                        if rooms:
                            idx        = (rooms.index(nav_target) + 1) % len(rooms) \
                                         if nav_target in rooms else 0
                            nav_target = rooms[idx]
                            slam.load_room(nav_target)
                            last_nav_spk = 0.0
                            ctx.feedback.speak(f"Navigating to {nav_target}.")

                elif gesture == "EDIT":
                    if mode == "navigation":
                        mode       = "mapping"
                        nav_target = None
                        ctx.feedback.speak("Mapping mode. Next to save.")

                # ── SLAM update ───────────────────────────────────────────────
                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()

                # ── Obstacle detection — all three sectors ────────────────────
                if now - last_haptic >= HAPTIC_INTERVAL:
                    front_m = _sector_min(scan,   0)
                    left_m  = _sector_min(scan,  90)
                    right_m = _sector_min(scan, 270)

                    # Priority: front > left > right
                    front_ms = _dist_ms(front_m)
                    left_ms  = _dist_ms(left_m)
                    right_ms = _dist_ms(right_m)

                    if front_ms:
                        ctx.feedback._pulse(3, front_ms)   # MT3 — front
                        last_haptic = now
                    elif left_ms:
                        ctx.feedback._pulse(2, left_ms)    # MT2 — left
                        last_haptic = now
                    elif right_ms:
                        ctx.feedback._pulse(1, right_ms)   # MT1 — right
                        last_haptic = now
                    elif mode == "navigation" and nav_target:
                        # No obstacle — fire direction cue
                        d = slam.direction_to_room(nav_target)
                        if d:
                            _, _, _, bearing = d
                            if bearing > 15:
                                ctx.feedback._pulse(2, _bearing_ms(abs(bearing)))
                                last_haptic = now
                            elif bearing < -15:
                                ctx.feedback._pulse(1, _bearing_ms(abs(bearing)))
                                last_haptic = now

                # ── Navigation TTS ────────────────────────────────────────────
                if mode == "navigation" and nav_target and now - last_nav_spk >= NAV_SPEAK_S:
                    d = slam.direction_to_room(nav_target)
                    if d:
                        _, _, dist_m, bearing = d
                        ctx.feedback.speak(
                            f"{nav_target}: {dist_m:.1f} meters, {_bearing_str(bearing)}."
                        )
                    last_nav_spk = now

                # ── Room auto-recognition (mapping only) ──────────────────────
                if mode == "mapping" and result.room_match:
                    ctx.feedback.speak(f"Recognised: {result.room_match}.")

        finally:
            ctx.abort.set()
            t.join(timeout=2.0)
            adapter.stop()


# ─── save helper ─────────────────────────────────────────────────────────────

def _do_save(slam: "SLAMEngine", ctx: "FeatureContext", port: str) -> None:
    """Save current map, speak result, upload PNG to server."""
    if len(slam.keyframes) < MIN_KF_TO_SAVE:
        ctx.feedback.speak(
            f"Not enough data yet — only {len(slam.keyframes)} keyframes. "
            "Walk more to build the map."
        )
        return

    n         = len(slam.list_rooms()) + 1
    room_name = f"room_{n}"
    slam.save_room(room_name)
    ctx.feedback.speak(f"Room {n} saved.")

    # Send map image to server (best-effort)
    png = _map_to_png(slam)
    if png and ctx.link:
        reply = ctx.link.send("lidar", {
            "action"    : "map_save",
            "room_name" : room_name,
            "frame"     : png,          # ServerLink base64-encodes bytes
        })
        if reply:
            print(f"[LIDAR] Server reply: {reply}")
        else:
            print("[LIDAR] Map sent but no reply (server may not handle lidar yet).")
