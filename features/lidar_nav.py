"""
features/lidar_nav.py — LiDAR mapping, localization, and navigation.

Motor vibration for directions (called from Feedback._pulse):
  MT 1 (right)  — turn right
  MT 2 (left)   — turn left
  MT 3 (bottom) — obstacle / proximity warning

Modes
  MAPPING    : builds occupancy map via SLAM; NEXT saves the room; EDIT
               switches to navigation if saved rooms exist.
  NAVIGATING : loaded a saved room; continuous haptic + periodic TTS
               guides the user toward the target origin.

Gesture controls (via ctx.gesture_queue while feature is RUNNING)
  NEXT  → (mapping) save current map as "room_N"
        → (navigating) cycle to next saved room as target
  EDIT  → (mapping) switch to navigation mode (first saved room)
        → (navigating) switch back to mapping mode

Port is set by hub.py via the --lidar-port argument and stored in the
class attribute LidarNavigation.port before the feature runs.
"""

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
DEFAULT_LIDAR_PORT   = "/dev/ttyUSB0"
LIDAR_BAUD           = 230400

SLAM_RESOLUTION      = 0.05   # metres per grid cell
SLAM_SIZE_M          = 30.0   # map extent

# Front arc for obstacle detection (radians either side of 0)
FORWARD_ARC_RAD      = math.radians(45.0)

# Proximity thresholds (metres)
PROX_DANGER_M        = 0.50   # MT3 strong pulse
PROX_CLOSE_M         = 0.80   # MT3 medium pulse
PROX_WARN_M          = 1.50   # MT3 light pulse

# Direction dead-zone: inside ±BEARING_DZ degrees no lateral vibration fires
BEARING_DZ_DEG       = 15.0

# How often to speak a navigation status update
NAV_ANNOUNCE_S       = 6.0

# Minimum interval between any haptic pulse (seconds)
HAPTIC_MIN_INTERVAL  = 0.25


# ─────────────────────────────────────────────────────────────────────────────

def _front_min_range(scan) -> float:
    """Minimum valid range in the forward ±45° arc of a LaserScan."""
    mask = (
        (scan.angles_rad <= FORWARD_ARC_RAD) |
        (scan.angles_rad >= 2 * math.pi - FORWARD_ARC_RAD)
    ) & (scan.ranges_m > 0.0)
    if not mask.any():
        return float("inf")
    return float(scan.ranges_m[mask].min())


def _bearing_to_words(bearing_deg: float) -> str:
    if bearing_deg > BEARING_DZ_DEG:
        return "turn left"
    if bearing_deg < -BEARING_DZ_DEG:
        return "turn right"
    return "straight ahead"


# ─────────────────────────────────────────────────────────────────────────────

class LidarNavigation(Feature):
    name  = "lidar"
    title = "Lidar Navigation"

    # Set from hub.py --lidar-port before run() is called
    port: str = DEFAULT_LIDAR_PORT

    def run(self, ctx: FeatureContext) -> None:
        if not _LIDAR_OK:
            ctx.feedback.speak(
                "Lidar libraries not installed. "
                "Run: pip install pyserial numpy scipy"
            )
            return

        # ── Open serial adapter ───────────────────────────────────────────────
        try:
            adapter = MS200Adapter(
                port          = self.port,
                baud          = LIDAR_BAUD,
                median_kernel = 5,
                max_jump_m    = 0.5,
            )
            adapter.start()
        except Exception as e:
            ctx.feedback.speak(f"Lidar error: {e}")
            return

        # ── SLAM engine ───────────────────────────────────────────────────────
        slam = SLAMEngine(
            map_resolution = SLAM_RESOLUTION,
            map_size_m     = SLAM_SIZE_M,
            db_dir         = ROOMS_DIR,
            debug          = False,
        )

        # ── Mutable state ─────────────────────────────────────────────────────
        mode          = "mapping"
        nav_target    = None        # name of room we're navigating toward
        last_announce = 0.0
        last_haptic   = 0.0

        # Background scan queue — adapter thread pushes, main loop pops
        scan_q: "queue.Queue" = queue.Queue(maxsize=2)

        def _scan_thread():
            while not ctx.abort.is_set():
                scan = adapter.get_scan(timeout=1.0)
                if scan is not None and not scan_q.full():
                    scan_q.put_nowait(scan)

        t = threading.Thread(target=_scan_thread, daemon=True)
        t.start()

        saved = slam.list_rooms()
        mode_hint = (
            "Next to save a room. Edit to switch to navigation."
            if not saved else
            f"{len(saved)} rooms saved. Edit to navigate."
        )
        ctx.feedback.speak(f"Lidar navigation. Mapping mode. {mode_hint}")

        try:
            while not ctx.abort.is_set():

                # ── Gesture commands ──────────────────────────────────────────
                gesture = None
                if ctx.gesture_queue:
                    try:
                        gesture = ctx.gesture_queue.get_nowait()
                    except queue.Empty:
                        pass

                if gesture == "NEXT":
                    if mode == "mapping":
                        if len(slam.keyframes) < 3:
                            ctx.feedback.speak(
                                "Not enough data yet. Walk around to build the map."
                            )
                        else:
                            n = len(slam.list_rooms()) + 1
                            room_name = f"room_{n}"
                            slam.save_room(room_name)
                            ctx.feedback.speak(f"Room {n} saved.")
                    else:
                        # Cycle to next saved room
                        rooms = slam.list_rooms()
                        if rooms:
                            idx = (rooms.index(nav_target) + 1) % len(rooms) if nav_target in rooms else 0
                            nav_target = rooms[idx]
                            slam.load_room(nav_target)
                            last_announce = 0.0
                            ctx.feedback.speak(f"Navigating to {nav_target}.")

                elif gesture == "EDIT":
                    if mode == "mapping":
                        rooms = slam.list_rooms()
                        if not rooms:
                            ctx.feedback.speak(
                                "No saved rooms. Save a map first with Next."
                            )
                        else:
                            mode = "navigation"
                            nav_target = rooms[0]
                            slam.load_room(nav_target)
                            last_announce = 0.0
                            ctx.feedback.speak(
                                f"Navigation mode. Navigating to {nav_target}. "
                                "Next to cycle targets. Edit to return to mapping."
                            )
                    else:
                        mode = "mapping"
                        nav_target = None
                        ctx.feedback.speak("Mapping mode.")

                # ── SLAM update ───────────────────────────────────────────────
                try:
                    scan = scan_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                pts    = MS200Adapter.to_xy(scan)
                result = slam.update(pts, rpm=scan.rpm)
                now    = time.time()

                # ── Proximity haptic (MT3 — bottom) ───────────────────────────
                if now - last_haptic >= HAPTIC_MIN_INTERVAL:
                    min_front = _front_min_range(scan)

                    if min_front < PROX_DANGER_M:
                        ctx.feedback._pulse(3, 350)
                        last_haptic = now
                    elif min_front < PROX_CLOSE_M:
                        ctx.feedback._pulse(3, 180)
                        last_haptic = now
                    elif min_front < PROX_WARN_M:
                        ctx.feedback._pulse(3, 80)
                        last_haptic = now
                    else:
                        # No proximity issue → direction vibration can fire
                        if mode == "navigation" and nav_target:
                            d = slam.direction_to_room(nav_target)
                            if d:
                                _, _, dist_m, bearing_deg = d

                                if bearing_deg > BEARING_DZ_DEG:
                                    # Need to turn left → MT2
                                    ms = _bearing_to_ms(bearing_deg)
                                    ctx.feedback._pulse(2, ms)
                                    last_haptic = now
                                elif bearing_deg < -BEARING_DZ_DEG:
                                    # Need to turn right → MT1
                                    ms = _bearing_to_ms(abs(bearing_deg))
                                    ctx.feedback._pulse(1, ms)
                                    last_haptic = now

                # ── Periodic TTS status (navigation mode only) ────────────────
                if mode == "navigation" and nav_target and now - last_announce >= NAV_ANNOUNCE_S:
                    d = slam.direction_to_room(nav_target)
                    if d:
                        _, _, dist_m, bearing_deg = d
                        direction = _bearing_to_words(bearing_deg)
                        ctx.feedback.speak(
                            f"{nav_target}: {dist_m:.1f} meters, {direction}."
                        )
                    last_announce = now

                # ── Room recognition TTS (mapping mode) ───────────────────────
                if mode == "mapping" and result.room_match:
                    ctx.feedback.speak(f"Recognised: {result.room_match}.")

        finally:
            ctx.abort.set()
            t.join(timeout=2.0)
            adapter.stop()


def _bearing_to_ms(abs_bearing_deg: float) -> int:
    """Convert bearing magnitude to haptic pulse duration (ms)."""
    if abs_bearing_deg >= 60:
        return 220
    if abs_bearing_deg >= 30:
        return 130
    return 70
