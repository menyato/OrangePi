"""
engine.py — turn the 10 Hz SensorFrame stream into named gesture events.

Pose check:
    Default (flex_exact=False):
        (frame.flex_bits & spec.flex_mask) == spec.flex_mask
        Extra bent fingers are allowed.

    Exact (flex_exact=True):
        frame.flex_bits == spec.flex_mask
        Only the precise finger combination fires the gesture.
        Used for START so a fist (all fingers bent) does not trigger it.

Motion check:
    STATIC — imu check must HOLD for `hold_frames` consecutive frames.
    FLICK  — imu_mask bits must RISE (0→set) from the previous frame
             while the pose is already satisfied.

imu_mask == 0 means pose-only — IMU check always passes.

Refractory cooldown after each fire prevents burst events.
"""

from gesture_hub.specs import GestureSpec, Motion


class GestureEngine:
    def __init__(self, gestures: dict[str, GestureSpec], on_event=None, fps: int = 10):
        self.gestures          = gestures
        self.on_event          = on_event
        self.refractory_frames = max(1, int(0.7 * fps))
        self._prev_imu_bits:   int            = 0
        self._static_count:    dict[str, int] = {}
        self._cooldown:        dict[str, int] = {}
        self._reset_counters()

    # ── public API ────────────────────────────────────────────────────────────
    def set_gestures(self, gestures: dict[str, GestureSpec]) -> None:
        self.gestures = gestures
        self._reset_counters()

    def feed(self, frame) -> None:
        flex = frame.flex_bits
        imu  = frame.imu_bits

        for name, spec in self.gestures.items():
            # ── cooldown ──────────────────────────────────────────────────────
            if self._cooldown.get(name, 0) > 0:
                self._cooldown[name] -= 1
                self._static_count[name] = 0
                continue

            # ── pose check ────────────────────────────────────────────────────
            if spec.flex_exact:
                pose_ok = (flex == spec.flex_mask)
            else:
                pose_ok = (flex & spec.flex_mask) == spec.flex_mask

            fired = False

            if spec.motion == Motion.STATIC:
                imu_ok = (spec.imu_mask == 0) or \
                         ((imu & spec.imu_mask) == spec.imu_mask)
                if pose_ok and imu_ok:
                    self._static_count[name] = self._static_count.get(name, 0) + 1
                    if self._static_count[name] >= spec.hold_frames:
                        fired = True
                else:
                    self._static_count[name] = 0

            else:  # FLICK
                prev_imu_clear = (self._prev_imu_bits & spec.imu_mask) == 0
                this_imu_set   = (spec.imu_mask == 0) or \
                                 ((imu & spec.imu_mask) == spec.imu_mask)
                rising = prev_imu_clear and this_imu_set
                if pose_ok and rising:
                    fired = True
                elif not pose_ok:
                    self._static_count[name] = 0

            if fired:
                self._static_count[name] = 0
                self._cooldown[name]      = self.refractory_frames
                if self.on_event:
                    try:
                        self.on_event(name)
                    except Exception as e:
                        print(f"[ENGINE] event handler error: {e}")

        self._prev_imu_bits = imu

    # ── private ───────────────────────────────────────────────────────────────
    def _reset_counters(self) -> None:
        self._static_count  = {n: 0 for n in self.gestures}
        self._cooldown      = {n: 0 for n in self.gestures}
        self._prev_imu_bits = 0