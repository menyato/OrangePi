"""
engine.py — turn the 10 Hz SensorFrame stream into named gesture events.

Matching:
  * POSE  — the frame's bent-finger set must EXACTLY equal the spec's.
  * STATIC — orientation flag held for `hold_frames` consecutive frames.
  * FLICK  — orientation flag RISES from the resting baseline (clear -> set)
             while the pose is held. The baseline is simply "the flag was
             clear on the previous frame", which is the cheap, firmware-free
             way to detect a flick "from base to forward".

A refractory cooldown after each fire prevents one physical gesture from
emitting a burst of events.
"""

from gesture_hub.specs import GestureSpec, Motion


class GestureEngine:
    def __init__(self, gestures: dict[str, GestureSpec], on_event=None, fps: int = 10):
        self.gestures = gestures
        self.on_event = on_event                       # callable(name) -> None
        self.refractory_frames = max(1, int(0.7 * fps))
        self._prev_flags: dict[str, bool] = {}         # the rolling "baseline"
        self._static_count: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._static_count = {n: 0 for n in self.gestures}
        self._cooldown = {n: 0 for n in self.gestures}

    def set_gestures(self, gestures: dict[str, GestureSpec]) -> None:
        self.gestures = gestures
        self._reset_counters()

    def feed(self, frame) -> None:
        flags = frame.imu_flags
        bent = frozenset(i for i, b in enumerate(frame.finger_bent) if b)

        for name, spec in self.gestures.items():
            if self._cooldown.get(name, 0) > 0:
                self._cooldown[name] -= 1
                self._static_count[name] = 0
                continue

            pose_ok = (bent == spec.fingers_set())
            fired = False

            if spec.motion == Motion.STATIC:
                held = (spec.orientation is None) or flags.get(spec.orientation, False)
                if pose_ok and held:
                    self._static_count[name] = self._static_count.get(name, 0) + 1
                    if self._static_count[name] >= spec.hold_frames:
                        fired = True
                else:
                    self._static_count[name] = 0

            else:  # FLICK — rising edge from baseline
                rising = False
                if spec.orientation is not None:
                    prev = self._prev_flags.get(spec.orientation, False)
                    rising = flags.get(spec.orientation, False) and not prev
                if pose_ok and rising:
                    fired = True

            if fired:
                self._static_count[name] = 0
                self._cooldown[name] = self.refractory_frames
                if self.on_event:
                    try:
                        self.on_event(name)
                    except Exception as e:
                        print(f"[ENGINE] event handler error: {e}")

        # update the baseline last, so this frame is "previous" next time
        self._prev_flags = dict(flags)
