"""
recorder.py — capture a gesture "by example".

The hub routes raw frames here (instead of to the engine) during a recording
window. After the window closes, analyze() distils the frames into a
(pose, motion, orientation) sample. The hub captures TWO samples; if they are
equal, build_spec() turns the sample into a GestureSpec to save.
"""

from collections import Counter

from gesture_hub.specs import GestureSpec, Motion


class GestureRecorder:
    def __init__(self, fps: int = 10, window_s: float = 2.5):
        self.window_frames = int(window_s * fps)
        self.reset()

    def reset(self) -> None:
        self._frames: list[tuple[frozenset, dict]] = []
        self._prev_flags: dict | None = None
        self._risen: set[str] = set()      # flags that went clear -> set in window

    def feed(self, frame) -> None:
        flags = frame.imu_flags
        if self._prev_flags is not None:
            for k, v in flags.items():
                if v and not self._prev_flags.get(k, False):
                    self._risen.add(k)
        self._prev_flags = dict(flags)
        bent = frozenset(i for i, b in enumerate(frame.finger_bent) if b)
        self._frames.append((bent, flags))

    def analyze(self) -> tuple | None:
        """Return (pose:frozenset, motion:Motion, orientation:str|None) or None."""
        posed = [f for f in self._frames if f[0]]   # frames with at least one bent finger
        if not posed:
            return None

        pose = Counter(f[0] for f in posed).most_common(1)[0][0]

        held = Counter()
        for bent, flags in posed:
            if bent != pose:
                continue
            for k, v in flags.items():
                if v:
                    held[k] += 1

        if self._risen:
            candidates = [k for k in self._risen if held.get(k, 0) > 0] or sorted(self._risen)
            return (pose, Motion.FLICK, candidates[0])
        if held:
            return (pose, Motion.STATIC, held.most_common(1)[0][0])
        return (pose, Motion.STATIC, None)

    @staticmethod
    def build_spec(name: str, sample: tuple) -> GestureSpec:
        pose, motion, orientation = sample
        hold = 2 if motion == Motion.FLICK else 3
        return GestureSpec(name, tuple(sorted(pose)), motion, orientation, hold)
