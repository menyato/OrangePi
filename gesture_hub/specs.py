"""
specs.py — gesture definitions.

A gesture has two parts:
  * a finger POSE  — the exact set of bent fingers (from SensorFrame.flex_bits)
  * a MOTION       — how the IMU must behave while the pose is held:
        STATIC  -> an orientation flag must be held       (e.g. tilt_right)
        FLICK   -> an orientation flag must RISE from the resting baseline,
                   i.e. go clear -> set ("from base to forward")

Finger indices follow glove_controller.FINGER_NAMES:
    0 Thumb   1 Index   2 Middle   3 Ring   4 Pinky

Orientation keys follow SensorFrame.imu_flags:
    tilt_right tilt_left tilt_forward tilt_backward rotate_cw rotate_ccw
"""

from dataclasses import dataclass
from enum import Enum

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_INDEX = {n.lower(): i for i, n in enumerate(FINGER_NAMES)}


class Motion(str, Enum):
    STATIC = "static"   # orientation flag held together with the pose
    FLICK = "flick"     # orientation flag rises from baseline (clear -> set)


@dataclass
class GestureSpec:
    name: str
    fingers: tuple          # required bent finger indices, e.g. (0, 4)
    motion: Motion
    orientation: str | None  # imu_flags key, or None for "pose only"
    hold_frames: int = 3     # frames the condition must persist (STATIC debounce)

    def fingers_set(self) -> frozenset:
        return frozenset(self.fingers)

    def describe(self) -> str:
        names = " + ".join(FINGER_NAMES[i] for i in self.fingers) or "no fingers"
        if self.motion == Motion.FLICK:
            return f"{names}, flick {self.orientation or 'any'}"
        return f"{names}, hold {self.orientation or 'flat'}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fingers": list(self.fingers),
            "motion": self.motion.value,
            "orientation": self.orientation,
            "hold_frames": self.hold_frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GestureSpec":
        return cls(
            name=d["name"],
            fingers=tuple(d["fingers"]),
            motion=Motion(d["motion"]),
            orientation=d.get("orientation"),
            hold_frames=int(d.get("hold_frames", 3)),
        )


# ── Built-in defaults (used when gestures.json is missing/corrupt) ────────────
#   START : Thumb + Pinky, held right tilt              -> wake / select
#   NEXT  : Thumb + Ring,  flick up (tilt_backward)     -> scroll
#   EDIT  : Thumb + Index, flick down (tilt_forward)    -> edit a gesture
DEFAULT_GESTURES: dict[str, GestureSpec] = {
    "START": GestureSpec("START", (0, 4), Motion.STATIC, "tilt_right", 3),
    "NEXT":  GestureSpec("NEXT",  (0, 3), Motion.FLICK,  "tilt_backward", 2),
    "EDIT":  GestureSpec("EDIT",  (0, 1), Motion.FLICK,  "tilt_forward", 2),
}
