"""
specs.py — gesture definitions, stored as ATmega hardware bit-masks.

flex_mask  — which bits in SensorFrame.flex_bits must be SET
             bit 0 = Thumb, 1 = Index, 2 = Middle, 3 = Ring, 4 = Pinky

imu_mask   — which bits in SensorFrame.imu_bits must be SET
             bit 0 = tilt_right,    bit 1 = tilt_left
             bit 2 = tilt_forward,  bit 3 = tilt_backward
             bit 4 = rotate_cw,     bit 5 = rotate_ccw

flex_exact — if True the flex match is EXACT: frame.flex_bits == flex_mask.
             If False (default) it is a SUBSET match: extra bent fingers are
             allowed.  System gesture START uses exact=True so a fist (all
             fingers bent) does NOT accidentally trigger it.

Motion:
    STATIC  — both masks must match and HOLD for `hold_frames` frames.
    FLICK   — imu_mask bits must RISE (0→set) while flex matches.

System gestures (hardcoded, never user-assignable):
    START : Thumb(0) + Pinky(4)   flex=0x11  imu=0x01  STATIC  exact=True
    NEXT  : Thumb(0) + Ring(3)    flex=0x09  imu=0x08  FLICK
    EDIT  : Thumb(0) + Middle(2)  flex=0x05  imu=0x04  FLICK
"""

from dataclasses import dataclass, field
from enum import Enum

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_INDEX = {n.lower(): i for i, n in enumerate(FINGER_NAMES)}

IMU_BIT_NAMES = [
    "tilt_right",    # bit 0
    "tilt_left",     # bit 1
    "tilt_forward",  # bit 2
    "tilt_backward", # bit 3
    "rotate_cw",     # bit 4
    "rotate_ccw",    # bit 5
]
IMU_BIT_INDEX = {n: i for i, n in enumerate(IMU_BIT_NAMES)}


class Motion(str, Enum):
    STATIC = "static"
    FLICK  = "flick"


@dataclass
class GestureSpec:
    name:        str
    flex_mask:   int
    imu_mask:    int
    motion:      Motion
    hold_frames: int  = 3
    flex_exact:  bool = False   # True → frame.flex_bits must == flex_mask exactly

    # ── human-readable helpers ───────────────────────────────────────────────
    def finger_names(self) -> list[str]:
        return [FINGER_NAMES[i] for i in range(5) if self.flex_mask & (1 << i)]

    def imu_names(self) -> list[str]:
        return [IMU_BIT_NAMES[i] for i in range(6) if self.imu_mask & (1 << i)]

    def describe(self) -> str:
        fingers = " + ".join(self.finger_names()) or "no fingers"
        imu     = " + ".join(self.imu_names())    or "any"
        exact   = " [exact]" if self.flex_exact else ""
        if self.motion == Motion.FLICK:
            return f"{fingers}{exact}, flick {imu}"
        return f"{fingers}{exact}, hold {imu}"

    # ── serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "flex_mask":   self.flex_mask,
            "imu_mask":    self.imu_mask,
            "motion":      self.motion.value,
            "hold_frames": self.hold_frames,
            "flex_exact":  self.flex_exact,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GestureSpec":
        return cls(
            name        = d["name"],
            flex_mask   = int(d["flex_mask"]),
            imu_mask    = int(d["imu_mask"]),
            motion      = Motion(d["motion"]),
            hold_frames = int(d.get("hold_frames", 3)),
            flex_exact  = bool(d.get("flex_exact", False)),
        )


# ── Built-in system gestures ──────────────────────────────────────────────────
#   START uses flex_exact=True so only Thumb+Pinky fires it,
#   not a full fist or any other superset of those two fingers.

DEFAULT_GESTURES: dict[str, GestureSpec] = {
    "START": GestureSpec("START", flex_mask=0x11, imu_mask=0x01,
                         motion=Motion.STATIC, hold_frames=3,
                         flex_exact=True),   # ← exact: only Thumb+Pinky
    "NEXT":  GestureSpec("NEXT",  flex_mask=0x09, imu_mask=0x08,
                         motion=Motion.FLICK,  hold_frames=2),
    "EDIT":  GestureSpec("EDIT",  flex_mask=0x05, imu_mask=0x04,
                         motion=Motion.FLICK,  hold_frames=2),
}