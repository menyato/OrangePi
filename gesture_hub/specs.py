"""
specs.py — gesture definitions, stored as ATmega hardware bit-masks.

Instead of abstract finger names, every gesture is defined by two integers
that come directly off the wire from the ATmega compact frame:

    flex_mask  — which bits in SensorFrame.flex_bits must be SET
                 bit 0 = Thumb, 1 = Index, 2 = Middle, 3 = Ring, 4 = Pinky

    imu_mask   — which bits in SensorFrame.imu_bits must be SET
                 bit 0 = tilt_right,    bit 1 = tilt_left
                 bit 2 = tilt_forward,  bit 3 = tilt_backward
                 bit 4 = rotate_cw,     bit 5 = rotate_ccw

Motion:
    STATIC  — both masks must match and HOLD for `hold_frames` frames.
    FLICK   — both masks must match on a RISING edge (prev frame had
              imu_mask clear, this frame has it set) while flex matches.

Default system gestures (hardcoded bit values, always the same):
    START : Thumb(0) + Pinky(4)  → flex_mask 0x11 (bits 0,4)
            tilt_right            → imu_mask  0x01 (bit 0)
            STATIC, hold 3 frames

    NEXT  : Thumb(0) + Ring(3)   → flex_mask 0x09 (bits 0,3)
            tilt_backward         → imu_mask  0x08 (bit 3)
            FLICK

    EDIT  : Thumb(0) + Middle(2) → flex_mask 0x05 (bits 0,2)
            tilt_forward          → imu_mask  0x04 (bit 2)
            FLICK

Finger bit positions
--------------------
  Thumb=0  Index=1  Middle=2  Ring=3  Pinky=4

IMU bit positions (SensorFrame.imu_bits from glove_controller.py)
------------------------------------------------------------------
  tilt_right=0   tilt_left=1   tilt_forward=2  tilt_backward=3
  rotate_cw=4    rotate_ccw=5
"""

from dataclasses import dataclass
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
    STATIC = "static"   # hold both masks for hold_frames consecutive frames
    FLICK  = "flick"    # imu_mask rises from 0->set while flex_mask is held


@dataclass
class GestureSpec:
    name:        str
    flex_mask:   int           # ATmega flex_bits mask
    imu_mask:    int           # ATmega imu_bits mask (0 = pose-only)
    motion:      Motion
    hold_frames: int = 3       # STATIC debounce / FLICK refractory frames

    # ── human-readable helpers ───────────────────────────────────────────────
    def finger_names(self) -> list[str]:
        return [FINGER_NAMES[i] for i in range(5) if self.flex_mask & (1 << i)]

    def imu_names(self) -> list[str]:
        return [IMU_BIT_NAMES[i] for i in range(6) if self.imu_mask & (1 << i)]

    def describe(self) -> str:
        fingers = " + ".join(self.finger_names()) or "no fingers"
        imu     = " + ".join(self.imu_names())    or "any"
        if self.motion == Motion.FLICK:
            return f"{fingers}, flick {imu}"
        return f"{fingers}, hold {imu}"

    # ── serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "flex_mask":   self.flex_mask,
            "imu_mask":    self.imu_mask,
            "motion":      self.motion.value,
            "hold_frames": self.hold_frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GestureSpec":
        return cls(
            name        = d["name"],
            flex_mask   = int(d["flex_mask"]),
            imu_mask    = int(d["imu_mask"]),
            motion      = Motion(d["motion"]),
            hold_frames = int(d.get("hold_frames", 3)),
        )


# ── Built-in defaults ─────────────────────────────────────────────────────────
#
#   Flex masks:
#     Thumb(bit0) + Pinky(bit4)  = 0b10001 = 0x11 = 17
#     Thumb(bit0) + Ring(bit3)   = 0b01001 = 0x09 =  9
#     Thumb(bit0) + Middle(bit2) = 0b00101 = 0x05 =  5
#
#   IMU masks (from ATmega imu_bits):
#     tilt_right    = bit0 = 0x01
#     tilt_backward = bit3 = 0x08
#     tilt_forward  = bit2 = 0x04

DEFAULT_GESTURES: dict[str, GestureSpec] = {
    "START": GestureSpec("START", flex_mask=0x11, imu_mask=0x01,
                         motion=Motion.STATIC, hold_frames=3),
    "NEXT":  GestureSpec("NEXT",  flex_mask=0x09, imu_mask=0x08,
                         motion=Motion.FLICK,  hold_frames=2),
    "EDIT":  GestureSpec("EDIT",  flex_mask=0x05, imu_mask=0x04,
                         motion=Motion.FLICK,  hold_frames=2),
}