"""
specs.py — gesture definitions, stored as ATmega hardware bit-masks.

flex_mask  — which bits in SensorFrame.flex_bits must be SET
             bit 0 = Thumb, 1 = Index, 2 = Middle, 3 = Ring, 4 = Pinky

imu_mask   — which bits in SensorFrame.imu_bits must be SET
             bit 0 = tilt_left,     bit 1 = tilt_right
             bit 2 = tilt_backward, bit 3 = tilt_forward
             bit 4 = rotate_cw,     bit 5 = rotate_ccw
             (bits 0-3 are swapped from the ATmega firmware's own bit order —
             the MPU sits mirrored on this glove, so its raw forward/back and
             left/right bits read backwards; corrected here in one place so
             every consumer of IMU_BIT_NAMES / imu_mask sees the true
             physical direction without reflashing the ATmega.)

flex_exact — if True the flex match is EXACT: frame.flex_bits == flex_mask.
             If False (default) it is a SUBSET match: extra bent fingers are
             allowed.

Motion:
    STATIC  — both masks must match and HOLD for `hold_frames` frames.
    FLICK   — imu_mask bits must RISE (0→set) while flex matches.

System gestures (hardcoded, never user-assignable):
    START     : Thumb+Pinky          0x11  tilt_right     STATIC  exact
    NEXT      : Pinky only           0x10  tilt_right     FLICK   exact
    EDIT      : Thumb+Middle         0x05  (any tilt)     STATIC  exact

Book-reader OCR gestures (only acted on while Book Reader is RUNNING):
    OCR_PAUSE : Thumb+Ring+Pinky     0x19  tilt_backward  STATIC  exact
    OCR_FWD   : Thumb only           0x01  tilt_right     FLICK   exact
    OCR_BWD   : Thumb only           0x01  tilt_left      FLICK   exact

All system gestures use flex_exact=True so that extra bent fingers never
accidentally trigger them.  OCR_PAUSE (0x19) is distinct from NEXT (0x09)
because NEXT requires exactly Thumb+Ring; adding Pinky changes the byte.
"""

from dataclasses import dataclass, field
from enum import Enum

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_INDEX = {n.lower(): i for i, n in enumerate(FINGER_NAMES)}

IMU_BIT_NAMES = [
    "tilt_left",     # bit 0
    "tilt_right",    # bit 1
    "tilt_backward", # bit 2
    "tilt_forward",  # bit 3
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
    # ── global navigation ─────────────────────────────────────────────────────
    "START": GestureSpec("START", flex_mask=0x11, imu_mask=0x02,
                         motion=Motion.STATIC, hold_frames=3, flex_exact=True),
    # Pinky only, flick wrist right — scroll / next
    "NEXT":  GestureSpec("NEXT",  flex_mask=0x10, imu_mask=0x02,
                         motion=Motion.FLICK,  hold_frames=2, flex_exact=True),
    # Thumb + Middle only, hold in any position — edit / change gesture
    "EDIT":  GestureSpec("EDIT",  flex_mask=0x05, imu_mask=0x00,
                         motion=Motion.STATIC, hold_frames=3, flex_exact=True),

    # ── Book Reader playback controls (only acted on inside OCR feature) ──────
    # Pause/resume: Thumb + Ring + Pinky closed, tilt wrist backward, hold.
    "OCR_PAUSE": GestureSpec("OCR_PAUSE", flex_mask=0x19, imu_mask=0x04,
                             motion=Motion.STATIC, hold_frames=3, flex_exact=True),
    # Skip forward ~5 s: Thumb only, flick wrist right.
    "OCR_FWD":   GestureSpec("OCR_FWD",   flex_mask=0x01, imu_mask=0x02,
                             motion=Motion.FLICK,  hold_frames=2, flex_exact=True),
    # Skip backward ~5 s: Thumb only, flick wrist left.
    "OCR_BWD":   GestureSpec("OCR_BWD",   flex_mask=0x01, imu_mask=0x01,
                             motion=Motion.FLICK,  hold_frames=2, flex_exact=True),
}