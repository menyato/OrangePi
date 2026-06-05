"""
recorder.py — capture a gesture "by example" using raw ATmega bit-masks.

During a recording window the hub routes raw SensorFrames here.
analyze() distils the captured frames into (flex_mask, imu_mask, motion):

    flex_mask — the most common bent-finger bitmask seen while any finger
                was bent, OR'd with itself so every finger present in the
                majority of frames makes it into the mask.

    imu_mask / motion — if an IMU bit ROSE during the window (clear → set)
                        while the pose was held → FLICK on the risen bits.
                        Otherwise → STATIC with the most frequently held
                        IMU bits.

The hub captures TWO samples; if they agree, build_spec() turns the sample
into a GestureSpec and saves it to gestures.json.
"""

from collections import Counter

from gesture_hub.specs import GestureSpec, Motion


class GestureRecorder:
    def __init__(self, fps: int = 10, window_s: float = 2.5):
        self.window_frames = int(window_s * fps)
        self.reset()

    def reset(self) -> None:
        self._frames: list[tuple[int, int]] = []   # (flex_bits, imu_bits)
        self._prev_imu: int | None = None
        self._risen_imu: int = 0                   # OR of all rising imu bits

    def feed(self, frame) -> None:
        flex = frame.flex_bits
        imu  = frame.imu_bits

        if self._prev_imu is not None:
            # bits that went 0 → 1 this frame
            self._risen_imu |= (~self._prev_imu & imu) & 0xFF

        self._prev_imu = imu
        self._frames.append((flex, imu))

    # ── analysis ──────────────────────────────────────────────────────────────
    def analyze(self) -> tuple | None:
        """Return (flex_mask:int, imu_mask:int, motion:Motion) or None."""
        posed = [(f, i) for f, i in self._frames if f != 0]
        if not posed:
            return None

        # ── flex_mask: most common flex_bits value while any finger is bent ──
        flex_counter = Counter(f for f, _ in posed)
        flex_mask    = flex_counter.most_common(1)[0][0]

        # ── restrict to frames where this pose is held ────────────────────────
        pose_frames = [(f, i) for f, i in posed
                       if (f & flex_mask) == flex_mask]
        if not pose_frames:
            pose_frames = posed   # fallback

        # ── imu during pose ───────────────────────────────────────────────────
        imu_counter = Counter()
        for _, imu in pose_frames:
            for bit in range(6):
                if imu & (1 << bit):
                    imu_counter[bit] += 1

        # risen IMU bits that were also present during the pose
        risen_during_pose = self._risen_imu
        # filter to bits that actually appeared during pose frames too
        pose_imu_bits = 0
        for bit, cnt in imu_counter.items():
            if cnt >= len(pose_frames) // 2:   # present in >50% of pose frames
                pose_imu_bits |= (1 << bit)

        risen_filtered = risen_during_pose & pose_imu_bits

        if risen_filtered:
            # FLICK — use the risen bits as the imu_mask
            return (flex_mask, risen_filtered, Motion.FLICK)
        elif pose_imu_bits:
            # STATIC — held orientation
            return (flex_mask, pose_imu_bits, Motion.STATIC)
        else:
            # pose only
            return (flex_mask, 0, Motion.STATIC)

    # ── spec builder ──────────────────────────────────────────────────────────
    @staticmethod
    def build_spec(name: str, sample: tuple) -> GestureSpec:
        flex_mask, imu_mask, motion = sample
        hold = 2 if motion == Motion.FLICK else 3
        return GestureSpec(name, flex_mask=flex_mask, imu_mask=imu_mask,
                           motion=motion, hold_frames=hold)