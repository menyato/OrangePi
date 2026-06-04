"""
feedback.py — non-visual feedback for the hub.

Haptics use the glove's 3 vibration motors (via GloveController.vibrate).
Speech uses espeak-ng (always present per requirements). If an ALSA device is
given, audio is routed through it so it lands on the same speaker the money
feature uses; otherwise espeak plays on the system default.
"""

import subprocess
import time


class Feedback:
    def __init__(self, controller, alsa: str | None = None, speed: int = 150):
        self.ctrl = controller
        self.alsa = alsa
        self.speed = speed

    # ── haptics ──────────────────────────────────────────────────────────────
    def _pulse(self, motor: int, ms: int) -> None:
        try:
            self.ctrl.vibrate(motor, ms)
        except Exception:
            pass

    def confirm(self) -> None:           # single short buzz — "got it"
        self._pulse(1, 120)

    def tick(self) -> None:              # tiny blip — menu moved
        self._pulse(2, 60)

    def select(self) -> None:            # double buzz — launched
        self._pulse(1, 90); time.sleep(0.12); self._pulse(1, 90)

    def error(self) -> None:             # long buzz — rejected
        self._pulse(3, 400)

    def beep(self) -> None:              # haptic "go" marker for recording
        self._pulse(2, 150)

    # ── speech ───────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        print(f"[HUB] {text}")
        try:
            if self.alsa:
                es = subprocess.Popen(
                    ["espeak-ng", "-s", str(self.speed), "--stdout", text],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                ap = subprocess.Popen(
                    ["aplay", "-D", self.alsa, "-q"],
                    stdin=es.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                es.stdout.close()
                ap.wait()
            else:
                subprocess.run(["espeak-ng", "-s", str(self.speed), text],
                               stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[HUB] espeak error: {e}")
