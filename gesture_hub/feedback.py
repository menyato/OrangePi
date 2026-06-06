"""
feedback.py — non-visual feedback for the hub.

Haptics use the glove's 3 vibration motors (via GloveController.vibrate).
Speech uses espeak-ng routed through the best available ALSA device.
Errors are always printed so audio problems are visible in the terminal.
"""

import re
import subprocess
import time


def _auto_alsa() -> str | None:
    """
    Find the best ALSA playback device.
    Prefers non-HDMI cards (3.5mm jack / I2S speaker on OrangePi).
    Falls back to the first card found, then to espeak system default.
    """
    try:
        out = subprocess.check_output(["aplay", "-l"],
                                      stderr=subprocess.DEVNULL, text=True)
        # Parse every card/device entry and its human name
        entries = re.findall(
            r"card\s+(\d+):\s*(\S+)[^,]*,\s*device\s+(\d+)", out
        )
        if not entries:
            print("[AUDIO] aplay -l returned no devices.")
            return None

        # Prefer non-HDMI devices (OrangePi card 0 is often HDMI)
        non_hdmi = [(c, n, d) for c, n, d in entries
                    if "hdmi" not in n.lower()]
        candidates = non_hdmi if non_hdmi else entries

        card, name, dev = candidates[0]
        alsa_dev = f"plughw:{card},{dev}"
        print(f"[AUDIO] ALSA device selected: {alsa_dev}  ({name})")
        return alsa_dev

    except FileNotFoundError:
        print("[AUDIO] aplay not found — install alsa-utils.")
    except Exception as e:
        print(f"[AUDIO] ALSA detection error: {e}")
    return None


class Feedback:
    def __init__(self, controller, alsa: str | None = None, speed: int = 150):
        self.ctrl  = controller
        self.alsa  = alsa if alsa is not None else _auto_alsa()
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                ap = subprocess.Popen(
                    ["aplay", "-D", self.alsa, "-q"],
                    stdin=es.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE)
                es.stdout.close()
                _, ap_err = ap.communicate()
                es.wait()
                if ap.returncode != 0:
                    err = ap_err.decode(errors="replace").strip()
                    print(f"[AUDIO] aplay error (device={self.alsa}): {err}")
                    # Try system default as fallback
                    subprocess.run(
                        ["espeak-ng", "-s", str(self.speed), text],
                        stderr=subprocess.PIPE)
            else:
                r = subprocess.run(
                    ["espeak-ng", "-s", str(self.speed), text],
                    stderr=subprocess.PIPE)
                if r.returncode != 0:
                    print(f"[AUDIO] espeak-ng error: "
                          f"{r.stderr.decode(errors='replace').strip()}")

        except FileNotFoundError:
            print("[AUDIO] espeak-ng not found — run: sudo apt install espeak-ng")
        except Exception as e:
            print(f"[AUDIO] speak error: {e}")
