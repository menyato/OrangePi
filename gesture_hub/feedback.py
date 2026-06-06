"""
feedback.py — non-visual feedback for the hub.

Haptics use the glove's 3 vibration motors (via GloveController.vibrate).
Speech uses espeak-ng. On startup:
  1. All ALSA devices are listed for debugging.
  2. Common mixer controls are unmuted and set to 90%.
  3. The first working ALSA device is selected by testing it with a short tone.
  4. Errors are always printed — nothing fails silently.
"""

import array
import math
import os
import re
import subprocess
import tempfile
import time
import wave


# ── Audio device selection ────────────────────────────────────────────────────

def _list_alsa_devices() -> list[tuple[str, str]]:
    """Return [(alsa_dev, card_name), ...] for all playback devices."""
    try:
        out = subprocess.check_output(["aplay", "-l"],
                                      stderr=subprocess.DEVNULL, text=True)
        print("[AUDIO] Playback devices found:")
        entries = []
        for line in out.splitlines():
            m = re.search(r"card\s+(\d+):\s*(\S+)[^,]*,\s*device\s+(\d+)", line)
            if m:
                card, name, dev = m.group(1), m.group(2), m.group(3)
                alsa_dev = f"plughw:{card},{dev}"
                print(f"  {alsa_dev}  ({name})")
                entries.append((alsa_dev, name))
        if not entries:
            print("[AUDIO]   (none found)")
        return entries
    except FileNotFoundError:
        print("[AUDIO] aplay not found — install alsa-utils: sudo apt install alsa-utils")
        return []
    except Exception as e:
        print(f"[AUDIO] aplay -l error: {e}")
        return []


def _unmute_volume() -> None:
    """Try to unmute and set 90% volume on common OrangePi mixer controls."""
    controls = ["Speaker", "Headphone", "Master", "PCM", "DAC",
                "LINEOUT", "Line Out", "Output", "Digital"]
    for ctrl in controls:
        subprocess.run(
            ["amixer", "-q", "sset", ctrl, "90%", "unmute"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def _make_test_wav() -> str:
    """Write a 0.4-second 440 Hz sine wave to a temp file. Returns path."""
    rate = 16000
    samples = array.array(
        "h",
        [int(28000 * math.sin(2 * math.pi * 440 * i / rate))
         for i in range(int(rate * 0.4))]
    )
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


def _test_device(alsa_dev: str, wav_path: str) -> bool:
    """Return True if aplay can play wav_path on alsa_dev without error."""
    r = subprocess.run(
        ["aplay", "-D", alsa_dev, "-q", wav_path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, timeout=5
    )
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace").strip()
        print(f"[AUDIO]   {alsa_dev} failed: {err}")
        return False
    return True


def _auto_alsa() -> str | None:
    """
    Select the best working ALSA device:
      1. List all devices.
      2. Unmute mixer controls.
      3. Play a test tone on each device; return the first that works.
      4. Prefer non-HDMI devices.
    """
    _unmute_volume()

    devices = _list_alsa_devices()
    if not devices:
        return None

    wav = _make_test_wav()
    try:
        # Prefer non-HDMI
        ordered = (
            [(d, n) for d, n in devices if "hdmi" not in n.lower()] +
            [(d, n) for d, n in devices if "hdmi"     in n.lower()]
        )
        for alsa_dev, name in ordered:
            print(f"[AUDIO] Testing {alsa_dev} ({name}) ...")
            if _test_device(alsa_dev, wav):
                print(f"[AUDIO] Selected: {alsa_dev} ({name})")
                return alsa_dev
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass

    print("[AUDIO] No working ALSA device — espeak will use system default.")
    return None


# ── Feedback class ────────────────────────────────────────────────────────────

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
                    print(f"[AUDIO] aplay error ({self.alsa}): {err}")
                    # Fallback: let espeak use system default
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
            print("[AUDIO] espeak-ng not found — sudo apt install espeak-ng")
        except Exception as e:
            print(f"[AUDIO] speak error: {e}")
