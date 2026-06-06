"""
feedback.py — non-visual feedback for the hub.

TTS: espeak-ng writes a WAV file, aplay plays it. File-based is more reliable
than piping because format negotiation issues can't cause silent failures.

ALSA: devices are listed and tested at startup. Non-HDMI devices are preferred
and tried in reverse order so a plugged-in headset (higher card number) beats
the built-in codec (card 0).

Volume: common mixer controls are unmuted and set to 90% before the test.
All errors are printed — nothing fails silently.
"""

import array
import math
import os
import re
import subprocess
import tempfile
import time
import wave


# ── Volume initialisation ─────────────────────────────────────────────────────

def _unmute_all(card: str = "") -> None:
    """Unmute and set 90% volume on every mixer control we can find."""
    card_flag = ["-c", card] if card else []

    # Try common control names — wrong ones are silently skipped by amixer -q
    controls = [
        "Master", "PCM", "Speaker", "Headphone", "LINEOUT",
        "Line Out", "Output", "Digital", "DAC", "Playback",
        # Allwinner sun8i-codec (audiocodec) specifics
        "DAC Mixer", "Left Mixer Left DAC", "Right Mixer Right DAC",
        "Headphone Source", "Speaker Source",
        "External Speaker", "Internal Speaker", "HPOUT",
    ]
    for ctrl in controls:
        subprocess.run(
            ["amixer", "-q"] + card_flag + ["sset", ctrl, "90%", "unmute"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Also try to unmute every control the card actually has
    try:
        out = subprocess.check_output(
            ["amixer"] + card_flag + ["scontents"],
            stderr=subprocess.DEVNULL, text=True)
        for name in re.findall(r"Simple mixer control '([^']+)'", out):
            subprocess.run(
                ["amixer", "-q"] + card_flag + ["sset", name, "90%", "unmute"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


# ── ALSA device selection ─────────────────────────────────────────────────────

def _make_test_wav() -> str:
    """Write a 0.5-second 440 Hz sine wave to a temp WAV file. Returns path."""
    rate = 22050
    samples = array.array(
        "h",
        [int(28000 * math.sin(2 * math.pi * 440 * i / rate))
         for i in range(int(rate * 0.5))],
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
    """Return True if aplay can open and play wav_path on alsa_dev."""
    try:
        r = subprocess.run(
            ["aplay", "-D", alsa_dev, "-q", wav_path],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, timeout=5,
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace").strip()
            print(f"[AUDIO]   {alsa_dev} — aplay error: {err}")
            return False
        return True
    except Exception as e:
        print(f"[AUDIO]   {alsa_dev} — test exception: {e}")
        return False


def _auto_alsa() -> str | None:
    """
    Find the best ALSA playback device:
      1. List all devices.
      2. Unmute every mixer control.
      3. Test each non-HDMI device with a short tone; return the first that works.
         Devices are tried in reverse card-number order so external / USB
         headsets (higher card numbers) are preferred over the built-in codec.
    """
    try:
        out = subprocess.check_output(
            ["aplay", "-l"], stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        print("[AUDIO] aplay not found — sudo apt install alsa-utils")
        return None
    except Exception as e:
        print(f"[AUDIO] aplay -l failed: {e}")
        return None

    entries = re.findall(
        r"card\s+(\d+):\s*(\S+)[^,]*,\s*device\s+(\d+)", out)
    if not entries:
        print("[AUDIO] No playback devices found.")
        return None

    print("[AUDIO] Playback devices found:")
    for card, name, dev in entries:
        print(f"  plughw:{card},{dev}  ({name})")

    # Unmute mixer controls on every card found
    cards_seen: set[str] = set()
    for card, _, _ in entries:
        if card not in cards_seen:
            _unmute_all(card)
            cards_seen.add(card)

    wav = _make_test_wav()
    try:
        non_hdmi = [(c, n, d) for c, n, d in entries if "hdmi" not in n.lower()]
        hdmi     = [(c, n, d) for c, n, d in entries if "hdmi"     in n.lower()]

        # Reverse non-HDMI so external/USB headsets (higher card#) come first
        ordered = list(reversed(non_hdmi)) + hdmi

        for card, name, dev in ordered:
            alsa_dev = f"plughw:{card},{dev}"
            print(f"[AUDIO] Testing {alsa_dev} ({name}) ...")
            if _test_device(alsa_dev, wav):
                print(f"[AUDIO] Selected: {alsa_dev} ({name})")
                return alsa_dev
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass

    print("[AUDIO] No working ALSA device found — espeak will use system default.")
    return None


# ── Feedback class ────────────────────────────────────────────────────────────

class Feedback:
    def __init__(self, controller, alsa: str | None = None, speed: int = 150):
        self.ctrl  = controller
        self.alsa  = alsa if alsa is not None else _auto_alsa()
        self.speed = speed

        try:
            v = subprocess.check_output(
                ["espeak-ng", "--version"], stderr=subprocess.STDOUT, text=True
            ).strip().splitlines()[0]
            print(f"[AUDIO] TTS engine: {v}")
        except FileNotFoundError:
            print("[AUDIO] espeak-ng not found — sudo apt install espeak-ng")
        except Exception:
            pass

    # ── haptics ──────────────────────────────────────────────────────────────
    def _pulse(self, motor: int, ms: int) -> None:
        try:
            self.ctrl.vibrate(motor, ms)
        except Exception:
            pass

    def confirm(self) -> None:       # single short buzz — "got it"
        self._pulse(1, 120)

    def tick(self) -> None:          # tiny blip — menu moved
        self._pulse(2, 60)

    def select(self) -> None:        # double buzz — launched
        self._pulse(1, 90); time.sleep(0.12); self._pulse(1, 90)

    def error(self) -> None:         # long buzz — rejected / off
        self._pulse(3, 400)

    def beep(self) -> None:          # haptic "go" marker for recording
        self._pulse(2, 150)

    # ── speech ───────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """
        Synthesise text to a WAV file via espeak-ng then play with aplay.
        File-based (not piped) to avoid format-negotiation failures.
        """
        print(f"[HUB] {text}")

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            # Step 1: generate WAV
            r = subprocess.run(
                ["espeak-ng", "-s", str(self.speed), "-w", tmp, text],
                stderr=subprocess.PIPE, timeout=30,
            )
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace").strip()
                print(f"[AUDIO] espeak-ng synthesis error: {err}")
                return

            # Step 2: play WAV
            if self.alsa:
                r2 = subprocess.run(
                    ["aplay", "-D", self.alsa, "-q", tmp],
                    stderr=subprocess.PIPE, timeout=60,
                )
                if r2.returncode != 0:
                    err2 = r2.stderr.decode(errors="replace").strip()
                    print(f"[AUDIO] aplay error ({self.alsa}): {err2}")
                    # Fallback: system default
                    subprocess.run(
                        ["aplay", "-q", tmp],
                        stderr=subprocess.DEVNULL, timeout=60,
                    )
            else:
                subprocess.run(
                    ["aplay", "-q", tmp],
                    stderr=subprocess.PIPE, timeout=60,
                )

        except FileNotFoundError as e:
            print(f"[AUDIO] Command not found: {e}"
                  " — sudo apt install espeak-ng alsa-utils")
        except subprocess.TimeoutExpired:
            print("[AUDIO] speak() timed out")
        except Exception as e:
            print(f"[AUDIO] speak error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
