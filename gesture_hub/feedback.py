"""
feedback.py — non-visual feedback for the hub.

TTS: espeak-ng writes a WAV file then aplay plays it in a daemon thread.
     speak() is always non-blocking: a new call kills the previous audio
     immediately so gestures are never delayed by ongoing speech.
     silence() can be called explicitly to clear audio before a recording
     window so the user hears only the haptic "go" buzz.

ALSA: devices listed and tested at startup. Non-HDMI devices are tried in
      reverse card-number order so USB/external headsets (higher card #)
      are preferred over the built-in codec (card 0).

Volume: every mixer control on every card is unmuted and set to 90%.
All errors are printed — nothing fails silently.
"""

import array
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import wave


# ── Volume initialisation ─────────────────────────────────────────────────────

def _unmute_all(card: str = "") -> None:
    card_flag = ["-c", card] if card else []

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

    # Enumerate every control the card actually exposes
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
    try:
        r = subprocess.run(
            ["aplay", "-D", alsa_dev, "-q", wav_path],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, timeout=5,
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace").strip()
            print(f"[AUDIO]   {alsa_dev} — {err}")
            return False
        return True
    except Exception as e:
        print(f"[AUDIO]   {alsa_dev} — {e}")
        return False


def _auto_alsa() -> "str | None":
    try:
        out = subprocess.check_output(
            ["aplay", "-l"], stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        print("[AUDIO] aplay not found — sudo apt install alsa-utils")
        return None
    except Exception as e:
        print(f"[AUDIO] aplay -l failed: {e}")
        return None

    entries = re.findall(r"card\s+(\d+):\s*(\S+)[^,]*,\s*device\s+(\d+)", out)
    if not entries:
        print("[AUDIO] No playback devices found.")
        return None

    print("[AUDIO] Playback devices found:")
    for card, name, dev in entries:
        print(f"  plughw:{card},{dev}  ({name})")

    cards_seen: set = set()
    for card, _, _ in entries:
        if card not in cards_seen:
            _unmute_all(card)
            cards_seen.add(card)

    wav = _make_test_wav()
    try:
        non_hdmi = [(c, n, d) for c, n, d in entries if "hdmi" not in n.lower()]
        hdmi     = [(c, n, d) for c, n, d in entries if "hdmi"     in n.lower()]
        # Reverse non-HDMI: higher card number = more recently connected device
        ordered  = list(reversed(non_hdmi)) + hdmi

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

    print("[AUDIO] No working device found — will use system default.")
    return None


# ── Feedback class ────────────────────────────────────────────────────────────

class Feedback:
    def __init__(self, controller, alsa: "str | None" = None, speed: int = 150):
        self.ctrl  = controller
        self.alsa  = alsa if alsa is not None else _auto_alsa()
        self.speed = speed

        # Background audio state (protected by _audio_lock)
        self._audio_proc: "subprocess.Popen | None" = None
        self._audio_tmp:  "str | None"               = None
        self._audio_lock  = threading.Lock()

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
    # Hand wiring: motor 1 = bottom, motor 2 = right, motor 3 = left
    # (see features/lidar_nav.py's module docstring for the same mapping).
    def _pulse(self, motor: int, ms: int) -> None:
        try:
            self.ctrl.vibrate(motor, ms)
        except Exception:
            pass

    def confirm(self) -> None:
        self._pulse(1, 120)   # bottom — positive/completion cue

    def tick(self) -> None:
        self._pulse(2, 60)    # right — neutral menu-scroll cue

    def select(self) -> None:
        self._pulse(1, 90); time.sleep(0.12); self._pulse(1, 90)   # bottom, double-pulse

    def error(self) -> None:
        self._pulse(3, 400)   # left — alert cue

    def beep(self) -> None:
        self._pulse(2, 150)   # right — "go" cue

    # ── audio control ────────────────────────────────────────────────────────
    def silence(self) -> None:
        """Kill current audio immediately (call before a recording window)."""
        self._stop_audio()

    def is_speaking(self) -> bool:
        """Return True while an aplay process is still running."""
        with self._audio_lock:
            return (self._audio_proc is not None
                    and self._audio_proc.poll() is None)

    def _stop_audio(self) -> None:
        with self._audio_lock:
            proc = self._audio_proc
            tmp  = self._audio_tmp
            self._audio_proc = None
            self._audio_tmp  = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ── speech ───────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """
        Speak text non-blocking: kills any current audio, synthesises a WAV
        file via espeak-ng, then plays it in a daemon thread.

        Gesture priority: a new speak() call always interrupts the previous
        one immediately — the user never waits for speech to finish.
        """
        print(f"[HUB] {text}")
        self._stop_audio()   # gesture priority — kill old speech first

        # Synthesise WAV synchronously (fast, < 1 s even for long text)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            r = subprocess.run(
                ["espeak-ng", "-s", str(self.speed), "-w", tmp, text],
                stderr=subprocess.PIPE, timeout=30)
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace").strip()
                print(f"[AUDIO] espeak-ng: {err}")
                os.unlink(tmp)
                return
        except FileNotFoundError:
            print("[AUDIO] espeak-ng not found — sudo apt install espeak-ng")
            try: os.unlink(tmp)
            except OSError: pass
            return
        except Exception as e:
            print(f"[AUDIO] espeak-ng: {e}")
            try: os.unlink(tmp)
            except OSError: pass
            return

        # --buffer-size=32768 gives aplay ~1.5 s of headroom at 22050 Hz, which
        # prevents the "xrun in PREPARED state" that happens when the USB headset
        # needs extra time to transition from PREPARED → RUNNING.
        if self.alsa:
            cmd = ["aplay", "-D", self.alsa, "-q", "--buffer-size=32768", tmp]
        else:
            cmd = ["aplay", "-q", tmp]

        # Launch aplay immediately so _stop_audio() can kill it at any point
        try:
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[AUDIO] aplay launch: {e}")
            try: os.unlink(tmp)
            except OSError: pass
            return

        with self._audio_lock:
            self._audio_proc = proc
            self._audio_tmp  = tmp

        # Daemon thread monitors for errors and cleans up the temp file
        def _monitor(p: "subprocess.Popen", t: str) -> None:
            try:
                _, err = p.communicate(timeout=60)
                # -15=SIGTERM, -9=SIGKILL (normal kill paths); also
                # aplay exits with code 1 + "Interrupted system call" when
                # terminated mid-stream — that is expected, not an error.
                err_str = err.decode(errors="replace").strip()
                is_normal_kill = (
                    p.returncode in (0, -15, -9)
                    or "Interrupted system call" in err_str
                )
                if not is_normal_kill and err_str:
                    # xrun messages are expected on first try — retry on same
                    # device with a larger buffer instead of falling back to the
                    # system default (which may be HDMI and inaudible).
                    if "xrun" not in err_str:
                        print(f"[AUDIO] aplay: {err_str}")
                    if self.alsa:
                        retry = ["aplay", "-D", self.alsa, "-q",
                                 "--buffer-size=65536", t]
                    else:
                        retry = ["aplay", "-q", t]
                    try:
                        subprocess.run(retry, stderr=subprocess.DEVNULL, timeout=60)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception as e:
                print(f"[AUDIO] monitor: {e}")
            finally:
                with self._audio_lock:
                    if self._audio_proc is p:
                        self._audio_proc = None
                        self._audio_tmp  = None
                try: os.unlink(t)
                except OSError: pass

        threading.Thread(target=_monitor, args=(proc, tmp), daemon=True).start()

    def play_raw(self, wav_bytes: bytes) -> float:
        """Play server-generated WAV bytes, non-blocking. Returns audio duration."""
        import io
        import wave as _wave
        self._stop_audio()
        try:
            with _wave.open(io.BytesIO(wav_bytes)) as wf:
                dur = wf.getnframes() / max(1, wf.getframerate())
        except Exception:
            dur = max(1.0, len(wav_bytes) / (22050 * 2))
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(wav_bytes)
        except Exception as e:
            print(f"[AUDIO] play_raw write: {e}")
            try: os.unlink(tmp)
            except OSError: pass
            return dur
        if self.alsa:
            cmd = ["aplay", "-D", self.alsa, "-q", "--buffer-size=32768", tmp]
        else:
            cmd = ["aplay", "-q", tmp]
        try:
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[AUDIO] play_raw launch: {e}")
            try: os.unlink(tmp)
            except OSError: pass
            return dur
        with self._audio_lock:
            self._audio_proc = proc
            self._audio_tmp  = tmp
        def _mon(p: "subprocess.Popen", t: str) -> None:
            try: p.communicate(timeout=60)
            except Exception: pass
            finally:
                with self._audio_lock:
                    if self._audio_proc is p:
                        self._audio_proc = None
                        self._audio_tmp  = None
                try: os.unlink(t)
                except OSError: pass
        threading.Thread(target=_mon, args=(proc, tmp), daemon=True).start()
        return dur

    def wait(self, timeout: float = 12.0) -> None:
        """Block until current speech finishes or timeout expires.

        Use before recording windows so the instruction is fully heard
        before the haptic 'go' buzz fires.
        """
        time.sleep(0.15)   # give aplay a moment to start
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_speaking():
                return
            time.sleep(0.05)
