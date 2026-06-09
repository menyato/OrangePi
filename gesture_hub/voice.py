"""
gesture_hub/voice.py — speech-to-command translator for the smart glove hub.

Runs a daemon thread that continuously listens via the microphone and injects
recognised utterances into the state machine via on_command() — the same
callback the gesture engine uses.

Recognised commands
-------------------
  "next" / "skip"            → NEXT
  "edit" / "change"          → EDIT
  "start" / "stop"           → START
  any phrase with "gesture" / "move" / "how" / "what"
                             → QUERY:<full text>  (answered by state machine)

Install on OrangePi
-------------------
  pip install SpeechRecognition
  sudo apt install python3-pyaudio portaudio19-dev   # or: pip install PyAudio
"""

import re
import threading

try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False

_COMMAND_MAP = {
    "next":    "NEXT",
    "skip":    "NEXT",
    "forward": "NEXT",
    "edit":    "EDIT",
    "change":  "EDIT",
    "modify":  "EDIT",
    "start":   "START",
    "stop":    "START",
    # book reader — scan a page / close the feature
    "scan":    "OCR_SCAN",
    "capture": "OCR_SCAN",
    "take":    "OCR_SCAN",
    "close":   "OCR_CLOSE",
    # programmable gesture-recording confirmation
    # also used in book reader pause: yes = add a page, no = resume
    "confirm": "PROG_CONFIRM",
    "yes":     "PROG_CONFIRM",
    "save":    "PROG_CONFIRM",
    "discard": "PROG_DISCARD",
    "cancel":  "PROG_DISCARD",
    "no":      "PROG_DISCARD",
}

_QUERY_TRIGGERS = {"gesture", "move", "how", "what"}


class VoiceListener:
    """Background speech-to-command translator.

    on_command(cmd) is called from the background thread for every recognised
    utterance where cmd is a gesture name (e.g. "NEXT") or "QUERY:<text>".
    """

    def __init__(self, on_command, energy_threshold: int = 3000):
        self._on_cmd = on_command
        self._energy = energy_threshold
        self._stop   = threading.Event()
        self._thread = None

    @property
    def available(self) -> bool:
        return _SR_OK

    def start(self) -> None:
        if not _SR_OK:
            print("[VOICE] speech_recognition not installed — voice disabled.")
            print("[VOICE]   pip install SpeechRecognition")
            print("[VOICE]   sudo apt install python3-pyaudio portaudio19-dev")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VoiceListener"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        r = sr.Recognizer()
        r.energy_threshold         = self._energy
        r.dynamic_energy_threshold = True
        r.pause_threshold          = 0.6
        try:
            mic = sr.Microphone()
        except OSError as e:
            print(f"[VOICE] Cannot open microphone: {e}")
            return

        with mic as source:
            print("[VOICE] Calibrating ambient noise (1.5 s) …")
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("[VOICE] Ready — speak 'next', 'edit', or ask about gestures.")
            while not self._stop.is_set():
                try:
                    audio = r.listen(source, timeout=5, phrase_time_limit=7)
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    continue
                self._recognise(r, audio)

    def _recognise(self, recogniser, audio) -> None:
        try:
            text = recogniser.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            return
        except Exception as e:
            print(f"[VOICE] recognition error: {e}")
            return
        print(f"[VOICE] heard: {text!r}")
        self._dispatch(text)

    def _dispatch(self, text: str) -> None:
        words = set(re.findall(r"[a-z]+", text))
        # Query intent: "what is the gesture for …" / "how do I …"
        if words & _QUERY_TRIGGERS:
            self._on_cmd(f"QUERY:{text}")
            return
        # Simple commands — first match wins
        for word, cmd in _COMMAND_MAP.items():
            if word in words:
                self._on_cmd(cmd)
                return
