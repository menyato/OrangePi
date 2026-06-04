"""
features/money_recognition.py — currency reader, wired into the hub.

This is a thin orchestration layer over your existing orangepi_client.py:
all the heavy lifting (VAD, Whisper STT, camera capture, Piper/espeak TTS,
earcons) is imported and reused unchanged. The only thing that changes is the
transport — instead of the client owning its own socket, messages go through
the hub's shared ServerLink, tagged "money", so the server routes them to the
money helper.
"""

import time

import orangepi_client as mc          # your existing, UNCHANGED client, used as a library
from features.base import Feature, FeatureContext


class MoneyRecognition(Feature):
    name = "money"
    title = "Money recognition"
    _models_ready = False              # load Whisper/cues/devices once per process

    def _ensure_ready(self) -> None:
        if MoneyRecognition._models_ready:
            return
        mc.init_cues()
        mc.auto_detect_all()
        mc.load_models()
        MoneyRecognition._models_ready = True

    def run(self, ctx: FeatureContext) -> None:
        self._ensure_ready()
        link, abort = ctx.link, ctx.abort

        resp = link.send(self.name, {"type": "hello"})
        if resp:
            mc._speak_resp(resp)
            mc.wait_speaking()

        while not abort.is_set():
            text, _audit = mc.listen()
            if abort.is_set():
                break
            if not text:
                continue

            payload = {"type": "voice", "text": text}

            # Capture a frame only for scan/redo words (never on yes/no confirmations).
            if any(t in text for t in mc.SCAN_TRIGGERS):
                mc.speak("Hold the bill steady. Capturing now.")
                mc.wait_speaking()
                jpeg = mc.capture_best_frame_local()
                if jpeg is None:
                    mc.speak("Camera error. Could not capture a frame. Please try again.")
                    mc.wait_speaking()
                    continue
                mc.play_cue(mc._CUE_SHOT)
                mc.speak("Captured. Analyzing now.")
                mc.wait_speaking()
                payload["frame"] = jpeg

            resp = link.send(self.name, payload)
            if resp is None:
                mc.speak("Connection lost. Returning to the menu.")
                mc.wait_speaking()
                break

            mc._speak_resp(resp)
            mc.wait_speaking()

            if resp.get("quit"):
                break

        # If we got here by abort (not by a server "quit"), give a short cue.
        if abort.is_set():
            mc.speak("Money recognition aborted.")
            mc.wait_speaking()
