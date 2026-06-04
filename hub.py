#!/usr/bin/env python3
"""
hub.py — THE FIRST SCRIPT on the OrangePi.

Boots the glove, taps its 10 Hz sensor stream, recognizes the 3 control
gestures, and runs the feature menu. One feature is money recognition; more
can be added by appending to FEATURES below.

Run from the orangepi/ directory (so glove_controller.py and orangepi_client.py
are importable as top-level modules):

    python3 hub.py --host 10.254.249.159 --port 9000 --uart /dev/ttyS5 --alsa plughw:0,0

Gestures live in gestures.json next to this file (created on first edit; until
then the built-in defaults are used). Re-record any gesture in-glove via the
EDIT gesture — no need to touch this file.
"""

import argparse
import os
import sys

from glove_controller import GloveController          # your existing driver

from gesture_hub.store import GestureStore
from gesture_hub.engine import GestureEngine
from gesture_hub.recorder import GestureRecorder
from gesture_hub.registry import FeatureRegistry
from gesture_hub.feedback import Feedback
from gesture_hub.state_machine import HubStateMachine
from net.client import ServerLink

from features.money_recognition import MoneyRecognition

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GESTURES_PATH = os.path.join(HERE, "gestures.json")

# Register features here (menu order = scroll order):
FEATURES = [MoneyRecognition()]


def main() -> None:
    ap = argparse.ArgumentParser(description="OrangePi gesture hub")
    ap.add_argument("--host", default="10.254.249.159", help="Server IP")
    ap.add_argument("--port", default=9000, type=int, help="Server port")
    ap.add_argument("--uart", default="/dev/ttyS5", help="ATmega UART device")
    ap.add_argument("--baud", default=115200, type=int, help="UART baud")
    ap.add_argument("--alsa", default=None,
                    help="ALSA device for hub speech, e.g. plughw:0,0 "
                         "(defaults to system default)")
    ap.add_argument("--gestures", default=DEFAULT_GESTURES_PATH,
                    help="Path to gestures.json")
    args = ap.parse_args()

    store = GestureStore(args.gestures)
    for name, spec in store.gestures.items():
        print(f"[HUB] gesture {name:6s}: {spec.describe()}")

    controller = GloveController(port=args.uart, baud=args.baud)
    recorder = GestureRecorder()
    registry = FeatureRegistry(FEATURES)
    feedback = Feedback(controller, alsa=args.alsa)
    link = ServerLink(args.host, args.port)

    engine = GestureEngine(store.gestures, on_event=None)
    sm = HubStateMachine(controller, engine, recorder, registry, store, feedback, link)

    # wire the stream: frames -> router; gesture events -> dispatch
    engine.on_event = sm.dispatch
    controller.on_frame = sm.on_frame

    if not controller.start(wait_ready=False):
        print(f"[HUB] Cannot open glove on {args.uart}. Check wiring.")
        sys.exit(1)

    link.connect()   # best-effort; features will reconnect on demand

    print("[HUB] Running. Ctrl-C to quit.")
    try:
        sm.run()
    except KeyboardInterrupt:
        print("\n[HUB] Shutting down...")
    finally:
        sm.stop()
        controller.stop()
        link.close()


if __name__ == "__main__":
    main()
