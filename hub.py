#!/usr/bin/env python3
"""
hub.py — THE FIRST SCRIPT on the OrangePi.

Boots the glove, taps its 10 Hz sensor stream, recognizes the 3 control
gestures, and runs the feature menu. One feature is money recognition.

Modes:
  python3 hub.py                 normal run (menu + features)
  python3 hub.py --monitor       live feed + gesture recognition view (no server)
  python3 hub.py --calibrate     per-hand flex calibration, THEN run normally
  python3 hub.py --threshold 45  set firmware bend threshold %, THEN run normally

Bend range differs hand to hand, so calibrate once per wearer:
  python3 hub.py --calibrate --host <SERVER_IP> --uart /dev/ttyS5 --alsa plughw:0,0

Run from the orangepi/ directory so glove_controller.py and orangepi_client.py
are importable as top-level modules.
"""

import argparse
import os
import sys
import time

from glove_controller import GloveController          # your existing driver

from gesture_hub.store import GestureStore
from gesture_hub.engine import GestureEngine
from gesture_hub.recorder import GestureRecorder
from gesture_hub.registry import FeatureRegistry
from gesture_hub.feedback import Feedback
from gesture_hub.state_machine import HubStateMachine
from gesture_hub import diagnostics
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
                    help="ALSA device for hub speech, e.g. plughw:0,0")
    ap.add_argument("--gestures", default=DEFAULT_GESTURES_PATH,
                    help="Path to gestures.json")
    # diagnostic / calibration modes
    ap.add_argument("--monitor", action="store_true",
                    help="Show the live feed + gesture recognition, then exit "
                         "(no server, no menu). Use this to calibrate gestures.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run per-hand flex calibration first, then run normally.")
    ap.add_argument("--threshold", type=int, default=None,
                    help="Set firmware bend threshold percent (10-90), then run.")
    args = ap.parse_args()

    store = GestureStore(args.gestures)
    for name, spec in store.gestures.items():
        print(f"[HUB] gesture {name:6s}: {spec.describe()}")

    controller = GloveController(port=args.uart, baud=args.baud)
    feedback = Feedback(controller, alsa=args.alsa)

    if not controller.start(wait_ready=False):
        print(f"[HUB] Cannot open glove on {args.uart}. Check wiring.")
        sys.exit(1)
    time.sleep(0.5)   # let the first frames arrive

    # ── MONITOR: feed + gesture view, no server, no menu ──────────────────────
    if args.monitor:
        try:
            diagnostics.run_monitor(controller, store.gestures, say=feedback.speak)
        finally:
            controller.stop()
        return

    # ── optional one-shot setup before the normal run ─────────────────────────
    if args.threshold is not None:
        try:
            controller.set_threshold(args.threshold)
            print(f"[HUB] bend threshold set to {args.threshold}%")
            time.sleep(0.3)
        except ValueError as e:
            print(f"[HUB] threshold error: {e}")

    if args.calibrate:
        diagnostics.run_calibration(controller, say=feedback.speak)

    # ── normal hub ─────────────────────────────────────────────────────────────
    recorder = GestureRecorder()
    registry = FeatureRegistry(FEATURES)
    link = ServerLink(args.host, args.port)

    engine = GestureEngine(store.gestures, on_event=None)
    sm = HubStateMachine(controller, engine, recorder, registry, store, feedback, link)

    engine.on_event = sm.dispatch
    controller.on_frame = sm.on_frame

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
