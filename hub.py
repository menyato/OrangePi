#!/usr/bin/env python3
"""
hub.py — THE FIRST SCRIPT on the OrangePi.

Boots the glove, runs optional calibration, then listens for the 3 control
gestures (matched by raw ATmega bitmasks) and runs the feature menu.

System gestures (hardcoded defaults, adjustable via EDIT):
    START : Thumb + Pinky,  hold tilt_right    (flex 0x11, imu 0x01, STATIC)
    NEXT  : Thumb + Ring,   flick tilt_backward (flex 0x09, imu 0x08, FLICK)
    EDIT  : Thumb + Middle, flick tilt_forward  (flex 0x05, imu 0x04, FLICK)

gestures.json      — gesture bitmasks, written on first run, updated via EDIT.
calibration.json   — REST+BENT per finger, saved after --calibrate and restored
                     automatically on every subsequent boot via "setcal" command.

Modes:
  python3 hub.py                 normal run (restores calibration if saved)
  python3 hub.py --monitor       live ATmega feed + gesture match view
  python3 hub.py --calibrate     calibrate, SAVE to calibration.json, then run
  python3 hub.py --threshold 35  override bend threshold %, then run

Run from the orangepi/ directory.
"""

import argparse
import os
import sys
import time

from glove_controller import GloveController

from gesture_hub.store        import GestureStore
from gesture_hub.cal_store    import CalStore
from gesture_hub.engine       import GestureEngine
from gesture_hub.recorder     import GestureRecorder
from gesture_hub.registry     import FeatureRegistry
from gesture_hub.feedback     import Feedback
from gesture_hub.state_machine import HubStateMachine
from gesture_hub               import diagnostics
from net.client                import ServerLink

from features.money_recognition import MoneyRecognition

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GESTURES_PATH = os.path.join(HERE, "gestures.json")
DEFAULT_CAL_PATH      = os.path.join(HERE, "calibration.json")

# ── Register features here (menu scroll order) ────────────────────────────────
FEATURES = [MoneyRecognition()]


def main() -> None:
    ap = argparse.ArgumentParser(description="OrangePi gesture hub")
    ap.add_argument("--host",      default="10.254.249.159", help="Server IP")
    ap.add_argument("--port",      default=9000, type=int,   help="Server port")
    ap.add_argument("--uart",      default="/dev/ttyS5",     help="ATmega UART device")
    ap.add_argument("--baud",      default=115200, type=int, help="UART baud rate")
    ap.add_argument("--alsa",      default=None,
                    help="ALSA device for hub speech, e.g. plughw:0,0")
    ap.add_argument("--gestures",  default=DEFAULT_GESTURES_PATH,
                    help="Path to gestures.json")
    ap.add_argument("--cal",       default=DEFAULT_CAL_PATH,
                    help="Path to calibration.json")
    ap.add_argument("--monitor",   action="store_true",
                    help="Show live ATmega feed + gesture matches, then exit.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run per-hand flex calibration, save it, then run normally.")
    ap.add_argument("--threshold", type=int, default=None,
                    help="Override firmware bend threshold percent (10-90).")
    args = ap.parse_args()

    # ── Load gesture store + calibration store ────────────────────────────────
    store     = GestureStore(args.gestures)
    cal_store = CalStore(args.cal)

    for name, spec in store.gestures.items():
        print(f"[HUB] gesture {name:6s}: {spec.describe()}"
              f"  (flex=0x{spec.flex_mask:02X} imu=0x{spec.imu_mask:02X}"
              f" {spec.motion.value})")

    # ── Open glove ────────────────────────────────────────────────────────────
    controller = GloveController(port=args.uart, baud=args.baud)
    feedback   = Feedback(controller, alsa=args.alsa)

    if not controller.start(wait_ready=False):
        print(f"[HUB] Cannot open glove on {args.uart}. Check wiring.")
        sys.exit(1)
    time.sleep(0.5)   # let first frames arrive

    # ── MONITOR mode ──────────────────────────────────────────────────────────
    if args.monitor:
        # Still restore calibration so monitor shows correct bend states
        if cal_store.has_calibration():
            print("[HUB] Restoring saved calibration before monitor...")
            cal_store.restore(controller)
            time.sleep(0.3)
        try:
            diagnostics.run_monitor(controller, store.gestures, say=feedback.speak)
        finally:
            controller.stop()
        return

    # ── Threshold override ────────────────────────────────────────────────────
    if args.threshold is not None:
        try:
            controller.set_threshold(args.threshold)
            print(f"[HUB] Bend threshold set to {args.threshold}%")
            time.sleep(0.3)
        except ValueError as e:
            print(f"[HUB] Threshold error: {e}")

    # ── Calibration (run + save) ──────────────────────────────────────────────
    if args.calibrate:
        result = diagnostics.run_calibration(controller, say=feedback.speak)
        if result:
            cal_store.save(
                rest=result["rest"],
                bent=result["bent"],
                thresh_pct=args.threshold or cal_store.thresh_pct,
            )
            feedback.speak("Calibration saved.")
        else:
            print("[HUB] Calibration aborted — continuing with existing values.")

    # ── Restore saved calibration on normal boot (no --calibrate flag) ────────
    elif cal_store.has_calibration():
        print("[HUB] Restoring saved calibration...")
        cal_store.restore(controller)
        time.sleep(0.3)
    else:
        print("[HUB] No saved calibration. Run with --calibrate to set one.")

    # ── Normal hub ────────────────────────────────────────────────────────────
    recorder = GestureRecorder()
    registry = FeatureRegistry(FEATURES)
    link     = ServerLink(args.host, args.port)

    engine = GestureEngine(store.gestures, on_event=None)
    sm     = HubStateMachine(controller, engine, recorder, registry,
                             store, feedback, link)

    engine.on_event     = sm.dispatch
    controller.on_frame = sm.on_frame

    link.connect()   # best-effort; features reconnect on demand

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