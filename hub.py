#!/usr/bin/env python3
"""
hub.py — OrangePi gesture hub (new flow).

Boot sequence
─────────────
1. Load GestureStore (gestures.json) and CalStore (calibration.json).
2. Open GloveController (UART to ATmega).
3. Restore saved calibration if present.
4. Start HubStateMachine — waits for START gesture.

Gesture flow (see state_machine.py for full detail)
────────────────────────────────────────────────────
  START (hardcoded) — global on/off toggle for the whole glove.

  First boot / missing gestures:
    After START the hub walks through any feature that has no gesture yet.
    The user performs the gesture twice; on match it is saved and the walk
    moves to the next unassigned feature.  NEXT skips to the next one.

  Normal operation:
    Each feature gesture is a toggle — fire once to open, fire again to close.
    While a feature is running, only its own toggle (to close) is heard.

  Programmable Gestures feature:
    Opens a panel where NEXT scrolls all features and EDIT re-binds gestures.

Hardcoded gestures (never user-assignable):
  START  Thumb+Pinky  tilt_right     STATIC  flex=0x11 imu=0x01
  NEXT   Thumb+Ring   tilt_backward  FLICK   flex=0x09 imu=0x08
  EDIT   Thumb+Middle tilt_forward   FLICK   flex=0x05 imu=0x04

Modes:
  python3 hub.py                 normal run
  python3 hub.py --monitor       live ATmega feed + gesture match view
  python3 hub.py --calibrate     run per-hand calibration, save, then run
  python3 hub.py --threshold 35  override bend threshold %, then run
"""

import argparse
import os
import sys
import time

from glove_controller import GloveController

from gesture_hub.store         import GestureStore
from gesture_hub.cal_store     import CalStore
from gesture_hub.engine        import GestureEngine
from gesture_hub.recorder      import GestureRecorder
from gesture_hub.registry      import FeatureRegistry
from gesture_hub.feedback      import Feedback
from gesture_hub.state_machine import HubStateMachine
from gesture_hub               import diagnostics
from net.client                import ServerLink

from features.money_recognition import MoneyRecognition
from features.ocr_reader        import OCRReader
from features.programmable      import ProgrammableGestures

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GESTURES_PATH = os.path.join(HERE, "gestures.json")
DEFAULT_CAL_PATH      = os.path.join(HERE, "calibration.json")

# ── Register features here (enrollment + Programmable panel order) ─────────
# ProgrammableGestures should always be last.
FEATURES = [
    MoneyRecognition(),
    OCRReader(),
    ProgrammableGestures(),
]


def _probe_devices(feedback, link) -> None:
    """
    Boot-time device check for a blind user.
    Tests camera, microphone, and server connection, then speaks the results.
    The speaker test is implicit: if the user hears anything, audio is working.
    """
    results = []

    # Server
    server_ok = link.sock is not None
    results.append("Server " + ("connected" if server_ok else "not reachable, will retry when needed"))

    # Camera
    camera_ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        camera_ok = cap.isOpened()
        cap.release()
    except Exception:
        pass
    results.append("Camera " + ("ready" if camera_ok else "not found"))

    # Microphone
    mic_ok = False
    mic_name = ""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        for d in devs:
            if d["max_input_channels"] > 0:
                mic_ok  = True
                mic_name = d["name"].split("(")[0].strip()
                break
    except Exception:
        pass
    results.append("Microphone " + (f"ready, {mic_name}" if mic_ok else "not found"))

    feedback.speak("Startup: " + ", ".join(results) + ". Speaker working.")


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

    # ── Load gesture store + calibration store ─────────────────────────────
    store     = GestureStore(args.gestures)
    cal_store = CalStore(args.cal)

    # Print current gesture table
    print("[HUB] Loaded gestures:")
    for name, spec in store.gestures.items():
        print(f"  {name:30s}: {spec.describe()}"
              f"  (flex=0x{spec.flex_mask:02X} imu=0x{spec.imu_mask:02X}"
              f" {spec.motion.value})")

    # ── Open glove ─────────────────────────────────────────────────────────
    controller = GloveController(port=args.uart, baud=args.baud)
    feedback   = Feedback(controller, alsa=args.alsa)

    if not controller.start(wait_ready=False):
        print(f"[HUB] Cannot open glove on {args.uart}. Check wiring.")
        sys.exit(1)
    time.sleep(0.5)   # let first frames arrive

    # ── MONITOR mode ────────────────────────────────────────────────────────
    if args.monitor:
        if cal_store.has_calibration():
            print("[HUB] Restoring saved calibration before monitor...")
            cal_store.restore(controller)
            time.sleep(0.3)
        try:
            diagnostics.run_monitor(controller, store.gestures,
                                    say=feedback.speak)
        finally:
            controller.stop()
        return

    # ── Threshold override ──────────────────────────────────────────────────
    if args.threshold is not None:
        try:
            controller.set_threshold(args.threshold)
            print(f"[HUB] Bend threshold set to {args.threshold}%")
            time.sleep(0.3)
        except ValueError as e:
            print(f"[HUB] Threshold error: {e}")

    # ── Calibration ─────────────────────────────────────────────────────────
    # Always save after a successful calibration so the next boot auto-restores.
    if args.calibrate:
        result = diagnostics.run_calibration(controller, say=feedback.speak)
        if result:
            cal_store.save(
                rest=result["rest"],
                bent=result["bent"],
                thresh_pct=args.threshold or cal_store.thresh_pct,
            )
            feedback.speak("Calibration saved.")
            # Restore immediately so this session also uses the new values
            cal_store.restore(controller)
            time.sleep(0.3)
        else:
            print("[HUB] Calibration aborted.")
            if cal_store.has_calibration():
                print("[HUB] Restoring previous calibration...")
                cal_store.restore(controller)
                time.sleep(0.3)
            else:
                print("[HUB] No saved calibration available.")
    elif cal_store.has_calibration():
        print("[HUB] Restoring saved calibration...")
        cal_store.restore(controller)
        time.sleep(0.3)
    else:
        print("[HUB] No saved calibration. Run with --calibrate to set one.")

    # ── Build engine with ALL known gestures ────────────────────────────────
    # The engine always knows the full store (system gestures + feature gestures
    # that have already been assigned).  The state machine decides which fired
    # gesture names to act on based on the current state.
    engine   = GestureEngine(store.gestures, on_event=None)
    recorder = GestureRecorder()
    registry = FeatureRegistry(FEATURES)
    link     = ServerLink(args.host, args.port)

    sm = HubStateMachine(
        controller=controller,
        engine=engine,
        recorder=recorder,
        registry=registry,
        store=store,
        feedback=feedback,
        link=link,
    )

    # Wire callbacks
    engine.on_event     = sm.dispatch
    controller.on_frame = sm.on_frame

    link.connect()   # best-effort; features reconnect on demand

    _probe_devices(feedback, link)

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