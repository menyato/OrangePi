"""
diagnostics.py — calibration + live monitor tools for the hub.

run_monitor()      shows the raw glove feed and which gesture poses match right
                   now, plus prints each gesture the moment it fires. Use it to
                   verify the 3 gestures and see how YOUR hand bends.

run_calibration()  runs the ATmega's per-finger flex calibration (open/bent per
                   finger). Bend range differs hand to hand, so do this once for
                   each new wearer — otherwise finger_bent (and every gesture)
                   will be unreliable.
"""

import threading
import time

from gesture_hub.specs import FINGER_NAMES, Motion
from gesture_hub.engine import GestureEngine


def run_monitor(controller, gestures: dict, say=print) -> None:
    say("Monitor mode. Move your hand to see the live feed.")
    print("\n[MONITOR] Live glove feed — Ctrl-C to exit.")
    print("[MONITOR] flex = raw bend per finger (Thumb..Pinky)")
    print("[MONITOR] match column shows which gesture POSE is satisfied right now\n")

    def on_event(name: str) -> None:
        print(f"\n>>> GESTURE FIRED: {name}\n")

    engine = GestureEngine(gestures, on_event=on_event)
    last_print = [0.0]

    def on_frame(frame) -> None:
        engine.feed(frame)                       # so real gesture events fire too
        now = time.time()
        if now - last_print[0] < 0.2:            # throttle the status line to ~5 Hz
            return
        last_print[0] = now

        bent_idx = [i for i, b in enumerate(frame.finger_bent) if b]
        bent = ",".join(FINGER_NAMES[i] for i in bent_idx) or "-"
        flags = ",".join(k for k, v in frame.imu_flags.items() if v) or "-"
        bent_set = frozenset(bent_idx)

        marks = []
        for nm, sp in gestures.items():
            ok = (bent_set == sp.fingers_set())
            if ok and sp.motion == Motion.STATIC and sp.orientation:
                ok = frame.imu_flags.get(sp.orientation, False)
            marks.append(f"{nm}:{'OK' if ok else '..'}")

        raw = " ".join(f"{r:3d}" for r in frame.raw_flex)
        print(f"\rflex[{raw}]  bent:{bent:<22} imu:{flags:<22} | {'  '.join(marks)}   ",
              end="", flush=True)

    controller.on_frame = on_frame
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MONITOR] exit.")


def run_calibration(controller, say=print) -> bool:
    """Run the firmware's open/bent calibration for all 5 fingers. Returns True
    on completion. Advance each step by pressing Enter at the terminal."""
    done = threading.Event()

    def on_cal(ev: str, data: dict) -> None:
        if ev == "START":
            say("Calibration started. Follow the prompts.")
        elif ev == "OPEN":
            say(f"Hold your {data['name']} straight and relaxed, then press Enter.")
        elif ev == "BENT":
            say(f"Now bend your {data['name']} all the way, then press Enter.")
        elif ev == "REST":
            print(f"   rest[{FINGER_NAMES[data['finger']]}] = {data['value']}")
        elif ev == "BENT_VAL":
            print(f"   bent[{FINGER_NAMES[data['finger']]}] = {data['value']}")
        elif ev == "DONE":
            say("Calibration complete.")
            print(f"   REST: {data.get('rest')}")
            print(f"   BENT: {data.get('bent')}")
            done.set()

    controller.on_cal_event = on_cal
    print("\n[CAL] Per-hand flex calibration. Press Enter to advance each step.")
    controller.calibrate()

    while not done.is_set():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("\n[CAL] aborted.")
            controller.on_cal_event = None
            return False
        controller.cal_confirm()
        if done.wait(0.4):
            break

    controller.on_cal_event = None
    return True
