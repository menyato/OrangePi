"""
diagnostics.py — calibration + live monitor tools for the hub.

run_monitor()
    Shows the raw ATmega output: flex_bits and imu_bits as hex, which
    fingers are bent, which IMU flags are active, and a live "match"
    column for every gesture in the store. Use this to verify the 3
    system gestures and tune per-hand thresholds.

run_calibration()
    Runs the ATmega's open/bent calibration for all 5 fingers so that
    the firmware's bend thresholds are accurate for this wearer's hand.
"""

import threading
import time

from gesture_hub.specs import FINGER_NAMES, IMU_BIT_NAMES, Motion
from gesture_hub.engine import GestureEngine


def run_monitor(controller, gestures: dict, say=print) -> None:
    say("Monitor mode. Move your hand to see the live feed.")
    print("\n[MONITOR] Live ATmega feed — Ctrl-C to exit.")
    print("[MONITOR] flex = raw flex byte (hex)  |  imu = raw IMU byte (hex)")
    print("[MONITOR] match column: gesture name : OK / ..\n")

    def on_event(name: str) -> None:
        print(f"\n>>> GESTURE FIRED: {name}\n")

    engine = GestureEngine(gestures, on_event=on_event)
    last_print = [0.0]

    def on_frame(frame) -> None:
        engine.feed(frame)
        now = time.time()
        if now - last_print[0] < 0.2:     # throttle to ~5 Hz display
            return
        last_print[0] = now

        fb  = frame.flex_bits
        ib  = frame.imu_bits
        raw = " ".join(f"{r:3d}" for r in frame.raw_flex)

        bent  = ",".join(FINGER_NAMES[i]  for i in range(5) if fb & (1 << i)) or "-"
        flags = ",".join(IMU_BIT_NAMES[i] for i in range(6) if ib & (1 << i)) or "-"

        marks = []
        for nm, sp in gestures.items():
            flex_ok = (fb & sp.flex_mask) == sp.flex_mask
            imu_ok  = (sp.imu_mask == 0) or ((ib & sp.imu_mask) == sp.imu_mask)
            ok = flex_ok and imu_ok
            marks.append(f"{nm}:{'OK' if ok else '..'}")

        # Per-finger state: show name and [BENT] or [open]
        finger_states = "  ".join(
            f"{FINGER_NAMES[i]}:{'[BENT]' if fb & (1 << i) else '[open]'}"
            for i in range(5)
        )
        gesture_matches = "  ".join(marks)
        print(
            f"\r{finger_states}  |  imu:{flags:<28}| {gesture_matches}   ",
            end="", flush=True
        )

    controller.on_frame = on_frame
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MONITOR] exit.")


def run_calibration(controller, say=print) -> dict | None:
    """Run the firmware's open/bent calibration for all 5 fingers.

    Returns a dict {"rest": [...], "bent": [...]} on success, or None if
    aborted. The caller (hub.py) saves this to calibration.json via CalStore.
    """
    done       = threading.Event()
    result     = {}

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
            result["rest"] = data.get("rest", [])
            result["bent"] = data.get("bent", [])
            print(f"   REST: {result['rest']}")
            print(f"   BENT: {result['bent']}")
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
            return None
        controller.cal_confirm()
        if done.wait(0.4):
            break

    controller.on_cal_event = None
    return result if result else None