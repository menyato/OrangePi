#!/usr/bin/env python3
"""
motor_test.py
Quick test: stream glove frames and fire vibration motors interactively.

Usage:
    python3 motor_test.py
    python3 motor_test.py --port /dev/ttyS5

Commands at the prompt:
    1 / 2 / 3        fire motor 1, 2, or 3 for 300 ms
    1 500            fire motor 1 for 500 ms
    all              fire all three motors in sequence
    q                quit
"""

import sys
import threading
import time
import argparse
from glove_controller import GloveController, SensorFrame

UART_DEV = "/dev/ttyS5"

# ── frame callback (runs in RX thread) ────────────────────────────────────────
frame_count = 0
last_print  = time.time()

def on_frame(f: SensorFrame):
    global frame_count, last_print
    frame_count += 1
    now = time.time()
    # Print a summary once per second so the terminal isn't spammed
    if now - last_print >= 1.0:
        bent  = [str(i) for i, v in enumerate(f.finger_bent) if v]
        flags = [k for k, v in f.imu_flags.items() if v]
        print(
            f"\r[{frame_count:5d} frames] "
            f"aX={f.imu.aX:+.1f} aY={f.imu.aY:+.1f} aZ={f.imu.aZ:+.1f} | "
            f"flex={f.raw_flex} | "
            f"bent={''.join(bent) or '-'} imu={flags or '-'}",
            end="", flush=True
        )
        last_print = now

# ── CLI ───────────────────────────────────────────────────────────────────────
def run_cli(ctrl: GloveController):
    print("\nMotor test — commands:")
    print("  1 / 2 / 3          fire motor for 300 ms")
    print("  1 <ms>             fire motor 1 for custom duration")
    print("  all                fire motors 1→2→3 in sequence")
    print("  all <ms>           same with custom duration")
    print("  q                  quit\n")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        if raw.lower() in ("q", "quit", "exit"):
            break

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd == "all":
            ms = int(parts[1]) if len(parts) > 1 else 300
            for motor in (1, 2, 3):
                print(f"  → motor {motor} for {ms} ms")
                ctrl.vibrate(motor, ms)
                time.sleep((ms / 1000) + 0.1)

        elif cmd in ("1", "2", "3"):
            motor = int(cmd)
            ms    = int(parts[1]) if len(parts) > 1 else 300
            print(f"  → motor {motor} for {ms} ms")
            ctrl.vibrate(motor, ms)

        else:
            print(f"  Unknown: {raw!r}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Glove motor test")
    ap.add_argument("--port", default=UART_DEV)
    ap.add_argument("--baud", default=115200, type=int)
    args = ap.parse_args()

    ctrl = GloveController(port=args.port, baud=args.baud, on_frame=on_frame)

    if not ctrl.start(wait_ready=False):
        print(f"Cannot open {args.port}")
        sys.exit(1)

    print(f"Connected to {args.port} — frames streaming in background")

    try:
        run_cli(ctrl)
    finally:
        print("\nStopping...")
        ctrl.stop()
