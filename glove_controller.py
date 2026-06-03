#!/usr/bin/env python3
"""
glove_controller.py
OrangePi ↔ ATmega Glove Controller — UART-B driver

Wire:   OrangePi UART-B TX → ATmega RX
        OrangePi UART-B RX → ATmega TX
        Shared GND

Protocol (ATmega → OrangePi):
  RDY                          – boot ready
  V:<aX>,<aY>,<aZ>,<r0>...<r4>|<fb_hex><ib_hex>  – 10 Hz sensor frame
  CAL:START / CAL:OPEN:<n> / CAL:BENT:<n>
  CAL:R:<n>:<val> / CAL:B:<n>:<val> / CAL:DONE
  CFG:<r0>,<b0>,<t0>|...       – status reply
  MOT:<n>:<ms>                 – motor confirmation

Protocol (OrangePi → ATmega):
  cal\n          – start calibration
  thresh <10-90>\n
  status\n
  mt<1-3> <ms>\n
  <enter>\n      – advance calibration step
"""

import serial
import threading
import queue
import time
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── CONFIG ────────────────────────────────────────────────────────────────────
UART_DEV   = "/dev/ttyS5"   # UART-B on OrangePi Zero 2W
BAUD       = 115200
TIMEOUT_S  = 0.1
LOG_LEVEL  = logging.INFO

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ── DATA CLASSES ──────────────────────────────────────────────────────────────
@dataclass
class IMUData:
    aX: float = 0.0
    aY: float = 0.0
    aZ: float = 0.0

@dataclass
class SensorFrame:
    imu:       IMUData = field(default_factory=IMUData)
    raw_flex:  list    = field(default_factory=lambda: [0]*5)
    flex_bits: int     = 0
    imu_bits:  int     = 0
    timestamp: float   = field(default_factory=time.time)

    @property
    def finger_bent(self) -> list:
        return [(self.flex_bits >> i) & 1 == 1 for i in range(5)]

    @property
    def imu_flags(self) -> dict:
        b = self.imu_bits
        return {
            "tilt_right":    bool(b & (1 << 0)),
            "tilt_left":     bool(b & (1 << 1)),
            "tilt_forward":  bool(b & (1 << 2)),
            "tilt_backward": bool(b & (1 << 3)),
            "rotate_cw":     bool(b & (1 << 4)),
            "rotate_ccw":    bool(b & (1 << 5)),
        }

    def __str__(self):
        bent  = [FINGER_NAMES[i] for i, v in enumerate(self.finger_bent) if v]
        flags = [k for k, v in self.imu_flags.items() if v]
        return (
            f"IMU aX={self.imu.aX:+.1f} aY={self.imu.aY:+.1f} aZ={self.imu.aZ:+.1f} | "
            f"Flex {[f'{r:3d}' for r in self.raw_flex]} | "
            f"Bent: {bent or 'none'} | IMU: {flags or 'none'}"
        )

@dataclass
class CalibrationState:
    active:    bool = False
    finger:    int  = 0
    wait_bent: bool = False
    rest_vals: list = field(default_factory=lambda: [None]*5)
    bent_vals: list = field(default_factory=lambda: [None]*5)

# ── PARSER ────────────────────────────────────────────────────────────────────
_V_RE = re.compile(
    r"V:([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),"
    r"(\d+),(\d+),(\d+),(\d+),(\d+)\|([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"
)

def parse_frame(line: str) -> Optional[SensorFrame]:
    m = _V_RE.match(line)
    if not m:
        return None
    g = m.groups()
    return SensorFrame(
        imu       = IMUData(float(g[0]), float(g[1]), float(g[2])),
        raw_flex  = [int(g[i]) for i in range(3, 8)],
        flex_bits = int(g[8], 16),
        imu_bits  = int(g[9], 16),
    )

# ── CONTROLLER ────────────────────────────────────────────────────────────────
class GloveController:
    """
    Thread-safe driver for the ATmega glove controller.

    Usage:
        ctrl = GloveController()
        ctrl.on_frame = lambda f: print(f)
        ctrl.start()
        ctrl.calibrate()
        ctrl.stop()
    """

    def __init__(self,
                 port:         str  = UART_DEV,
                 baud:         int  = BAUD,
                 on_frame:     Optional[Callable[[SensorFrame], None]] = None,
                 on_cal_event: Optional[Callable[[str, dict],   None]] = None):

        self.port         = port
        self.baud         = baud
        self.on_frame     = on_frame
        self.on_cal_event = on_cal_event

        self._ser:       Optional[serial.Serial] = None
        self._tx_q:      queue.Queue             = queue.Queue()
        self._stop       = threading.Event()
        self._ready      = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        self.cal        = CalibrationState()
        self.last_frame: Optional[SensorFrame] = None

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=LOG_LEVEL,
            datefmt="%H:%M:%S",
        )
        self.log = logging.getLogger("GloveCtrl")

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, wait_ready: bool = True, timeout: float = 5.0) -> bool:
        """Open serial port and start background threads."""
        try:
            self._ser = serial.Serial(
                self.port, self.baud,
                timeout=TIMEOUT_S,
                write_timeout=1.0,
            )
        except serial.SerialException as e:
            self.log.error(f"Cannot open {self.port}: {e}")
            return False

        self._stop.clear()
        self._ready.clear()

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="glove-rx")
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="glove-tx")
        self._rx_thread.start()
        self._tx_thread.start()

        self.log.info(f"Opened {self.port} @ {self.baud} baud")

        if wait_ready:
            ok = self._ready.wait(timeout)
            if not ok:
                self.log.warning("ATmega RDY not received within timeout — continuing anyway")
        return True

    def stop(self):
        """Gracefully shut down threads and close port."""
        self._stop.set()
        self._tx_q.put(None)
        if self._rx_thread:
            self._rx_thread.join(timeout=2)
        if self._tx_thread:
            self._tx_thread.join(timeout=2)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self.log.info("Controller stopped")

    def send(self, cmd: str):
        """Queue a raw command (newline appended automatically)."""
        self._tx_q.put(cmd.strip() + "\n")

    def calibrate(self):
        """Start full 5-finger calibration."""
        self.send("cal")

    def cal_confirm(self):
        """Send Enter to advance the calibration step."""
        self.send("")

    def set_threshold(self, percent: int):
        """Set bend threshold percent (10–90)."""
        if not 10 <= percent <= 90:
            raise ValueError("threshold must be 10–90")
        self.send(f"thresh {percent}")

    def status(self):
        """Request calibration status from ATmega."""
        self.send("status")

    def vibrate(self, motor: int, ms: int):
        """Trigger a vibration motor (motor 1-3, ms > 0)."""
        if motor not in (1, 2, 3):
            raise ValueError("motor must be 1, 2, or 3")
        if ms <= 0:
            raise ValueError("ms must be > 0")
        self.send(f"mt{motor} {ms}")

    # ── internal threads ──────────────────────────────────────────────────────

    def _rx_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(256)
            except serial.SerialException as e:
                self.log.error(f"RX error: {e}")
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line_b, buf = buf.split(b"\n", 1)
                line = line_b.decode("ascii", errors="replace").strip()
                if line:
                    self._dispatch(line)

    def _tx_loop(self):
        while not self._stop.is_set():
            try:
                cmd = self._tx_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd is None:
                break
            try:
                self._ser.write(cmd.encode("ascii"))
                self._ser.flush()
                self.log.debug(f"TX → {cmd.strip()!r}")
            except serial.SerialException as e:
                self.log.error(f"TX error: {e}")

    def _dispatch(self, line: str):
        self.log.debug(f"RX ← {line!r}")

        if line == "RDY":
            self.log.info("ATmega ready")
            self._ready.set()
            return

        if line.startswith("V:"):
            frame = parse_frame(line)
            if frame:
                self.last_frame = frame
                if self.on_frame:
                    self.on_frame(frame)
            else:
                self.log.warning(f"Bad V frame: {line!r}")
            return

        if line.startswith("CAL:"):
            self._handle_cal(line)
            return

        if line.startswith("CFG:"):
            self.log.info(f"Config: {line}")
            if self.on_cal_event:
                self.on_cal_event("CFG", {"raw": line})
            return

        if line.startswith("MOT:"):
            parts = line[4:].split(":")
            self.log.info(f"Motor {parts[0]} fired for {parts[1]} ms")
            return

        self.log.warning(f"Unknown line: {line!r}")

    def _handle_cal(self, line: str):
        tag = line[4:]

        if tag == "START":
            self.cal = CalibrationState(active=True)
            self.log.info("Calibration started")
            if self.on_cal_event:
                self.on_cal_event("START", {})

        elif tag.startswith("OPEN:"):
            n = int(tag[5:])
            self.cal.finger    = n
            self.cal.wait_bent = False
            self.log.info(f"Hold {FINGER_NAMES[n]} OPEN then press Enter")
            if self.on_cal_event:
                self.on_cal_event("OPEN", {"finger": n, "name": FINGER_NAMES[n]})

        elif tag.startswith("BENT:"):
            n = int(tag[5:])
            self.cal.wait_bent = True
            self.log.info(f"Bend {FINGER_NAMES[n]} fully then press Enter")
            if self.on_cal_event:
                self.on_cal_event("BENT", {"finger": n, "name": FINGER_NAMES[n]})

        elif tag.startswith("R:"):
            parts = tag[2:].split(":")
            n, v = int(parts[0]), int(parts[1])
            self.cal.rest_vals[n] = v
            self.log.info(f"  REST[{FINGER_NAMES[n]}] = {v}")
            if self.on_cal_event:
                self.on_cal_event("REST", {"finger": n, "value": v})

        elif tag.startswith("B:"):
            parts = tag[2:].split(":")
            n, v = int(parts[0]), int(parts[1])
            self.cal.bent_vals[n] = v
            self.log.info(f"  BENT[{FINGER_NAMES[n]}] = {v}")
            if self.on_cal_event:
                self.on_cal_event("BENT_VAL", {"finger": n, "value": v})

        elif tag == "DONE":
            self.cal.active = False
            self.log.info("Calibration complete")
            self.log.info(f"  REST: {self.cal.rest_vals}")
            self.log.info(f"  BENT: {self.cal.bent_vals}")
            if self.on_cal_event:
                self.on_cal_event("DONE", {
                    "rest": self.cal.rest_vals,
                    "bent": self.cal.bent_vals,
                })



# ── MOTOR TEST CLI ────────────────────────────────────────────────────────────
_frame_count = 0
_last_print  = 0.0

def _on_frame(f: SensorFrame):
    global _frame_count, _last_print
    _frame_count += 1
    now = time.time()
    if now - _last_print >= 1.0:
        bent  = [str(i) for i, v in enumerate(f.finger_bent) if v]
        flags = [k for k, v in f.imu_flags.items() if v]
        print(
            f"\r[{_frame_count:5d} frames] "
            f"aX={f.imu.aX:+.1f} aY={f.imu.aY:+.1f} aZ={f.imu.aZ:+.1f} | "
            f"flex={f.raw_flex} | "
            f"bent={''.join(bent) or '-'} imu={flags or '-'}   ",
            end="", flush=True
        )
        _last_print = now

def motor_test_cli(ctrl: GloveController):
    print("\nMotor test — commands:")
    print("  1 / 2 / 3          fire motor for 300 ms")
    print("  1 <ms>             fire motor 1 for custom duration")
    print("  all                fire motors 1→2→3 in sequence")
    print("  all <ms>           same with custom duration")
    print("  cal                start finger calibration")
    print("  <enter>            confirm calibration step")
    print("  thresh <n>         set threshold percent (10-90)")
    print("  status             query ATmega config")
    print("  q                  quit\n")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            ctrl.cal_confirm()
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

        elif cmd == "cal":
            ctrl.calibrate()

        elif cmd == "status":
            ctrl.status()

        elif cmd.startswith("thresh"):
            try:
                ctrl.set_threshold(int(parts[1]))
            except (IndexError, ValueError) as e:
                print(f"  Error: {e}")

        else:
            print(f"  Unknown: {raw!r}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="OrangePi glove controller")
    ap.add_argument("--port",    default=UART_DEV, help="Serial device (default: %(default)s)")
    ap.add_argument("--baud",    default=BAUD, type=int, help="Baud rate (default: %(default)s)")
    ap.add_argument("--verbose", action="store_true", help="Show raw debug lines")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger("GloveCtrl").setLevel(logging.DEBUG)

    _last_print = time.time()

    ctrl = GloveController(
        port=args.port,
        baud=args.baud,
        on_frame=_on_frame,
    )

    if not ctrl.start(wait_ready=False):
        print(f"Cannot open {args.port}. Check wiring and device path.")
        exit(1)

    print(f"Connected to {args.port} — frames streaming in background")

    try:
        motor_test_cli(ctrl)
    finally:
        print("\nStopping...")
        ctrl.stop()