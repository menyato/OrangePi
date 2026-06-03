import glove_controller
import threading
import queue
import time
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── CONFIG ────────────────────────────────────────────────────────────────────
UART_DEV   = "/dev/ttyS5"   # UART-B on most OrangePi variants; change if needed
BAUD       = 115200
TIMEOUT_S  = 0.1            # serial read timeout
LOG_LEVEL  = logging.INFO

# Finger index names for human-readable output
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ── DATA CLASSES ──────────────────────────────────────────────────────────────
@dataclass
class IMUData:
    aX: float = 0.0
    aY: float = 0.0
    aZ: float = 0.0

@dataclass
class SensorFrame:
    imu:        IMUData        = field(default_factory=IMUData)
    raw_flex:   list           = field(default_factory=lambda: [0]*5)
    flex_bits:  int            = 0   # 0x00–0x1F, bit i set → finger i bent
    imu_bits:   int            = 0   # see ATmega comments for bit mapping
    timestamp:  float          = field(default_factory=time.time)

    @property
    def finger_bent(self) -> list:
        """Return list of booleans, one per finger."""
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
        bent = [FINGER_NAMES[i] for i, v in enumerate(self.finger_bent) if v]
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
_V_RE   = re.compile(
    r"V:([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),"
    r"(\d+),(\d+),(\d+),(\d+),(\d+)\|([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"
)
_CFG_RE = re.compile(
    r"CFG:(\d+),(\d+),(\d+)\|(\d+),(\d+),(\d+)\|(\d+),(\d+),(\d+)"
    r"\|(\d+),(\d+),(\d+)\|(\d+),(\d+),(\d+)"
)

def parse_frame(line: str) -> Optional[SensorFrame]:
    m = _V_RE.match(line)
    if not m:
        return None
    g = m.groups()
    frame = SensorFrame(
        imu       = IMUData(float(g[0]), float(g[1]), float(g[2])),
        raw_flex  = [int(g[i]) for i in range(3, 8)],
        flex_bits = int(g[8], 16),
        imu_bits  = int(g[9], 16),
    )
    return frame

# ── CONTROLLER ────────────────────────────────────────────────────────────────
class GloveController:
    """
    Thread-safe driver for the ATmega glove controller.

    Usage:
        ctrl = GloveController()
        ctrl.on_frame = lambda f: print(f)
        ctrl.start()
        ...
        ctrl.calibrate()
        ctrl.stop()
    """

    def __init__(self,
                 port:    str = UART_DEV,
                 baud:    int = BAUD,
                 on_frame:    Optional[Callable[[SensorFrame], None]] = None,
                 on_cal_event: Optional[Callable[[str, dict], None]]  = None):

        self.port        = port
        self.baud        = baud
        self.on_frame    = on_frame     # called from reader thread
        self.on_cal_event = on_cal_event

        self._ser:   Optional[glove_controller.Serial] = None
        self._tx_q:  queue.Queue = queue.Queue()
        self._stop   = threading.Event()
        self._ready  = threading.Event()

        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        self.cal     = CalibrationState()
        self.last_frame: Optional[SensorFrame] = None

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=LOG_LEVEL,
            datefmt="%H:%M:%S",
        )
        self.log = logging.getLogger("GloveCtrl")

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, wait_ready: bool = True, timeout: float = 5.0) -> bool:
        """Open the serial port and start background threads."""
        try:
            self._ser = glove_controller.Serial(
                self.port, self.baud,
                timeout=TIMEOUT_S,
                write_timeout=1.0,
            )
        except glove_controller.SerialException as e:
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
            return ok
        return True

    def stop(self):
        """Gracefully shut down threads and close port."""
        self._stop.set()
        self._tx_q.put(None)   # unblock TX thread
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
        """
        Trigger a vibration motor.
        motor: 1, 2, or 3
        ms:    duration in milliseconds
        """
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
            except glove_controller.SerialException as e:
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
            except glove_controller.SerialException as e:
                self.log.error(f"TX error: {e}")

    def _dispatch(self, line: str):
        self.log.debug(f"RX ← {line!r}")

        # ── ready ──
        if line == "RDY":
            self.log.info("ATmega ready")
            self._ready.set()
            return

        # ── sensor frame ──
        if line.startswith("V:"):
            frame = parse_frame(line)
            if frame:
                self.last_frame = frame
                if self.on_frame:
                    self.on_frame(frame)
            else:
                self.log.warning(f"Bad V frame: {line!r}")
            return

        # ── calibration ──
        if line.startswith("CAL:"):
            self._handle_cal(line)
            return

        # ── config / motor confirmations ──
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
        tag = line[4:]   # strip "CAL:"

        if tag == "START":
            self.cal = CalibrationState(active=True)
            self.log.info("Calibration started")
            if self.on_cal_event:
                self.on_cal_event("START", {})

        elif tag.startswith("OPEN:"):
            n = int(tag[5:])
            self.cal.finger    = n
            self.cal.wait_bent = False
            msg = f"Hold {FINGER_NAMES[n]} OPEN then press Enter"
            self.log.info(msg)
            if self.on_cal_event:
                self.on_cal_event("OPEN", {"finger": n, "name": FINGER_NAMES[n]})

        elif tag.startswith("BENT:"):
            n = int(tag[5:])
            self.cal.wait_bent = True
            msg = f"Bend {FINGER_NAMES[n]} fully then press Enter"
            self.log.info(msg)
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


# ── INTERACTIVE CLI ───────────────────────────────────────────────────────────
def interactive_cli(ctrl: GloveController):
    """
    Minimal interactive shell so you can control the glove by hand
    while frames stream in the background.
    """
    print("\nGlove CLI — commands:")
    print("  cal          start calibration")
    print("  <enter>      confirm calibration step")
    print("  thresh <n>   set threshold percent (10-90)")
    print("  status       query ATmega config")
    print("  mt<1-3> <ms> fire motor")
    print("  q            quit\n")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd.lower() in ("q", "quit", "exit"):
            break
        elif cmd == "":
            ctrl.cal_confirm()
        elif cmd == "cal":
            ctrl.calibrate()
        elif cmd.startswith("thresh"):
            try:
                ctrl.set_threshold(int(cmd.split()[1]))
            except (IndexError, ValueError) as e:
                print(f"  Error: {e}")
        elif cmd == "status":
            ctrl.status()
        elif cmd.startswith("mt"):
            try:
                parts = cmd.split()
                motor = int(parts[0][2])
                ms    = int(parts[1])
                ctrl.vibrate(motor, ms)
            except (IndexError, ValueError) as e:
                print(f"  Error: {e}")
        else:
            print(f"  Unknown command: {cmd!r}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="OrangePi glove controller driver")
    ap.add_argument("--port",    default=UART_DEV,  help="Serial device (default: %(default)s)")
    ap.add_argument("--baud",    default=BAUD, type=int, help="Baud rate (default: %(default)s)")
    ap.add_argument("--verbose", action="store_true",    help="Show all raw frames")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger("GloveCtrl").setLevel(logging.DEBUG)

    def on_frame(f: SensorFrame):
        # Print only when something is actually happening to avoid spam
        if f.flex_bits or f.imu_bits:
            print(f"  ★ {f}")

    ctrl = GloveController(
        port=args.port,
        baud=args.baud,
        on_frame=on_frame,
    )

    if not ctrl.start(wait_ready=True, timeout=5.0):
        print(f"Failed to open {args.port}. Check wiring and device path.")
        exit(1)

    try:
        interactive_cli(ctrl)
    finally:
        ctrl.stop()