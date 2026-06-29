"""
lidar_adapter.py  –  MS200 serial → clean numpy point cloud
============================================================
Data format  (MS200 User Manual §5, Table 5-2 / 5-3)
  Byte  0     : 0x54  frame header
  Byte  1     : 0x2C  low-5 bits = 12 pts
  Bytes 2-3   : rotation speed °/s   (uint16 LE)
  Bytes 4-5   : start angle ×0.01°   (uint16 LE)
  Bytes 6-41  : 12 × (dist_mm uint16 LE, intensity uint8)
  Bytes 42-43 : end angle ×0.01°     (uint16 LE)
  Bytes 44-45 : timestamp ms         (uint16 LE)
  Byte  46    : CRC-8 over bytes 0-45
  Total       : 47 bytes
"""

import serial
import struct
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import medfilt


# ─── Complete 256-entry CRC-8 table (from MS200 manual §5) ───────────────────
_CRC8_TABLE = [
    0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
    0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
    0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
    0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
    0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
    0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
    0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
    0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
    0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
    0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
    0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
    0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
    0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
    0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
    0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
    0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8,
]

def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


@dataclass
class LaserScan:
    """
    One complete 360° sweep.
    angles_rad  : (N,) float32  radians [0, 2π)
    ranges_m    : (N,) float32  metres  (0.0 = invalid/filtered)
    intensities : (N,) uint8
    timestamp   : float  time.time() when sweep completed
    rpm         : float  measured rotation speed
    """
    angles_rad  : np.ndarray
    ranges_m    : np.ndarray
    intensities : np.ndarray
    timestamp   : float = field(default_factory=time.time)
    rpm         : float = 0.0


class MS200Adapter:
    """
    Reads the MS200 UART stream and produces LaserScan objects.

    Quick start
    -----------
        adapter = MS200Adapter("COM3")
        adapter.start()
        while True:
            scan = adapter.get_scan(timeout=2.0)
            if scan:
                pts = MS200Adapter.to_xy(scan)  # (N,2) float32
        adapter.stop()
    """

    HEADER        = 0x54
    TYPE_BYTE     = 0x2C
    FRAME_LEN     = 47
    PACKET_PTS    = 12
    MIN_DIST_M    = 0.03
    MAX_DIST_M    = 12.0
    MIN_INTENSITY = 16      # 0-15 reserved / error codes per manual

    def __init__(
        self,
        port          : str   = "COM3",
        baud          : int   = 230400,
        rotation_deg  : float = 0.0,
        median_kernel : int   = 5,
        max_jump_m    : float = 0.5,
        use_crc       : bool  = True,
    ):
        self.port          = port
        self.baud          = baud
        self.rotation_rad  = math.radians(rotation_deg)
        self.median_kernel = max(1, median_kernel | 1)   # ensure odd
        self.max_jump_m    = max_jump_m
        self.use_crc       = use_crc

        self._ser          : Optional[serial.Serial] = None
        self._buf          = bytearray()
        self._packets      : List[list] = []
        self._scan_started = False
        self._stats        = {
            "frames_ok": 0, "frames_bad_crc": 0,
            "points_total": 0, "points_filtered": 0,
        }

    # ── auto-detect ───────────────────────────────────────────────────────────
    @staticmethod
    def find_port(baud: int = 230400, timeout: float = 2.0) -> Optional[str]:
        """
        Scan all USB/ACM serial ports and return the first one that streams
        valid MS200 frames (0x54 0x2C header) within `timeout` seconds.
        Returns None if nothing found.

        Usage:
            port = MS200Adapter.find_port()
            if port:
                adapter = MS200Adapter(port)
        """
        import glob
        candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        if not candidates:
            print("[MS200] find_port: no USB/ACM serial devices found.")
            return None

        print(f"[MS200] find_port: probing {candidates}")
        for port in candidates:
            try:
                s = serial.Serial(port, baud, timeout=0.1)
                s.reset_input_buffer()
                deadline = time.time() + timeout
                found = False
                buf = bytearray()
                while time.time() < deadline:
                    chunk = s.read(256)
                    if chunk:
                        buf.extend(chunk)
                    # look for 0x54 0x2C anywhere in buffer
                    for i in range(len(buf) - 1):
                        if buf[i] == 0x54 and buf[i + 1] == 0x2C:
                            found = True
                            break
                    if found:
                        break
                s.close()
                if found:
                    print(f"[MS200] find_port: MS200 detected on {port}")
                    return port
                else:
                    print(f"[MS200] find_port: no MS200 frames on {port}")
            except Exception as e:
                print(f"[MS200] find_port: {port} error — {e}")

        print("[MS200] find_port: MS200 not found on any port.")
        return None

    def start(self) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        self._ser.reset_input_buffer()
        print(f"[MS200] connected {self.port} @ {self.baud}. Syncing sweep…")

    def stop(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        print(f"[MS200] stats: {self._stats}")

    def get_scan(self, timeout: float = 2.0) -> Optional[LaserScan]:
        """Block until a full 360° sweep arrives. Returns None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self._ser.read(2048)
            if raw:
                self._buf.extend(raw)
            scan = self._process_buffer()
            if scan is not None:
                return scan
        return None

    # ── internal buffer parser ─────────────────────────────────────────────────
    def _process_buffer(self) -> Optional[LaserScan]:
        while True:
            idx = self._buf.find(bytes([self.HEADER]))
            if idx == -1:
                self._buf.clear()
                return None
            if len(self._buf) < idx + self.FRAME_LEN:
                if idx > 0:
                    del self._buf[:idx]
                return None

            frame = bytes(self._buf[idx: idx + self.FRAME_LEN])
            del self._buf[:idx + self.FRAME_LEN]

            if frame[1] != self.TYPE_BYTE:
                continue

            if self.use_crc:
                if _crc8(frame[:self.FRAME_LEN - 1]) != frame[self.FRAME_LEN - 1]:
                    self._stats["frames_bad_crc"] += 1
                    continue
            self._stats["frames_ok"] += 1

            speed_dps   = struct.unpack_from("<H", frame, 2)[0]
            start_angle = struct.unpack_from("<H", frame, 4)[0]  / 100.0
            end_angle   = struct.unpack_from("<H", frame, 42)[0] / 100.0
            if end_angle < start_angle:
                end_angle += 360.0
            rpm = speed_dps / 6.0

            pts = []
            for i in range(self.PACKET_PTS):
                off       = 6 + i * 3
                dist_mm   = struct.unpack_from("<H", frame, off)[0]
                intensity = frame[off + 2]
                dist_m    = dist_mm / 1000.0
                angle     = (start_angle + (end_angle - start_angle) * i
                             / (self.PACKET_PTS - 1)) % 360.0
                if (dist_m < self.MIN_DIST_M or dist_m > self.MAX_DIST_M
                        or intensity < self.MIN_INTENSITY):
                    dist_m = 0.0
                pts.append((angle, dist_m, intensity))
            self._stats["points_total"] += self.PACKET_PTS

            # Wait for first angle-0 packet before collecting
            if not self._scan_started:
                if start_angle < 5.0:
                    self._scan_started = True
                    self._packets = [pts]
                continue   # always read more data during sync phase

            # Sweep done: angle wraps back near 0 with enough packets
            if start_angle < 5.0 and len(self._packets) > 20:
                scan = self._build_scan(self._packets, rpm)
                self._packets = [pts]
                return scan

            self._packets.append(pts)

    def _build_scan(self, packets: list, rpm: float) -> LaserScan:
        angles_deg  = []
        ranges_m    = []
        intensities = []
        for pkt in packets:
            for (a, d, i) in pkt:
                angles_deg.append(a)
                ranges_m.append(d)
                intensities.append(i)

        ang = np.array(angles_deg,  dtype=np.float32)
        rng = np.array(ranges_m,    dtype=np.float32)
        ity = np.array(intensities, dtype=np.uint8)

        order = np.argsort(ang)
        ang, rng, ity = ang[order], rng[order], ity[order]

        # Median filter – only on valid points; kernel must not exceed subset size
        valid = rng > 0.0
        n_valid = int(valid.sum())
        if n_valid > self.median_kernel:
            k = self.median_kernel
            if n_valid < k:
                k = n_valid if n_valid % 2 == 1 else n_valid - 1
            filtered = rng.copy()
            filtered[valid] = medfilt(rng[valid], kernel_size=k)
            rng = filtered

        # Jump filter
        if len(rng) > 4:
            fwd  = np.abs(np.diff(rng, append=rng[-1]))
            bwd  = np.abs(np.diff(rng, prepend=rng[0]))
            jump = (fwd > self.max_jump_m) & (bwd > self.max_jump_m)
            rng[jump] = 0.0
            self._stats["points_filtered"] += int(jump.sum())

        ang_rad = (np.radians(ang) + self.rotation_rad) % (2.0 * math.pi)

        return LaserScan(
            angles_rad  = ang_rad,
            ranges_m    = rng,
            intensities = ity,
            timestamp   = time.time(),
            rpm         = rpm,
        )

    @staticmethod
    def to_xy(scan: LaserScan) -> np.ndarray:
        """Returns (N,2) float32 XY points (sensor frame). Invalid dropped."""
        valid = scan.ranges_m > 0.0
        if not valid.any():
            return np.empty((0, 2), dtype=np.float32)
        r  = scan.ranges_m[valid]
        th = scan.angles_rad[valid]
        return np.column_stack([r * np.cos(th), r * np.sin(th)]).astype(np.float32)