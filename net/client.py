"""
net/client.py — the single socket link the hub keeps open to the server.

Every feature sends through here. The link injects the feature name into each
message, base64-encodes any JPEG `frame`, frames the payload with a 4-byte
big-endian length prefix, and returns the decoded JSON reply. Wire format is
identical to the original orangepi_client/server, so the server understands it.
"""

import base64
import json
import socket
import struct
import threading
import time


class ServerLink:
    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.host    = host
        self.port    = port
        self.timeout = timeout
        self.sock:   socket.socket | None = None
        self._lock   = threading.Lock()

    def connect(self) -> bool:
        try:
            print(f"[LINK] Connecting to {self.host}:{self.port} ...")
            s = socket.create_connection((self.host, self.port), timeout=20)
            s.settimeout(self.timeout)
            self.sock = s
            print("[LINK] Connected.")
            return True
        except OSError as e:
            print(f"[LINK] Connect failed: {e}")
            self.sock = None
            return False

    def send(self, feature: str, msg: dict) -> dict | None:
        """Send one message for `feature`, return the reply dict (or None)."""
        with self._lock:
            if self.sock is None and not self.connect():
                return None

            payload = dict(msg)
            payload["feature"] = feature
            payload["_t_sent"] = time.time()   # lets the server compute network+queue latency
            frame = payload.get("frame")
            if isinstance(frame, (bytes, bytearray)):
                payload["frame"] = base64.b64encode(frame).decode("ascii")

            try:
                data = json.dumps(payload).encode("utf-8")
                self.sock.sendall(struct.pack(">I", len(data)) + data)
                raw_len = self._recv_exact(4)
                if raw_len is None:
                    self._reset()
                    return None
                n = struct.unpack(">I", raw_len)[0]
                raw = self._recv_exact(n)
                if raw is None:
                    self._reset()
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[LINK] Bad response JSON: {e}")
                    self._reset()
                    return None
            except (OSError, TimeoutError) as e:
                print(f"[LINK] Send error: {e}")
                self._reset()
                return None

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _reset(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = None

    def close(self) -> None:
        self._reset()