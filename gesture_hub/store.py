"""
store.py — persist gesture definitions to a JSON file.

The JSON format stores raw ATmega bitmasks so there is zero translation
between what the firmware sends and what gets matched:

    {
      "START": { "name":"START", "flex_mask":17, "imu_mask":1,
                 "motion":"static", "hold_frames":3 },
      ...
    }

Falls back to specs.DEFAULT_GESTURES whenever the file is missing or
unreadable, so a corrupt store can never lock the user out of the hub.
"""

import json
import os

from gesture_hub.specs import GestureSpec, DEFAULT_GESTURES


class GestureStore:
    def __init__(self, path: str):
        self.path = path
        self.gestures: dict[str, GestureSpec] = dict(DEFAULT_GESTURES)
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            print(f"[STORE] {self.path} not found — using built-in defaults.")
            self._write_defaults()
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            loaded = {name: GestureSpec.from_dict(d) for name, d in data.items()}
            if loaded:
                # System gestures always come from the current DEFAULT_GESTURES so
                # renamed / replaced gestures (e.g. BACK → OCR_BWD) never survive
                # from an old JSON file.  User feature assignments (FEAT:*) come
                # from the file.
                merged = dict(DEFAULT_GESTURES)
                for name, spec in loaded.items():
                    if name not in DEFAULT_GESTURES:
                        merged[name] = spec
                self.gestures = merged
                print(f"[STORE] Loaded {len(merged)} gestures from {self.path} "
                      f"({len(merged) - len(DEFAULT_GESTURES)} user-assigned)")
        except Exception as e:
            print(f"[STORE] Load failed ({e}); keeping defaults.")

    def save(self) -> None:
        try:
            with open(self.path, "w") as f:
                data = {}
                for n, s in self.gestures.items():
                    d = s.to_dict()
                    d["_desc"] = s.describe()   # human-readable, ignored on load
                    data[n] = d
                json.dump(data, f, indent=2)
            print(f"[STORE] Saved {len(self.gestures)} gestures to {self.path}")
        except OSError as e:
            print(f"[STORE] Save failed: {e}")

    def set(self, name: str, spec: GestureSpec) -> None:
        self.gestures[name] = spec

    # ── private ───────────────────────────────────────────────────────────────
    def _write_defaults(self) -> None:
        """Write the built-in defaults to disk so the file exists next time."""
        try:
            with open(self.path, "w") as f:
                json.dump(
                    {n: s.to_dict() for n, s in self.gestures.items()},
                    f, indent=2
                )
            print(f"[STORE] Wrote default gestures to {self.path}")
        except OSError as e:
            print(f"[STORE] Could not write defaults: {e}")