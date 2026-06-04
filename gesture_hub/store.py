"""
store.py — persist gesture definitions to a JSON file.

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
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            loaded = {name: GestureSpec.from_dict(d) for name, d in data.items()}
            if loaded:
                self.gestures = loaded
                print(f"[STORE] Loaded {len(loaded)} gestures from {self.path}")
        except Exception as e:
            print(f"[STORE] Load failed ({e}); keeping defaults.")

    def save(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump({n: s.to_dict() for n, s in self.gestures.items()},
                          f, indent=2)
            print(f"[STORE] Saved gestures to {self.path}")
        except OSError as e:
            print(f"[STORE] Save failed: {e}")

    def set(self, name: str, spec: GestureSpec) -> None:
        self.gestures[name] = spec
