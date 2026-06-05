"""
cal_store.py — persist ATmega calibration (REST + BENT per finger) to JSON.

After a successful calibration the hub calls save() with the 5 REST values
and 5 BENT values reported by the ATmega.  On every subsequent boot hub.py
calls restore() which sends a single "setcal" command to the ATmega to reload
those values, then recomputes THRESH via the existing "thresh" command.

calibration.json lives next to gestures.json (same directory as hub.py):
{
  "rest":          [243, 51, 166, 113, 238],
  "bent":          [88,   4,  42,  50,  50],
  "thresh_pct":    35,
  "fingers":       ["Thumb","Index","Middle","Ring","Pinky"]
}

ATmega setcal command (add to .ino — see below):
  setcal <r0> <b0> <r1> <b1> <r2> <b2> <r3> <b3> <r4> <b4>
  e.g.  setcal 243 88 51 4 166 42 113 50 238 50

Arduino snippet to add inside handleSerial(), after the thresh block:

  } else if (strncmp(cmd, "setcal", 6) == 0) {
    int vals[10];
    char* p = cmd + 7;
    for (int i = 0; i < 10; i++) {
      vals[i] = atoi(p);
      p = strchr(p, ' ');
      if (!p) break;
      p++;
    }
    for (int i = 0; i < 5; i++) {
      REST[i] = vals[i * 2];
      BENT[i] = vals[i * 2 + 1];
    }
    recomputeThresh();
    Serial.println(F("CAL:RESTORED"));
  }
"""

import json
import os

DEFAULT_THRESH_PCT = 35   # firmware default is 35 %


class CalStore:
    def __init__(self, path: str):
        self.path = path
        self.rest:       list[int] | None = None
        self.bent:       list[int] | None = None
        self.thresh_pct: int              = DEFAULT_THRESH_PCT
        self._load()

    # ── public ────────────────────────────────────────────────────────────────
    def save(self, rest: list[int], bent: list[int],
             thresh_pct: int = DEFAULT_THRESH_PCT) -> None:
        self.rest       = list(rest)
        self.bent       = list(bent)
        self.thresh_pct = thresh_pct
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "rest":       self.rest,
                    "bent":       self.bent,
                    "thresh_pct": self.thresh_pct,
                    "fingers":    ["Thumb", "Index", "Middle", "Ring", "Pinky"],
                }, f, indent=2)
            print(f"[CAL] Saved calibration to {self.path}")
            print(f"[CAL]   REST: {self.rest}")
            print(f"[CAL]   BENT: {self.bent}")
            print(f"[CAL]   THRESH%: {self.thresh_pct}")
        except OSError as e:
            print(f"[CAL] Save failed: {e}")

    def restore(self, controller) -> bool:
        """Replay saved calibration to the ATmega. Returns True if sent."""
        if self.rest is None or self.bent is None:
            print("[CAL] No saved calibration found — skipping restore.")
            return False

        # Build: setcal r0 b0 r1 b1 r2 b2 r3 b3 r4 b4
        pairs = []
        for r, b in zip(self.rest, self.bent):
            pairs += [str(r), str(b)]
        cmd = "setcal " + " ".join(pairs)
        controller.send(cmd)

        # Also resend thresh so THRESH[] is recomputed from the restored values
        controller.set_threshold(self.thresh_pct)

        print(f"[CAL] Restored calibration → {cmd}")
        print(f"[CAL] Threshold reset to {self.thresh_pct}%")
        return True

    def has_calibration(self) -> bool:
        return self.rest is not None and self.bent is not None

    # ── private ───────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.rest       = data["rest"]
            self.bent       = data["bent"]
            self.thresh_pct = int(data.get("thresh_pct", DEFAULT_THRESH_PCT))
            print(f"[CAL] Loaded calibration from {self.path}")
            print(f"[CAL]   REST: {self.rest}")
            print(f"[CAL]   BENT: {self.bent}")
        except Exception as e:
            print(f"[CAL] Load failed ({e}) — will recalibrate if needed.")
