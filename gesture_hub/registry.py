"""
registry.py — ordered list of features, each with an optional user-assigned gesture.

A feature's gesture is stored in GestureStore under the key  "FEAT:<title>".
The registry itself only knows which features exist and their order; it does
NOT own gesture data — that belongs to GestureStore so it survives restarts.

Public API used by HubStateMachine:
    registry.features               list of feature objects (stable order)
    registry.unassigned(store)      features that have no gesture in store yet
    registry.gesture_key(feat)      "FEAT:<title>" key used in store
    registry.next_unassigned(store) advance cursor among unassigned features;
                                    returns current unassigned feature or None
    registry.current_unassigned     the feature currently being enrolled
    registry.advance(store)         scroll cursor (for Programmable panel)
    registry.current               feature at cursor
"""


class FeatureRegistry:
    def __init__(self, features: list):
        if not features:
            raise ValueError("FeatureRegistry needs at least one feature")
        self._features = list(features)
        self._cursor   = 0                  # used inside Programmable panel
        self._unassigned_cursor = 0         # used during enrollment

    # ── gesture key ──────────────────────────────────────────────────────────
    @staticmethod
    def gesture_key(feature) -> str:
        """Stable store key for a feature's user-assigned gesture."""
        return f"FEAT:{feature.title}"

    # ── feature list ─────────────────────────────────────────────────────────
    @property
    def features(self) -> list:
        return self._features

    def unassigned(self, store) -> list:
        """Return features that have no gesture saved in store yet."""
        return [f for f in self._features
                if self.gesture_key(f) not in store.gestures]

    # ── enrollment cursor (first-boot / missing gesture walk) ────────────────
    @property
    def current_unassigned(self):
        """Feature currently being enrolled (may be None if all assigned)."""
        return getattr(self, "_enroll_feat", None)

    def start_enrollment(self, store):
        """Initialise the enrollment walk. Returns first unassigned or None."""
        missing = self.unassigned(store)
        if not missing:
            self._enroll_feat = None
            return None
        self._enroll_feat = missing[0]
        self._enroll_list = missing
        self._enroll_idx  = 0
        return self._enroll_feat

    def next_enrollment(self, store):
        """Advance to the next unassigned feature. Returns it or None if done."""
        missing = self.unassigned(store)   # re-query after each save
        if not missing:
            self._enroll_feat = None
            return None
        # try to keep walking forward from where we were
        idx = 0
        if hasattr(self, "_enroll_feat") and self._enroll_feat in missing:
            idx = missing.index(self._enroll_feat) + 1
        if idx >= len(missing):
            self._enroll_feat = None
            return None
        self._enroll_feat = missing[idx]
        return self._enroll_feat

    # ── Programmable panel cursor ─────────────────────────────────────────────
    @property
    def current(self):
        return self._features[self._cursor]

    def advance(self) -> None:
        """Move the Programmable panel cursor one step forward (wrapping)."""
        self._cursor = (self._cursor + 1) % len(self._features)