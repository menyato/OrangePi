"""
registry.py — the ordered, scrollable list of features.

NEXT advances the cursor (wrapping at the end); current() is what SELECT
launches. Add a feature by appending its instance here from hub.py.
"""


class FeatureRegistry:
    def __init__(self, features: list):
        if not features:
            raise ValueError("FeatureRegistry needs at least one feature")
        self._features = list(features)
        self._i = 0

    def current(self):
        return self._features[self._i]

    def next(self):
        self._i = (self._i + 1) % len(self._features)
        return self.current()

    def titles(self) -> list[str]:
        return [f.title for f in self._features]
