"""
features/programmable.py — the "Programmable Gestures" feature.

This feature is special: toggling it on enters PROGRAMMABLE state in the
state machine, which handles all the scrolling and re-binding logic directly.
The run() method is therefore never called for this feature — the state
machine short-circuits to PROGRAMMABLE before reaching it.

It is listed in the feature registry like any other feature so that:
  • It appears in the enrollment walk (user assigns a gesture to open it).
  • It appears inside the Programmable panel so its own gesture can be rebound.
"""

from features.base import Feature, FeatureContext


class ProgrammableGestures(Feature):
    title = "Programmable Gestures"
    name  = "programmable"

    def run(self, ctx: FeatureContext) -> None:
        # Should never be called — state_machine handles this feature specially.
        ctx.feedback.speak("Programmable panel is already open.")