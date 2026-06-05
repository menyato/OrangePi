"""
state_machine.py — hub control loop implementing the new gesture-driven flow.

═══════════════════════════════════════════════════════════════════════════════
FLOW OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

START (hardcoded) is the global glove on/off toggle.
  • Perform START while off  → glove ACTIVE, begins listening for all gestures.
  • Perform START while on   → glove INACTIVE, ignores everything.
  Without START the engine is paused; no other gesture fires.

──────────────────────────────────────────────────────────────────────────────
FIRST BOOT / ENROLLMENT (some features have no gesture yet)
──────────────────────────────────────────────────────────────────────────────
After START, if any feature has no gesture assigned, the hub enters ENROLL:

  • TTS: "No gesture for <Feature>. Perform twice to register."
  • NEXT scrolls to the next unassigned feature (skipping already-saved ones).
  • User performs the desired gesture twice in a row (record-by-example).
    Both samples must agree — if not, TTS says so and the user tries again.
  • On match → saved to gestures.json under key "FEAT:<title>".
  • Walk continues until all features are assigned, then transitions to ACTIVE.

──────────────────────────────────────────────────────────────────────────────
ACTIVE (all features have gestures)
──────────────────────────────────────────────────────────────────────────────
  • Each feature gesture is a TOGGLE:
      first fire  → launch feature (RUNNING state)
      second fire → stop feature, return to ACTIVE
  • While a feature is RUNNING, only its own gesture is acted on (to stop it).
    All other gestures are ignored.
  • START always works as the global off switch (drops back to INACTIVE,
    stopping any running feature).

──────────────────────────────────────────────────────────────────────────────
PROGRAMMABLE FEATURE (one of the features is ProgrammableGestures)
──────────────────────────────────────────────────────────────────────────────
  When the Programmable feature is toggled on, the hub enters PROGRAMMABLE:
  • NEXT scrolls through ALL features (cursor wraps).
    TTS announces: "<Feature title>. Current gesture: <describe>."
  • EDIT re-binds the highlighted feature's gesture (record-twice, same flow).
  • Programmable's own gesture toggles it off → back to ACTIVE.

──────────────────────────────────────────────────────────────────────────────
HARDCODED GESTURES (never user-assignable):
  START  flex=0x11 imu=0x01 STATIC  — global on/off
  NEXT   flex=0x09 imu=0x08 FLICK   — scroll (enrollment + Programmable)
  EDIT   flex=0x05 imu=0x04 FLICK   — re-bind in Programmable
──────────────────────────────────────────────────────────────────────────────

Threads
-------
  RX thread  (GloveController) → on_frame() → engine.feed() / recorder.feed()
  RX thread  (engine fires)    → dispatch()  puts name on _queue
  Main thread                  → run() consumes _queue, blocks while feature runs
"""

import queue
import threading
import time
from enum import Enum, auto

from features.base import Feature, FeatureContext


class State(Enum):
    INACTIVE     = auto()   # START not yet done — engine paused
    ENROLL       = auto()   # walking through unassigned features
    ACTIVE       = auto()   # listening for feature gestures
    RUNNING      = auto()   # one feature is executing
    PROGRAMMABLE = auto()   # gesture-management panel is open


class HubStateMachine:
    def __init__(self,
                 controller,
                 engine,
                 recorder,
                 registry,
                 store,
                 feedback,
                 link,
                 record_window_s: float = 2.5):

        self.controller      = controller
        self.engine          = engine
        self.recorder        = recorder
        self.registry        = registry
        self.store           = store
        self.feedback        = feedback
        self.link            = link
        self.record_window_s = record_window_s

        self.state     = State.INACTIVE
        self._queue:   queue.Queue             = queue.Queue()
        self._abort:   threading.Event | None  = None
        self._shutdown = threading.Event()

        # recording gate — set True while capture window is open
        self.recording = False

        # feature currently active in RUNNING / PROGRAMMABLE
        self._active_feature  = None
        self._active_key: str = ""      # gesture key that launched it

    # ═══════════════════════════════════════════════════════════════════════
    # Frame routing  (RX thread)
    # ═══════════════════════════════════════════════════════════════════════

    def on_frame(self, frame) -> None:
        if self.recording:
            self.recorder.feed(frame)
        elif self.state != State.INACTIVE:
            self.engine.feed(frame)
        # INACTIVE: engine is paused; frames are discarded

    # ═══════════════════════════════════════════════════════════════════════
    # Gesture dispatch  (RX thread → queue)
    # ═══════════════════════════════════════════════════════════════════════

    def dispatch(self, name: str) -> None:
        """Called by GestureEngine when a gesture fires."""
        # START is handled inline on every state
        if name == "START":
            self._queue.put("START")
            return

        if self.state == State.INACTIVE:
            return  # everything else blocked

        if self.state == State.RUNNING:
            # only the feature's own toggle gesture (or START) is forwarded
            if name == self._active_key and self._abort is not None:
                print(f"[SM] Toggle-off gesture for running feature.")
                self._abort.set()
            return

        self._queue.put(name)

    # ═══════════════════════════════════════════════════════════════════════
    # Main loop  (main thread)
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        self.feedback.speak(
            "Glove ready. Perform the start gesture to activate."
        )
        while not self._shutdown.is_set():
            try:
                name = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle(name)

    def stop(self) -> None:
        self._shutdown.set()
        if self._abort:
            self._abort.set()

    # ═══════════════════════════════════════════════════════════════════════
    # Central dispatcher
    # ═══════════════════════════════════════════════════════════════════════

    def _handle(self, name: str) -> None:

        # ── global START toggle ───────────────────────────────────────────
        if name == "START":
            if self.state == State.INACTIVE:
                self._activate()
            else:
                self._deactivate()
            return

        # ── per-state handling ────────────────────────────────────────────
        if self.state == State.ENROLL:
            self._handle_enroll(name)

        elif self.state == State.ACTIVE:
            self._handle_active(name)

        elif self.state == State.PROGRAMMABLE:
            self._handle_programmable(name)

        # RUNNING is handled inline in dispatch() (abort only)

    # ═══════════════════════════════════════════════════════════════════════
    # INACTIVE → ACTIVE / ENROLL
    # ═══════════════════════════════════════════════════════════════════════

    def _activate(self) -> None:
        self.feedback.confirm()
        missing = self.registry.unassigned(self.store)
        if missing:
            self.state = State.ENROLL
            feat = self.registry.start_enrollment(self.store)
            self._announce_enroll(feat)
        else:
            self.state = State.ACTIVE
            self.feedback.speak("Glove active.")

    def _deactivate(self) -> None:
        if self._abort:
            self._abort.set()
        self.state = State.INACTIVE
        self._active_feature = None
        self._active_key     = ""
        self._drain_queue()
        self.feedback.speak("Glove off.")

    # ═══════════════════════════════════════════════════════════════════════
    # ENROLL state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_enroll(self, name: str) -> None:
        feat = self.registry.current_unassigned
        if feat is None:
            # shouldn't happen, but guard
            self.state = State.ACTIVE
            self.feedback.speak("All gestures assigned. Glove active.")
            return

        if name == "NEXT":
            # skip to next unassigned feature
            nxt = self.registry.next_enrollment(self.store)
            if nxt is None:
                self.state = State.ACTIVE
                self.feedback.speak("All gestures assigned. Glove active.")
            else:
                self._announce_enroll(nxt)
            return

        # Any other gesture name means: user performed a gesture → record it
        # (The engine fired it, so it's a real gesture attempt.)
        # We ignore the name itself; we do record-by-example here.
        # But actually, record-by-example requires the user to hold the gesture
        # through a capture window — so we start capture on any non-system
        # gesture event. If it's NEXT we skip; any other name triggers capture.
        if name in ("EDIT",):
            # EDIT has no meaning in ENROLL; ignore
            return

        # Trigger the two-sample capture for the current feature
        self._record_feature_gesture(feat)

    def _announce_enroll(self, feat) -> None:
        key = self.registry.gesture_key(feat)
        if key in self.store.gestures:
            # already assigned (race), move on
            self._advance_enroll()
            return
        self.feedback.speak(
            f"No gesture for {feat.title}. "
            f"Perform the gesture twice to register it. "
            f"Or do the next gesture to skip to the next feature."
        )

    def _advance_enroll(self) -> None:
        nxt = self.registry.next_enrollment(self.store)
        if nxt is None:
            self.state = State.ACTIVE
            self.feedback.speak("All gestures assigned. Glove active.")
        else:
            self._announce_enroll(nxt)

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVE state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_active(self, name: str) -> None:
        # NEXT / EDIT have no role in ACTIVE
        if name in ("NEXT", "EDIT"):
            return

        # Check if 'name' matches any feature's assigned gesture key
        for feat in self.registry.features:
            key = self.registry.gesture_key(feat)
            if key == name:
                self._launch_feature(feat, key)
                return

        # Unknown gesture — ignore silently
        print(f"[SM] Unknown gesture in ACTIVE: {name!r}")

    # ═══════════════════════════════════════════════════════════════════════
    # PROGRAMMABLE state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_programmable(self, name: str) -> None:
        prog_key = self._active_key

        if name == prog_key:
            # toggle off
            self.state           = State.ACTIVE
            self._active_feature = None
            self._active_key     = ""
            self.feedback.speak("Programmable closed.")
            return

        if name == "NEXT":
            self.registry.advance()
            feat = self.registry.current
            self._announce_prog_feature(feat)
            return

        if name == "EDIT":
            feat = self.registry.current
            self._record_feature_gesture(feat, in_programmable=True)
            return

    def _announce_prog_feature(self, feat) -> None:
        key  = self.registry.gesture_key(feat)
        spec = self.store.gestures.get(key)
        if spec:
            desc = spec.describe()
        else:
            desc = "no gesture assigned"
        self.feedback.speak(f"{feat.title}. Current gesture: {desc}.")

    # ═══════════════════════════════════════════════════════════════════════
    # Launching a feature
    # ═══════════════════════════════════════════════════════════════════════

    def _launch_feature(self, feat, key: str) -> None:
        from features.programmable import ProgrammableGestures

        self.feedback.select()
        self.feedback.speak(f"Opening {feat.title}.")

        self._active_feature = feat
        self._active_key     = key

        if isinstance(feat, ProgrammableGestures):
            # Enter Programmable panel — no blocking thread needed
            self.state = State.PROGRAMMABLE
            self.feedback.speak(
                "Programmable open. Next to scroll features, "
                "edit to change a gesture, or do the programmable gesture again to close."
            )
            self._announce_prog_feature(self.registry.current)
            return

        # Normal feature: run in main thread, blocking
        self._abort = threading.Event()
        self.state  = State.RUNNING
        try:
            feat.run(FeatureContext(link=self.link,
                                    abort=self._abort,
                                    feedback=self.feedback))
        except Exception as e:
            print(f"[SM] Feature crashed: {e}")
            self.feedback.error()
        finally:
            self._abort          = None
            self._active_feature = None
            self._active_key     = ""
            self.state           = State.ACTIVE
            self._drain_queue()
            self.feedback.speak(f"{feat.title} closed.")

    # ═══════════════════════════════════════════════════════════════════════
    # Record-by-example (two-sample match)
    # ═══════════════════════════════════════════════════════════════════════

    def _record_feature_gesture(self, feat, in_programmable: bool = False) -> None:
        """Record a gesture for feat via two matching samples."""
        title = feat.title
        key   = self.registry.gesture_key(feat)

        self.feedback.speak(
            f"Recording gesture for {title}. "
            f"Perform the gesture after the buzz."
        )
        sample1 = self._capture_sample()
        if sample1 is None:
            self.feedback.error()
            self.feedback.speak("No gesture detected. Try again.")
            return

        self.feedback.speak("Good. Perform it again to confirm.")
        sample2 = self._capture_sample()

        if sample2 is None or sample1 != sample2:
            self.feedback.error()
            self.feedback.speak(
                "The two tries did not match. Nothing was saved. Try again."
            )
            return

        spec = self.recorder.build_spec(key, sample1)
        self.store.set(key, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)

        self.feedback.confirm()
        self.feedback.speak(
            f"Gesture for {title} saved. Now {spec.describe()}."
        )

        # After saving in enrollment, advance to next unassigned
        if not in_programmable:
            self._advance_enroll()

    def _capture_sample(self):
        """Open a recording window and return analyze() result."""
        self.recorder.reset()
        self.feedback.beep()
        time.sleep(0.4)          # brief pause before window opens
        self.recording = True
        time.sleep(self.record_window_s)
        self.recording = False
        return self.recorder.analyze()

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass