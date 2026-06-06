"""
state_machine.py — hub control loop with full TTS narration and START lockout.

TTS contract (for blind users):
  Every state transition, every action, every error is spoken aloud.
  Haptic confirmation (vibration) accompanies every successful action.
  Nothing happens silently.

START lockout:
  After START fires (activate OR deactivate), a cooldown of START_LOCKOUT_S
  seconds is enforced before START can fire again.  This prevents the user
  from accidentally toggling the glove back off immediately with the same
  sustained gesture hold.

States
------
  INACTIVE     — glove off; engine paused; only START is ever acted on.
  ENROLL       — walking unassigned features; NEXT skips, any gesture records.
  ACTIVE       — listening for feature toggle gestures.
  RUNNING      — one feature executing; only its own toggle + START accepted.
  PROGRAMMABLE — gesture-management panel open.
"""

import queue
import threading
import time
from enum import Enum, auto

from features.base import FeatureContext

# Seconds START is locked out after each toggle to prevent accidental re-fire.
START_LOCKOUT_S = 2.0


class State(Enum):
    INACTIVE     = auto()
    ENROLL       = auto()
    ACTIVE       = auto()
    RUNNING      = auto()
    PROGRAMMABLE = auto()


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
        self._queue:   queue.Queue            = queue.Queue()
        self._abort:   threading.Event | None = None
        self._shutdown = threading.Event()

        self.recording = False

        self._active_feature  = None
        self._active_key: str = ""

        # Timestamp of the last START toggle; enforces lockout.
        self._last_start_ts: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Frame routing  (RX thread)
    # ═══════════════════════════════════════════════════════════════════════

    def on_frame(self, frame) -> None:
        if self.recording:
            self.recorder.feed(frame)
        else:
            self.engine.feed(frame)   # always feed — dispatch() filters by state

    # ═══════════════════════════════════════════════════════════════════════
    # Gesture dispatch  (RX thread → queue)
    # ═══════════════════════════════════════════════════════════════════════

    def dispatch(self, name: str) -> None:
        if name == "START":
            # Enforce lockout: drop the event if fired too soon after last toggle.
            if time.time() - self._last_start_ts < START_LOCKOUT_S:
                print(f"[SM] START ignored — within lockout window.")
                return
            self._queue.put("START")
            return

        if self.state == State.INACTIVE:
            return

        if self.state == State.RUNNING:
            if name == self._active_key and self._abort is not None:
                print(f"[SM] Toggle-off for running feature.")
                self._abort.set()
            return

        self._queue.put(name)

    # ═══════════════════════════════════════════════════════════════════════
    # Main loop  (main thread)
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        start_spec = self.store.gestures.get("START")
        start_desc = f" — {start_spec.describe()}" if start_spec else ""
        self.feedback.speak(
            f"Glove ready. "
            f"Perform the start gesture{start_desc} to activate."
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
        if name == "START":
            if self.state == State.INACTIVE:
                self._activate()
            else:
                self._deactivate()
            return

        if self.state == State.ENROLL:
            self._handle_enroll(name)
        elif self.state == State.ACTIVE:
            self._handle_active(name)
        elif self.state == State.PROGRAMMABLE:
            self._handle_programmable(name)

    # ═══════════════════════════════════════════════════════════════════════
    # Activate / Deactivate
    # ═══════════════════════════════════════════════════════════════════════

    def _activate(self) -> None:
        self._last_start_ts = time.time()
        self.feedback.confirm()                          # haptic: short buzz

        missing = self.registry.unassigned(self.store)
        if missing:
            self.state = State.ENROLL
            missing_names = ", ".join(f.title for f in missing)
            available = [f for f in self.registry.features
                         if self.registry.gesture_key(f) in self.store.gestures]
            msg = (
                f"Glove active. "
                f"{len(missing)} feature{'s' if len(missing) > 1 else ''} "
                f"need a gesture: {missing_names}. "
            )
            if available:
                avail_names = ", ".join(f.title for f in available)
                msg += f"Available features: {avail_names}. "
            msg += "Starting enrollment now."
            self.feedback.speak(msg)
            feat = self.registry.start_enrollment(self.store)
            self._announce_enroll(feat)
        else:
            self.state = State.ACTIVE
            feat_list  = ", ".join(f.title for f in self.registry.features)
            self.feedback.speak(
                f"Glove active. "
                f"Available features: {feat_list}. "
                f"Perform a feature gesture to open it."
            )

    def _deactivate(self) -> None:
        self._last_start_ts = time.time()
        prev_state   = self.state
        prev_feature = self._active_feature          # save before clearing

        if self._abort:
            self._abort.set()

        self.state           = State.INACTIVE
        self._active_feature = None
        self._active_key     = ""
        self._drain_queue()

        self.feedback.error()                            # haptic: long buzz = off
        if prev_state == State.RUNNING and prev_feature:
            self.feedback.speak(
                f"{prev_feature.title} stopped. Glove off."
            )
        elif prev_state == State.PROGRAMMABLE:
            self.feedback.speak("Programmable closed. Glove off.")
        else:
            self.feedback.speak("Glove off.")

    # ═══════════════════════════════════════════════════════════════════════
    # ENROLL state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_enroll(self, name: str) -> None:
        feat = self.registry.current_unassigned
        if feat is None:
            self.state = State.ACTIVE
            self.feedback.speak("All gestures assigned. Glove active.")
            return

        if name == "NEXT":
            nxt = self.registry.next_enrollment(self.store)
            if nxt is None:
                self.state = State.ACTIVE
                self.feedback.speak("All gestures assigned. Glove active.")
            else:
                self.feedback.tick()                     # haptic: tiny blip
                self._announce_enroll(nxt)
            return

        if name == "EDIT":
            # EDIT is the explicit trigger to start two-sample recording
            self.feedback.tick()                         # haptic: starting record
            self._record_feature_gesture(feat)
            return

        # Any other gesture: check if it's an already-assigned feature → activate it
        for f in self.registry.features:
            key = self.registry.gesture_key(f)
            if key == name and key in self.store.gestures:
                self._launch_feature(f, key)
                # After the feature closes, resume enrollment if still needed
                if self.state == State.ACTIVE and self.registry.unassigned(self.store):
                    self.state = State.ENROLL
                    cur = self.registry.current_unassigned
                    if cur:
                        self._announce_enroll(cur)
                return

        # Unknown gesture — re-read the enrollment prompt so the user knows
        # exactly which gesture to make next.
        self._announce_enroll(feat)

    def _announce_enroll(self, feat) -> None:
        key = self.registry.gesture_key(feat)
        if key in self.store.gestures:
            self._advance_enroll()
            return
        remaining = len(self.registry.unassigned(self.store))

        # Include physical gesture descriptions so the blind user knows
        # exactly which finger+wrist combination to make.
        edit_spec = self.store.gestures.get("EDIT")
        next_spec  = self.store.gestures.get("NEXT")
        edit_desc  = f" — {edit_spec.describe()}" if edit_spec else ""
        next_desc  = f" — {next_spec.describe()}" if next_spec else ""

        self.feedback.speak(
            f"You are now enrolling: {feat.title}. "
            f"No gesture has been assigned to this feature yet. "
            f"To record a gesture: perform the edit gesture{edit_desc}. "
            f"To skip to the next feature: perform the next gesture{next_desc}. "
            f"To turn the glove off: perform the start gesture. "
            f"{remaining} feature{'s' if remaining > 1 else ''} still need a gesture."
        )

    def _advance_enroll(self) -> None:
        nxt = self.registry.next_enrollment(self.store)
        if nxt is None:
            self.state = State.ACTIVE
            self.feedback.confirm()                      # haptic: all done
            feat_list  = ", ".join(f.title for f in self.registry.features)
            self.feedback.speak(
                f"All gestures registered. Glove active. "
                f"Available features: {feat_list}."
            )
        else:
            self._announce_enroll(nxt)

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVE state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_active(self, name: str) -> None:
        if name == "NEXT":
            self.feedback.speak(
                "Next gesture is for scrolling inside the programmable panel. "
                "Open programmable gestures first."
            )
            return

        if name == "EDIT":
            self.feedback.speak(
                "Edit gesture is for the programmable panel. "
                "Open programmable gestures first."
            )
            return

        for feat in self.registry.features:
            key = self.registry.gesture_key(feat)
            if key == name:
                self._launch_feature(feat, key)
                return

        print(f"[SM] Unrecognised gesture in ACTIVE: {name!r}")

    # ═══════════════════════════════════════════════════════════════════════
    # PROGRAMMABLE state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_programmable(self, name: str) -> None:
        prog_key = self._active_key

        if name == prog_key:
            # Toggle off
            self.state           = State.ACTIVE
            self._active_feature = None
            self._active_key     = ""
            self.feedback.confirm()                      # haptic: closed
            self.feedback.speak(
                "Programmable gestures closed. Glove active. "
                "Perform a feature gesture to open it."
            )
            return

        if name == "NEXT":
            self.registry.advance()
            feat = self.registry.current
            self.feedback.tick()                         # haptic: scrolled
            self._announce_prog_feature(feat)
            return

        if name == "EDIT":
            feat = self.registry.current
            self.feedback.tick()                         # haptic: starting edit
            self.feedback.speak(f"Editing gesture for {feat.title}.")
            self._record_feature_gesture(feat, in_programmable=True)
            return

        # Feature gesture fired while in programmable — tell the user
        self.feedback.speak(
            "A feature gesture was detected. "
            "Close programmable gestures first to use features."
        )

    def _announce_prog_feature(self, feat) -> None:
        key  = self.registry.gesture_key(feat)
        spec = self.store.gestures.get(key)
        desc = spec.describe() if spec else "no gesture assigned"
        self.feedback.speak(
            f"{feat.title}. Current gesture: {desc}. "
            f"Perform the edit gesture to change it."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Launching a feature
    # ═══════════════════════════════════════════════════════════════════════

    def _launch_feature(self, feat, key: str) -> None:
        from features.programmable import ProgrammableGestures

        self._active_feature = feat
        self._active_key     = key

        if isinstance(feat, ProgrammableGestures):
            self.state = State.PROGRAMMABLE
            self.feedback.select()                       # haptic: launched
            self.feedback.speak(
                f"Programmable gestures open. "
                f"Perform the next gesture to scroll through features. "
                f"Perform the edit gesture to change a gesture. "
                f"Perform the programmable gesture again to close."
            )
            self._announce_prog_feature(self.registry.current)
            return

        # Normal feature — blocking run
        self.feedback.select()                           # haptic: launched
        self.feedback.speak(
            f"Opening {feat.title}. "
            f"Perform the same gesture again to close it."
        )
        self._abort = threading.Event()
        self.state  = State.RUNNING
        crashed = False
        try:
            feat.run(FeatureContext(link=self.link,
                                    abort=self._abort,
                                    feedback=self.feedback))
        except Exception as e:
            print(f"[SM] Feature crashed: {e}")
            crashed = True
            self.feedback.error()
            self.feedback.speak(f"{feat.title} encountered an error and closed.")
        finally:
            # Check deactivation BEFORE clearing state: if START was pressed
            # while this feature ran, _deactivate already set state to INACTIVE
            # and spoke its own message — don't override it.
            deactivated = (self.state != State.RUNNING)
            self._abort          = None
            self._active_feature = None
            self._active_key     = ""
            if not deactivated:
                self.state = State.ACTIVE
            self._drain_queue()
            if not deactivated and not crashed:
                self.feedback.confirm()                  # haptic: closed
                self.feedback.speak(
                    f"{feat.title} closed. Glove active. "
                    f"Perform a feature gesture to open it."
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Record-by-example (two-sample match)
    # ═══════════════════════════════════════════════════════════════════════

    def _record_feature_gesture(self, feat, in_programmable: bool = False) -> None:
        title = feat.title
        key   = self.registry.gesture_key(feat)

        self.feedback.speak(
            f"Recording gesture for {title}. "
            f"Hold your hand still, then perform the gesture "
            f"after you feel the buzz."
        )
        sample1 = self._capture_sample()

        if sample1 is None:
            self.feedback.error()                        # haptic: failed
            self.feedback.speak(
                f"No gesture was detected for {title}. "
                f"Please try again."
            )
            if not in_programmable:
                self._announce_enroll(feat)
            else:
                self._announce_prog_feature(feat)
            return

        # Tell the blind user exactly what was captured so they can
        # decide whether to repeat or retry.
        preview = self.recorder.build_spec("preview", sample1)
        self.feedback.confirm()                          # haptic: first sample OK
        self.feedback.speak(
            f"First sample recorded. "
            f"Detected: {preview.describe()}. "
            f"Now perform the exact same gesture again after the buzz."
        )
        sample2 = self._capture_sample()

        if sample2 is None or sample1 != sample2:
            self.feedback.error()                        # haptic: mismatch
            self.feedback.speak(
                f"The two gestures did not match. Nothing was saved for {title}. "
                f"Please try again."
            )
            if not in_programmable:
                self._announce_enroll(feat)
            else:
                self._announce_prog_feature(feat)
            return

        spec = self.recorder.build_spec(key, sample1)
        self.store.set(key, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)

        self.feedback.select()                           # haptic: saved (double buzz)
        self.feedback.speak(
            f"Gesture for {title} saved. "
            f"It is: {spec.describe()}. "
            f"Saved to gestures file."
        )

        if not in_programmable:
            self._advance_enroll()
        else:
            self._announce_prog_feature(feat)

    def _capture_sample(self):
        self.recorder.reset()
        self.feedback.beep()                             # haptic: window opening
        time.sleep(0.4)
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