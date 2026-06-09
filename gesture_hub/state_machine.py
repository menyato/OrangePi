"""
state_machine.py — hub control loop.

TTS contract (blind users):
  Every action is announced. speech() is non-blocking; gestures always
  take priority and cut the current announcement short. The new speech
  then explains what happened and what to do next.
  Before recording windows silence() is called so the user hears only
  the haptic "go" buzz with no competing audio.

States
------
  INACTIVE     — glove off; only START is ever acted on.
  ENROLL       — walking unassigned features; NEXT skips, EDIT records.
  ACTIVE       — listening for feature toggle gestures.
  RUNNING      — one feature executing.
  PROGRAMMABLE — gesture-management panel open.
"""

import queue
import re
import threading
import time
from enum import Enum, auto

from features.base import FeatureContext

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

        self._last_start_ts: float = 0.0

        # Queue for forwarding gestures to a running feature (e.g. OCR reader).
        # Set in _launch_feature(), cleared in finally.
        self._feature_gesture_q = None

    # ═══════════════════════════════════════════════════════════════════════
    # Frame routing  (RX thread)
    # ═══════════════════════════════════════════════════════════════════════

    def on_frame(self, frame) -> None:
        if self.recording:
            self.recorder.feed(frame)
        else:
            self.engine.feed(frame)

    # ═══════════════════════════════════════════════════════════════════════
    # Gesture dispatch  (RX thread → queue)
    # ═══════════════════════════════════════════════════════════════════════

    def dispatch(self, name: str) -> None:
        if name == "START":
            if time.time() - self._last_start_ts < START_LOCKOUT_S:
                print("[SM] START ignored — lockout.")
                return
            self._queue.put("START")
            return

        if self.state == State.INACTIVE:
            return

        if self.state == State.RUNNING:
            if name == self._active_key and self._abort is not None:
                self._abort.set()
            elif self._feature_gesture_q is not None:
                # Forward all other gestures to the running feature's queue
                self._feature_gesture_q.put(name)
            return

        self._queue.put(name)

    # ═══════════════════════════════════════════════════════════════════════
    # Main loop  (main thread)
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        start_spec = self.store.gestures.get("START")
        desc = f" — {start_spec.describe()}" if start_spec else ""
        self.feedback.speak(f"Ready.{desc} to activate.")
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
        self.feedback.confirm()

        missing = self.registry.unassigned(self.store)
        if missing:
            self.state = State.ENROLL
            n     = len(missing)
            names = ", ".join(f.title for f in missing)
            self.feedback.speak(
                f"Active. {n} feature{'s' if n > 1 else ''} need gestures: {names}."
            )
            feat = self.registry.start_enrollment(self.store)
            self._announce_enroll(feat)
        else:
            self.state = State.ACTIVE
            feat_list  = ", ".join(f.title for f in self.registry.features)
            self.feedback.speak(f"Active. Features: {feat_list}. Gesture to open.")

    def _deactivate(self) -> None:
        self._last_start_ts = time.time()
        prev_state   = self.state
        prev_feature = self._active_feature

        if self._abort:
            self._abort.set()

        self.state           = State.INACTIVE
        self._active_feature = None
        self._active_key     = ""
        self._drain_queue()

        self.feedback.error()
        if prev_state == State.RUNNING and prev_feature:
            self.feedback.speak(f"{prev_feature.title} stopped. Glove off.")
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
            self.feedback.speak("All set. Glove active.")
            return

        if name == "NEXT":
            nxt = self.registry.next_enrollment(self.store)
            if nxt is None:
                self.state = State.ACTIVE
                self.feedback.speak("All set. Glove active.")
            else:
                self.feedback.tick()
                self._announce_enroll(nxt)
            return

        if name == "EDIT":
            self.feedback.tick()
            self._record_feature_gesture(feat)
            return

        # Already-assigned feature gesture — activate it, then resume enrollment
        for f in self.registry.features:
            key = self.registry.gesture_key(f)
            if key == name and key in self.store.gestures:
                self._launch_feature(f, key)
                if self.state == State.ACTIVE and self.registry.unassigned(self.store):
                    self.state = State.ENROLL
                    cur = self.registry.current_unassigned
                    if cur:
                        self._announce_enroll(cur)
                return

        # Unknown gesture — re-announce current feature
        self._announce_enroll(feat)

    def _announce_enroll(self, feat) -> None:
        key = self.registry.gesture_key(feat)
        if key in self.store.gestures:
            self._advance_enroll()
            return
        remaining = len(self.registry.unassigned(self.store))

        edit_spec = self.store.gestures.get("EDIT")
        next_spec  = self.store.gestures.get("NEXT")
        edit_desc  = f" — {edit_spec.describe()}" if edit_spec else ""
        next_desc  = f" — {next_spec.describe()}" if next_spec else ""

        self.feedback.speak(
            f"Enrolling: {feat.title}. "
            f"Edit{edit_desc} to record. "
            f"Next{next_desc} to skip. "
            f"{remaining} left."
        )

    def _advance_enroll(self) -> None:
        nxt = self.registry.next_enrollment(self.store)
        if nxt is None:
            self.state = State.ACTIVE
            self.feedback.confirm()
            feat_list  = ", ".join(f.title for f in self.registry.features)
            self.feedback.speak(f"All set. Features: {feat_list}.")
        else:
            self._announce_enroll(nxt)

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVE state
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_active(self, name: str) -> None:
        if name == "NEXT":
            self.feedback.speak("Next scrolls programmable panel. Open it first.")
            return

        if name == "EDIT":
            self.feedback.speak("Edit is for programmable panel. Open it first.")
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
            self.state           = State.ACTIVE
            self._active_feature = None
            self._active_key     = ""
            self.feedback.confirm()
            self.feedback.speak("Programmable closed. Gesture to open a feature.")
            return

        if name == "NEXT":
            self.registry.advance()
            self.feedback.tick()
            self._announce_prog_feature(self.registry.current)
            return

        if name == "EDIT":
            feat = self.registry.current
            self.feedback.tick()
            self.feedback.speak(f"Editing {feat.title}.")
            self._record_feature_gesture(feat, in_programmable=True)
            return

        self.feedback.speak("Close programmable panel first to use features.")

    def _announce_prog_feature(self, feat) -> None:
        key  = self.registry.gesture_key(feat)
        spec = self.store.gestures.get(key)
        desc = spec.describe() if spec else "no gesture assigned"
        self.feedback.speak(f"{feat.title}: {desc}. Edit to change.")

    # ═══════════════════════════════════════════════════════════════════════
    # Launching a feature
    # ═══════════════════════════════════════════════════════════════════════

    def _launch_feature(self, feat, key: str) -> None:
        from features.programmable import ProgrammableGestures

        self._active_feature = feat
        self._active_key     = key

        if isinstance(feat, ProgrammableGestures):
            self.state = State.PROGRAMMABLE
            self.feedback.select()
            self.feedback.speak(
                "Programmable open. Next scrolls, edit changes, same gesture closes."
            )
            self._announce_prog_feature(self.registry.current)
            return

        self.feedback.select()
        self.feedback.speak(f"Opening {feat.title}. Same gesture to close.")
        self._abort = threading.Event()
        self.state  = State.RUNNING
        self._feature_gesture_q = queue.Queue()
        crashed = False
        try:
            feat.run(FeatureContext(link=self.link,
                                    abort=self._abort,
                                    feedback=self.feedback,
                                    gesture_queue=self._feature_gesture_q))
        except Exception as e:
            print(f"[SM] Feature crashed: {e}")
            crashed = True
            self.feedback.error()
            self.feedback.speak(f"{feat.title} error. Closed.")
        finally:
            deactivated              = (self.state != State.RUNNING)
            self._abort              = None
            self._active_feature     = None
            self._active_key         = ""
            self._feature_gesture_q  = None
            if not deactivated:
                self.state = State.ACTIVE
            self._drain_queue()
            if not deactivated and not crashed:
                self.feedback.confirm()
                self.feedback.speak(f"{feat.title} closed. Gesture to open.")

    # ═══════════════════════════════════════════════════════════════════════
    # Record-by-example (two-sample match)
    # ═══════════════════════════════════════════════════════════════════════

    def _record_feature_gesture(self, feat, in_programmable: bool = False) -> None:
        title = feat.title
        key   = self.registry.gesture_key(feat)

        # Instruction plays non-blocking; sleep gives it time to finish
        # before the haptic "go" buzz fires.
        self.feedback.speak(f"Recording {title}. Hold still, gesture after buzz.")
        time.sleep(3.5)
        sample1 = self._capture_sample()

        if sample1 is None:
            self.feedback.error()
            self.feedback.speak("Nothing detected. Try again.")
            if not in_programmable:
                self._announce_enroll(feat)
            else:
                self._announce_prog_feature(feat)
            return

        preview = self.recorder.build_spec("preview", sample1)
        self.feedback.confirm()
        self.feedback.speak(
            f"Got it: {preview.describe()}. Same gesture after buzz."
        )
        time.sleep(3.0)
        sample2 = self._capture_sample()

        if sample2 is None or sample1 != sample2:
            self.feedback.error()
            self.feedback.speak("No match. Try again.")
            if not in_programmable:
                self._announce_enroll(feat)
            else:
                self._announce_prog_feature(feat)
            return

        spec = self.recorder.build_spec(key, sample1)
        self.store.set(key, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)

        self.feedback.select()
        self.feedback.speak(f"Saved for {title}: {spec.describe()}.")

        if not in_programmable:
            self._advance_enroll()
        else:
            self._announce_prog_feature(feat)

    def _capture_sample(self):
        self.recorder.reset()
        self.feedback.silence()  # clear any ongoing TTS before the haptic "go"
        self.feedback.beep()
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

    # ═══════════════════════════════════════════════════════════════════════
    # Voice commands  (VoiceListener background thread)
    # ═══════════════════════════════════════════════════════════════════════

    def on_voice(self, cmd: str) -> None:
        """Called from VoiceListener thread. Queries answered immediately;
        all other voice commands are injected as gestures via dispatch()."""
        if cmd.startswith("QUERY:"):
            self._handle_query(cmd[6:])
            return
        self.dispatch(cmd)

    def _handle_query(self, text: str) -> None:
        """Speak the gesture assigned to whichever feature best matches text.

        Example utterances:
          "what is the gesture for money"
          "what is the move for book reader"
          "how do I use programmable"
        """
        words = set(re.findall(r"[a-z]+", text.lower()))
        # Score each feature by title-word overlap with the spoken query
        best_feat, best_score = None, 0
        for feat in self.registry.features:
            title_words = set(re.findall(r"[a-z]+", feat.title.lower()))
            score = len(title_words & words)
            if score > best_score:
                best_feat, best_score = feat, score

        if best_feat is None or best_score == 0:
            names = ", ".join(f.title for f in self.registry.features)
            self.feedback.speak(f"Available features: {names}.")
            return

        key  = self.registry.gesture_key(best_feat)
        spec = self.store.gestures.get(key)
        if spec is None:
            self.feedback.speak(f"{best_feat.title} has no gesture assigned yet.")
        else:
            self.feedback.speak(f"{best_feat.title}: {spec.describe()}.")
