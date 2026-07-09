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

from features.base import FeatureContext, MIC_OWNING_FEATURE_NAMES

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

        # Set by hub.py right after construction (same imperative-wiring
        # pattern as engine.on_event/controller.on_frame). Paused around
        # mic-owning features in _launch_feature() so the global
        # VoiceListener's sr.Microphone() stream never fights a feature's own
        # sounddevice.InputStream for the same physical mic.
        self.voice = None

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

        # Used by _wait_prog_confirm() to receive voice PROG_CONFIRM/PROG_DISCARD
        # from the VoiceListener thread while the main thread blocks on confirmation.
        self._prog_confirm_q    = queue.Queue()
        self._prog_confirm_mode = False

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

    def run(self, program: bool = False) -> None:
        if program:
            # hub --program: jump straight into the Programmable panel so the
            # user can register/re-bind every feature's gesture (including Home
            # and each relay) without first doing START.
            self.enter_programmable_direct()
        else:
            start_spec = self.store.gestures.get("START")
            desc = f" — {start_spec.describe()}" if start_spec else ""
            self.feedback.speak(f"Ready.{desc} to activate.")
        while not self._shutdown.is_set():
            try:
                name = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle(name)

    def enter_programmable_direct(self) -> None:
        """Open the Programmable panel directly (used by hub --program). Scroll
        with NEXT, record with EDIT (perform the gesture, then do it twice to
        confirm). Close with the Programmable gesture or START."""
        from features.programmable import ProgrammableGestures
        prog = next((f for f in self.registry.features
                     if isinstance(f, ProgrammableGestures)), None)
        self._active_feature = prog
        self._active_key     = self.registry.gesture_key(prog) if prog else ""
        self.state           = State.PROGRAMMABLE
        self.feedback.select()
        self.feedback.speak(
            "Programming mode. Flick next to scroll features, hold edit to "
            "record a gesture, then do the new gesture twice to confirm."
        )
        self._announce_prog_feature(self.registry.current, opening=True)

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
            # Build a per-feature gesture summary
            parts = []
            for f in self.registry.features:
                key  = self.registry.gesture_key(f)
                spec = self.store.gestures.get(key)
                desc = spec.describe() if spec else "no gesture assigned"
                parts.append(f"{f.title}: {desc}")
            features_desc = ". ".join(parts)
            self.feedback.speak(
                f"Glove on. {features_desc}. "
                "Say a feature name to ask about its gesture. "
                "Say stop to turn off."
            )

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
            self._drain_queue()   # discard EDIT bounces that queued while recording
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

    # System gesture names that are only valid inside a running feature —
    # silently ignore them in all other states.
    _FEATURE_ONLY_GESTURES = frozenset({"OCR_PAUSE", "OCR_FWD", "OCR_BWD"})

    def _handle_active(self, name: str) -> None:
        # OCR-only and nav-only gestures have no meaning while idle
        if name in self._FEATURE_ONLY_GESTURES or name in ("NEXT", "EDIT"):
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
            self._drain_queue()   # discard EDIT bounces that queued while recording
            return

        # Silently ignore any gesture that doesn't belong in programmable mode.

    def _announce_prog_feature(self, feat, opening: bool = False) -> None:
        key  = self.registry.gesture_key(feat)
        spec = self.store.gestures.get(key)
        if spec:
            desc    = spec.describe()
            detail  = f"{feat.title}: {desc}. Edit to change."
        else:
            detail  = f"{feat.title}: not recorded yet. Edit to record."
        if opening:
            self.feedback.speak(
                f"Programmable. {detail} Next to scroll. Same gesture to close."
            )
        else:
            self.feedback.speak(detail)

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
            # Drain any bounce gestures from the opening gesture before PROGRAMMABLE
            # state starts processing — otherwise rapid NEXT fires skip features.
            self._drain_queue()
            self._announce_prog_feature(self.registry.current, opening=True)
            return

        self.feedback.select()
        self.feedback.speak(f"Opening {feat.title}. Same gesture to close.")
        self._abort = threading.Event()
        self.state  = State.RUNNING
        self._feature_gesture_q = queue.Queue()
        crashed = False
        # Mic-owning features (money/ocr/env/home/lidar family) open their own
        # microphone stream — pause the global voice listener for the duration
        # so the two never fight over the same physical mic (see
        # features.base.MIC_OWNING_FEATURE_NAMES for why).
        pause_voice = self.voice is not None and feat.name in MIC_OWNING_FEATURE_NAMES
        if pause_voice:
            self.voice.stop()
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
            if pause_voice:
                self.voice.start()
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

        if in_programmable:
            self._record_prog_gesture(feat, key, title)
        else:
            self._record_enroll_gesture(feat, key, title)

    def _record_enroll_gesture(self, feat, key: str, title: str) -> None:
        """Enrollment flow: instruction → 5 s delay → sample1 → preview → sample2."""
        self.feedback.speak(
            f"Recording {title}. "
            "Release your hand completely. "
            "Perform the gesture after the buzz in 5 seconds."
        )
        self.feedback.wait(timeout=12)
        time.sleep(5.0)   # let the user release the EDIT hand and prepare
        sample1 = self._capture_sample()

        if sample1 is None:
            self.feedback.error()
            self.feedback.speak("Nothing detected. Try again.")
            self._announce_enroll(feat)
            return

        preview = self.recorder.build_spec("preview", sample1)
        self.feedback.confirm()
        self.feedback.speak(f"Got it: {preview.describe()}. Same gesture after buzz.")
        self.feedback.wait(timeout=10)
        sample2 = self._capture_sample()

        if sample2 is None or sample1 != sample2:
            self.feedback.error()
            self.feedback.speak("No match. Try again.")
            self._announce_enroll(feat)
            return

        spec = self.recorder.build_spec(key, sample1)
        self.store.set(key, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)
        self.feedback.select()
        self.feedback.speak(f"Saved for {title}: {spec.describe()}.")
        self._advance_enroll()

    def _record_prog_gesture(self, feat, key: str, title: str) -> None:
        """Programmable flow:
        timer → capture → announce → voice-or-gesture×2 confirm → save → next feature.
        """
        # Timer: tell the user to prepare, then a 5 s silence before buzz
        self.feedback.speak(
            f"Recording {title}. "
            "Release your hand completely. "
            "Perform the gesture after the buzz in 5 seconds."
        )
        self.feedback.wait(timeout=12)
        time.sleep(5.0)   # let the user release the EDIT hand and prepare

        sample1 = self._capture_sample()

        if sample1 is None:
            self.feedback.error()
            self.feedback.speak("Nothing detected. Edit to try again.")
            self._announce_prog_feature(feat)
            return

        preview = self.recorder.build_spec("preview", sample1)
        desc    = preview.describe()

        self.feedback.confirm()
        self.feedback.speak(
            f"Got: {desc}. "
            "Say confirm to save, discard to cancel. "
            "Or do the move twice after the buzzes."
        )
        self.feedback.wait(timeout=12)

        confirmed = self._wait_prog_confirm(sample1)

        if not confirmed:
            self.feedback.speak("Discarded. Edit to try again.")
            self._announce_prog_feature(feat)
            return

        spec = self.recorder.build_spec(key, sample1)
        self.store.set(key, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)

        self.feedback.select()
        self.feedback.speak(f"Saved: {spec.describe()}.")
        # Auto-advance to next feature after saving
        self.registry.advance()
        self._announce_prog_feature(self.registry.current)

    def _wait_prog_confirm(self, sample1) -> bool:
        """Block until the user confirms (voice or gesture×2) or discards.

        Voice path  — say 'confirm'/'yes'/'save' or 'discard'/'cancel'/'no'.
        Gesture path — do the same move twice in a row, once per buzz.
        Returns True if confirmed, False if discarded/timed-out.
        """
        self._prog_confirm_mode = True
        # Drain any stale voice responses
        try:
            while True: self._prog_confirm_q.get_nowait()
        except queue.Empty:
            pass

        try:
            deadline  = time.time() + 30
            first_try = True

            while time.time() < deadline:
                # Non-blocking voice check (user may have replied during the speech)
                try:
                    resp = self._prog_confirm_q.get_nowait()
                    return resp == "PROG_CONFIRM"
                except queue.Empty:
                    pass

                if not first_try:
                    self.feedback.speak(
                        "Do it twice, or say confirm or discard."
                    )
                    self.feedback.wait(timeout=8)
                    # Voice check after re-prompt
                    try:
                        resp = self._prog_confirm_q.get_nowait()
                        return resp == "PROG_CONFIRM"
                    except queue.Empty:
                        pass

                first_try = False

                # ── First repetition ─────────────────────────────────────
                s2 = self._capture_sample()

                try:
                    resp = self._prog_confirm_q.get_nowait()
                    return resp == "PROG_CONFIRM"
                except queue.Empty:
                    pass

                if s2 is None or s2 != sample1:
                    self.feedback.error()
                    self.feedback.speak("Different move.")
                    self.feedback.wait(timeout=3)
                    continue

                # ── Second repetition ────────────────────────────────────
                self.feedback.confirm()
                self.feedback.speak("One more.")
                self.feedback.wait(timeout=4)

                try:
                    resp = self._prog_confirm_q.get_nowait()
                    return resp == "PROG_CONFIRM"
                except queue.Empty:
                    pass

                s3 = self._capture_sample()

                try:
                    resp = self._prog_confirm_q.get_nowait()
                    return resp == "PROG_CONFIRM"
                except queue.Empty:
                    pass

                if s3 is not None and s3 == sample1:
                    return True

                self.feedback.error()
                self.feedback.speak("Second didn't match.")
                self.feedback.wait(timeout=3)

            # Timed out (30 s)
            self.feedback.speak("Timed out. Edit to try again.")
            return False

        finally:
            self._prog_confirm_mode = False

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
        PROG_CONFIRM/PROG_DISCARD go to the confirmation queue when active;
        all other voice commands are injected as gestures via dispatch()."""
        if cmd.startswith("QUERY:"):
            self._handle_query(cmd[6:])
            return
        if cmd in ("PROG_CONFIRM", "PROG_DISCARD") and self._prog_confirm_mode:
            self._prog_confirm_q.put(cmd)
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
