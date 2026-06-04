"""
state_machine.py — the hub's control loop.

Threads:
  * The GloveController RX thread calls on_frame() for every sensor frame.
    on_frame routes the frame to the recorder (while recording) or to the
    gesture engine (otherwise).
  * The engine calls dispatch() (still on the RX thread) when a gesture fires.
    dispatch() either queues the event for the main loop, or — while a feature
    is running — handles the abort gesture inline.
  * run() is the MAIN thread loop. It consumes events and may block while a
    feature runs or a gesture is being recorded; that's fine because the RX
    thread keeps the stream flowing.

States:  IDLE -> MENU -> FEATURE / EDIT_SELECT
"""

import queue
import threading
import time
from enum import Enum, auto

from features.base import FeatureContext


class State(Enum):
    IDLE = auto()
    MENU = auto()
    FEATURE = auto()
    EDIT_SELECT = auto()


class HubStateMachine:
    def __init__(self, controller, engine, recorder, registry, store, feedback, link,
                 abort_gesture: str = "START", record_window_s: float = 2.5):
        self.controller = controller
        self.engine = engine
        self.recorder = recorder
        self.registry = registry
        self.store = store
        self.feedback = feedback
        self.link = link
        self.abort_gesture = abort_gesture
        self.record_window_s = record_window_s

        self.state = State.IDLE
        self._queue: queue.Queue = queue.Queue()
        self._abort: threading.Event | None = None
        self._shutdown = threading.Event()
        self.recording = False

        self._editable = list(self.store.gestures.keys())   # e.g. ["START","NEXT","EDIT"]
        self._edit_index = 0

    # ── frame routing (RX thread) ─────────────────────────────────────────────
    def on_frame(self, frame) -> None:
        if self.recording:
            self.recorder.feed(frame)
        else:
            self.engine.feed(frame)

    # ── gesture events (RX thread) ────────────────────────────────────────────
    def dispatch(self, name: str) -> None:
        if self.state == State.FEATURE:
            if name == self.abort_gesture and self._abort is not None:
                print("[HUB] Abort gesture — stopping feature.")
                self._abort.set()
            return
        self._queue.put(name)

    # ── main loop (main thread) ───────────────────────────────────────────────
    def run(self) -> None:
        self.feedback.speak("Glove ready. Do the start gesture to begin.")
        while not self._shutdown.is_set():
            try:
                name = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle(name)

    def stop(self) -> None:
        self._shutdown.set()

    def _handle(self, name: str) -> None:
        if self.state == State.IDLE:
            if name == "START":
                self.state = State.MENU
                self.feedback.confirm()
                self._announce_feature()

        elif self.state == State.MENU:
            if name == "NEXT":
                self.registry.next()
                self.feedback.tick()
                self._announce_feature()
            elif name == "START":
                self._launch_current()
            elif name == "EDIT":
                self.state = State.EDIT_SELECT
                self._edit_index = 0
                self.feedback.confirm()
                self.feedback.speak("Edit mode. Scroll to a gesture, start to record, "
                                    "or edit again to cancel.")
                self._announce_edit_target()

        elif self.state == State.EDIT_SELECT:
            if name == "NEXT":
                self._edit_index = (self._edit_index + 1) % len(self._editable)
                self.feedback.tick()
                self._announce_edit_target()
            elif name == "START":
                self._record_gesture(self._editable[self._edit_index])
            elif name == "EDIT":
                self.feedback.speak("Edit cancelled.")
                self.state = State.MENU
                self._announce_feature()

    # ── announcements ─────────────────────────────────────────────────────────
    def _announce_feature(self) -> None:
        self.feedback.speak(f"{self.registry.current().title}. "
                            f"Start to open, next to scroll.")

    def _announce_edit_target(self) -> None:
        name = self._editable[self._edit_index]
        spec = self.store.gestures[name]
        self.feedback.speak(f"{name}. Currently {spec.describe()}.")

    # ── launching a feature ───────────────────────────────────────────────────
    def _launch_current(self) -> None:
        feature = self.registry.current()
        self.feedback.select()
        self.feedback.speak(f"Opening {feature.title}. "
                            f"Hold the start gesture to exit.")
        self._abort = threading.Event()
        self.state = State.FEATURE
        try:
            feature.run(FeatureContext(link=self.link,
                                       abort=self._abort,
                                       feedback=self.feedback))
        except Exception as e:
            print(f"[HUB] Feature crashed: {e}")
            self.feedback.error()
        finally:
            self._abort = None
            self.state = State.MENU
            self._drain_queue()
            self.feedback.speak(f"{feature.title} closed.")
            self._announce_feature()

    # ── record-by-example ─────────────────────────────────────────────────────
    def _record_gesture(self, name: str) -> None:
        self.feedback.speak(f"Recording {name}. Perform the gesture after the buzz.")
        sample1 = self._capture_sample()
        if sample1 is None:
            self.feedback.error()
            self.feedback.speak("No gesture detected. Cancelled.")
            return

        self.feedback.speak("Now do it again to confirm, after the buzz.")
        sample2 = self._capture_sample()
        if sample2 is None or sample1 != sample2:
            self.feedback.error()
            self.feedback.speak("The two tries did not match. Nothing changed.")
            return

        spec = self.recorder.build_spec(name, sample1)
        self.store.set(name, spec)
        self.store.save()
        self.engine.set_gestures(self.store.gestures)
        self.feedback.confirm()
        self.feedback.speak(f"{name} saved. Now {spec.describe()}.")
        self.state = State.MENU
        self._announce_feature()

    def _capture_sample(self):
        self.recorder.reset()
        self.feedback.beep()
        time.sleep(0.4)
        self.recording = True
        time.sleep(self.record_window_s)
        self.recording = False
        return self.recorder.analyze()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
