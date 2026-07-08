#!/usr/bin/env python3
"""
hub.py — OrangePi gesture hub (new flow).

Boot sequence
─────────────
1. Load GestureStore (gestures.json) and CalStore (calibration.json).
2. Open GloveController (UART to ATmega).
3. Restore saved calibration if present.
4. Start HubStateMachine — waits for START gesture.

Gesture flow (see state_machine.py for full detail)
────────────────────────────────────────────────────
  START (hardcoded) — global on/off toggle for the whole glove.

  First boot / missing gestures:
    After START the hub walks through any feature that has no gesture yet.
    The user performs the gesture twice; on match it is saved and the walk
    moves to the next unassigned feature.  NEXT skips to the next one.

  Normal operation:
    Each feature gesture is a toggle — fire once to open, fire again to close.
    While a feature is running, only its own toggle (to close) is heard.

  Programmable Gestures feature:
    Opens a panel where NEXT scrolls all features and EDIT re-binds gestures.

Hardcoded gestures (never user-assignable):
  START  Thumb+Pinky  tilt_right     STATIC  flex=0x11 imu=0x01
  NEXT   Thumb+Ring   tilt_backward  FLICK   flex=0x09 imu=0x08
  EDIT   Thumb+Middle tilt_forward   FLICK   flex=0x05 imu=0x04

Modes:
  python3 hub.py                 normal run
  python3 hub.py --monitor       live ATmega feed + gesture match view
  python3 hub.py --calibrate     run per-hand calibration, save, then run
  python3 hub.py --threshold 35  override bend threshold %, then run

Feature bypass (no gesture needed — for testing only):
  python3 hub.py --money         launch Money Recognition directly
  python3 hub.py --ocr           launch OCR Reader directly
  python3 hub.py --env           launch Environment Awareness directly
  python3 hub.py --home          launch Home Automation directly
  python3 hub.py --lidar         launch Lidar Navigation directly

Gesture management:
  python3 hub.py --reset-gestures   clear all user-assigned feature gestures
                                    (forces re-enrollment on next boot)
"""

import argparse
import os
import queue
import sys
import threading
import time

from glove_controller import GloveController
import orangepi_client as mc

from gesture_hub.store         import GestureStore
from gesture_hub.cal_store     import CalStore
from gesture_hub.engine        import GestureEngine
from gesture_hub.recorder      import GestureRecorder
from gesture_hub.registry      import FeatureRegistry
from gesture_hub.feedback      import Feedback
from gesture_hub.state_machine import HubStateMachine
from gesture_hub.voice         import VoiceListener
from gesture_hub               import diagnostics
from net.client                import ServerLink

from features.money_recognition import MoneyRecognition
from features.ocr_reader        import OCRReader
from features.env_awareness     import EnvAwareness
from features.home_automation   import HomeAutomation, RelaySwitch
from features.programmable      import ProgrammableGestures
from features.lidar_nav         import (LidarNavigation, LidarObstacleTest,
                                        LidarMappingTest, LidarNavigateTest)
from features.base              import FeatureContext, MIC_OWNING_FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GESTURES_PATH = os.path.join(HERE, "gestures.json")
DEFAULT_CAL_PATH      = os.path.join(HERE, "calibration.json")


def _probe_devices(feedback, link) -> None:
    """
    Boot-time device check for a blind user.
    Tests camera, microphone, and server connection, then speaks the results.
    The speaker test is implicit: if the user hears anything, audio is working.
    """
    results = []

    # Server
    server_ok = link.sock is not None
    results.append("Server " + ("connected" if server_ok else "not reachable, will retry when needed"))

    # Camera
    camera_ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        camera_ok = cap.isOpened()
        cap.release()
    except Exception:
        pass
    results.append("Camera " + ("ready" if camera_ok else "not found"))

    # Microphone
    mic_ok = False
    mic_name = ""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        for d in devs:
            if d["max_input_channels"] > 0:
                mic_ok  = True
                mic_name = d["name"].split("(")[0].strip()
                break
    except Exception:
        pass
    results.append("Microphone " + (f"ready, {mic_name}" if mic_ok else "not found"))

    feedback.speak("Startup: " + ", ".join(results) + ". Speaker working.")


def main() -> None:
    ap = argparse.ArgumentParser(description="OrangePi gesture hub")

    # ── connectivity ──────────────────────────────────────────────────────────
    ap.add_argument("--host",      default="10.254.249.159", help="Server IP")
    ap.add_argument("--port",      default=9000, type=int,   help="Server port")
    ap.add_argument("--uart",      default="/dev/ttyS5",     help="ATmega UART device")
    ap.add_argument("--baud",      default=115200, type=int, help="UART baud rate")
    ap.add_argument("--alsa",      default=None,
                    help="ALSA device for hub speech, e.g. plughw:0,0")
    ap.add_argument("--mic",       default=None, type=int,
                    help="Force sounddevice input index for feature voice "
                         "(money/env/home), instead of auto-detect. See: "
                         "python3 -c \"import sounddevice as sd; print(sd.query_devices())\"")

    # ── paths ─────────────────────────────────────────────────────────────────
    ap.add_argument("--gestures",  default=DEFAULT_GESTURES_PATH,
                    help="Path to gestures.json")
    ap.add_argument("--cal",       default=DEFAULT_CAL_PATH,
                    help="Path to calibration.json")

    # ── standard modes ────────────────────────────────────────────────────────
    ap.add_argument("--monitor",   action="store_true",
                    help="Show live ATmega feed + gesture matches, then exit.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run per-hand flex calibration, save it, then run normally.")
    ap.add_argument("--threshold", type=int, default=None,
                    help="Override firmware bend threshold percent (10-90).")
    ap.add_argument("--no-voice",  action="store_true",
                    help="Disable voice commands (useful if no microphone).")

    # ── feature bypass (testing — skips gesture recognition entirely) ─────────
    feat_grp = ap.add_mutually_exclusive_group()
    feat_grp.add_argument("--money",     action="store_true",
                           help="Launch Money Recognition directly.")
    feat_grp.add_argument("--ocr",       action="store_true",
                           help="Launch OCR Reader directly.")
    feat_grp.add_argument("--env",       action="store_true",
                           help="Launch Environment Awareness directly.")
    feat_grp.add_argument("--home",      action="store_true",
                           help="Launch Home Automation directly.")
    feat_grp.add_argument("--relay1",    action="store_true",
                           help="Fire Relay 1 toggle directly (no gesture needed).")
    feat_grp.add_argument("--relay2",    action="store_true",
                           help="Fire Relay 2 toggle directly (no gesture needed).")
    feat_grp.add_argument("--lidar",     action="store_true",
                           help="Full LiDAR: mapping + voice save + navigation.")
    feat_grp.add_argument("--obstacles", action="store_true",
                           help="LiDAR obstacle-detection test only (no SLAM).")
    feat_grp.add_argument("--mapping",   action="store_true",
                           help="LiDAR mapping test: build map, live server view, save.")
    feat_grp.add_argument("--navigate",  action="store_true",
                           help="LiDAR navigation test: navigate to a saved room.")

    # ── lidar-specific ────────────────────────────────────────────────────────
    ap.add_argument("--lidar-port", default="auto",
                    help="Serial port for the MS200 LiDAR. "
                         "Default 'auto' scans /dev/ttyUSB* and /dev/ttyACM*.")
    ap.add_argument("--navigate-room", default="",
                    help="Room name to navigate to (used with --navigate).")

    # ── gesture management ────────────────────────────────────────────────────
    ap.add_argument("--reset-gestures", action="store_true",
                    help="Clear all user-assigned feature gestures and exit. "
                         "Forces full re-enrollment on the next boot.")

    # ── book reader shelf ─────────────────────────────────────────────────────
    ap.add_argument("--sample-books", action="store_true",
                    help="Install the Book Reader practice books (garden, "
                         "ocean, mountain) before starting.")
    ap.add_argument("--fresh-books", action="store_true",
                    help="Delete ALL saved books, then install the practice "
                         "books, before starting.")

    args = ap.parse_args()

    if args.sample_books or args.fresh_books:
        import glob as _glob
        from features.ocr_reader import SESSIONS_DIR, _install_sample_books
        if args.fresh_books and os.path.isdir(SESSIONS_DIR):
            for _p in _glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
                os.remove(_p)
                print(f"[HUB] deleted saved book {_p}")
        _install_sample_books()

    if args.mic is not None:
        mc.set_mic_override(args.mic)

    # ── Connect early: needed to ask the server how many relays are actually
    #    paired, so we build exactly that many gesture-bindable RelaySwitch
    #    features below (instead of a hardcoded count). This same link is
    #    reused for the rest of main() — no second ServerLink is created.
    link = ServerLink(args.host, args.port)
    link.connect()
    relay_count = 2   # fallback if unreachable / nothing paired yet
    resp = link.send("home", {"action": "get_relay_count"})
    if resp and "relay_count" in resp:
        relay_count = int(resp["relay_count"])
    relay_features = [RelaySwitch(i, f"Relay {i}") for i in range(1, relay_count + 1)]
    print(f"[HUB] Home automation: {relay_count} relay(s) — "
          f"{', '.join(f.title for f in relay_features)}")

    # ── Feature list (built here so --lidar-port is available) ───────────────
    lidar_feat = LidarNavigation()
    lidar_feat.port = args.lidar_port

    obs_feat = LidarObstacleTest()
    obs_feat.port = args.lidar_port

    map_feat = LidarMappingTest()
    map_feat.port = args.lidar_port

    nav_feat = LidarNavigateTest()
    nav_feat.port      = args.lidar_port
    nav_feat.room_name = args.navigate_room

    # Gesture-menu features (what the user scrolls/assigns in the pipeline).
    # The unified Lidar feature asks obstacle / mapping / navigation at startup,
    # so its three test-only siblings are NOT listed here — they stay reachable
    # only via the --obstacles / --mapping / --navigate bypass flags.
    FEATURES = [
        MoneyRecognition(),
        OCRReader(),
        EnvAwareness(),
        HomeAutomation(),
        *relay_features,
        lidar_feat,
        ProgrammableGestures(),  # always last
    ]
    # Extra features reachable only by bypass flag, never in the gesture menu.
    BYPASS_ONLY_FEATURES = [obs_feat, map_feat, nav_feat]

    # ── Load gesture store + calibration store ────────────────────────────────
    store     = GestureStore(args.gestures)
    cal_store = CalStore(args.cal)

    # Print current gesture table
    print("[HUB] Loaded gestures:")
    for name, spec in store.gestures.items():
        print(f"  {name:30s}: {spec.describe()}"
              f"  (flex=0x{spec.flex_mask:02X} imu=0x{spec.imu_mask:02X}"
              f" {spec.motion.value})")

    # ── --reset-gestures: wipe FEAT:* keys, save, exit ───────────────────────
    if args.reset_gestures:
        feat_keys = [k for k in store.gestures if k.startswith("FEAT:")]
        for k in feat_keys:
            del store.gestures[k]
        store.save()
        print(f"[HUB] Reset {len(feat_keys)} user gesture(s): {feat_keys}")
        print("[HUB] Re-enrollment will run on next boot. Exiting.")
        return

    # ── Open glove ────────────────────────────────────────────────────────────
    controller = GloveController(port=args.uart, baud=args.baud)
    feedback   = Feedback(controller, alsa=args.alsa)

    if not controller.start(wait_ready=False):
        print(f"[HUB] Cannot open glove on {args.uart}. Check wiring.")
        sys.exit(1)
    time.sleep(0.5)   # let first frames arrive

    # ── MONITOR mode ──────────────────────────────────────────────────────────
    if args.monitor:
        if cal_store.has_calibration():
            print("[HUB] Restoring saved calibration before monitor...")
            cal_store.restore(controller)
            time.sleep(0.3)
        try:
            diagnostics.run_monitor(controller, store.gestures,
                                    say=feedback.speak)
        finally:
            controller.stop()
        return

    # ── Threshold override ────────────────────────────────────────────────────
    if args.threshold is not None:
        try:
            controller.set_threshold(args.threshold)
            print(f"[HUB] Bend threshold set to {args.threshold}%")
            time.sleep(0.3)
        except ValueError as e:
            print(f"[HUB] Threshold error: {e}")

    # ── Calibration ───────────────────────────────────────────────────────────
    # Always save after a successful calibration so the next boot auto-restores.
    if args.calibrate:
        result = diagnostics.run_calibration(controller, say=feedback.speak)
        if result:
            cal_store.save(
                rest=result["rest"],
                bent=result["bent"],
                thresh_pct=args.threshold or cal_store.thresh_pct,
            )
            feedback.speak("Calibration saved.")
            # Restore immediately so this session also uses the new values
            cal_store.restore(controller)
            time.sleep(0.3)
        else:
            print("[HUB] Calibration aborted.")
            if cal_store.has_calibration():
                print("[HUB] Restoring previous calibration...")
                cal_store.restore(controller)
                time.sleep(0.3)
            else:
                print("[HUB] No saved calibration available.")
    elif cal_store.has_calibration():
        print("[HUB] Restoring saved calibration...")
        cal_store.restore(controller)
        time.sleep(0.3)
    else:
        print("[HUB] No saved calibration. Run with --calibrate to set one.")

    # ── Feature bypass (--money / --ocr / --env / --lidar) ───────────────────
    # Bypasses the state machine entirely; runs the feature directly.
    # Voice "stop" (→ START) sets the abort event to exit cleanly.
    bypass_name = (
        "money"     if args.money     else
        "ocr"       if args.ocr       else
        "env"       if args.env       else
        "home"      if args.home      else
        "relay1"    if args.relay1    else
        "relay2"    if args.relay2    else
        "lidar"     if args.lidar     else
        "obstacles" if args.obstacles else
        "mapping"   if args.mapping   else
        "navigate"  if args.navigate  else
        None
    )
    if bypass_name:
        feat_map = {f.name: f for f in FEATURES + BYPASS_ONLY_FEATURES}
        feat = feat_map.get(bypass_name)
        if feat is None:
            print(f"[HUB] Feature {bypass_name!r} not in registry.")
            controller.stop()
            return

        abort         = threading.Event()
        gesture_queue = queue.Queue()
        # link is already connected — created early, before FEATURES, so we
        # could ask the server how many relays are paired (see above).

        # In the normal (non-bypass) flow, HubStateMachine.dispatch() forwards
        # every recognized gesture/voice token into the running feature's own
        # gesture_queue (see gesture_hub/state_machine.py) — that's how OCR's
        # pause/skip/scan controls and Home's relay toggles work in practice.
        # The bypass path used to skip all of this (gesture_queue=None, no
        # GestureEngine wired to the glove at all), so a feature launched via
        # --money/--ocr/etc. only ever heard its own internal voice loop and
        # never got hand-gesture or global-voice commands. Replicate the same
        # forwarding here so bypass mode behaves like the real thing.
        active_key = FeatureRegistry.gesture_key(feat)

        def _forward(name: str) -> None:
            if name in ("START", active_key):
                abort.set()
                return
            gesture_queue.put(name)

        engine = GestureEngine(store.gestures, on_event=_forward)
        controller.on_frame = engine.feed

        def _voice_abort(cmd: str) -> None:
            if cmd in ("START", "OCR_CLOSE"):
                abort.set()
            _forward(cmd)

        voice = VoiceListener(on_command=_voice_abort)
        # Mic-owning features (money/ocr/env/home/lidar family) manage their
        # own speech listener — skip hub's global one entirely in bypass mode
        # to avoid two simultaneous mic consumers (see
        # features.base.MIC_OWNING_FEATURE_NAMES).
        if not args.no_voice and bypass_name not in MIC_OWNING_FEATURE_NAMES:
            voice.start()

        feedback.speak(
            f"Test mode: {feat.title}. Say stop or press Ctrl-C to quit."
        )
        try:
            feat.run(FeatureContext(
                link          = link,
                abort         = abort,
                feedback      = feedback,
                gesture_queue = gesture_queue,
            ))
        except KeyboardInterrupt:
            pass
        finally:
            voice.stop()
            controller.stop()
            link.close()
        return

    # ── Normal run — build engine + state machine ─────────────────────────────
    # The engine always knows the full store (system gestures + feature gestures
    # that have already been assigned).  The state machine decides which fired
    # gesture names to act on based on the current state.
    engine   = GestureEngine(store.gestures, on_event=None)
    recorder = GestureRecorder()
    registry = FeatureRegistry(FEATURES)
    # link is already connected — created early, before FEATURES, so we could
    # ask the server how many relays are paired (see top of main()).

    sm = HubStateMachine(
        controller=controller,
        engine=engine,
        recorder=recorder,
        registry=registry,
        store=store,
        feedback=feedback,
        link=link,
    )

    # Wire callbacks
    engine.on_event     = sm.dispatch
    controller.on_frame = sm.on_frame

    # link is already connected (or will reconnect on demand via ServerLink's
    # own lazy-reconnect logic) — see the early connect at the top of main().

    voice = VoiceListener(on_command=sm.on_voice)
    if not args.no_voice:
        voice.start()
        # Only wire sm.voice when voice is actually enabled — _launch_feature()
        # unconditionally restarts it after a mic-owning feature closes, which
        # would defeat --no-voice if wired unconditionally here. See
        # features.base.MIC_OWNING_FEATURE_NAMES for why this pause/resume
        # exists at all: it and a feature's own voice loop must never fight
        # over the same physical microphone.
        sm.voice = voice

    _probe_devices(feedback, link)

    print("[HUB] Running. Ctrl-C to quit.")
    try:
        sm.run()
    except KeyboardInterrupt:
        print("\n[HUB] Shutting down...")
    finally:
        voice.stop()
        sm.stop()
        controller.stop()
        link.close()


if __name__ == "__main__":
    main()