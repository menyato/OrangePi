"""
features/base.py — the contract every OrangePi feature implements.

A feature is launched from the menu and OWNS the user interaction until it
returns (user said "quit"/"done") or ctx.abort is set (panic gesture). It uses
ctx.link to talk to the server, tagging every message with its own `name`.
"""

from dataclasses import dataclass

# Feature .name values whose run() opens its own microphone stream (directly
# via orangepi_client's mc.listen()/voice_listen_loop, or lidar_nav.py's own
# self-contained _LidarVoice) rather than relying on gesture_hub/voice.py's
# global VoiceListener for spoken input. The global VoiceListener's own
# sr.Microphone() stream and these features' sounddevice.InputStream both
# want exclusive access to the same physical mic — running both at once causes
# garbled/truncated captures ("Too short, discarding", "Nothing understood")
# on either or both sides. Callers that own a VoiceListener (hub.py's normal
# run and bypass-mode branch) must stop() it before launching one of these and
# start() it again after, exactly as was already done for the LiDAR family
# alone before this was generalized to every mic-owning feature.
MIC_OWNING_FEATURE_NAMES = {
    "money", "ocr", "env", "home", "lidar", "obstacles", "mapping", "navigate",
}


@dataclass
class FeatureContext:
    link: object       # net.client.ServerLink
    abort: object      # threading.Event — set when the user does the abort gesture
    feedback: object   # gesture_hub.feedback.Feedback
    # Features that need in-session gesture control (e.g. OCR reader) receive
    # a queue here.  The state machine forwards all non-START non-toggle
    # gesture names to it while the feature is RUNNING.
    gesture_queue: object = None   # queue.Queue | None


class Feature:
    name: str = "base"          # routing key sent to the server
    title: str = "Base feature"  # spoken name in the menu

    def run(self, ctx: FeatureContext) -> None:
        raise NotImplementedError
