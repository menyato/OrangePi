"""
features/base.py — the contract every OrangePi feature implements.

A feature is launched from the menu and OWNS the user interaction until it
returns (user said "quit"/"done") or ctx.abort is set (panic gesture). It uses
ctx.link to talk to the server, tagging every message with its own `name`.
"""

from dataclasses import dataclass


@dataclass
class FeatureContext:
    link: object       # net.client.ServerLink
    abort: object      # threading.Event — set when the user does the abort gesture
    feedback: object   # gesture_hub.feedback.Feedback


class Feature:
    name: str = "base"          # routing key sent to the server
    title: str = "Base feature"  # spoken name in the menu

    def run(self, ctx: FeatureContext) -> None:
        raise NotImplementedError
