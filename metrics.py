"""
metrics.py — tiny helper for reporting OrangePi-side timing to the server.

Per-request latency and server processing time are already captured centrally
by feature_server.py for every feature, using the "_t_sent" timestamp that
net/client.py's ServerLink now stamps on every outgoing message — no feature
code has to do anything for that.

This module is for the timing that DOESN'T fit that pattern: one-time model /
voice-stack load time, camera capture duration, or any other client-side
duration a feature wants recorded. Reports are fire-and-forget (best effort —
a metrics failure must never affect the feature itself) and land in
server_app/handlers/client_metrics.jsonl.

Usage:
    import time
    import metrics

    t0 = time.time()
    mc.load_models()
    metrics.report_load(ctx.link, "ocr", (time.time() - t0) * 1000)
"""

import time


def report_metric(link, source_feature: str, event: str, ms: float, **extra) -> None:
    """Best-effort, non-blocking-in-spirit metric report. Never raises."""
    try:
        link.send("metrics", {
            "event": event,
            "source_feature": source_feature,
            "ms": round(ms, 1),
            **extra,
        })
    except Exception:
        pass


def report_load(link, source_feature: str, ms: float, **extra) -> None:
    """Report a one-time model/voice-stack/device init duration."""
    report_metric(link, source_feature, "client_load", ms, **extra)


def report_action(link, source_feature: str, action: str, ms: float, **extra) -> None:
    """Report the duration of a discrete client-side action (e.g. a capture)."""
    report_metric(link, source_feature, "client_action", ms, action=action, **extra)
