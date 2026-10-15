"""Faithfulness alert: notify when the latest RAGAS faithfulness drops below a threshold.

Reads reports/ragas_summary.json (written by evaluate_ragas.py). If faithfulness < threshold
it (1) prints an ALERT line, (2) appends it to reports/alerts.log and (3) POSTs a message to a
Discord or Slack incoming webhook when ALERT_WEBHOOK_URL is set.

    python -m arabic_legal_rag.alerting                      # threshold 0.80
    python -m arabic_legal_rag.alerting --threshold 0.99     # force an alert to test it

Exit code: 0 = OK, 2 = alert triggered, 1 = no usable score. So CI can fail/notify on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_THRESHOLD = 0.80


def build_message(faithfulness: float, threshold: float, summary: dict) -> str:
    return (
        f"ALERT: RAG faithfulness {faithfulness:.3f} < {threshold:.2f} "
        f"(questions={summary.get('num_questions')}, judge={summary.get('judge_model')}, "
        f"generator={summary.get('generator')})"
    )


def payload_for(url: str, message: str) -> dict:
    """Discord webhooks expect {'content': ...}; Slack expects {'text': ...}."""
    return {"content": message} if "discord" in url else {"text": message}


def send_webhook(url: str, message: str, timeout: float = 10.0) -> bool:
    import httpx

    try:
        r = httpx.post(url, json=payload_for(url, message), timeout=timeout)
        return r.status_code < 300
    except httpx.HTTPError:
        return False


def check(summary: dict, threshold: float, webhook_url: str | None, log_path: Path | None) -> int:
    faith = summary.get("faithfulness")
    if faith is None:
        print("No faithfulness score in summary; nothing to check.", file=sys.stderr)
        return 1
    if faith >= threshold:
        print(f"OK: faithfulness {faith:.3f} >= {threshold:.2f}")
        return 0

    message = build_message(faith, threshold, summary)
    print(message)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {message}\n")
    if webhook_url:
        sent = send_webhook(webhook_url, message)
        print("Notification sent." if sent else "Webhook call failed.", file=sys.stderr)
    else:
        print("ALERT_WEBHOOK_URL not set: alert printed and logged only.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--summary", default="reports/ragas_summary.json")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--log", default="reports/alerts.log")
    args = p.parse_args(argv)

    path = Path(args.summary)
    if not path.exists():
        print(f"{path} not found. Run evaluate_ragas first.", file=sys.stderr)
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))
    return check(summary, args.threshold, os.getenv("ALERT_WEBHOOK_URL"), Path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
