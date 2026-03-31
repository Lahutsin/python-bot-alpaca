from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def main() -> int:
    heartbeat_path = Path(os.getenv("ALPACA_HEARTBEAT_PATH", "bot_heartbeat.json"))
    if not heartbeat_path.exists():
        return 1

    try:
        with heartbeat_path.open("r", encoding="utf-8") as file_pointer:
            heartbeat = json.load(file_pointer)
    except (OSError, json.JSONDecodeError):
        return 1

    updated_at = heartbeat.get("updated_at")
    sleep_timeout_minutes = heartbeat.get("sleep_timeout_minutes", 60)
    if not updated_at:
        return 1

    try:
        heartbeat_time = datetime.fromisoformat(updated_at)
        max_age_seconds = int(os.getenv("ALPACA_HEALTHCHECK_MAX_AGE_SECONDS", int(sleep_timeout_minutes) * 60 + 300))
    except (TypeError, ValueError):
        return 1

    if datetime.utcnow() - heartbeat_time > timedelta(seconds=max_age_seconds):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())