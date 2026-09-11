"""Persistence helpers for live and completed training statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON without ever exposing a partially-written file to readers."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)


def load_json(path: str | Path, default: Any) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


class TrainingHistory:
    """Keep a bounded episode history and the strongest car run on disk."""

    def __init__(self, path: str | Path, history_limit: int = 100):
        self.path = Path(path)
        self.history_limit = history_limit
        saved = load_json(self.path, {})
        self.best = saved.get("best") if isinstance(saved, dict) else None
        history = saved.get("history", []) if isinstance(saved, dict) else []
        self.history = history if isinstance(history, list) else []

    @staticmethod
    def _rank(run: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(bool(run.get("completed_course"))),
            float(run.get("waypoints_reached", 0)),
            float(run.get("episode_reward", 0)),
            -float(run.get("episode_steps", 0)),
        )

    def record(self, run: dict[str, Any]) -> None:
        self.history.append(run)
        self.history = self.history[-self.history_limit :]
        if self.best is None or self._rank(run) > self._rank(self.best):
            self.best = dict(run)
        atomic_write_json(
            self.path,
            {"best": self.best, "history": self.history},
        )
