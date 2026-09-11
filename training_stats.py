"""Persistence helpers for live and completed training statistics."""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: str | Path, payload: dict[str, Any], *, best_effort: bool = False
) -> bool:
    """Publish complete JSON, retrying temporary permission failures.

    Best-effort telemetry keeps the previous snapshot if a file stays locked.
    Other IO errors and serialization errors still propagate.
    """
    destination = Path(path)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=destination.name + ".", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary, destination)
                return True
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except PermissionError:
        if not best_effort:
            raise
        warnings.warn(
            f"Could not update {destination}: permission denied; "
            "keeping the previous statistics snapshot.",
            RuntimeWarning,
        )
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Cleanup must not mask the write result.


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
            best_effort=True,
        )
