"""Permanent, run-scoped storage for training metadata and episodes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_stats import atomic_write_json, load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class RunArchive:
    def __init__(self, root: str | Path, run_id: str, worker_id: int = 0):
        self.root = Path(root)
        self.run_id = run_id
        self.worker_id = worker_id
        self.run_dir = self.root / run_id
        self.status_dir = self.run_dir / "status"
        self.episode_dir = self.run_dir / "episodes"
        self.status_path = self.status_dir / f"worker-{worker_id}.json"
        self.episodes_path = self.episode_dir / f"worker-{worker_id}.jsonl"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def publish_status(self, stats: dict[str, Any]) -> None:
        payload = dict(stats)
        payload.update({"run_id": self.run_id, "worker_id": self.worker_id, "updated_at": utc_now()})
        atomic_write_json(self.status_path, payload, best_effort=True)

    def record_episode(self, episode: dict[str, Any]) -> None:
        payload = dict(episode)
        payload.update({"run_id": self.run_id, "worker_id": self.worker_id, "saved_at": utc_now()})
        # Each worker owns one append-only file, so parallel environments never
        # contend for the same archive handle.
        with self.episodes_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            file.flush()
            os.fsync(file.fileno())


def read_recent_episodes(root: str | Path, limit: int | None = 100) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    paths = sorted(Path(root).glob("*/episodes/worker-*.jsonl"), reverse=True)
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        episodes.append(value)
        except OSError:
            continue
    episodes.sort(key=lambda item: item.get("saved_at", ""), reverse=True)
    return episodes[:limit] if limit is not None else episodes


def best_episode(episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    def rank(run: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(bool(run.get("completed_course"))),
            float(run.get("waypoints_reached", 0)),
            float(run.get("episode_reward", 0)),
            -float(run.get("episode_steps", 0)),
        )

    return max(episodes, key=rank) if episodes else None


def load_run_status(root: str | Path, run_id: str | None) -> list[dict[str, Any]]:
    if not run_id:
        return []
    statuses = []
    for path in sorted((Path(root) / run_id / "status").glob("worker-*.json")):
        value = load_json(path, {})
        if isinstance(value, dict) and value:
            statuses.append(value)
    return statuses
