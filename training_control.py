"""Shared, validated settings for training and the web dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training_stats import atomic_write_json, load_json


DEFAULT_CONFIG: dict[str, Any] = {
    "total_timesteps": 200_000,
    "episode_max_steps": 2_000,
    "parallel_envs": 1,
    "simulation_speed": 0.0,
    "render": True,
    "render_every": 1,
    "checkpoint_frequency": 25_000,
    "stop_requested": False,
}

LIMITS = {
    "total_timesteps": (2_048, 100_000_000),
    "episode_max_steps": (100, 1_000_000),
    "parallel_envs": (1, 16),
    "simulation_speed": (0.0, 20.0),
    "render_every": (1, 1_000),
    "checkpoint_frequency": (1_000, 10_000_000),
}


def _bounded_number(name: str, value: Any, integer: bool) -> int | float:
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    low, high = LIMITS[name]
    if not low <= converted <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return converted


def validate_config(values: Any, *, partial: bool = False) -> dict[str, Any]:
    """Validate a complete config or a dashboard patch."""
    if not isinstance(values, dict):
        raise ValueError("settings must be a JSON object")
    unknown = set(values) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown setting: {sorted(unknown)[0]}")

    result = {} if partial else dict(DEFAULT_CONFIG)
    for name, value in values.items():
        if name in {"render", "stop_requested"}:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
            result[name] = value
        elif name == "simulation_speed":
            result[name] = _bounded_number(name, value, integer=False)
        else:
            result[name] = _bounded_number(name, value, integer=True)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    saved = load_json(path, {})
    try:
        return validate_config(saved)
    except ValueError:
        return dict(DEFAULT_CONFIG)


def update_config(path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    config = load_config(path)
    config.update(validate_config(updates, partial=True))
    atomic_write_json(path, config)
    return config
