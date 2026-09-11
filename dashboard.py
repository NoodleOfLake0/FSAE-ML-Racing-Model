from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from training_archive import best_episode, load_run_status, new_run_id, read_recent_episodes, utc_now
from training_control import load_config, update_config
from training_stats import atomic_write_json, load_json


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "training_status.json"
HISTORY_FILE = BASE_DIR / "training_history.json"
CONFIG_FILE = BASE_DIR / "training_config.json"
RUNS_DIR = BASE_DIR / "runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_run.json"

_training_process = None
_training_log = None
_archive_signature = None
_archive_cache = []


def current_run_id():
    pointer = load_json(CURRENT_RUN_FILE, {})
    return pointer.get("run_id") if isinstance(pointer, dict) else None


def run_manifest(run_id):
    if not run_id:
        return {}
    value = load_json(RUNS_DIR / run_id / "run.json", {})
    return value if isinstance(value, dict) else {}


def seconds_since(value):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(value)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def live_run_state():
    global _training_process, _training_log
    run_id = current_run_id()
    manifest = run_manifest(run_id)
    if _training_process is not None:
        return_code = _training_process.poll()
        if return_code is None:
            manifest["state"] = "running"
        else:
            if manifest.get("state") == "running":
                manifest["state"] = "failed" if return_code else "completed"
            if _training_log is not None:
                _training_log.close()
                _training_log = None
            _training_process = None
    return run_id, manifest


def archived_episodes():
    """Avoid reparsing the permanent archive on every telemetry poll."""
    global _archive_signature, _archive_cache
    paths = sorted(RUNS_DIR.glob("*/episodes/worker-*.jsonl"))
    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )
    if signature != _archive_signature:
        _archive_cache = read_recent_episodes(RUNS_DIR, limit=None)
        _archive_signature = signature
    return _archive_cache


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    run_id, manifest = live_run_state()
    statuses = load_run_status(RUNS_DIR, run_id)

    if statuses:
        stats = dict(max(statuses, key=lambda value: value.get("updated_at", "")))
        stats["episode"] = sum(int(value.get("episode", 0)) for value in statuses)
        stats["total_steps"] = sum(int(value.get("total_steps", 0)) for value in statuses)
        stats["active_workers"] = len(statuses)
        stats["waiting"] = (
            manifest.get("state") not in {"running", "starting"}
            or seconds_since(stats.get("updated_at")) > 5
        )
    else:
        stats = load_json(STATS_FILE, {})
        if not isinstance(stats, dict) or not stats:
            stats = {
                "episode": 0, "total_steps": 0, "episode_step": 0,
                "waypoint": 0, "total_waypoints": 0, "track_distance": 0,
                "reward": 0, "episode_reward": 0, "throttle": 0,
                "steering": 0, "speed": 0,
            }
        stats["waiting"] = manifest.get("state") != "running"
        stats["active_workers"] = 0

    archived = archived_episodes()
    legacy = load_json(HISTORY_FILE, {})
    legacy_history = legacy.get("history", []) if isinstance(legacy, dict) else []
    legacy_history = list(reversed(legacy_history)) if isinstance(legacy_history, list) else []
    all_history = archived + legacy_history
    stats.update({
        "best_car": best_episode(all_history),
        "history": all_history[:12],
        "run": manifest,
        "run_id": run_id,
        "config": load_config(CONFIG_FILE),
    })
    response = jsonify(stats)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "GET":
        return jsonify(load_config(CONFIG_FILE))
    try:
        saved = update_config(CONFIG_FILE, request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(saved)


@app.route("/api/training/start", methods=["POST"])
def start_training():
    global _training_process, _training_log
    _, manifest = live_run_state()
    if _training_process is not None and _training_process.poll() is None:
        return jsonify({"error": "training is already running"}), 409
    if manifest.get("state") in {"running", "starting"}:
        statuses = load_run_status(RUNS_DIR, manifest.get("run_id"))
        latest_update = max((item.get("updated_at") for item in statuses), default=manifest.get("started_at"))
        if seconds_since(latest_update) <= 10:
            return jsonify({"error": "another training run is already active"}), 409
        manifest.update({"state": "interrupted", "finished_at": utc_now()})
        previous_run_id = manifest.get("run_id")
        if previous_run_id:
            atomic_write_json(RUNS_DIR / previous_run_id / "run.json", manifest)

    try:
        requested = request.get_json(silent=True) or {}
        if not isinstance(requested, dict):
            raise ValueError("settings must be a JSON object")
        config = update_config(CONFIG_FILE, {**requested, "stop_requested": False})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    queued = {
        "run_id": run_id, "state": "starting", "started_at": utc_now(),
        "finished_at": None, "config_at_start": config, "timesteps_completed": 0,
    }
    atomic_write_json(run_dir / "run.json", queued)
    atomic_write_json(CURRENT_RUN_FILE, {"run_id": run_id})
    _training_log = (run_dir / "training.log").open("a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _training_process = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "train.py"), "--run-id", run_id, "--config", str(CONFIG_FILE)],
        cwd=BASE_DIR,
        stdout=_training_log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    return jsonify({"run_id": run_id, "state": "starting", "config": config}), 202


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    _, manifest = live_run_state()
    if manifest.get("state") not in {"running", "starting"}:
        return jsonify({"error": "no training run is active"}), 409
    config = update_config(CONFIG_FILE, {"stop_requested": True})
    return jsonify({"state": "stopping", "config": config}), 202


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
