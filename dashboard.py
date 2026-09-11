from pathlib import Path

from flask import Flask, jsonify, render_template

from training_stats import load_json


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "training_status.json"
HISTORY_FILE = BASE_DIR / "training_history.json"


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    stats = load_json(STATS_FILE, {})
    saved = load_json(HISTORY_FILE, {})
    if not isinstance(saved, dict):
        saved = {}
    if not stats:
        stats = {
            "episode": 0,
            "total_steps": 0,
            "episode_step": 0,
            "waypoint": 0,
            "total_waypoints": 0,
            "track_distance": 0,
            "reward": 0,
            "throttle": 0,
            "steering": 0,
            "speed": 0,
            "waiting": True
        }
    stats["best_car"] = saved.get("best")
    stats["history"] = saved.get("history", [])[-12:]
    response = jsonify(stats)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
