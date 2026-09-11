import json
from pathlib import Path

from flask import Flask, render_template


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "training_status.json"


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():

    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return {
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


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )