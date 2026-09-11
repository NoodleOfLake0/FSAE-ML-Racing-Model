import json
import os
import time


STATS_FILE = "training_status.json"


while True:

    # Clear terminal
    os.system("cls")

    print("=" * 42)
    print("       VT FSAE RL TRAINING STATUS")
    print("=" * 42)
    print()

    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)

        print(f"Episode:        {stats.get('episode', 0)}")
        print(f"Total Steps:    {stats.get('total_steps', 0)}")
        print(f"Episode Step:   {stats.get('episode_step', 0)}")
        print()

        print(
            f"Waypoint:       "
            f"{stats.get('waypoint', 0)} / "
            f"{stats.get('total_waypoints', 0)}"
        )

        print(
            f"Track Distance: "
            f"{stats.get('track_distance', 0):.4f}"
        )

        print()
        print("-" * 42)

        print(
            f"Reward:         "
            f"{stats.get('reward', 0):.4f}"
        )

        print(
            f"Throttle:       "
            f"{stats.get('throttle', 0):+.3f}"
        )

        print(
            f"Steering:       "
            f"{stats.get('steering', 0):+.3f}"
        )

        print()
        print("-" * 42)

        print(
            f"Speed:          "
            f"{stats.get('speed', 0):.4f}"
        )

        print()
        print("=" * 42)
        print("Ctrl+C to close this monitor.")

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):
        print("Waiting for training data...")

    # Refresh 10x / second
    time.sleep(0.1)