import tempfile
import unittest
from pathlib import Path

from training_archive import RunArchive, best_episode, load_run_status, read_recent_episodes
from training_control import DEFAULT_CONFIG, load_config, update_config, validate_config


class TrainingControlTests(unittest.TestCase):
    def test_config_is_persistent_and_partial_updates_keep_other_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            self.assertEqual(load_config(path), DEFAULT_CONFIG)
            saved = update_config(path, {"simulation_speed": 0.5, "parallel_envs": 4})
            self.assertEqual(saved["simulation_speed"], 0.5)
            self.assertEqual(load_config(path)["parallel_envs"], 4)
            self.assertEqual(load_config(path)["total_timesteps"], 200_000)

    def test_invalid_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({"parallel_envs": 0}, partial=True)
        with self.assertRaises(ValueError):
            validate_config({"surprise": 1}, partial=True)


class TrainingArchiveTests(unittest.TestCase):
    def test_worker_archives_are_append_only_and_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RunArchive(directory, "run-a", 0)
            second = RunArchive(directory, "run-a", 1)
            first.record_episode({"iteration": 1, "waypoints_reached": 2, "episode_reward": 4})
            second.record_episode({"iteration": 1, "waypoints_reached": 3, "episode_reward": 1})
            first.record_episode({"iteration": 2, "waypoints_reached": 1, "episode_reward": 9})
            first.publish_status({"episode": 2})
            second.publish_status({"episode": 1})

            episodes = read_recent_episodes(directory)
            self.assertEqual(len(episodes), 3)
            self.assertEqual(best_episode(episodes)["worker_id"], 1)
            self.assertEqual(len(load_run_status(directory, "run-a")), 2)
            self.assertEqual(len(first.episodes_path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
