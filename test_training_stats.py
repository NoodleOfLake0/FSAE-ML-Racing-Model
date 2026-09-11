import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_stats import TrainingHistory, atomic_write_json


class TrainingStatsTests(unittest.TestCase):
    def test_temporary_lock_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            import os
            replace = os.replace
            with patch("training_stats.os.replace", side_effect=[PermissionError(), None]) as mocked:
                # Perform the real replacement on the successful retry.
                def retry(source, target):
                    if mocked.call_count == 1:
                        raise PermissionError()
                    return replace(source, target)
                mocked.side_effect = retry
                with patch("training_stats.time.sleep"):
                    self.assertTrue(atomic_write_json(path, {"speed": 2}))
            self.assertEqual(json.loads(path.read_text()), {"speed": 2})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_persistent_lock_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            atomic_write_json(path, {"speed": 1})
            with patch("training_stats.os.replace", side_effect=PermissionError()), patch("training_stats.time.sleep"):
                with self.assertWarns(RuntimeWarning):
                    self.assertFalse(atomic_write_json(path, {"speed": 2}, best_effort=True))
                with self.assertRaises(PermissionError):
                    atomic_write_json(path, {"speed": 3})
            self.assertEqual(json.loads(path.read_text()), {"speed": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_atomic_write_replaces_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            atomic_write_json(path, {"speed": 1.25})
            self.assertEqual(json.loads(path.read_text()), {"speed": 1.25})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_history_keeps_best_run_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            history = TrainingHistory(path, history_limit=2)
            history.record({"iteration": 1, "waypoints_reached": 2, "episode_reward": 3})
            history.record({"iteration": 2, "waypoints_reached": 4, "episode_reward": 1})
            history.record({"iteration": 3, "waypoints_reached": 1, "episode_reward": 9})

            reloaded = TrainingHistory(path, history_limit=2)
            self.assertEqual(reloaded.best["iteration"], 2)
            self.assertEqual([run["iteration"] for run in reloaded.history], [2, 3])


if __name__ == "__main__":
    unittest.main()
