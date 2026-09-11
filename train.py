"""Train the FSAE agent using dashboard-controlled, permanently archived runs."""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from fsae_env import FSAEEnv
from training_archive import new_run_id, utc_now
from training_control import load_config, update_config
from training_stats import atomic_write_json


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
CONFIG_FILE = BASE_DIR / "training_config.json"
CURRENT_RUN_FILE = RUNS_DIR / "current_run.json"


def make_environment(worker_id, run_id, controls_path, render, run_dir):
    env = FSAEEnv(
        xml_path=str(BASE_DIR / "car.xml"),
        render_mode="human" if render else None,
        run_id=run_id,
        worker_id=worker_id,
        controls_path=controls_path,
        runs_dir=RUNS_DIR,
    )
    return Monitor(env, filename=str(run_dir / "logs" / f"worker-{worker_id}"))


class StopRequestedCallback(BaseCallback):
    """Stop a rollout promptly when the dashboard asks the run to stop."""

    def __init__(self, controls_path):
        super().__init__()
        self.controls_path = controls_path
        self.calls_since_read = 25

    def _on_step(self):
        self.calls_since_read += 1
        if self.calls_since_read >= 25:
            self.calls_since_read = 0
            if load_config(self.controls_path)["stop_requested"]:
                return False
        return True


def write_manifest(run_dir, payload):
    atomic_write_json(run_dir / "run.json", payload)
    atomic_write_json(CURRENT_RUN_FILE, {"run_id": payload["run_id"]})


def train(run_id=None, controls_path=CONFIG_FILE):
    config = load_config(controls_path)
    run_id = run_id or new_run_id()
    run_dir = RUNS_DIR / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)

    parallel_envs = config["parallel_envs"]
    render = config["render"] and parallel_envs == 1
    manifest = {
        "run_id": run_id,
        "state": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "pid": os.getpid(),
        "config_at_start": config,
        "render_effective": render,
        "timesteps_completed": 0,
    }
    write_manifest(run_dir, manifest)
    atomic_write_json(run_dir / "config.json", config)

    env = None
    model = None
    final_state = "completed"
    failure = None

    try:
        factories = [
            partial(make_environment, worker, run_id, str(controls_path), render, run_dir)
            for worker in range(parallel_envs)
        ]
        vector_env = DummyVecEnv(factories) if parallel_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
        env = VecMonitor(vector_env, filename=str(run_dir / "logs" / "combined"))
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(run_dir / "logs" / "tensorboard"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
        )
        next_checkpoint = config["checkpoint_frequency"]

        while True:
            live_config = load_config(controls_path)
            target = live_config["total_timesteps"]
            if live_config["stop_requested"]:
                final_state = "stopped"
                break
            if model.num_timesteps >= target:
                break

            # One rollout at a time lets the total target be extended live.
            remaining = target - model.num_timesteps
            rollout_size = 2048 * parallel_envs
            model.learn(
                total_timesteps=min(remaining, rollout_size),
                callback=StopRequestedCallback(controls_path),
                reset_num_timesteps=False,
                progress_bar=False,
            )
            manifest["timesteps_completed"] = model.num_timesteps
            write_manifest(run_dir, manifest)

            live_config = load_config(controls_path)
            checkpoint_frequency = live_config["checkpoint_frequency"]
            if model.num_timesteps >= next_checkpoint:
                model.save(run_dir / "models" / "checkpoints" / f"fsae_ppo_{model.num_timesteps}_steps")
                next_checkpoint = model.num_timesteps + checkpoint_frequency
    except KeyboardInterrupt:
        final_state = "stopped"
    except Exception as error:
        final_state = "failed"
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            model.save(run_dir / "models" / "fsae_ppo_final")
            (BASE_DIR / "models").mkdir(exist_ok=True)
            model.save(BASE_DIR / "models" / "fsae_ppo_final")
        if env is not None:
            env.close()
        manifest.update({
            "state": final_state,
            "finished_at": utc_now(),
            "timesteps_completed": model.num_timesteps if model is not None else 0,
            "error": failure,
        })
        write_manifest(run_dir, manifest)
        update_config(controls_path, {"stop_requested": False})

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    train(args.run_id, args.config)


if __name__ == "__main__":
    main()
