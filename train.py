import os
import subprocess
import sys
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from fsae_env import FSAEEnv


os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

monitor_process = subprocess.Popen(
    [
        sys.executable,
        "live_status.py"
    ],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

# ----------------------------------
# Create environment
# ----------------------------------

env = FSAEEnv(
    xml_path="car.xml",
    render_mode="human"
)

# Records episode reward and length
env = Monitor(
    env,
    filename="logs/training"
)


# ----------------------------------
# Save every 25,000 steps
# ----------------------------------

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path="models/checkpoints",
    name_prefix="fsae_ppo",
    verbose=2
)


# ----------------------------------
# Create PPO neural network
# ----------------------------------

model = PPO(
    policy="MlpPolicy",
    env=env,

    verbose=1,

    tensorboard_log="logs/tensorboard",

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64
)


# ----------------------------------
# Train
# ----------------------------------

model.learn(
    total_timesteps=200_000,
    callback=checkpoint_callback
)


# ----------------------------------
# Save final model
# ----------------------------------

model.save(
    "models/fsae_ppo_final"
)

env.close()