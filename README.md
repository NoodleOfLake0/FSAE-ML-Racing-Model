# FSAE ML Racing Model

A learning-focused autonomous racing demo that uses **MuJoCo**, **Gymnasium**, and **Stable-Baselines3** to train a reinforcement-learning agent to control a simplified Formula SAE-style race car.

The purpose of this project is not to build a competition-ready autonomous controller. Instead, it is a sandbox for learning how vehicle dynamics, simulation environments, reward functions, observations, actions, and reinforcement learning fit together in an autonomous racing pipeline.

---

## Project Goals

This project is intended to build practical experience with:

- Modeling a vehicle in MuJoCo
- Creating a custom Gymnasium environment
- Defining observations and action spaces
- Designing reinforcement-learning reward functions
- Training a PPO agent with Stable-Baselines3
- Logging and visualizing training progress
- Saving and loading trained policies
- Evaluating autonomous driving behavior in simulation
- Understanding how ML interacts with vehicle dynamics and controls

Longer term, the project can serve as a stepping stone toward more realistic autonomous racing work involving path planning, perception, state estimation, vehicle control, and Formula SAE Driverless-style systems.

---

## Current Stack

| Tool | Purpose |
| --- | --- |
| Python | Main development language |
| MuJoCo | Physics and vehicle simulation |
| Gymnasium | Reinforcement-learning environment interface |
| Stable-Baselines3 | RL algorithms and training infrastructure |
| PPO | Initial reinforcement-learning algorithm |
| TensorBoard | Training metrics and visualization |
| NumPy | Numerical operations |

---

## System Overview

The basic learning pipeline is:

```text
MuJoCo Vehicle Model
        |
        v
Custom Gymnasium Environment
        |
        +--> Observations
        |      - vehicle state
        |      - position
        |      - velocity
        |      - orientation
        |      - track-relative information
        |
        +--> Actions
        |      - steering
        |      - throttle / drive command
        |
        +--> Reward Function
               - forward progress
               - staying on track
               - stable driving
               - penalties for failure

        |
        v
Stable-Baselines3 PPO Agent
        |
        v
Training + Checkpoints + TensorBoard Logs
        |
        v
Evaluation in MuJoCo
```

The agent repeatedly observes the simulated car state, selects an action, receives a reward, and updates its policy based on its experience.

---

## Project Structure

A typical project layout is:

```text
FSAE-ML-Racing-Model/
|
├── train.py
├── evaluate.py
├── environment/
│   ├── __init__.py
│   └── racing_env.py
|
├── models/
│   └── car.xml
|
├── checkpoints/
|
├── logs/
│   └── tensorboard/
|
├── saved_models/
|
├── requirements.txt
└── README.md
```

The exact directory structure may change as the project develops.

---

## Environment

The custom Gymnasium environment acts as the interface between MuJoCo and the reinforcement-learning agent.

At every simulation step, the environment:

1. Reads the current MuJoCo simulation state.
2. Converts that state into an observation.
3. Sends the observation to the PPO policy.
4. Receives an action from the policy.
5. Applies the action to the simulated vehicle.
6. Advances the physics simulation.
7. Calculates the reward.
8. Determines whether the episode has terminated.

Conceptually:

```python
observation = env.reset()

while not done:
    action = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
```

---

## Observation Space

The observation space describes what information the agent is allowed to use when making a driving decision.

Possible observations include:

```text
Vehicle position
Vehicle velocity
Vehicle heading
Angular velocity
Steering state
Distance from track center
Heading error relative to track
Distance to upcoming waypoints
Previous control inputs
```

A major design goal is keeping the observation space informative without giving the agent unnecessary or unrealistic information.

---

## Action Space

The agent controls the simulated vehicle through continuous actions.

A simplified action vector may look like:

```text
[steering, throttle]
```

For example:

```text
steering ∈ [-1, 1]
throttle ∈ [-1, 1]
```

These normalized values are converted into commands that MuJoCo applies to the vehicle actuators.

Future versions may separate:

```text
steering
throttle
braking
```

or model lower-level actuator behavior more realistically.

---

## Reward Function

The reward function defines what behavior the reinforcement-learning agent is encouraged to learn.

A simple reward may combine:

```text
+ forward progress
+ velocity in the desired direction
+ staying near the track center
- leaving the track
- excessive steering
- unstable motion
- collisions
- moving backward
```

Conceptually:

```python
reward = (
    progress_reward
    + speed_reward
    - track_error_penalty
    - control_penalty
)
```

Reward design is one of the most important parts of the project.

A poorly designed reward can cause the agent to discover behavior that technically maximizes reward while failing to drive correctly.

---

## Reinforcement Learning Algorithm

The current model uses **Proximal Policy Optimization (PPO)** through Stable-Baselines3.

PPO is a policy-gradient reinforcement-learning algorithm that works well with continuous control problems and provides a relatively stable starting point for experimentation.

Example:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/tensorboard/"
)

model.learn(total_timesteps=200_000)
```

The initial goal is not to aggressively optimize hyperparameters. The priority is first proving that the environment, physics, observations, actions, rewards, and training loop all behave correctly.

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd FSAE-ML-Racing-Model
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install mujoco gymnasium stable-baselines3 tensorboard numpy
```

If a `requirements.txt` file is present:

```bash
pip install -r requirements.txt
```

---

## Training

Start training with:

```bash
python train.py
```

The training script should:

- create the custom racing environment
- initialize PPO
- begin training
- write TensorBoard logs
- periodically save checkpoints
- save the final model

A typical run may use:

```python
model.learn(
    total_timesteps=200_000,
    callback=checkpoint_callback
)
```

Early runs should be treated primarily as debugging experiments rather than attempts to produce a strong racing policy.

---

## TensorBoard

Training metrics can be viewed with TensorBoard.

```bash
tensorboard --logdir logs/tensorboard
```

Then open the local TensorBoard address shown in the terminal.

Useful metrics include:

- episode reward
- episode length
- policy loss
- value loss
- entropy
- approximate KL divergence
- explained variance

The most important metric is still actual vehicle behavior in simulation. A rising reward is only useful if the learned behavior matches the intended task.

---

## Checkpoints

Training checkpoints can be saved periodically so that progress is not lost during long runs.

Example:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="ppo_fsae"
)
```

This also makes it possible to compare policies from different stages of training.

---

## Evaluating a Trained Model

A trained PPO model can be loaded with:

```python
from stable_baselines3 import PPO

model = PPO.load("saved_models/ppo_fsae")
```

The evaluation loop should run the policy without training and visualize its behavior in MuJoCo.

Important evaluation questions include:

- Does the vehicle stay on the intended path?
- Does steering oscillate?
- Does the model exploit the reward function?
- Can it recover from small disturbances?
- Does performance generalize to different initial conditions?
- Does it fail when speed increases?

---

## Development Strategy

The project is being developed incrementally.

### Phase 1 — Simulation

- Build a functioning MuJoCo vehicle model
- Verify joints, actuators, mass, and basic dynamics
- Confirm steering and propulsion work manually

### Phase 2 — Gymnasium Environment

- Wrap the MuJoCo model in a Gymnasium environment
- Define observation space
- Define action space
- Implement reset logic
- Implement termination conditions

### Phase 3 — Basic RL

- Create a simple reward function
- Train PPO
- Verify the training loop works
- Add checkpoints
- Add TensorBoard logging

### Phase 4 — Behavior Improvement

- Tune the reward function
- Improve track-relative observations
- Penalize unstable or unrealistic control
- Randomize starting conditions
- Improve robustness

### Phase 5 — More Realistic Racing

Potential future additions:

- waypoint following
- racing-line optimization
- track boundaries
- lap timing
- tire-force modeling
- more realistic steering dynamics
- sensor noise
- actuator delay
- domain randomization
- path planning
- classical controller baselines
- comparison against PID / pure pursuit / MPC
- perception-based observations

---

## ML vs. Traditional Controls

Reinforcement learning is only one approach to autonomous vehicle control.

A realistic autonomous racing system may combine:

```text
Perception
    ↓
State Estimation
    ↓
Path Planning
    ↓
Trajectory Generation
    ↓
Vehicle Controller
    ↓
Actuators
```

The controller itself could use:

- PID
- Pure Pursuit
- Stanley Controller
- Model Predictive Control
- Reinforcement Learning
- Hybrid learning + classical control methods

Part of the purpose of this project is learning where ML is actually useful rather than forcing ML into every component.

---

## Known Limitations

This repository is currently a learning/demo project and should not be interpreted as a complete autonomous Formula SAE system.

Current or expected simplifications include:

- simplified vehicle dynamics
- simplified tire behavior
- idealized state information
- limited sensor modeling
- limited track complexity
- simplified actuator dynamics
- no safety-critical guarantees
- no real-time embedded deployment
- no physical vehicle validation

Simulation success does not guarantee performance on a real race car.

---

## Windows / OneDrive Note

For training runs that generate frequent checkpoints, logs, or JSON/configuration files, storing the repository inside a synced OneDrive directory can sometimes cause file-locking or permission issues.

If write errors occur, move the repository to a normal local development directory such as:

```text
C:\dev\FSAE-ML-Racing-Model
```

rather than:

```text
C:\Users\<user>\OneDrive\...
```

This is especially useful for TensorBoard logs, model checkpoints, and frequently rewritten training artifacts.

---

## Learning Priorities

The main engineering priorities for this project are:

1. Understand the simulation before training the model.
2. Verify observations and actions before tuning PPO.
3. Test the reward function independently.
4. Visualize what the agent is actually doing.
5. Change one major variable at a time.
6. Keep experiments reproducible.
7. Compare ML approaches against conventional controls where possible.

The objective is not simply to obtain a high reward value. The objective is to understand **why the vehicle behaves the way it does**.

---

## Future Experiments

Possible experiments include:

- PPO vs. SAC
- different observation spaces
- different reward formulations
- curriculum learning
- randomized initial vehicle states
- sensor noise injection
- track randomization
- speed-vs-stability reward tradeoffs
- model generalization across multiple tracks
- RL vs. pure pursuit
- RL vs. MPC
- hybrid RL + classical controller architectures

---

## Disclaimer

This project is an independent educational demo inspired by autonomous racing and Formula SAE engineering concepts.

It is not production software, not safety-certified, and should not be used to control a real vehicle without substantial additional engineering, validation, and safety systems.

---

## Author

Developed as an engineering and machine-learning learning project focused on reinforcement learning, autonomous racing, vehicle dynamics, and Formula SAE-style systems.
