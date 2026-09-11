import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer
import numpy as np
import json


class FSAEEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
    }

    def __init__(
        self,
        xml_path="car.xml",
        render_mode=None,
    ):
        self.step_count = 0          # current episode
        self.total_env_steps = 0     # entire run
        self.episode_count = 0       # completed episodes
        

        super().__init__()
        self.render_mode = render_mode

        # -------------------------
        # MuJoCo model
        # -------------------------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.car_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "car"
        )

        self.forward_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "forward"
        )

        self.turn_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "turn"
        )

        # -------------------------
        # Gym action space
        # -------------------------
        #
        # action[0] = throttle
        # action[1] = turning
        #

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # -------------------------
        # Observation
        # -------------------------
        #
        # [
        #   target_x relative to car,
        #   target_y relative to car,
        #   speed,
        #   yaw rate,
        #   current throttle,
        #   current steering
        # ]
        #

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Rough centerline of the course you made.
        self.waypoints = np.array([
            [0.20, 0.00],
            [0.30, 0.00],
            [0.40, 0.00],
            [0.50, 0.01],
            [0.60, 0.04],
            [0.70, 0.09],
            [0.79, 0.15],
            [0.87, 0.22],
            [0.95, 0.27],
            [1.05, 0.29],
            [1.15, 0.28],
            [1.25, 0.24],
            [1.35, 0.18],
            [1.45, 0.10],
            [1.55, 0.05],
            [1.65, 0.03],
            [1.75, 0.03],
        ], dtype=np.float32)

        self.waypoint_index = 0

        # One RL action advances several physics frames
        self.frame_skip = 5

        self.step_count = 0
        self.max_steps = 2000

        self.last_action = np.zeros(2, dtype=np.float32)
        self.viewer = None
        self.render_counter = 0
        self.render_every = 1

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False
            )

    # Define the renderer for the program 
    def render(self):

        if self.viewer is not None:

            if self.viewer.is_running():
                self.viewer.sync()

    # --------------------------------------------------
    # Vehicle position
    # --------------------------------------------------

    def _position(self):

        return self.data.xpos[self.car_body_id][:2].copy()


    # --------------------------------------------------
    # Vector from car to target, expressed in CAR frame
    # --------------------------------------------------

    def _target_local(self):

        index = min(
            self.waypoint_index,
            len(self.waypoints) - 1
        )

        target = self.waypoints[index]

        car_pos = self.data.xpos[self.car_body_id]

        world_delta = np.array([
            target[0] - car_pos[0],
            target[1] - car_pos[1],
            0.0
        ])

        # Rotation matrix: car frame -> world frame
        rotation = (
            self.data.xmat[self.car_body_id]
            .reshape(3, 3)
        )

        # Convert world vector -> car frame
        local_delta = rotation.T @ world_delta

        return local_delta[:2]


    # --------------------------------------------------
    # Observation
    # --------------------------------------------------

    def _get_obs(self):

        target = self._target_local()

        speed = np.linalg.norm(
            self.data.qvel[0:2]
        )

        yaw_rate = self.data.qvel[5]

        return np.array([
            target[0],
            target[1],
            speed,
            yaw_rate,
            self.last_action[0],
            self.last_action[1]
        ], dtype=np.float32)


    # --------------------------------------------------
    # Distance to current target
    # --------------------------------------------------

    def _target_distance(self):

        index = min(
            self.waypoint_index,
            len(self.waypoints) - 1
        )

        return np.linalg.norm(
            self._position()
            - self.waypoints[index]
        )


    # --------------------------------------------------
    # Distance from approximate track centerline
    # --------------------------------------------------

    def _track_distance(self):

        pos = self._position()

        distances = np.linalg.norm(
            self.waypoints - pos,
            axis=1
        )

        return np.min(distances)


    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.waypoint_index = 0
        self.step_count = 0

        self.last_action[:] = 0.0

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), {}


    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        old_distance = self._target_distance()

        # Apply neural-network output directly
        # to MuJoCo actuators
        self.data.ctrl[self.forward_id] = action[0]
        self.data.ctrl[self.turn_id] = action[1]

        self.last_action = action.astype(
            np.float32
        )

        # Advance MuJoCo physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data
            )

        self.step_count += 1
        self.total_env_steps += 1

        new_distance = self._target_distance()

        # ----------------------------------------
        # Reward
        # ----------------------------------------

        # Moving toward target = positive
        # moving away = negative
        reward = 10.0 * (
            old_distance - new_distance
        )

        terminated = False

        # Reached waypoint
        if new_distance < 0.06:

            reward += 2.0

            self.waypoint_index += 1

            # Completed course
            if self.waypoint_index >= len(self.waypoints):

                reward += 20.0
                terminated = True


        # ----------------------------------------
        # Off-track
        # ----------------------------------------

        track_distance = self._track_distance()

        if track_distance > 0.30:

            reward -= 10.0
            terminated = True


        # ----------------------------------------
        # Time limit
        # ----------------------------------------

        truncated = (
            self.step_count >= self.max_steps
        )


        observation = self._get_obs()

        info = {
            "waypoint": self.waypoint_index,
            "track_distance": track_distance
        }

        self.render_counter += 1
        if self.render_mode == "human" and self.render_counter % self.render_every == 0:
            self.render()

        if terminated or truncated:
            self.episode_count += 1

        stats = {
            "episode": self.episode_count,
            "total_steps": self.total_env_steps,
            "episode_step": self.step_count,

            "waypoint": self.waypoint_index,
            "total_waypoints": len(self.waypoints),

            "track_distance": float(track_distance),
            "reward": float(reward),

            "throttle": float(action[0]),
            "steering": float(action[1]),

            "speed": float(
                np.linalg.norm(self.data.qvel[0:2])
            )
        }
        
        with open("training_status.json", "w") as f:
            json.dump(stats, f)

        # if self.render_mode == "human":
            #print(
            #    f"\r"
            #    f"Episode: {self.episode_count} | "
            #    f"Total Steps: {self.total_env_steps} | "
            #    f"Episode Step: {self.step_count} | "
            #    f"Waypoint: {self.waypoint_index}/{len(self.waypoints)} | "
            #    f"Reward: {reward:.3f}",
            #    end="",
            #    flush=True
            # )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )
        


    def close(self):
         if self.viewer is not None:
            self.viewer.close()
            self.viewer = None