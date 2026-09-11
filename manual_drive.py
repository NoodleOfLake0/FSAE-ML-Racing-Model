import time
import mujoco
import mujoco.viewer
from typing import Literal


model = mujoco.MjModel.from_xml_path("car.xml")
data = mujoco.MjData(model)

# 1. Define standard GLFW integer codes for Pylance validation
MuJoCoArrowKey = Literal[265, 264, 263, 262]

UP_ARROW: MuJoCoArrowKey = 265
DOWN_ARROW: MuJoCoArrowKey = 264
LEFT_ARROW: MuJoCoArrowKey = 263
RIGHT_ARROW: MuJoCoArrowKey = 262

# 2. Define your key callback
def my_key_callback(keycode: int):
    # Pylance safely checks the integers here
    if keycode == UP_ARROW:
        print("Pressed Up Arrow")
    elif keycode == DOWN_ARROW:
        print("Pressed Down Arrow")
    elif keycode == LEFT_ARROW:
        print("Pressed Left Arrow")
    elif keycode == RIGHT_ARROW:
        print("Pressed Right Arrow")

forward_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "forward"
)

turn_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "turn"
)


throttle = 0.0
steering = 0.0


def key_callback(keycode: int):
    global throttle, steering

    key = chr(keycode).lower()

     # Handle special GLFW integer keys first
    if keycode == UP_ARROW:
        throttle = 1.0
    elif keycode == DOWN_ARROW:
        throttle = -1.0
    elif keycode == LEFT_ARROW:
        steering = 1.0
    elif keycode == RIGHT_ARROW:
        steering = -1.0
    
    # Handle standard character keys safely
    else:
        try:
            key = chr(keycode).lower()
            
            if key == "x":
                throttle = 0.0
            elif key == "c":
                steering = 0.0
            elif key == " ":
                throttle = 0.0
                steering = 0.0
        except ValueError:
            # Ignore codes that can't be converted to a character
            pass


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    while viewer.is_running():

        step_start = time.time()

        data.ctrl[forward_id] = throttle
        data.ctrl[turn_id] = steering

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = (
            model.opt.timestep -
            (time.time() - step_start)
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)