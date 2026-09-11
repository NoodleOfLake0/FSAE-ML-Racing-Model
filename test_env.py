from stable_baselines3.common.env_checker import check_env

from fsae_env import FSAEEnv


env = FSAEEnv()

check_env(
    env,
    warn=True
)

print("Environment passed.")

obs, info = env.reset()

print("Observation:")
print(obs)

print("Action space:")
print(env.action_space)