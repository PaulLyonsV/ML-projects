from collections.abc import Callable
import gymnasium as gym
from IPython.display import Image
from moviepy import ImageSequenceClip
import os
from PIL import Image as PILImage
import torch
from typing import Optional
import matplotlib.pyplot as plt
import requests
from io import BytesIO
config = {
    "discount": 0.99,
    "env_id": "FrozenLake-v1",
    "env_kwargs": {
        "is_slippery": True,
        "map_name": "4x4",
    },
    "gif_fps": 5,
    "seed": 0,
    "videos_directory": "videos",
}

torch.manual_seed(config["seed"])

def get_seed(
    upper=1 << 31
) -> int:
    return int(torch.randint(upper, size=()))

print(get_seed())

env = gym.make(
    config["env_id"],
    render_mode="rgb_array",
    **config["env_kwargs"]
)
print(env.observation_space, env.action_space)

print(
    env.observation_space.n, 
    env.action_space.n
)

env.action_space.seed(get_seed())

url = "https://users.renyi.hu/~zsamboki/teaching/dml-autumn-2025/lab_notebooks/images/walking.png"
response = requests.get(url)
PILImage.open(BytesIO(response.content))

episode_return = 0
frames = []
step_id = 0
env.reset(seed=get_seed())
os.makedirs(config["videos_directory"], exist_ok=True)
while True:
    action = env.action_space.sample()
    _, reward, terminated, _,  _ = env.step(action)
    episode_return += reward * config["discount"] ** step_id
    if terminated:
        break

    step_id += 1

# https://stackoverflow.com/a/64796174
def run_episode(
    config: dict,
    env: gym.Env,
    gif_name: Optional[str] = None,
    policy: Optional[Callable[[int], int]]=None,
) -> float:

    if policy is None:
        policy = lambda observation: env.action_space.sample()

    episode_return = 0
    frames = []
    step_id = 0
    observation, _ = env.reset(seed=get_seed())
    if gif_name is not None:
        os.makedirs(config["videos_directory"], exist_ok=True)
        frames.append(env.render())

    while True:
        action = policy(observation)
        observation, reward, _, terminated, _ = env.step(action)
        episode_return += reward * config["discount"] ** step_id
        if gif_name is not None:
            frames.append(env.render())

        if terminated:
            break

        step_id += 1
    if gif_name is not None:
        # https://stackoverflow.com/a/64796174
        clip = ImageSequenceClip(frames, fps=config["gif_fps"])
        gif_path = os.path.join(config["videos_directory"], gif_name)
        clip.write_gif(gif_path, fps=config["gif_fps"])

    return episode_return

episode_return = run_episode(config, env, gif_name=None)

#hard coded policy
policy_list = [
    0, 3, 1, 3, 
    0, 1, 2, 0, 
    3, 1, 0, 0, 
    2, 2, 2, 1
]


policy = lambda observation: policy_list[observation]

returns = []
for i in range(1000):
    episode_return = run_episode(config, env, gif_name=None, policy=policy)
    returns.append(episode_return)

success = sum(k>0 for k in returns)
print(success)
returns_tensor = torch.tensor(returns)
idx = torch.argsort(returns_tensor)

plt.plot(returns_tensor[idx])
plt.scatter(range(len(idx)), returns_tensor[idx], s=5)

plt.title("Sorted Episode Returns (Scatter + Line)")
plt.xlabel("Sorted Episode Index")
plt.ylabel("Return")
plt.show()


gif = "handmade.gif"
episode_return = run_episode(config, env, gif_name=gif, policy=policy)
gif_path = os.path.join(config["videos_directory"], "handmade.gif")
Image(gif_path)