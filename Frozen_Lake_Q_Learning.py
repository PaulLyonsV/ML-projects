import gymnasium as gym
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
import tqdm

from util_1015 import (
    line_plot_confidence_band,
    get_seed,
    run_episode
)

config = {
    "discount": 0.99,
    "env_id": "FrozenLake-v1",
    "env_kwargs": {
        "is_slippery": True,
        "map_name": "4x4",
    },
    "env_num_eval": 16,
    "env_num_train": 13,
    "eval_interval": 1000,
    "gif_fps": 20,
    "improvement_threshold": 1e-4,
    "learning_rate": 10 ** torch.linspace(-5, 1, 13, dtype=torch.float32),
    "q_init": 1.,
    "seed": 0,
    "steps_num": 10_001,
    "videos_directory": "videos",
    "alpha_init": 0.5,
    "learning_rate_schedule_temperature": 10.0
}


torch.manual_seed(config["seed"])

env = gym.make_vec(
    config["env_id"],
    num_envs=config["env_num_train"],
    **config["env_kwargs"]
)


def get_init_q_matrix(
    config: dict,
    env: gym.vector.VectorEnv
) -> torch.Tensor:

    num_envs = env.num_envs
    observation_space_n = env.single_observation_space.n
    action_space_n = env.single_action_space.n
    q_values = config["q_init"] * torch.ones(
        (num_envs, observation_space_n, action_space_n),
        dtype=torch.float32
    )

    return q_values

q_values = get_init_q_matrix(config, env)


def argmax_q_policy(
    observations: torch.Tensor,
    q_values: torch.Tensor,
    epsilon=0.01
) -> torch.Tensor:
        
    observations = torch.as_tensor(observations)
    env_num = len(observations)
    observation_num, action_num = q_values.shape[-2:]

    q_values = q_values.broadcast_to(
        (env_num, observation_num, action_num)
    )

    actions = torch.empty(observations.shape, dtype=torch.int64)
    
    #epsilon greedy strategy: greedy policy with probability
    # of (1-epsilon). Otherwise, random policy.
    greedy = torch.rand(actions.shape) > epsilon
    actions[greedy] = q_values[greedy, observations[greedy]].argmax(dim=-1)
    actions[~greedy] = torch.randint(
        high=action_num,
        size=((~greedy).sum(),)
    )
    return actions

observations = torch.zeros(env.num_envs, dtype=torch.int64)
print(argmax_q_policy(observations, q_values[0]))
print(argmax_q_policy(observations, q_values, epsilon=0.5))

def evaluate_q_values(
    config: dict,
    q_values: torch.Tensor
) -> torch.Tensor:
    
    env = gym.make_vec(
        config["env_id"],
        num_envs=config["env_num_eval"],
        **config["env_kwargs"]
    )
    
    observations = torch.tensor(env.reset(seed=get_seed())[0])

    ongoing = torch.ones(
        config["env_num_eval"],
        dtype=torch.bool
    )
    returns = torch.zeros(
        config["env_num_eval"],
        dtype=torch.float32
    )
    step_id = 0
    while ongoing.any():
        actions = argmax_q_policy(observations, q_values).numpy()
        observations, rewards, truncated, terminal = env.step(actions)[:-1]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        returns[ongoing] += config["discount"] ** step_id * rewards[ongoing]
        ongoing &= torch.tensor(~(truncated | terminal), dtype=torch.bool)
        

        step_id += 1

    env.close()

    return returns

def q_learning(
    config: dict
) -> dict:
    
    env = gym.make_vec(
        config["env_id"],
        num_envs=config["env_num_train"],
        **config["env_kwargs"]
    )
    env_arange = torch.arange(config["env_num_train"])
    observations = env.reset(seed=get_seed())[0]
    
    
    progress_bar = tqdm.tqdm(
        torch.linspace(1, 0, config["steps_num"] + 1)[:-1],
        total=config["steps_num"]
    )
    
    q_values = get_init_q_matrix(config, env)
    visits = torch.zeros_like(q_values)
    
    eval_num = (config["steps_num"] // config["eval_interval"]) + 1
    output = dict()

    output["best_avg_return"] = -torch.inf * torch.ones(
        config["env_num_train"],
        dtype=torch.float32
    )
    output["best_q_values"] = q_values.clone()
    output["eval_returns"] = torch.empty(
        (eval_num, config["env_num_train"], config["env_num_eval"]),
        dtype=torch.float32
    )
    output["eval_steps"] = torch.empty(eval_num, dtype=torch.int64)
    
    alpha_init = config["learning_rate"].float()

    temperature = config["learning_rate_schedule_temperature"]
    eval_id = 0
    

    for step_id, epsilon in enumerate(progress_bar):
        if step_id % config["eval_interval"] == 0:
            for env_id, q_values_env in enumerate(q_values):
                eval_returns = evaluate_q_values(config, q_values_env)
                if (
                    eval_returns.mean()
                > output["best_avg_return"][env_id] + config["improvement_threshold"]
                ):
                    output["best_avg_return"][env_id] = eval_returns.mean()
                    output["best_q_values"][env_id] = q_values_env

                output["eval_returns"][eval_id, env_id] = eval_returns
            
            output["eval_steps"][eval_id] = step_id
            eval_id += 1

        actions = argmax_q_policy(observations, q_values, epsilon)
        observations_next, rewards, truncated, terminal = env.step(
            actions.numpy()
        )[:-1]
        
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        ongoing = torch.tensor(~(truncated | terminal), dtype=torch.bool)
        
        #find target w Bellman equation
        target = rewards + config["discount"] * ongoing * q_values[
            env_arange, observations_next
        ].max(dim=-1)[0]


        error = target - q_values[
            env_arange,
            observations,
            actions
        ]
        
        visits[env_arange, observations, actions] += 1
        counts = visits[env_arange, observations, actions]
        #learning rate decays over a visits by a factor determined by temperature param.
        alpha = alpha_init * (counts.float().clamp(min=1.0)).pow(-1.0 / temperature)
        
        q_values[env_arange, observations, actions] \
            += alpha.squeeze() * error

        observations = (observations_next)

    env.close()
    progress_bar.close()

    return output

taus = [1.0, 5.0, 10.0]
q_inits = [0.0, 1.0, 5.0]

best_returns = []

for tau in taus:
    print("tau =", tau)
    tau_row = []

    for q_init in q_inits:
        config_modified = config | {
            "learning_rate_schedule_temperature": tau,
            "q_init": q_init,
        }

        output = q_learning(config_modified)
        tau_row.append(output["best_avg_return"].mean().cpu())

    best_returns.append(torch.tensor(tau_row))

best_returns = torch.stack(best_returns)
print(best_returns)

#Visualizes hyperparameter configurations (g_init and temperature).
plt.imshow(best_returns.numpy(), cmap="viridis", origin="lower")
plt.colorbar(label="Best Average Return")
plt.xticks(range(len(q_inits)), [f"q_init={v}" for v in q_inits])
plt.yticks(range(len(taus)), [f"tau={v}" for v in taus])
plt.title("Grid Search: Best Avg Return by (tau, q_init)")
plt.show()
