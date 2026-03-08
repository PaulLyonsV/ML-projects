from collections import defaultdict
from collections.abc import Callable
import gymnasium as gym
from IPython.display import Video
import itertools
from moviepy import ImageSequenceClip
import os
import torch
import tqdm
from typing import Optional
import matplotlib.pyplot as plt

from util_1114 import (
    AdamW,
    get_mlp,
    get_mse,
    get_seed,
    Optimizer,
    pbt_init,
    pbt_update
)

config = {
    "device": "cuda",
    "discount": 0.99,
    "ensemble_shape": (16,),
    "env_id": "LunarLander-v3",
    "env_kwargs": {"render_mode": "rgd-array"},
    "eval_interval": 1_000,
    "hyperparameter_raw_init_distributions": {
        "ema_decay": torch.distributions.Uniform(
            torch.tensor(-3, device="cuda", dtype=torch.float32),
            torch.tensor(0, device="cuda", dtype=torch.float32)
        ),
        "epsilon": torch.distributions.Uniform(
            torch.tensor(-10, device="cuda", dtype=torch.float32),
            torch.tensor(-5, device="cuda", dtype=torch.float32)
        ),
        "epsilon_greedy": torch.distributions.Uniform(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "first_moment_decay": torch.distributions.Uniform(
            torch.tensor(-3, device="cuda", dtype=torch.float32),
            torch.tensor(0, device="cuda", dtype=torch.float32)
        ),
        "learning_rate": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        ),
        "second_moment_decay": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        ),
        "weight_decay": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        ),
    },
    "hyperparameter_raw_perturb": {
        "ema_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "epsilon": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "epsilon_greedy": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(.1, device="cuda", dtype=torch.float32)
        ),
        "first_moment_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "learning_rate": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "second_moment_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "weight_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
    },
    "hyperparameter_transforms": {
        "ema_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
        "epsilon": lambda log10: 10 ** log10,
        "epsilon_greedy": lambda x: x.clamp(0, 1),
        "first_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
        "learning_rate": lambda log10: 10 ** log10,
        "second_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
        "weight_decay": lambda log10: 10 ** log10,
    },
    "improvement_threshold": 1e-4,
    "minibatch_size": 128,
    "replay_buffer_capacity": 10_000,
    "seed": 0,
    "steps_num": 20_000,
    "steps_without_improvement": 100_000,
    "videos_directory": "videos",
    "warmup_steps": 1_000,
    "welch_confidence_level": .8,
    "welch_sample_size": 10,
    "td_steps": 1
}

torch.manual_seed(config["seed"])

env: gym.vector.VectorEnv = gym.make_vec(
    config["env_id"],
    num_envs = config["ensemble_shape"][0],
    **config["env_kwargs"]
)
td_steps_list = [1, 2, 3, 5]
plt.figure(figsize=(10, 6))



print("Observation space shape:", env.single_observation_space.shape)
print("Number of actions:", env.single_action_space.n)

target_network = get_mlp(
    config,
    env.single_observation_space.shape[0],
    env.single_action_space.n,
    3,
    128
)

observations = torch.asarray(
    env.reset(seed=get_seed())[0],
    device = config["device"],
    dtype = torch.float32
)

q_values = target_network(observations.unsqueeze(1))

print("Q-values shape:", q_values.shape)

#Epsilon-greedy policy.
def argmax_q_policy(
    observations: torch.Tensor,
    q_function: Callable[[torch.Tensor], torch.Tensor],
    epsilon=0.01
) -> torch.Tensor:
    q_values = q_function(observations)

    actions = torch.empty(q_values.shape[:-1], device=q_values.device, dtype=torch.int64)
    greedy = torch.rand(actions.shape, device=actions.device) > epsilon
    actions[greedy] = q_values[greedy].argmax(dim=-1)
    actions[~greedy] = torch.randint(
        device=actions.device,
        high=q_values.shape[-1],
        size=((~greedy).sum(),),
    )

    return actions

q_function = lambda observation: target_network(observation.unsqueeze(-2)).squeeze(-2)

actions = argmax_q_policy(
    observations,
    q_function,
    epsilon=0.1
)
print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)

def make_video(
    config: dict,
    policy: Callable[[torch.Tensor], torch.Tensor],
    ensemble_id: Optional[int] = 0,
    fps: Optional[int] = None,
    video_name="test.mp4",
) -> tuple[float, float, str]:
    env = gym.make(
        config["env_id"],
        render_mode = "rgb_array",
        **config["env_kwargs"]
    )

    discounted_return = 0
    if fps is None:
        fps = env.metadata["render_fps"]

    frames = []
    observation, _ = env.reset(seed=get_seed())
    step_id = 0
    undiscounted_return = 0

    ensemble_size = config["ensemble_shape"][0]

    frames.append(env.render())
    while True:
        action = policy(torch.asarray(observation, device=config["device"]).expand(ensemble_size, -1))[ensemble_id]
        observation, reward, truncated, terminal, _ = env.step(action.cpu().numpy())
        discounted_return += config["discount"] ** step_id * reward
        frames.append(env.render())
        undiscounted_return += reward
        if truncated or terminal:
            break

        step_id += 1

    env.close()

    os.makedirs(config["videos_directory"], exist_ok=True)

    clip = ImageSequenceClip(
        frames,
        fps=fps
    )
    gif_path = os.path.join(config["videos_directory"], video_name)
    clip.write_videofile(
        gif_path,
        fps=fps
    )

    return discounted_return, undiscounted_return, gif_path

policy = lambda observation: argmax_q_policy(
    observation,
    q_function,
)
discounted_return, undiscounted_return, video_path = make_video(
    config,
    policy
)
print(f"Discounted return: {discounted_return}")
print(f"Undiscounted return: {undiscounted_return}")
Video(video_path)

def get_returns(
    config: dict,
    policy: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    num_envs = config["ensemble_shape"][0]
    env = gym.make_vec(
        config["env_id"],
        num_envs = num_envs,
        **config["env_kwargs"]
    )

    discount = config["discount"]
    returns = torch.zeros(
        num_envs,
        device=config["device"],
        dtype=torch.float32
    )
    step = 0
    step_observations = torch.asarray(
        env.reset(seed=get_seed())[0],
        device=config["device"],
        dtype=torch.float32
    )
    step_ongoing = torch.ones(
        num_envs,
        device=config["device"],
        dtype=torch.bool
    )
    
    while step_ongoing.any():
        step_actions = policy(step_observations)
        step_observations, step_rewards, truncated, terminal = (
            torch.asarray(
                t,
                device=config["device"],
                dtype=dtype
            )
            for t, dtype in zip(
                env.step(step_actions.cpu().numpy())[:4],
                (torch.float32,) * 2 + (torch.bool,) * 2
            )
        )

        returns += step_rewards * step_ongoing
        step_ongoing &= ~(truncated | terminal)

        step += 1

    return returns

returns = get_returns(
    config,
    policy
)

print("Returns:", returns)

class ReplayBuffer:
    def __init__(
        self,
        config: dict,
        observation_init: torch.Tensor,
    ):
        self.capacity = config["replay_buffer_capacity"]
        self.device = config["device"]
        self.ensemble_shape = config["ensemble_shape"]
        self.ensemble_dim = len(self.ensemble_shape)
        self.minibatch_size = config["minibatch_size"]
        self.td_steps = config["td_steps"]
        self.discount = config["discount"]
        observation_shape = observation_init.shape[self.ensemble_dim:]
        self.warmup_steps = config["warmup_steps"]

        self.actions = torch.empty(
            self.ensemble_shape + (self.capacity,) ,
            device=self.device,
            dtype=torch.int64
        )
        self.observations = torch.empty(
            self.ensemble_shape + (self.capacity,) + observation_shape,
            device=self.device,
            dtype=torch.float32
        )
        self.rewards = torch.empty(
            self.ensemble_shape + (self.capacity,),
            device=self.device,
            dtype=torch.float32
        )

        self.episode_ids = torch.empty(
            self.ensemble_shape + (self.capacity,),
            device=self.device,
            dtype=torch.int64
        )

        self.full = False
        self.cursor = 0
        self.size = 0
        
        index = [slice(None)] * self.ensemble_dim + [0]
        self.observations[*index] = observation_init
        self.episode_ids[*index] = 0



    def get_minibatch(self) -> dict:
        start_indices = torch.randint(
            device=self.device,
            high=self.size,
            size=self.ensemble_shape + (self.minibatch_size,)
        )
        
        #creates buffer to overwrite oldest experiences.
        start_indices = (start_indices + self.cursor - self.size) % self.capacity
        #gather sequence of states to compute multi step target
        step_range = torch.arange(
            self.td_steps + 1, 
            device=self.device
        )
        indices = start_indices.unsqueeze(-1) + step_range
        indices = indices % self.capacity

        ids = torch.gather(
            self.episode_ids,
            dim=self.ensemble_dim,
            index=indices
        )

        #If episode ends mid-sequence, the rest is masked out.
        valid_mask = (ids == ids[..., 0:1])

        actions = torch.gather(
            self.actions,
            dim=self.ensemble_dim,
            index=indices[..., 0]
        )

        obs_indices = indices[..., 0].unsqueeze(-1).expand(
            *indices[..., 0].shape,
            self.observations.shape[-1]
        )
        observations = torch.gather(
            self.observations,
            dim=self.ensemble_dim,
            index=obs_indices
        )

        obs_next_indices = indices[..., -1].unsqueeze(-1).expand(
            *indices[..., -1].shape,
            self.observations.shape[-1]
        )
        observations_next = torch.gather(
            self.observations,
            dim=self.ensemble_dim,
            index=obs_next_indices
        )

        rewards_seq = torch.gather(
            self.rewards,
            dim=self.ensemble_dim,
            index=indices[..., :-1]
        )

        rewards_seq = rewards_seq * valid_mask[..., :-1]

        #Compute discounted cumulative rewards sum
        discounts = self.discount ** torch.arange(self.td_steps, device=self.device)
        discounts = discounts.reshape((1,) * (rewards_seq.ndim - 1) + (-1,))
        rewards = (rewards_seq * discounts).sum(dim=-1)

        terminal = ~valid_mask[..., -1]

        minibatch = {
            "actions": actions,
            "observations": observations,
            "observations_next": observations_next,
            "rewards": rewards,
            "terminal": terminal,
        }

        return minibatch


    def update(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        terminal: torch.Tensor
    ):
        index = [slice(None)] * self.ensemble_dim + [self.cursor]

        self.actions[*index] = actions
        self.rewards[*index] = rewards
        
        current_ids = self.episode_ids[*index]

        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0
            self.full = True
            
        self.size = self.capacity - 1 if self.full else self.cursor
        
        index_next = [slice(None)] * self.ensemble_dim + [self.cursor]
        
        next_ids = current_ids + terminal.long()
        self.episode_ids[*index_next] = next_ids
        self.observations[*index_next] = observations

    def warmup(self, env: gym.vector.VectorEnv):
        env.action_space.seed(get_seed())
        progress_bar = tqdm.trange(self.warmup_steps)
        for _ in progress_bar:
            action = env.action_space.sample()
            observation, reward, terminal, truncated, meta = env.step(action)
            terminal |= truncated
            self.update(*(
                torch.asarray(
                    tensor,
                    device=self.device
                )
                for tensor in (action, observation, reward, terminal)
            ))
replay_buffer = ReplayBuffer(
    config,
    observations
)
replay_buffer.warmup(env)
minibatch = replay_buffer.get_minibatch()
for key, value in minibatch.items():
    print(f"{key}: {value.shape} {value.dtype}")

    training_network = get_mlp(
    config,
    env.single_observation_space.shape[0],
    env.single_action_space.n,
    3,
    128
)
training_network.load_state_dict(target_network.state_dict())

def reshape_hyperparameter(
    hyperparameter: torch.Tensor,
    parameter: torch.Tensor
) -> torch.Tensor:
    hyperparameter = hyperparameter.reshape(
        hyperparameter.shape + (1,) * (parameter.ndim - hyperparameter.ndim)
    )

    return hyperparameter

# update target w exponential moving average.
def ema_update(
    config: dict,
    ema_model: torch.nn.Module,
    trained_model: torch.nn.Module
):
    ema_state_dict, trained_state_dict = (
        model.state_dict()
        for model in (ema_model, trained_model)
    )
    for key, trained_parameter in trained_state_dict.items():
        ema_decay = reshape_hyperparameter(
            config["ema_decay"],
            trained_parameter
        )
        ema_parameter = ema_state_dict[key]
        ema_state_dict[key] = (
            ema_decay * ema_parameter
          + (1 - ema_decay) * trained_parameter
        )

    ema_model.load_state_dict(ema_state_dict)

pbt_init(config, defaultdict(list))
ema_update(
    config,
    target_network,
    training_network
)

def get_target(
    config: dict,
    minibatch: dict,
    target_network: torch.nn.Module
) -> torch.Tensor:
    with torch.no_grad():
        value_predict_next = torch.max(
            target_network(minibatch["observations_next"]),
            dim=-1
        ).values

    discount_factor = config["discount"] ** config["td_steps"]
    
    #Compute target with Bellman equation.
    target = (
        minibatch["rewards"]
        + discount_factor
        * ~minibatch["terminal"]
        * value_predict_next
    )

    return target

target = get_target(
    config,
    minibatch,
    target_network
)
print("Target shape:", target.shape)

def train_q_network(
    config: dict,
    env: gym.vector.VectorEnv,
    optimizer: Optimizer,
    target_network: torch.nn.Module,
    training_network: torch.nn.Module
):
    best_max_return = -torch.inf
    config_local = dict(config)
    log = defaultdict(list)
    steps_without_improvement = 0

    target_q_function = lambda observation: target_network(observation.unsqueeze(-2)).squeeze(-2)
    training_q_function = lambda observation: training_network(observation.unsqueeze(-2)).squeeze(-2)

    target_greedy_policy = lambda observation: argmax_q_policy(observation, target_q_function)

    env.action_space.seed(get_seed())
    observation = torch.asarray(
        env.reset(seed=get_seed())[0], 
        device=config["device"]
    )
    replay_buffer = ReplayBuffer(
        config,
        observation
    )
    replay_buffer.warmup(env)

    pbt_init(config_local, log)
    optimizer.update_config(config_local)

    progress_bar = tqdm.trange(config["steps_num"])

    for step_id in progress_bar:

        with torch.no_grad():
            action = argmax_q_policy(
                observation,
                training_q_function,
                config_local["epsilon_greedy"]
            )

        observation, reward, terminal, truncated = (
            torch.asarray(array, device=config["device"])
            for array in env.step(action.cpu().numpy())[:4]
        )
        replay_buffer.update(
            action,
            observation,
            reward,
            terminal | truncated
        )

        minibatch = replay_buffer.get_minibatch()
        optimizer.zero_grad()
        action_values_predict = training_network(minibatch["observations"])
        values_predict = torch.gather(
            action_values_predict,
            -1,
            minibatch["actions"].unsqueeze(-1)
        ).squeeze(-1)

        values_target = get_target(config, minibatch, target_network)
        loss = get_mse(values_predict, values_target).sum()
        loss.backward()
        optimizer.step()

        ema_update(config_local, target_network, training_network)
        if (step_id + 1) % config["eval_interval"] == 0:
            print("Commencing evaluation.")
            eval_progress_bar = tqdm.trange(config["welch_sample_size"])
            with torch.no_grad():
                evaluations = torch.stack([
                    get_returns(config_local, target_greedy_policy)
                    for _ in eval_progress_bar
                ])
                eval_progress_bar.close()

                evaluations_mean = evaluations.mean(dim=0)
                print(
                    "Evaluation results:",
                    f"min {evaluations_mean.min().cpu().item():.2f}",
                    f"mean {evaluations_mean.mean().cpu().item():.2f}",
                    f"median {evaluations_mean.median().cpu().item():.2f}",
                    f"max {evaluations_mean.max().cpu().item():.2f}",
                    f"std {evaluations_mean.std().cpu().item():.2f}",
                )
                ensemble_id = evaluations_mean.argmax()

                evaluation_mean_max = evaluations_mean.max().cpu().item()
                make_video(
                    config_local,
                    lambda observation: argmax_q_policy(observation, target_network)[ensemble_id],
                    ensemble_id=evaluations_mean.argmax().cpu().item(),
                    video_name=f"{config['env_id']}_step_{step_id}.mp4",
                )

                if evaluation_mean_max > best_max_return + config["improvement_threshold"]:
                    best_max_return = evaluation_mean_max
                    print("New best evaluation!")
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += config["eval_interval"]
                    print(
                        f"Best evaluation: {best_max_return:.2f}",
                        f"steps without improvement: {steps_without_improvement}"
                    )
                    if steps_without_improvement >= config["steps_without_improvement"]:
                        print("Early stopping.")
                        break

                log["evaluations"].append(evaluations)

                with torch.no_grad():
                    pbt_update(
                        config_local,
                        evaluations,
                        log,
                        itertools.chain(optimizer.get_parameters(), target_network.parameters())
                    )

    for key, value in log.items():
        if isinstance(value, list):
            log[key] = torch.stack(value)

    return log

optimizer = AdamW(training_network.parameters())
td_steps_list = [1, 2, 3, 5]

plt.figure(figsize=(10, 6))

for steps in td_steps_list:
    print(f"training with td steps: {steps} ---")

    config["td_steps"] = steps
    
    target_network = get_mlp(
        config,
        env.single_observation_space.shape[0],
        env.single_action_space.n,
        3,
        128
    )
    training_network = get_mlp(
        config,
        env.single_observation_space.shape[0],
        env.single_action_space.n,
        3,
        128
    )
    training_network.load_state_dict(target_network.state_dict())
    optimizer = AdamW(training_network.parameters())

    log = train_q_network(
        config,
        env,
        optimizer,
        target_network,
        training_network
    )

    evaluations = log["evaluations"]
    
    evaluations_mean = evaluations.mean(dim=-1)
    
    best_ensemble_id = evaluations_mean.argmax(-1)
    eval_num = evaluations.shape[0]
    best_samples = torch.stack([
        evaluations[i, best_ensemble_id[i], :]
        for i in range(eval_num)
    ])
    
    
    means = best_samples.mean(dim=-1)
    stds = best_samples.std(dim=-1)
    x_axis = torch.arange(1, len(means) + 1) * config["eval_interval"]
    
    line = plt.plot(x_axis, means, label=f"td-{steps}")
    plt.fill_between(
        x_axis, 
        means - stds, 
        means + stds, 
        alpha=0.2, 
        color=line[0].get_color()
    )

plt.xlabel("training steps")
plt.ylabel("return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()