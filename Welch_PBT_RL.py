from collections import defaultdict
from collections.abc import Iterable
import gymnasium as gym
from IPython.display import Video
from moviepy import ImageSequenceClip
import os
import torch
import tqdm
from typing import Optional
import matplotlib.pyplot as plt

from util_1107 import (
    AdamW,
    get_mlp,
    get_seed,
    Optimizer,
    welch_one_sided
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

config = {
        "device": device,
        "discount": 0.99,
        "ensemble_shape": (16,),
        "env_id": "LunarLander-v3",
        "exploit_method": "welch",
        "truncate_proportion": 0.2, 
        "env_kwargs": {},
        "eval_interval": 100,
        "hyperparameter_raw_init_distributions": {
            "epsilon": torch.distributions.Uniform(
                torch.tensor(-10, device=device, dtype=torch.float32),
                torch.tensor(-5, device=device, dtype=torch.float32)
            ),
            "first_moment_decay": torch.distributions.Uniform(
                torch.tensor(-3, device=device, dtype=torch.float32),
                torch.tensor(0, device=device, dtype=torch.float32)
            ),
            "learning_rate": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            ),
            "second_moment_decay": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            ),
            "weight_decay": torch.distributions.Uniform(
                torch.tensor(-5, device=device, dtype=torch.float32),
                torch.tensor(-1, device=device, dtype=torch.float32)
            ),
        },
        "hyperparameter_raw_perturb": {
            "epsilon": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "first_moment_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "learning_rate": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "second_moment_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
            "weight_decay": torch.distributions.Normal(
                torch.tensor(0, device=device, dtype=torch.float32),
                torch.tensor(1, device=device, dtype=torch.float32)
            ),
        },
        "hyperparameter_transforms": {
            "epsilon": lambda log10: 10 ** log10,
            "first_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
            "learning_rate": lambda log10: 10 ** log10,
            "second_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
            "weight_decay": lambda log10: 10 ** log10,
        },
        "improvement_threshold": 1e-4,
        "seed": 0,
        "steps_num": 10_000,
        "steps_without_improvement": 1000,
        "videos_directory": "videos",
        "welch_confidence_level": .8,
        "welch_sample_size": 10,
    }


torch.manual_seed(config["seed"])

env = gym.make(
    config["env_id"],
    **config["env_kwargs"]
)

observation, _ = env.reset(seed=get_seed())

policy = get_mlp(
    config,
    env.observation_space.shape[0],
    env.action_space.n,
    2,
    128
)
observation = torch.asarray(
    observation,
    device=config["device"],
    dtype=torch.float32
)


def get_logits(
    config: dict,
    observation: torch.Tensor,
    policy: torch.nn.Module
) -> torch.Tensor:
    ensemble_shape = config["ensemble_shape"]
    ensemble_dim = len(ensemble_shape)

    ensembled_input = observation.shape[:ensemble_dim] == ensemble_shape
    batch_dim = observation.dim() - ensembled_input * ensemble_dim - 1

    #Handles different input shapes depending on whether its 
    #single env or ensemble training.
    if batch_dim == 0:
        observation = observation[..., None, :]

    logits = policy(observation)
    if batch_dim == 0:
        logits = logits[..., 0, :]

    return logits

logits = get_logits(config, observation, policy)

def make_video(
    config: dict,
    policy: torch.nn.Module,
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

    frames.append(env.render())
    while True:
        logits = get_logits(
            config,
            torch.asarray(
                observation,
                device=config["device"],
                dtype=torch.float32
            ),
            policy
        )
        action = logits[ensemble_id].argmax().cpu().numpy()

        observation, reward, truncated, terminal, _ = env.step(action)
        discounted_return += config["discount"] ** step_id * reward
        frames.append(env.render())
        undiscounted_return += reward
        if truncated or terminal:
            break

        step_id += 1

    env.close()

    os.makedirs(config["videos_directory"], exist_ok=True)
    
    # https://stackoverflow.com/a/64796174
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

discounted_return, undiscounted_return, video_path = make_video(
    config,
    policy
)
print(
    f"Discounted return: {discounted_return:.2f}",
    f"Undiscounted return: {undiscounted_return:.2f}"
)
Video(video_path)

env = gym.make_vec(
    config["env_id"],
    num_envs=config["ensemble_shape"][0],
    **config["env_kwargs"]
)
observation = env.reset(seed=get_seed())[0]
logits = get_logits(
    config,
    torch.asarray(
        observation,
        device=config["device"],
        dtype=torch.float32
    ),
    policy
)

#Gumbel code was written by our professor. 
#We use gumbel before argmaxing to 
# take a stochastic sample from 
# deterministic logits in action space.
gumbel = torch.distributions.Gumbel(
    torch.tensor(0., device=config["device"]),
    torch.tensor(1., device=config["device"])
)
actions = (logits + gumbel.sample(logits.shape)).argmax(dim=-1)
print(actions)


def get_episode_data(
    config: dict,
    env: gym.vector.VectorEnv,
    policy: torch.nn.Module,
    gumbel: Optional[torch.distributions.Gumbel] = None
) -> dict:
    
    env_num = env.observation_space.shape[0]

    step_observations = torch.asarray(
        env.reset(seed=get_seed())[0],
        device=config["device"],
        dtype=torch.float32
    )
    step_ongoing = torch.ones(
        env_num,
        device=config["device"],
        dtype=torch.bool
    )

    episode_actions = []
    episode_observations = [step_observations]
    episode_rewards = []    
    
    while step_ongoing.any():
        logits = get_logits(config, step_observations, policy)
        if gumbel is not None:
            logits += gumbel.sample(logits.shape)

        step_actions = logits.argmax(dim=-1)
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
        
        for t, l in zip(
            (
                step_actions, step_observations, step_rewards * step_ongoing
            ),
            (
                episode_actions,
                episode_observations,
                episode_rewards
            )
        ):
            l.append(t)
        
        step_ongoing = step_ongoing & ~(truncated | terminal)
    
    return {
        key: torch.stack(collection, dim=1)
        for key, collection in zip(
            ("actions", "observations", "rewards"),
            (
                episode_actions,
                episode_observations,
                episode_rewards
            )
        )
    }

deterministic_episode_data = get_episode_data(config, env, policy)
stochastic_episode_data = get_episode_data(config, env, policy, gumbel)

(
    deterministic_undiscounted_return,
    stochastic_undiscounted_return
) = (
    data["rewards"].sum(dim=-1)
    for data in (deterministic_episode_data, stochastic_episode_data)
)

print(f"Average undiscounted return for deterministic actions: {deterministic_undiscounted_return.mean().cpu().item():.2f} stochastic actions: {stochastic_undiscounted_return.mean().cpu().item():.2f}")

def get_discounted_returns(
    config: dict,
    rewards: torch.Tensor
) -> torch.Tensor:
    step_num = rewards.shape[-1]

    arange = torch.arange(
        step_num,
        device=config["device"],
        dtype=torch.float32
    )
    discounts = (config["discount"] ** (arange[:, None] - arange)).tril()

    discounted_returns = rewards @ discounts

    return discounted_returns

deterministic_discounted_returns = get_discounted_returns(
    config,
    deterministic_episode_data["rewards"]
)


def exploit_welch(
    config: dict,
    evaluations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

    population_size = evaluations.shape[1]

    target_indices_all = torch.randint(
        high=population_size,
        size=(population_size,),
        device=evaluations.device
    )

    source_mask = welch_one_sided(
        evaluations,
        evaluations[:, target_indices_all],
        confidence_level=config["welch_confidence_level"]
    )

    source_indices = source_mask.nonzero(as_tuple=False).flatten()
    target_indices = target_indices_all[source_indices]

    return source_indices, target_indices

def exploit_truncation(
    config: dict,
    evaluations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    
    population_size = evaluations.shape[1]
    mean_returns = evaluations.mean(dim=0)

    k = int(round(population_size * config["truncate_proportion"]))
    k = max(1, k)

    best_indices = torch.topk(mean_returns, k).indices
    worst_indices = torch.topk(-mean_returns, k).indices
    worst_indices, _ = worst_indices.sort()
    best_indices, _ = best_indices.sort()
    
    target_indices = best_indices

    return worst_indices, target_indices

def pbt_init(
    config: dict,
    log: dict,
):
        
    for name, distribution in config[
        "hyperparameter_raw_init_distributions"
    ].items():
        value_raw = distribution.sample(config["ensemble_shape"])
        config[name + "_raw"] = value_raw
        value = config[
            "hyperparameter_transforms"
        ][name](value_raw)
        config[name] = value
        log[name].append(value)


def pbt_update(
    config: dict,
    evaluations: torch.Tensor,
    log: dict,
    parameters: Iterable[torch.nn.Parameter]
):
    if config["exploit_method"] == "welch":
        source_indices, target_indices = exploit_welch(config, evaluations)
    elif config["exploit_method"] == "truncation":
        source_indices, target_indices = exploit_truncation(config, evaluations)
    else:
        raise ValueError(f"exploit method error")
    if len(source_indices) == 0:
        return

    #Overwrite models that underperform by exploit metric with weights from target models.
    for r in parameters:
        r[source_indices] = r[target_indices]

    for name, transform in config["hyperparameter_transforms"].items():
        value_raw = config[name + "_raw"]
        
        #add noise to cloned models.
        noise = config["hyperparameter_raw_perturb"][name].sample(
            (len(source_indices),)
        )

        value_raw[source_indices] = value_raw[target_indices] + noise
        config[name] = transform(value_raw)

        log[name].append(config[name])
        

config_local = dict(config)
log = defaultdict(list)
pbt_init(config_local, log)
print(config_local["learning_rate"], config_local["learning_rate_raw"])
print(log)
optimizer = AdamW(policy.parameters())
optimizer.update_config(config_local)
print(optimizer.config)


def reinforce_step(
    config: dict,
    episode_data: dict,
    optimizer: Optimizer,
    policy: torch.nn.Module,
) -> torch.Tensor:
    optimizer.zero_grad()
    
    actions, observations, rewards = (
        episode_data[key]
        for key in ("actions", "observations", "rewards")
    )

    discounted_returns = get_discounted_returns(config, rewards)

    logits = get_logits(
        config,
        observations[:, :-1],
        policy
    )
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    logits = torch.gather(
        logits,
        2,
        actions[..., None]
    )[..., 0]

    loss = -(discounted_returns * logits).mean(dim=-1).sum()
    loss.backward()
    optimizer.step()


reinforce_step(config, stochastic_episode_data, optimizer, policy)
new_deterministic_episode_data = get_episode_data(config, env, policy)
print(new_deterministic_episode_data["rewards"].sum(dim=-1).mean().cpu().numpy())

def reinforce(
    config: dict,
    env: gym.vector.VectorEnv,
    optimizer: Optimizer,
    policy: torch.nn.Module,
):
    ensemble_shape = config["ensemble_shape"]
    if len(ensemble_shape) != 1:
        raise ValueError(f"The number of dimensions in the ensemble shape should be 1 for the  population size, but it is {len(ensemble_shape)}")

    best_max_return = -torch.inf
    config_local = dict(config)
    gumbel = torch.distributions.Gumbel(
        torch.tensor(0., device=config["device"]),
        torch.tensor(1., device=config["device"])
    )
    log = defaultdict(list)
    steps_without_improvement = 0

    pbt_init(config_local, log)
    optimizer.update_config(config_local)

    progress_bar = tqdm.trange(config["steps_num"])

    for step_id in progress_bar:
        with torch.no_grad():
            episode_data = get_episode_data(
                config,
                env,
                policy,
                gumbel=gumbel
            )

        reinforce_step(
            config,
            episode_data,
            optimizer,
            policy
        )

        if (step_id + 1) % config["eval_interval"] == 0:
            print("Commencing evaluation.")
            eval_progress_bar = tqdm.trange(config["welch_sample_size"])
            with torch.no_grad():
                evaluations = torch.stack([
                    get_episode_data(
                        config_local,
                        env,
                        policy
                    )["rewards"].sum(dim=-1)
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

                evaluation_mean_max = evaluations_mean.max().cpu().item()
                make_video(
                    config_local,
                    policy,
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
                        optimizer.get_parameters()
                    )

    for key, value in log.items():
        if isinstance(value, list):
            log[key] = torch.stack(value)

    return log

policy = get_mlp(
    config,
    env.single_observation_space.shape[0],
    env.single_action_space.n,
    2,
    128
)
optimizer = AdamW(policy.parameters())


config["exploit_method"] = "welch"
log_welch = reinforce(config, env, optimizer, policy)


config["exploit_method"] = "truncation"
log_trunc = reinforce(config, env, optimizer, policy)


def best_curve(log):
    evals = log["evaluations"]         
    mean_over_samples = evals.mean(dim=1)  
    best_per_step = mean_over_samples.max(dim=1).values
    return best_per_step.cpu()

welch_curve = best_curve(log_welch)
trunc_curve = best_curve(log_trunc)

plt.plot(welch_curve, label="Welch")
plt.plot(trunc_curve, label="Truncation")
plt.legend()
plt.xlabel("Evaluation step")
plt.ylabel("Best Mean Return")
plt.show()