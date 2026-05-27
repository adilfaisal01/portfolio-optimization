"""
Test GRPOContinuousLoss on Pendulum-v1 (continuous action space).

Adapted from TorchRL's PPO tutorial, but:
  - No critic network (yeeted)
  - No GAE advantage
  - Advantage = group-relative across N parallel trajectories
  - Uses your GRPOContinuousLoss module
"""

import torch
import torch.nn as nn
from torch import optim

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, StepCounter, ObservationNorm
from torchrl.envs.utils import check_env_specs, set_exploration_type, ExplorationType
from torchrl.modules import TanhNormal, ProbabilisticActor

# Import YOUR loss module
from grpo import GRPOContinuousLoss

device = torch.device("cpu")
print(f"Device: {device}")

# ── Hyperparams ──────────────────────────────────────────
n_state = 3          # Pendulum: cos(theta), sin(theta), theta_dot
n_action = 1         # Pendulum: torque
num_cells = 64
lr = 3e-4
total_frames = 50_000
frames_per_batch = 2000   # steps per collector batch
sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
entropy_coeff = 0.01
gamma = 0.99
group_size = 8            # N parallel trajectories for group advantage

# ── Environment ──────────────────────────────────────────
base_env = GymEnv("Pendulum-v1", device=device)
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(max_steps=200),
    ),
)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
check_env_specs(env)

print(f"Action spec: {env.action_spec}")
print(f"Observation spec: {env.observation_spec}")

# ── Policy Network (actor only, no critic!) ──────────────
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net,
    in_keys=["observation"],
    out_keys=["loc", "scale"],
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec_unbatched.space.low,
        "high": env.action_spec_unbatched.space.high,
    },
    return_log_prob=True,
)

print(f"Policy module: {policy_module}")

# ── YOUR Loss Module ─────────────────────────────────────
loss_module = GRPOContinuousLoss(
    actor_network=policy_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=True,
    entropy_coeff=entropy_coeff,
    reduction="mean",
    beta=0.6,
)

optimizer = optim.Adam(loss_module.parameters(), lr)

# ── Collector & Replay Buffer ───────────────────────────
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

# ── Training Loop ───────────────────────────────────────
logs = []
episode_reward_mean_list = []

for i, tensordict_data in enumerate(collector):
    # tensordict_data has shape [frames_per_batch] — a bunch of steps
    
    # ── Reshape into group trajectories ──
    # Split into group_size trajectories of equal length
    # Each "trajectory" represents one macro bridge branch
    traj_length = frames_per_batch // group_size
    data_grouped = tensordict_data[:traj_length * group_size]  # trim to fit
    
    # Reshape: [group_size, traj_length] 
    # Each group member = one trajectory with its own total return
    data_grouped = data_grouped.reshape(group_size, traj_length)
    
    # ── Compute group-relative advantage ──
    # Total return per trajectory (sum of rewards over timesteps)
    # reward shape: [group_size, traj_length, 1]
    returns = data_grouped["next", "reward"].sum(dim=-2)  # [group_size, 1]
    
    # Group-relative advantage
    group_mean = returns.mean()
    group_std = returns.std() + 1e-8
    advantages = (returns - group_mean) / group_std  # [group_size, 1]
    
    # Broadcast advantage back to each step in the trajectory
    advantages = advantages.expand(-1, traj_length).unsqueeze(-1)  # [group_size, traj_length, 1]
    
    # Set advantage in the grouped tensordict
    data_grouped["advantage"] = advantages
    
    # Flatten back for replay buffer
    data_flat = data_grouped.reshape(-1)
    
    # ── Fill replay buffer ──
    data_view = data_grouped.reshape(-1)
    replay_buffer.empty()
    replay_buffer.extend(data_view.cpu())
    
    # ── Optimize ──
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size).to(device)
            
            # Get log probs from current policy for these actions/states
            loss_vals = loss_module(subdata)
            
            loss_value = loss_vals["loss_objective"]
            loss_value.backward()
            
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # ── Logging ──
    episode_rewards = tensordict_data["next", "episode_reward"][tensordict_data["next", "done"]]
    if len(episode_rewards) > 0:
        episode_reward_mean = episode_rewards.mean().item()
        episode_reward_mean_list.append(episode_reward_mean)
    else:
        episode_reward_mean = 0
    
    logs.append({
        "batch": i,
        "loss": loss_value.item(),
        "reward_mean": episode_reward_mean,
        "clip_ratio": loss_vals["clip_ratio"].item(),
        "entropy": loss_vals["entropy"].item(),
    })
    
    print(f"Batch {i:3d} | Loss: {loss_value.item():.3f} | "
          f"Reward: {episode_reward_mean:.1f} | "
          f"Clip: {loss_vals['clip_ratio'].item():.2f} | "
          f"Entropy: {loss_vals['entropy'].item():.2f}")
    
    if i >= 20:  # enough for a test
        break

print("\n✅ Training test complete!")
print(f"Final reward mean: {episode_reward_mean_list[-1] if episode_reward_mean_list else 'N/A'}")
