"""Simple GRPO test on Pendulum-v1 — no TorchRL collector complexity."""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor

from torchrl.modules import ProbabilisticActor, TanhNormal

from grpo import GRPOContinuousLoss

device = torch.device("cpu")
print(f"Device: {device}")

# ── Hyperparams ──
group_size = 8       # N parallel trajectories
traj_length = 100     # steps per trajectory
n_state = 3
n_action = 1
num_cells = 64
lr = 3e-4
num_iterations = 1000

# ── Policy Network ──
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells), nn.Tanh(),
    nn.LazyLinear(num_cells), nn.Tanh(),
    nn.LazyLinear(2 * n_action),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    return_log_prob=True,
)

loss_module = GRPOContinuousLoss(policy_module, clip_epsilon=0.2, entropy_bonus=True, beta=0.08)
optimizer = torch.optim.Adam(loss_module.parameters(), lr)

# ── Environment ──
env = gym.make("Pendulum-v1")

reward_log = []

for iteration in range(num_iterations):
    # Collect N parallel trajectories
    all_observations = []
    all_actions = []
    all_log_probs = []
    all_returns = []

    for _ in range(group_size):
        obs, _ = env.reset(seed=np.random.randint(0, 10000))
        trajectory_obs = []
        trajectory_actions = []
        trajectory_log_probs = []
        total_return = 0

        for t in range(traj_length):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            td = TensorDict({"observation": obs_tensor}, batch_size=[])
            with torch.no_grad():
                td_out = policy_module(td)
            
            action = td_out["action"]
            log_prob = td_out["action_log_prob"]
            
            # Ensure consistent shape [1, 1]
            if log_prob.dim() < 2:
                log_prob = log_prob.unsqueeze(-1)
            
            # Step environment — TanhNormal gives [1, 1], squeeze to [1]
            obs, reward, terminated, truncated, _ = env.step(action.squeeze().numpy().reshape(-1))
            done = terminated or truncated
            
            trajectory_obs.append(obs_tensor)
            trajectory_actions.append(action)
            trajectory_log_probs.append(log_prob)
            total_return += reward
            
            if done:
                # Pad with zeros to keep fixed length
                remaining = traj_length - t - 1
                for _ in range(remaining):
                    trajectory_obs.append(torch.zeros(1, n_state))
                    trajectory_actions.append(torch.zeros(1, n_action))
                    trajectory_log_probs.append(torch.zeros(1, 1))
                break
        
        all_observations.append(torch.cat(trajectory_obs))
        all_actions.append(torch.cat(trajectory_actions))
        all_log_probs.append(torch.cat(trajectory_log_probs))
        all_returns.append(total_return)

    # ── Stack into group tensor ──
    obs_tensor = torch.stack(all_observations)     # [group, T, state]
    act_tensor = torch.stack(all_actions)          # [group, T, 1]
    # action_log_prob from ProbabilisticActor, remap to sample_log_prob
    lp_tensor = torch.stack(all_log_probs)         # [group, T, 1]
    ret_tensor = torch.tensor(all_returns)         # [group]

    # ── Group-relative advantage ──
    group_mean = ret_tensor.mean()
    group_std = ret_tensor.std() + 1e-8
    advantages = (ret_tensor - group_mean) / group_std  # [group]
    advantages = advantages.unsqueeze(-1).unsqueeze(-1).expand(-1, traj_length, 1)  # [group, T, 1]

    # ── Flatten for loss module ──
    td = TensorDict({
        "observation": obs_tensor.reshape(-1, n_state),
        "action": act_tensor.reshape(-1, n_action),
        "sample_log_prob": lp_tensor.reshape(-1, 1),
        "advantage": advantages.reshape(-1, 1),
    }, batch_size=[group_size * traj_length])

    # ── Forward pass + optimize ──
    loss_vals = loss_module(td)
    loss = loss_vals["loss_objective"]
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
    optimizer.step()

    # ── Log ──
    reward_log.append(ret_tensor.mean().item())
    if iteration % 10 == 0:
        print(f"Iter {iteration:3d} | Loss: {loss.item():.3f} | "
              f"Avg Return: {ret_tensor.mean().item():.1f} ± {ret_tensor.std().item():.1f} | "
              f"Clip: [{loss_vals['clip_low'].item():.2f}, {loss_vals['clip_high'].item():.2f}] | "
              f"Entropy: {loss_vals['entropy'].item():.2f}")

print(f"\n✅ Done! {num_iterations} iterations of GRPO on Pendulum-v1")
print(f"Avg return (last 10): {np.mean(reward_log[-10:]):.1f}")
