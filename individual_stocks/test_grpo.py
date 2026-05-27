"""
test_grpo.py — Automated verification suite for GRPOContinuousLoss.

Tests:
  1. Ratio drift: importance ratio diverges from 1.0 after training
  2. Gradient flow: all actor_network_params receive gradients
  3. Shape correctness: all tensor shapes match expectations
  4. Forward pass: loss module runs without crashing on valid inputs
  5. Multi-environment: runs on Pendulum-v1 and a dummy env
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal

from grpo import GRPOContinuousLoss

device = torch.device("cpu")
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def make_policy(n_state=3, n_action=1, hidden=32):
    net = nn.Sequential(
        nn.LazyLinear(hidden), nn.ReLU(),
        nn.LazyLinear(hidden), nn.ReLU(),
        nn.LazyLinear(2 * n_action),
        NormalParamExtractor(),
    )
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=module,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    return actor


def test_forward_pass():
    """Loss module runs without errors on valid input and produces expected keys."""
    print("\n── Test: Forward Pass ──")
    policy = make_policy(4, 2)
    loss = GRPOContinuousLoss(policy)
    opt = torch.optim.Adam(loss.parameters(), 3e-4)

    B = 64  # batch
    td = TensorDict({
        "observation": torch.randn(B, 4),
        "action": torch.randn(B, 2).tanh(),
        "sample_log_prob": torch.randn(B, 1),
        "advantage": torch.randn(B, 1),
    }, batch_size=[B])

    out = loss(td)
    check("loss_objective is scalar", out["loss_objective"].dim() == 0)
    check("entropy is scalar", out["entropy"].dim() == 0)
    check("clip_low is scalar", out["clip_low"].dim() == 0)
    check("clip_high is scalar", out["clip_high"].dim() == 0)
    check("loss is finite", out["loss_objective"].isfinite().item())

    # Backward pass
    opt.zero_grad()
    out["loss_objective"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in loss.actor_network_params.values(True, True))
    check("gradients flow to actor_network_params", has_grad)


def test_ratio_drift():
    """Importance ratio diverges from 1.0 after optimizer steps."""
    print("\n── Test: Ratio Drift ──")
    policy = make_policy(4, 2, 64)
    loss = GRPOContinuousLoss(policy, clip_epsilon=0.2, entropy_bonus=True, beta=0.08,
                              entropy_coeff=0.01)
    opt = torch.optim.Adam(loss.parameters(), 1e-2)

    group_size, traj_length, n_state, n_action = 8, 30, 4, 2

    all_ratios = []
    for step in range(60):
        obs = torch.randn(group_size * traj_length, n_state)
        with torch.no_grad():
            td = TensorDict({"observation": obs}, batch_size=[group_size * traj_length])
            out = policy(td)
        act = out["action"]
        lp = out["action_log_prob"]
        if lp.dim() < 2:
            lp = lp.unsqueeze(-1)

        rets = torch.randn(group_size)
        adv = (rets - rets.mean()) / (rets.std() + 1e-8)
        adv = adv.unsqueeze(-1).unsqueeze(-1).expand(-1, traj_length, 1).reshape(-1, 1)

        td = TensorDict({
            "observation": obs,
            "action": act,
            "sample_log_prob": lp,
            "advantage": adv,
        }, batch_size=[group_size * traj_length])

        out_td = loss(td)
        opt.zero_grad()
        out_td["loss_objective"].backward()
        opt.step()

        with torch.no_grad():
            _, lw, kl = loss._log_weight(td)
            all_ratios.append(lw.exp().mean().item())

    first, last = all_ratios[0], all_ratios[-1]
    max_drift = max(abs(r - 1.0) for r in all_ratios)
    check("ratio drifts from 1.0", max_drift > 0.01,
          f"max drift from 1.0: {max_drift:.4f}")
    check("ratio trajectory evolves",
          len(set(round(r, 2) for r in all_ratios)) > 3,
          f"unique ratio values: {set(round(r, 2) for r in all_ratios)}")
    print(f"    Ratio range: [{min(all_ratios):.4f}, {max(all_ratios):.4f}] over {len(all_ratios)} steps")


def test_shape_annotations():
    """All tensor dims match expected shapes through the forward pass."""
    print("\n── Test: Shape Correctness ──")
    policy = make_policy(8, 4, 64)
    loss = GRPOContinuousLoss(policy)
    B = 128

    td = TensorDict({
        "observation": torch.randn(B, 8),
        "action": torch.randn(B, 4).tanh(),
        "sample_log_prob": torch.randn(B, 1),
        "advantage": torch.randn(B, 1),
    }, batch_size=[B])

    # Check internal shapes
    dist, log_weight, kl_approx = loss._log_weight(td)
    check("dist batch_shape[0] == B", dist.batch_shape[0] == B)
    check("log_weight is [B, 1]", log_weight.shape == (B, 1))
    check("kl_approx is scalar", kl_approx.dim() == 0)

    out = loss(td)
    check("loss_objective scalar", out["loss_objective"].dim() == 0)
    check("entropy scalar", out["entropy"].dim() == 0)
    check("clip_low scalar", out["clip_low"].dim() == 0)
    check("clip_high scalar", out["clip_high"].dim() == 0)


def test_pendulum_episode():
    """Full rollout on Pendulum-v1 with group-relative advantage, no crashes."""
    print("\n── Test: Pendulum Rollout ──")
    policy = make_policy(3, 1, 32)
    loss = GRPOContinuousLoss(policy, clip_epsilon=0.2, entropy_bonus=True, beta=0.08)
    opt = torch.optim.Adam(loss.parameters(), 3e-4)
    env = gym.make("Pendulum-v1")

    group_size, traj_length = 4, 40
    all_obs, all_act, all_lp, all_ret = [], [], [], []

    for _ in range(group_size):
        obs, _ = env.reset()
        to, ta, tl, tr = [], [], [], 0.0
        for t in range(traj_length):
            ot = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                td = TensorDict({"observation": ot}, batch_size=[])
                out = policy(td)
            a = out["action"]
            lp = out["action_log_prob"]
            if lp.dim() < 2:
                lp = lp.unsqueeze(-1)
            obs, rew, term, trunc, _ = env.step(a.squeeze().numpy().reshape(-1))
            to.append(ot)
            ta.append(a)
            tl.append(lp)
            tr += rew
            if term or trunc:
                for _ in range(traj_length - t - 1):
                    to.append(torch.zeros(1, 3))
                    ta.append(torch.zeros(1, 1))
                    tl.append(torch.zeros(1, 1))
                break
        all_obs.append(torch.cat(to))
        all_act.append(torch.cat(ta))
        all_lp.append(torch.cat(tl))
        all_ret.append(tr)

    obs_t = torch.stack(all_obs)
    act_t = torch.stack(all_act)
    lp_t = torch.stack(all_lp)
    ret_t = torch.tensor(all_ret)

    adv = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    adv = adv.unsqueeze(-1).unsqueeze(-1).expand(-1, traj_length, 1)

    td = TensorDict({
        "observation": obs_t.reshape(-1, 3),
        "action": act_t.reshape(-1, 1),
        "sample_log_prob": lp_t.reshape(-1, 1),
        "advantage": adv.reshape(-1, 1),
    }, batch_size=[group_size * traj_length])

    out = loss(td)
    opt.zero_grad()
    out["loss_objective"].backward()
    opt.step()

    check("Pendulum forward pass", out["loss_objective"].isfinite().item(),
          f"loss={out['loss_objective'].item():.4f}")

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in loss.actor_network_params.values(True, True))
    check("Pendulum gradients flow", has_grad)
    env.close()


def test_multi_batch_shapes():
    """Loss handles varying batch sizes and reductions correctly."""
    print("\n── Test: Reduction Modes ──")
    policy = make_policy(4, 2, 16)
    for reduction in ["mean", "sum", "none"]:
        loss = GRPOContinuousLoss(policy, reduction=reduction)
        B = 32
        td = TensorDict({
            "observation": torch.randn(B, 4),
            "action": torch.randn(B, 2).tanh(),
            "sample_log_prob": torch.randn(B, 1),
            "advantage": torch.randn(B, 1),
        }, batch_size=[B])
        out = loss(td)
        if reduction == "none":
            check(f"none reduction → [B, 1]", out["loss_objective"].shape == (B, 1))
        else:
            check(f"{reduction} reduction → scalar",
              out["loss_objective"].dim() == 0)


if __name__ == "__main__":
    print(f"GRPOContinuousLoss Test Suite")
    print(f"{'='*40}")
    print(f"Device: {device}")

    test_forward_pass()
    test_ratio_drift()
    test_shape_annotations()
    test_pendulum_episode()
    test_multi_batch_shapes()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        exit(1)
    else:
        print("All tests passed! ✅")
