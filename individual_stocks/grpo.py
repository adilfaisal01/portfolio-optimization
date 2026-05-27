import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import is_tensor_collection, TensorDictBase, TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.objectives.common import LossModule


class GRPOContinuousLoss(LossModule):
    """Group Relative Policy Optimization for continuous actions.

    No critic network — the group of parallel trajectories is the baseline.
    Advantage is computed externally as (R_i - mu_group) / sigma_group
    and passed in via the TensorDict.

    Input TensorDict shapes (batch_size=[B] where B = group_size * traj_length):
        'action':           [B, action_dim]    — continuous actions taken
        'sample_log_prob':  [B, 1]             — log probs from old policy
        'advantage':        [B, 1]             — group-relative z-score
        'observation':      [B, obs_dim]       — states (for dist rebuild)
    """

    @dataclass
    class _AcceptedKeys:
        action: NestedKey = "action"
        sample_log_prob: NestedKey = "sample_log_prob"
        advantage: NestedKey = "advantage"

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        entropy_coeff: float = 0.01,
        reduction: str = "mean",
        device: Optional[torch.device] = None,
        beta: float = 0.08,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.convert_to_functional(actor_network, "actor_network")

        self.clip_epsilon = clip_epsilon
        self.entropy_bonus = entropy_bonus
        self.entropy_coeff = entropy_coeff
        self.reduction = reduction
        self.beta = beta

        self._keys = self._AcceptedKeys()
        self.samples_mc_entropy = 1

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor:
        """Compute entropy of action distribution.

        Input:
            dist:      distribution with batch_shape [B]
            adv_shape: torch.Size([B, 1]) for broadcasting

        Returns:
            [B, 1] entropy per sample

        Tries analytical entropy first, falls back to MC sampling.
        """
        try:
            entropy = dist.entropy()  # -> [B]
            if not entropy.isfinite().all():
                del entropy
                if VERBOSE:
                    torchrl_logger.info(
                        "Entropy is not finite. Using Monte Carlo sampling."
                    )
                raise NotImplementedError
        except NotImplementedError:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)} or is not finite. "
                    "Using Monte Carlo sampling."
                )
            # MC estimate: sample actions, compute -mean(log_prob)
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))  # -> [1, B, action_dim]
            else:
                x = dist.sample((self.samples_mc_entropy,))    # -> [1, B, action_dim]
            log_prob = dist.log_prob(x)                        # -> [1, B]
            if is_tensor_collection(log_prob):
                if isinstance(self._keys.sample_log_prob, NestedKey):
                    log_prob = log_prob.get(self._keys.sample_log_prob)
                else:
                    log_prob = log_prob.select(*self._keys.sample_log_prob)
            entropy = -log_prob.mean(0)                        # -> [B]
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)  # -> [B, 1]

    def _log_weight(self, tensordict: TensorDictBase) -> tuple:
        """Compute importance sampling ratio and KL approx.

        Returns:
            dist:       distribution with batch_shape [B]
            log_weight: [B, 1] — log(pi_theta / pi_theta_old) for each transition
            kl_approx:  scalar — D_KL(pi_old || pi_theta) approximation
        """
        # Load functional params into the module before computing current distribution.
        # convert_to_functional() extracts params into actor_network_params (a TensorDict).
        # The optimizer steps on those params, so we must copy them back before get_dist().
        with self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)  # batch_shape: [B]
        action = tensordict.get(self._keys.action)      # -> [B, action_dim]
        old_log_prob = tensordict.get(self._keys.sample_log_prob)  # -> [B, 1]

        # dist.log_prob squeezes last dim: [B, action_dim] -> [B]
        # Unsqueeze to match old_log_prob shape [B, 1]
        new_log_prob = dist.log_prob(action).unsqueeze(-1)  # -> [B, 1]
        log_weight = new_log_prob - old_log_prob            # -> [B, 1]

        # KL approx: E[exp(log_weight) - 1 - log_weight] = KL(pi_old || pi_theta)
        kl_approx = (log_weight.exp() - 1 - log_weight).mean()  # scalar

        return dist, log_weight, kl_approx

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute GRPO loss.

        1. Get importance weight from current policy vs old policy
        2. Clip the ratio to [1-eps, 1+eps] for stability
        3. Take the tighter bound: min(ratio * A, clip(ratio) * A)
        4. Add entropy bonus for exploration
        5. Add KL penalty to prevent policy from drifting too far
        """
        dist, log_weight, kl_approx = self._log_weight(tensordict)
        # dist:       batch_shape [B]
        # log_weight: [B, 1]
        # kl_approx:  scalar

        ratio = log_weight.exp()                       # -> [B, 1]
        adv = tensordict.get(self._keys.advantage)     # -> [B, 1]

        # Clipped surrogate objective
        loss_term_1 = adv * ratio                                       # -> [B, 1]
        clipping_methods = torch.clamp(ratio, 1 - self.clip_epsilon,
                                       1 + self.clip_epsilon)           # -> [B, 1]
        loss_term_2 = clipping_methods * adv                            # -> [B, 1]
        loss = -torch.min(loss_term_1, loss_term_2)                     # -> [B, 1]

        # Reduction + entropy bonus + KL penalty
        if self.reduction == "mean":
            loss = loss.mean()  # scalar
            if self.entropy_bonus:
                entropy = self._get_entropy(dist, adv.shape).mean()  # scalar
                loss = loss - self.entropy_coeff * entropy
            else:
                entropy = torch.tensor(0.0)
            loss = loss + self.beta * kl_approx

        elif self.reduction == "sum":
            loss = loss.sum()  # scalar
            if self.entropy_bonus:
                entropy = self._get_entropy(dist, adv.shape).sum()  # scalar
                loss = loss - self.entropy_coeff * entropy
            else:
                entropy = torch.tensor(0.0)

        else:  # "none"
            entropy = torch.tensor(0.0)

        # Monitor: fraction of ratios hitting the clip bounds
        clip_low = (ratio < 1 - self.clip_epsilon).float().mean()
        clip_high = (ratio > 1 + self.clip_epsilon).float().mean()

        return TensorDict(
            {
                "loss_objective": loss,  # scalar
            },
            [],
        ).set("entropy", entropy  # scalar
        ).set("clip_low", clip_low  # scalar
        ).set("clip_high", clip_high)  # scalar

    def set_keys(self, **kwargs) -> None:
        """Remap input keys (e.g. loss.set_keys(advantage='group_adv'))."""
        for key, value in kwargs.items():
            if hasattr(self._keys, key):
                setattr(self._keys, key, value)
