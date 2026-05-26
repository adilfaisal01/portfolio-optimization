from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import ProbabilisticTensorDictSequential
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule


class GRPOContinuousLoss(LossModule):
    """Group Relative Policy Optimization for continuous actions.

    No critic network — the group of parallel trajectories is the baseline.
    Advantage is computed externally as (R_i - mu_group) / sigma_group
    and passed in via the TensorDict.
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

        self._keys = self._AcceptedKeys()

    def _log_weight(self, tensordict: TensorDictBase) -> tuple:
        """Compute importance sampling log-weight from current policy."""
        dist = self.actor_network.get_dist(tensordict)
        action = tensordict.get(self._keys.action)
        old_log_prob = tensordict.get(self._keys.sample_log_prob)

        new_log_prob = dist.log_prob(action)
        log_weight = new_log_prob - old_log_prob

        # KL approximation for monitoring
        kl_approx = (log_weight.exp() - 1 - log_weight).mean()

        return dist, log_weight, kl_approx

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute GRPO loss.

        Implement the clipped surrogate objective with group-relative advantages.
        
        Steps:
        1. Get importance weight via _log_weight()
        2. Pull advantage from tensordict using self._keys.advantage
        3. Compute clipped surrogate: min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
        4. Apply reduction (mean/sum/none)
        5. Add entropy bonus if enabled
        6. Return TensorDict with loss_objective, entropy, clip_fraction, etc.
        """
        dist, log_weight,kl_approx= self._log_weight(tensordict)
        ratio=log_weight.exp()
        adv=tensordict.get(self._keys.advantage)
        loss_term_1=adv*ratio
        clipping_methods= torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        loss_term2=clipping_methods*adv
        loss=torch.min(loss_term_1,loss_term2)

        #reduction of loss function
        
        
        if self.entropy_bonus:
            entropy=dist.entropy().mean()
            loss=loss-self.entropy_coeff*entropy
        else:
            entropy=torch.tensor(0.0)

        return TensorD

        
        
        
        raise NotImplementedError("You write this bestie 💅")

    def set_keys(self, **kwargs) -> None:
        """Remap input keys (e.g. loss.set_keys(advantage='group_adv'))."""
        for key, value in kwargs.items():
            if hasattr(self._keys, key):
                setattr(self._keys, key, value)
