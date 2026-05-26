from tensordict.nn import ProbabilisticTensorDictSequential
from torchrl.objectives.common import LossModule
import torch
from tensordict.utils import NestedKey
from collections.abc import Mapping
from dataclasses import dataclass
from torch import distributions as d

class GRPOTrading(LossModule):
    @dataclass
    class _AcceptedKeys:
        """You control what keys your loss reads from the TensorDict"""
        action: NestedKey = "action"
        sample_log_prob: NestedKey = "sample_log_prob"
        advantage: NestedKey = "advantage"

    def __init__(self,
        actor_network: ProbabilisticTensorDictSequential,
        entropy_bonus: bool=True,
        gamma:float | None=None,
        normalize_advantage:bool=False,
        functional:bool=True,
        actor:ProbabilisticTensorDictSequential=None,
        clip_values: float | None=None,
        device:torch.device | None=None,
        entropy_coeff:float | Mapping[NestedKey, float]| None=None,
        reduction:str | None=None,
        samples_mc_entropy:int=1,
        **kwargs
        
    ):
        super().__init__()
        if actor is not None:
            actor_network = actor
            del actor

        if reduction is None:
            reduction = "mean"

        if device is None:
                    try:
                        device = next(self.parameters()).device
                    except (AttributeError, StopIteration):
                        device = getattr(
                            torch, "get_default_device", lambda: torch.device("cpu")
                        )()
        # Set default value if None
        if entropy_coeff is None:
            entropy_coeff = 0.01

        self.entropy_coeff=entropy_coeff
        self.convert_to_functional(actor_network,"actor_network")
        self.entropy_bonus=entropy_bonus
        self.gamma=gamma
        self.clip_value=torch.tensor(clip_values,device=device)
        self._functional=functional
        self.reduction=reduction
        self.samples_mc_entropy=samples_mc_entropy
    @property
    def functional(self):
        return self._functional

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
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
                    f"Entropy not implemented for {type(dist)} or is not finite. Using Monte Carlo sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            with (
                set_composite_lp_aggregate(False)
                if isinstance(dist, CompositeDistribution)
                else contextlib.nullcontext()
            ):
                log_prob = dist.log_prob(x)
                if is_tensor_collection(log_prob):
                    if isinstance(self.tensor_keys.sample_log_prob, NestedKey):
                        log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
                    else:
                        log_prob = log_prob.select(*self.tensor_keys.sample_log_prob)

            entropy = -log_prob.mean(0)
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)
        

    
    
    
        
        
       

