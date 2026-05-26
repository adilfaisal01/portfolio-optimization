from tensordict.nn import ProbabilisticTensorDictSequential
from torchrl.objectives.common import LossModule
import torch
from tensordict.utils import NestedKey
from collections.abc import Mapping
from dataclasses import dataclass


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
        **kwargs
        
    ):
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
        
        # entropy_coef has been removed in v0.11
        if "entropy_coef" in kwargs:
            raise TypeError(
                "'entropy_coef' has been removed in torchrl v0.11. Please use 'entropy_coeff' instead."
            )

        # Set default value if None
        if entropy_coeff is None:
            entropy_coeff = 0.01
        
        super().__init__()

