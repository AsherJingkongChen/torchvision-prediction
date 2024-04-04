from dataclasses import dataclass
from torch import nn, optim, Tensor
from typing import Callable, Iterable

from .extensions.optim import Momentum
from .extensions.lr_scheduler import CosineLR


@dataclass
class TrainingHyperParameters:
    hidden_node_count: int
    activation_function: nn.Module
    weight_initializer: Callable[[Tensor], Tensor]
    loss_function: Callable[[Tensor, Tensor], Tensor]
    regularization_factor: float
    optimizer: optim.Optimizer
    learning_epochs: int
    learning_rate_scheduler: optim.lr_scheduler.LRScheduler | None
    normalizer: nn.modules.batchnorm._NormBase | None

    @staticmethod
    def DOMAIN():
        return {
            "hidden_node_count": (
                5,
                8,
                11,
            ),
            "activation_function": (
                nn.Tanh,
                nn.ReLU,
            ),
            "weight_initializer": (
                nn.init.normal_,
                nn.init.xavier_normal_,
                nn.init.kaiming_normal_,
            ),
            "loss_function": (
                # Loss functions
                nn.functional.mse_loss,
            ),
            "regularization_factor": (
                0.001,
                0.0001,
            ),
            "optimizer": (
                optim.SGD,
                Momentum,
                optim.Adam,
            ),
            "learning_epochs": (
                100,
                200,
                300,
            ),
            "learning_rate_scheduler": (
                None,
                CosineLR,
            ),
            "normalizer": (
                None,
                nn.BatchNorm1d,
            ),
        }

    @staticmethod
    def get_all_combination_count() -> int:
        from functools import reduce

        return reduce(
            lambda x, y: x * y,
            map(
                lambda values: len(values),
                TrainingHyperParameters.DOMAIN().values(),
            ),
        )

    @staticmethod
    def get_all_combinations() -> Iterable["TrainingHyperParameters"]:
        from itertools import product

        domain = TrainingHyperParameters.DOMAIN()
        return map(
            lambda values: TrainingHyperParameters(
                **{k: v for k, v in zip(domain, values)}
            ),
            product(*domain.values()),
        )
