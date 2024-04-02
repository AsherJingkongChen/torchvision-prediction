from torch.optim import SGD
from typing import Optional


class Momentum(SGD):
    __doc__ = SGD.__doc__

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
