from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer


class CosineLR(CosineAnnealingLR):
    __doc__ = CosineAnnealingLR.__doc__

    def __init__(
        self,
        optimizer: Optimizer,
        T_max=20,
        eta_min=0,
        last_epoch=-1,
        verbose="deprecated",
    ):
        super().__init__(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )
