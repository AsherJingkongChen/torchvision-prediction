def construct_KMNIST():
    from pathlib import Path
    from torch.utils.data import ConcatDataset
    from torchvision.datasets import KMNIST
    from torchvision.transforms import ToTensor
    from torch import get_default_dtype, tensor

    dtype = get_default_dtype()
    def to_target_tensor(target: int):
        return tensor([target], dtype=dtype)

    return ConcatDataset(
        (
            KMNIST(
                root=(Path(__file__) / "../../../datasets").resolve(),
                train=True,
                download=True,
                transform=ToTensor(),
                target_transform=to_target_tensor,
            ),
            KMNIST(
                root=(Path(__file__) / "../../../datasets").resolve(),
                train=False,
                download=True,
                transform=ToTensor(),
                target_transform=to_target_tensor,
            ),
        )
    )
