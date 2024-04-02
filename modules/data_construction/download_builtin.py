def construct_KMNIST():
    from pathlib import Path
    from torch.utils.data import ConcatDataset
    from torchvision.datasets import KMNIST
    from torchvision.transforms import ToTensor
    from torch import tensor, get_default_dtype

    def to_tensor_target(target: int):
        return tensor([target], dtype=get_default_dtype())

    return ConcatDataset(
        (
            KMNIST(
                root=(Path(__file__) / "../../../datasets").resolve(),
                train=True,
                download=True,
                transform=ToTensor(),
                target_transform=to_tensor_target,
            ),
            KMNIST(
                root=(Path(__file__) / "../../../datasets").resolve(),
                train=False,
                download=True,
                transform=ToTensor(),
                target_transform=to_tensor_target,
            ),
        )
    )
