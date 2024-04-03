from pathlib import Path
from pprint import pprint
from torch import float32, int32, Generator, load, nn, save, tensor
from torch.types import Device
from torch.utils.data import DataLoader, random_split

from ..data_construction.download_builtin import construct_KMNIST
from .environment import request_best_device, request_snapshot_path
from .hyper_parameters import TrainingHyperParameters
from .test import test_model
from .top_k import TopK
from .train import train_model


# Define the settings of the experiment
DEVICE: Device = request_best_device()
RANDOM_SEED: int = 96
ENSEMBLE_COUNT: int = 5
DATA = construct_KMNIST()
DATA_COUNT: int = 2000
DATA_TRAIN_RATIO: float = 0.80
DATA_TRAIN_COUNT = int(DATA_TRAIN_RATIO * DATA_COUNT)
DATA_TEST_COUNT = DATA_COUNT - DATA_TRAIN_COUNT
DATA_TRAIN, DATA_TEST, _ = random_split(
    dataset=DATA,
    lengths=[
        DATA_TRAIN_COUNT,
        DATA_TEST_COUNT,
        len(DATA) - DATA_TRAIN_COUNT - DATA_TEST_COUNT,
    ],
    generator=Generator().manual_seed(RANDOM_SEED),
)
DATA_TRAIN = DataLoader(DATA_TRAIN, batch_size=10000)
DATA_TEST = DataLoader(DATA_TEST, batch_size=10000)
FEATURE_COUNT = (
    tensor(
        next(iter(DATA_TEST))[0].shape[1:],
        device=DEVICE,
    )
    .prod(dtype=int32)
    .item()
)
SNAPSHOT_NUMBER: int | None = 1
"""
Select the snapshot to load.
If `None` or Falsy, a new snapshot will be created.

It allows faster inference by loading the pre-trained models.
"""

_ = None


def train_and_test_all() -> list[tuple[nn.Module, TrainingHyperParameters, float]]:
    """
    Train and test on all hyper-parameters and collect the top K models

    ## Returns
    - Top K entries (`List[Tuple[nn.Module, TrainingHyperParameters, float]]`)
        - An entry contains the model, training hyper-parameters and validation loss.
    """

    from tqdm.auto import tqdm

    progress_bar = tqdm(
        desc="Training with all hyper-parameters",
        dynamic_ncols=True,
        leave=True,
        total=sum(
            map(
                lambda p: p.learning_epochs,
                TrainingHyperParameters.get_all_combinations(),
            )
        ),
        mininterval=1,
    )
    top_models = TopK(k=ENSEMBLE_COUNT)

    with progress_bar:
        for hyper_parameters in TrainingHyperParameters.get_all_combinations():
            model = train_model(
                data_train=DATA_TRAIN,
                feature_count=FEATURE_COUNT,
                device=DEVICE,
                hyper_parameters=hyper_parameters,
                progress_bar=progress_bar,
            )
            validation_loss = test_model(
                data_test=DATA_TEST,
                model=model,
                device=DEVICE,
                loss_function=hyper_parameters.loss_function,
            )

            # Update the top K models
            top_models.update(
                (model, hyper_parameters, validation_loss),
                key=lambda t: t[2],
            )

    return list(top_models)


snapshot_path = request_snapshot_path(SNAPSHOT_NUMBER)
top_models_snapshot_path = Path(snapshot_path) / "top_models.zip"

if top_models_snapshot_path.is_file():
    print(f'Loading the top models from "{top_models_snapshot_path}"')
    top_models = load(top_models_snapshot_path)
    pprint(top_models)

    validation_loss = test_model(
        data_test=DATA_TEST,
        model=top_models[0][0],
        device=DEVICE,
        loss_function=top_models[0][1].loss_function,
    )
    print(f"Re-validation loss: {validation_loss}")

    from torch.func import stack_module_state

elif top_models_snapshot_path.exists():
    raise OSError(f"{top_models_snapshot_path} should be a file")
else:
    top_models = train_and_test_all()

    print(f'Saving the top models at "{top_models_snapshot_path}"')
    save(top_models, top_models_snapshot_path)
    pprint(top_models)
