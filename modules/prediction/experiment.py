from pathlib import Path
from pprint import pprint
from torch import Generator, int32, load, nn, save, tensor
from torch.types import Device
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data_construction.download_builtin import construct_KMNIST
from .ensemble import test_ensemble
from .environment import request_best_device, request_snapshot_path
from .hyper_parameters import TrainingHyperParameters
from .test import test_model
from .top_k import TopK
from .train import train_model


# Define the settings of the experiment

SNAPSHOT_NUMBER: int | None = 2000
"""
Select the snapshot to load.
If `None` or Falsy, a new snapshot will be created.

It allows faster inference by loading the pre-trained models.
"""

DEVICE: Device = request_best_device()
RANDOM_SEED: int = 96
ENSEMBLE_COUNT: int = 5
DATA = construct_KMNIST()
DATA_COUNT: int = 2000
TAG_COUNT = 10
DATA_TRAIN_RATIO: float = 0.80
STOP_CRITERIA_THRESHOLD_LOSS: float = 5e-3

# Auto generated settings

DATA_TRAIN_COUNT = int(DATA_TRAIN_RATIO * DATA_COUNT)
DATA_TEST_COUNT = DATA_COUNT - DATA_TRAIN_COUNT
DATA_TRAIN, DATA_TEST, _ = random_split(
    dataset=DATA,
    lengths=[
        DATA_TRAIN_COUNT,
        DATA_TEST_COUNT,
        len(DATA) - DATA_COUNT,
    ],
    generator=Generator().manual_seed(RANDOM_SEED),
)
DATA_TRAIN = DataLoader(DATA_TRAIN, batch_size=min(max(DATA_COUNT, 1000), 20000))
DATA_TEST = DataLoader(DATA_TEST, batch_size=min(max(DATA_COUNT, 1000), 20000))
FEATURE_COUNT = (
    tensor(
        next(iter(DATA_TEST))[0].shape[1:],
        device=DEVICE,
    )
    .prod(dtype=int32)
    .item()
)

_ = None

print("Settings of the experiment:")
pprint(
    {
        "Data range of X": [DATA[0][0].min().item(), DATA[0][0].max().item()],
        "Data range of Y": [DATA[0][1].min().item(), DATA[0][1].max().item()],
        "Data shape of X": tuple(DATA[0][0].shape),
        "Data shape of Y": tuple(DATA[0][1].shape),
        "Data type of X": DATA[0][0].dtype,
        "Data type of Y": DATA[0][1].dtype,
        "Device": DEVICE.type,
        "Dimension of X": FEATURE_COUNT,
        "Dimension of Y": TAG_COUNT,
    }
)
pprint(DATA[0][0].shape)
print()

def train_and_test_all() -> list[tuple[nn.Module, TrainingHyperParameters, float]]:
    """
    Train and test on all hyper-parameters and collect the top K models

    ## Returns
    - Top K entries (`List[Tuple[nn.Module, TrainingHyperParameters, float]]`)
        - An entry contains the model, training hyper-parameters and validation loss.
    """

    progress_bar = tqdm(
        desc="Training with all hyper-parameters",
        dynamic_ncols=True,
        leave=True,
        total=sum(1 for _ in TrainingHyperParameters.get_all_combinations()),
        mininterval=0.1,
        bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
    )
    top_models = TopK(k=ENSEMBLE_COUNT)

    with progress_bar:
        for hyper_parameters in TrainingHyperParameters.get_all_combinations():
            model = train_model(
                data_train=DATA_TRAIN,
                feature_count=FEATURE_COUNT,
                tag_count=TAG_COUNT,
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


# Load the top models or train and test models with all hyper-parameters
snapshot_path = request_snapshot_path(SNAPSHOT_NUMBER)
top_models_snapshot_path = Path(snapshot_path) / "top_models.zip"

if top_models_snapshot_path.is_file():
    print(f'Loading the top models from "{top_models_snapshot_path}"')

    top_models = load(top_models_snapshot_path)

elif top_models_snapshot_path.exists():
    raise OSError(f"{top_models_snapshot_path} should be a file")
else:
    top_models = train_and_test_all()

    print(f'Saving the top models at "{top_models_snapshot_path}"')

    for entry in top_models:
        entry[0].to(device="cpu")
    save(top_models, top_models_snapshot_path)

# Ensemble the top models
hyper_parameters_base = [entry[1] for entry in top_models]
validation_losses_base = [float(f"{entry[2]:.4f}") for entry in top_models]
# validation_loss_ensemble, prediction_rate_ensemble = test_ensemble(
#     data_test=DATA_TEST,
#     models=[entry[0] for entry in top_models],
#     device=DEVICE,
#     loss_function=nn.functional.mse_loss,
# )

# # Show the output
# print(f"Hyper-parameters of the base models: ")
# pprint(hyper_parameters_base)
# print(f"Validation loss of the base models: {validation_losses_base}")
# print(f"Validation loss of the ensemble: {validation_loss_ensemble:.5f}")
# print(f"Prediction rate of the ensemble: {prediction_rate_ensemble:.5f}")

# Uses the best hyper-parameters to train the model with different stop criterias
best_hyper_parameters = hyper_parameters_base[0]
best_hyper_parameters_without_learning_epochs = TrainingHyperParameters(
    learning_epochs=None,
    **{k: v for k, v in best_hyper_parameters.__dict__.items() if k != "learning_epochs"}
)

# Show the best hyper-parameters
print(f"Best hyper-parameters: ")
pprint(best_hyper_parameters)
assert not best_hyper_parameters_without_learning_epochs.learning_epochs

progress_bar = tqdm(
    desc="Training with the best hyper-parameters on different stop criterias",
    dynamic_ncols=True,
    leave=True,
    total=3,
    mininterval=0.1,
    bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
)

with progress_bar:
    # Train the model stopping on epoch
    model_stop_on_epoch = train_model(
        data_train=DATA_TRAIN,
        feature_count=FEATURE_COUNT,
        tag_count=TAG_COUNT,
        device=DEVICE,
        hyper_parameters=best_hyper_parameters,
        progress_bar=progress_bar,
    )
    validation_loss_stop_on_epoch = test_model(
        data_test=DATA_TEST,
        model=model_stop_on_epoch,
        device=DEVICE,
        loss_function=best_hyper_parameters.loss_function,
    ) if model_stop_on_epoch else None

    # Train the model stopping on threshold of loss
    model_stop_on_threshold_loss = train_model(
        data_train=DATA_TRAIN,
        feature_count=FEATURE_COUNT,
        tag_count=TAG_COUNT,
        device=DEVICE,
        hyper_parameters=best_hyper_parameters_without_learning_epochs,
        progress_bar=progress_bar,
        threshold_loss=STOP_CRITERIA_THRESHOLD_LOSS,
    )
    validation_loss_stop_on_threshold_loss = test_model(
        data_test=DATA_TEST,
        model=model_stop_on_threshold_loss,
        device=DEVICE,
        loss_function=best_hyper_parameters_without_learning_epochs.loss_function,
    ) if model_stop_on_threshold_loss else None

    # Train the model stopping on epoch and threshold of loss
    model_stop_on_epoch_and_threshold_loss = train_model(
        data_train=DATA_TRAIN,
        feature_count=FEATURE_COUNT,
        tag_count=TAG_COUNT,
        device=DEVICE,
        hyper_parameters=best_hyper_parameters,
        progress_bar=progress_bar,
        threshold_loss=STOP_CRITERIA_THRESHOLD_LOSS,
    )
    validation_loss_stop_on_epoch_and_threshold_loss = test_model(
        data_test=DATA_TEST,
        model=model_stop_on_epoch_and_threshold_loss,
        device=DEVICE,
        loss_function=best_hyper_parameters.loss_function,
    ) if model_stop_on_epoch_and_threshold_loss else None

print(
    f"Validation loss of the model stopping on epoch: "
    f"{validation_loss_stop_on_epoch:.5f}"
)
print(
    f"Validation loss of the model stopping on threshold of loss: "
    f"{validation_loss_stop_on_threshold_loss:.5f}"
)
if model_stop_on_epoch_and_threshold_loss:
    print(
        f"Validation loss of the model stopping on epoch and threshold of loss: "
        f"{validation_loss_stop_on_epoch_and_threshold_loss:.5f}"
    )
else:
    print(
        f"Model stopping on epoch and threshold of loss is not acceptable"
    )
