from pprint import pprint
from torch import float32, int32, Generator, no_grad, nn, tensor
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from ..data_construction.download_builtin import construct_KMNIST
from .environment import request_best_device
from .hyper_parameters import TrainingHyperParameters
from .test import test_model
from .top_k import TopK
from .train import train_model

# Define the settings of the experiment
DEVICE = request_best_device()
SNAPSHOT_INDEX: int = 0
TOP_MODEL_MAXIMUM: int = 5
RANDOM_SEED: int = 96
DATA_TRAIN, DATA_TEST, _ = random_split(
    dataset=construct_KMNIST(),
    lengths=[0.0008, 0.0002, 1 - 0.0008 - 0.0002],
    generator=Generator().manual_seed(RANDOM_SEED),
)
DATA_TRAIN = DataLoader(DATA_TRAIN, batch_size=1 << 14)
DATA_TEST = DataLoader(DATA_TEST, batch_size=1 << 14)
FEATURE_COUNT = (
    tensor(
        next(iter(DATA_TEST))[0].shape[1:],
        device=DEVICE,
    )
    .prod(dtype=int32)
    .item()
)

progress_bar = tqdm(
    desc="Training with all hyper-parameters",
    total=TrainingHyperParameters.get_all_combination_count()
    * max(TrainingHyperParameters.DOMAIN()["learning_epochs"]),
    leave=True,
)
top_models = TopK(k=TOP_MODEL_MAXIMUM)

# Train on all hyper-parameters and collect the top K models
with progress_bar:
    for hyper_parameters in list(TrainingHyperParameters.get_all_combinations())[-2:]:
        model = train_model(
            data_train=DATA_TRAIN,
            feature_count=FEATURE_COUNT,
            device=DEVICE,
            hyper_parameters=hyper_parameters,
            progress_bar=progress_bar,
        )
        loss_val = test_model(
            data_test=DATA_TEST,
            model=model,
            device=DEVICE,
            loss_function=hyper_parameters.loss_function,
        )

        # Update the top K models
        top_models.update((loss_val, (hyper_parameters, model)), key=lambda t: t[0])

# Show outputs
print(f"Top {top_models.k} hyper-parameters: ")
for loss_val, (hyper_parameters, model) in top_models:
    pprint((loss_val, (hyper_parameters, model)))

# Save the top K models
from torch import save as save_model
