from pprint import pprint
from torch import float32, int32, Generator, no_grad, nn, tensor
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from ..data_construction.download_builtin import construct_KMNIST
from .environment import request_best_device
from .hyper_parameters import TrainingHyperParameters
from .top_k import TopK

# Define the settings of the experiment
DEVICE = request_best_device()
ENSEMBLE_COUNT: int = 5
RANDOM_SEED: int = 96
DATA_TRAIN, DATA_TEST = random_split(
    dataset=construct_KMNIST(),
    lengths=[0.8, 0.2],
    generator=Generator().manual_seed(RANDOM_SEED),
)
DATA_TRAIN = DataLoader(DATA_TRAIN, batch_size=1 << 14 >> 3)
DATA_TEST = DataLoader(DATA_TEST, batch_size=1 << 14 >> 3)

progress_bar = tqdm(
    desc="Training",
    total=TrainingHyperParameters.get_all_combination_count()
    * len(DATA_TRAIN)
    * TrainingHyperParameters.get_max_learning_epochs(),
    leave=True,
)
top_combinations = TopK(k=ENSEMBLE_COUNT)
feature_count = (
    tensor(
        next(iter(DATA_TEST))[0].shape[1:],
        device=DEVICE,
    )
    .prod(dtype=int32)
    .item()
)

# 1. Enumerate all hyper-parameters
# 2. update the top K combinations of training hyper-parameters
for HP in list(TrainingHyperParameters.get_all_combinations())[-1:]:
    # Define the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_count, HP.hidden_node_count, device=DEVICE),
        *(
            (HP.normalizer(HP.hidden_node_count, device=DEVICE),)
            if HP.normalizer
            else ()
        ),
        HP.activation_function(),
        nn.Linear(HP.hidden_node_count, 1, device=DEVICE),
    )

    def init_weights(module: nn.Module) -> None:
        """
        A helper function to initialize the weights of a `torch.nn.Module`
        """
        try:
            HP.weight_initializer(module.weight)
            nn.init.zeros_(module.bias)
        except Exception:
            return

    # Initialize the weights of the model
    model.apply(init_weights)

    # Define the optimizer and the learning rate scheduler
    optimizer = HP.optimizer(
        model.parameters(),
        weight_decay=HP.regularization_factor,  # In PyTorch, `weight_decay` is also called L2 penalty.
    )
    scheduler = (
        HP.learning_rate_scheduler(optimizer) if HP.learning_rate_scheduler else None
    )

    learning_epochs_step = (
        TrainingHyperParameters.get_max_learning_epochs() // HP.learning_epochs
    )

    # Train the model (Per epoch)
    model.train()
    for epoch in range(HP.learning_epochs):
        # Train the model (Per batched data)
        for DATA in DATA_TRAIN:
            X, Y = DATA
            X, Y = X.to(device=DEVICE), Y.to(device=DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss = HP.loss_function(model(X), Y)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Update the learning rate
        if scheduler:
            scheduler.step()

        progress_bar.update(learning_epochs_step)

    # Evaluate the model
    loss_mean = 0.0

    model.eval()
    with no_grad():
        for i, DATA in enumerate(DATA_TEST):
            X, Y = DATA
            X, Y = X.to(device=DEVICE), Y.to(device=DEVICE)

            loss_mean += HP.loss_function(model(X), Y).item()
        loss_mean = loss_mean / (i + 1)

    # Update the top K combinations of hyper-parameters
    top_combinations.update((loss_mean, HP), key=lambda t: t[0])

progress_bar.close()

# Show outputs
print(f"Top {ENSEMBLE_COUNT} hyper-parameters: ")
for entry in top_combinations:
    pprint(entry)
