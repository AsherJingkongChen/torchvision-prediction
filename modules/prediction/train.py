from torch import device, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .hyper_parameters import TrainingHyperParameters


def train_model(
    data_train: DataLoader,
    device: device,
    feature_count: int,
    tag_count: int,
    hyper_parameters: TrainingHyperParameters,
    progress_bar: tqdm = None,
) -> nn.Module:
    """
    Train a new model with the given arguments

    ## Returns
    - model (`torch.nn.Module`)
    """

    # Define the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_count, hyper_parameters.hidden_node_count),
        *(
            [hyper_parameters.normalizer(hyper_parameters.hidden_node_count)]
            if hyper_parameters.normalizer
            else []
        ),
        hyper_parameters.activation_function(),
        nn.Linear(hyper_parameters.hidden_node_count, tag_count),
        nn.Softmax(dim=1)
    )

    # Initialize the weights of the model (1/2)
    def init_weights(module: nn.Module) -> None:
        """
        A helper function to initialize the weights of a `torch.nn.Module`
        """
        try:
            hyper_parameters.weight_initializer(module.weight)
            nn.init.zeros_(module.bias)
        except Exception:
            return

    # Initialize the weights of the model (2/2)
    model.apply(init_weights)

    # Define the optimizer and the learning rate scheduler
    optimizer = hyper_parameters.optimizer(
        model.parameters(),
        weight_decay=hyper_parameters.regularization_factor,  # In PyTorch, `weight_decay` is also called L2 penalty.
    )
    scheduler = (
        hyper_parameters.learning_rate_scheduler(optimizer)
        if hyper_parameters.learning_rate_scheduler
        else None
    )

    # Train the model (Per epoch)
    model = model.to(device=device).train()
    for _ in range(hyper_parameters.learning_epochs):
        # Train the model (Per batched data)
        for data in data_train:
            X, Y = data
            X, Y = X.to(device=device), Y.to(device=device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss = hyper_parameters.loss_function(model(X), Y)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Update the learning rate
        if scheduler:
            scheduler.step()

        if progress_bar:
            progress_bar.update()

    # Return the trained model
    return model
