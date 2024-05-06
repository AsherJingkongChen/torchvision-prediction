from torch import device, nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .hyper_parameters import TrainingHyperParameters


def train_model(
    data_train: DataLoader,
    device: device,
    feature_count: int,
    tag_count: int,
    hyper_parameters: TrainingHyperParameters,
    progress_bar: tqdm,
    threshold_dist: float | None = None,
) -> nn.Module | None:
    """
    Train a new model with the given arguments

    ## Returns
    - model (`torch.nn.Module | None`)
        - An acceptable model or `None` if the model is not acceptable
            - A model can be not acceptable if the validation loss
              is not less than the threshold loss within the given learning epochs.
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

    epoch = 0
    while True:
        # Train the model (Per batched data)
        for data in data_train:
            X, Y = data
            X, Y = X.to(device=device), Y.to(device=device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            P = model(X)
            dist: Tensor = (P - Y).abs()
            loss = hyper_parameters.loss_function(P, Y)

            # Check if the loss is lower than the threshold.
            # If so, return the model as it is acceptable.
            if threshold_dist and dist.mean() < threshold_dist:
                # Update the progress bar by 1
                progress_bar.update()

                return model

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Update the learning rate
        if scheduler:
            scheduler.step()

        # Update the epoch
        if hyper_parameters.learning_epochs:
            if epoch < hyper_parameters.learning_epochs:
                epoch += 1
            else:
                break

    # Update the progress bar by 1
    progress_bar.update()

    # If the threshold loss presents,
    # Return `None` as the model is not acceptable
    if threshold_dist:
        return None

    # Otherwise, return the model.
    else:
        return model
