from torch import device, no_grad, nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Callable


def test_model(
    data_test: DataLoader,
    model: nn.Module,
    device: device,
    loss_function: Callable[[Tensor, Tensor], Tensor],
) -> float:
    """
    Test a model with the given arguments

    ## Returns
    - Validation loss (`float`)
        - Adopts the mean of validation losses on `data_test`
    """

    # Evaluate the model
    loss_sum = 0.0

    model = model.to(device=device).eval()
    with no_grad():
        for i, data in enumerate(data_test):
            X, Y = data
            X, Y = X.to(device=device), Y.to(device=device)

            loss_sum += loss_function(model(X), Y).item()
        loss_mean = loss_sum / (i + 1)

    # Return the validation loss
    return loss_mean
