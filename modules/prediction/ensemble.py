from torch import device, mean, nn, no_grad, stack, Tensor
from torch.utils.data import DataLoader
from typing import Callable, Iterable


def predict_ensemble_average(
    data_input: Tensor,
    models: Iterable[nn.Module],
) -> Tensor:
    """
    Predict the target values `Y` of the input data `X` using the ensemble of models

    ## Returns
    - predictions (`torch.Tensor`)
        - The predicted target values

    ## Details
    - Uses the average of the predictions of all models as the final prediction
    """

    predictions = stack([model(data_input) for model in models], dim=0)
    average_prediction = mean(predictions, dim=0)
    return average_prediction


def test_ensemble(
    data_test: DataLoader,
    models: list[nn.Module],
    device: device,
    loss_function: Callable[[Tensor, Tensor], Tensor],
) -> float:
    """
    Test an ensemble of models with the given arguments

    ## Returns
    - Validation loss (`float`)
        - Adopts the mean of validation losses on `data_test`
    """

    # Evaluate the models
    models = [model.to(device=device).eval() for model in models]
    with no_grad():
        loss_sum = 0.0
        for i, data in enumerate(data_test):
            X, Y = data
            X, Y = X.to(device=device), Y.to(device=device)

            loss_sum += loss_function(
                predict_ensemble_average(X, models),
                Y,
            ).item()
        loss_mean = loss_sum / (i + 1)

    # Return the validation loss
    return loss_mean
