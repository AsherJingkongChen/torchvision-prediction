from torch import argmax, device, float32, mean, nn, no_grad, stack, Tensor
from torch.utils.data import DataLoader
from typing import Callable, Iterable


def predict_ensemble_average(
    data_input: Tensor,
    models: Iterable[nn.Module],
) -> Tensor:
    """
    Predict the target values `Y` of the input data `X` using the ensemble of models

    ## Returns
    - average_predictions (`torch.Tensor`)
        - The avarage predicted target values from the ensemble of models
    """

    predictions = stack([model(data_input) for model in models], dim=0)
    average_predictions = mean(predictions, dim=0)
    return average_predictions


def test_ensemble(
    data_test: DataLoader,
    models: list[nn.Module],
    device: device,
    loss_function: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Test an ensemble of models with the given arguments

    ## Returns
    - Validation loss (`float`)
        - Uses the mean of validation losses on `data_test`
    - Prediction rate (`float`)
        - The rate of correct predictions on `data_test`
    """

    # Evaluate the models
    models = [model.to(device=device).eval() for model in models]
    with no_grad():
        loss_sum = 0.0
        rate_sum = 0
        for i, data in enumerate(data_test):
            X, Y = data
            X, Y = X.to(device=device), Y.to(device=device)

            Yp = predict_ensemble_average(X, models)
            loss_sum += loss_function(Yp, Y).item()
            rate_sum += (
                (argmax(Yp, dim=1) == argmax(Y, dim=1)).mean(dtype=float32).item()
            )
        loss_mean = loss_sum / (i + 1)
        rate_mean = rate_sum / (i + 1)

    return loss_mean, rate_mean
