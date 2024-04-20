# Torch-vision Prediction

## Introduction

Predict the label of some images from the built-in datasets of `torchvision` using a neural network.

## Prerequisites

- Python `3.10` or above

## Experiment

Please open your terminal and change the working directory to this project folder, and run these commands:

1. Install the required packages:

```shell
python3 -m pip install -Ur requirements.txt
```

2. Run the experiment:

```shell
python3 run.py
```

### Details

To switch snapshots, you can change the `SNAPSHOT_NUMBER` variable in the file `modules/prediction/experiment.py`:

```python
SNAPSHOT_NUMBER: int | None = 2000
"""
Select the snapshot to load.
If `None` or Falsy, a new snapshot will be created.

It allows faster inference by loading the pre-trained models.
"""
```

It is set to `2000` by default. You can set it to other values like `1000`, `9000`, etc.

## Outputs

The outputs stand for the execution results of the experiment (See `modules/prediction/experiment.py`). The output contains the settings of the experiment and models and the validation results.

1. \[Default\] Snapshot 2000, Dataset: `KMNIST Dataset`, Data count: `2000`, Task: `Classification`, X is real number (Pixel analogy data) and Y is one-hot encoded probability vector with 10 classes.

    ```plaintext
    Settings of the experiment:
    {'Data range of X': [0.0, 1.0],
    'Data range of Y': [0.0, 1.0],
    'Data shape of X': (1, 28, 28),
    'Data shape of Y': (10,),
    'Data type of X': torch.float32,
    'Data type of Y': torch.float32,
    'Device': 'mps',
    'Dimension of X': 784,
    'Dimension of Y': 10}
    torch.Size([1, 28, 28])

    Loading the top models from "snapshots/index/2000/top_models.zip"
    Best hyper-parameters: 
    TrainingHyperParameters(hidden_node_count=11,
                            activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                            weight_initializer=<function xavier_normal_ at 0x1051b8670>,
                            loss_function=<function mse_loss at 0x10511e050>,
                            regularization_factor=0.0001,
                            optimizer=<class 'torch.optim.adam.Adam'>,
                            learning_epochs=300,
                            learning_rate_scheduler=None,
                            normalizer=None)
    Training with the best hyper-parameters on different stop criterias: 100%|██| [01:03<00:00]
    Validation loss of the model stopping on epoch: 0.03696
    Validation loss of the model stopping on threshold of loss: 0.03654
    Model stopping on epoch and threshold of loss is not acceptable
    ```

## Attributions

Though all the datasets are built-in `torchvision`:

- "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
  - License: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  - Repository: [https://github.com/rois-codh/kmnist](https://github.com/rois-codh/kmnist)
