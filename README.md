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

## Output

1. \[Default\] Snapshot 2000, Data count: `2000`

```plaintext
Loading the top models from "snapshots/index/2000/top_models.zip"
Hyper-parameters of the base models: 
[TrainingHyperParameters(hidden_node_count=11,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x107fb83a0>,
                         loss_function=<function mse_loss at 0x107f19d80>,
                         regularization_factor=0.0001,
                         optimizer=<class 'torch.optim.adam.Adam'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None),
 TrainingHyperParameters(hidden_node_count=11,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x107fb83a0>,
                         loss_function=<function mse_loss at 0x107f19d80>,
                         regularization_factor=0.001,
                         optimizer=<class 'torch.optim.adam.Adam'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None),
 TrainingHyperParameters(hidden_node_count=11,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function kaiming_normal_ at 0x107fb8550>,
                         loss_function=<function mse_loss at 0x107f19d80>,
                         regularization_factor=0.0001,
                         optimizer=<class 'torch.optim.adam.Adam'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None),
 TrainingHyperParameters(hidden_node_count=11,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function xavier_normal_ at 0x107fb83a0>,
                         loss_function=<function mse_loss at 0x107f19d80>,
                         regularization_factor=0.0001,
                         optimizer=<class 'torch.optim.adam.Adam'>,
                         learning_epochs=200,
                         learning_rate_scheduler=None,
                         normalizer=None),
 TrainingHyperParameters(hidden_node_count=11,
                         activation_function=<class 'torch.nn.modules.activation.ReLU'>,
                         weight_initializer=<function kaiming_normal_ at 0x107fb8550>,
                         loss_function=<function mse_loss at 0x107f19d80>,
                         regularization_factor=0.001,
                         optimizer=<class 'torch.optim.adam.Adam'>,
                         learning_epochs=300,
                         learning_rate_scheduler=None,
                         normalizer=None)]
Validation loss of the base models: [0.0357, 0.0366, 0.0367, 0.0376, 0.0382]
Validation loss of the ensemble: 0.03355
Prediction rate of the ensemble: 0.80250
```

## Attributions

Though all the datasets are built-in `torchvision`:

- "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
  - License: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  - Repository: [https://github.com/rois-codh/kmnist](https://github.com/rois-codh/kmnist)
