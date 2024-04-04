# Torch-vision Prediction

## Introduction

Predict the label of some images from the built-in datasets of `torchvision` using a neural network.

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

## Output

1. \[Default\] Snapshot 2000, Data count: `2000`

```plaintext
Loading the top models from "snapshots/index/2000/top_models.zip"
Validation loss of the base models: 4.38, 4.46, 4.50, 4.51, 4.52
Validation loss of the ensemble: 3.866
```
