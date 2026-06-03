import numpy as np
import torch
import torch.nn as nn


class GasModel(nn.Module):
    """
    Feedforward neural network for Gas Sensor classification.
    Input: 128 sensor features
    Output: 6 gas classes
    """

    def __init__(self, input_size=128, hidden_size=128, num_classes=6):
        super(GasModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def get_model():
    return GasModel()


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)