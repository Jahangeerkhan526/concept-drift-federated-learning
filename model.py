import numpy as np
import torch
import torch.nn as nn


class HARModel(nn.Module):
    """
    Simple feedforward neural network for HAR classification.
    Input: 561 sensor features
    Output: 6 activity classes
    """

    def __init__(self, input_size=561, hidden_size=128, num_classes=6):
        super(HARModel, self).__init__()
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
    """Return a fresh instance of the model."""
    return HARModel()


def get_parameters(model):
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Load parameters into model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)