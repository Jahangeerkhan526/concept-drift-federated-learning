import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from model_gas import get_model, get_parameters, set_parameters
from utils_gas import get_gas_dataloader


class GasClient(fl.client.NumPyClient):
    """
    Federated Learning client for Gas Sensor classification.
    Includes gradient history storage for DAAW.
    """

    def __init__(self, client_id, X, y, indices):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.indices = indices
        self.model = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gradient_history = []

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        params_before = [p.clone().detach() for p in self.model.parameters()]
        loader = get_gas_dataloader(self.X, self.y, self.indices)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        params_after = [p.clone().detach() for p in self.model.parameters()]
        gradient = np.concatenate([
            (after - before).cpu().numpy().flatten()
            for before, after in zip(params_before, params_after)
        ])
        self.gradient_history.append(gradient)

        print(f"Client {self.client_id} trained — loss: {total_loss:.4f}")
        return get_parameters(self.model), len(self.indices), {
            "client_id": self.client_id,
            "loss": total_loss
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loader = get_gas_dataloader(self.X, self.y, self.indices)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        print(f"Client {self.client_id} eval — accuracy: {accuracy:.4f}")
        return float(total_loss), len(self.indices), {"accuracy": accuracy}