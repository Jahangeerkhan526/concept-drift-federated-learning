import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from model import get_model, get_parameters, set_parameters
from utils import get_client_dataloader


class HARClient(fl.client.NumPyClient):
    """
    Federated Learning client for HAR classification.
    Each client trains on its own local data and sends
    model updates to the server.
    """

    def __init__(self, client_id, X, y, indices):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.indices = indices
        self.model = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        """Return current model parameters to server."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """
        Receive global model from server,
        train on local data, return updated parameters.
        """
        # Load global model parameters
        set_parameters(self.model, parameters)

        # Get local dataloader
        loader = get_client_dataloader(self.X, self.y, self.indices)

        # Train
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

        print(f"Client {self.client_id} trained — loss: {total_loss:.4f}")

        return get_parameters(self.model), len(self.indices), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local data and return loss and accuracy."""
        set_parameters(self.model, parameters)
        loader = get_client_dataloader(self.X, self.y, self.indices)

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