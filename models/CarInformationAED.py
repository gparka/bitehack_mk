import torch
from torch import nn


class CarInformationAED(nn.Module):
    def __init__(self, max_window):
        self.max_window: int = max_window
        super(CarInformationAED, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.id = torch.nn.Identity(max_window)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
        ).to(self.device)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
            torch.nn.ReLU(),
            torch.nn.Linear(max_window, max_window),
        ).to(self.device)

    def decode(self, encoded):
        return self.decoder(encoded)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        # encode and decode
        id_x = self.id(x)
        encoded = self.encode(x) + id_x
        decoded = self.decode(encoded)
        return encoded, decoded


