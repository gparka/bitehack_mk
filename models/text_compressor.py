import torch
from torch import nn
import yaml
from yaml.loader import SafeLoader


class TextCompressor(nn.Module):
    def __init__(self, max_window):
        self.max_window: int = max_window
        super(TextCompressor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.max_window, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        ).to(self.device)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, self.max_window),
        ).to(self.device)

    def decode(self, encoded):
        return self.decoder(encoded)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        # encode and decode
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
