from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch.utils.data import IterableDataset
from torch.optim import Optimizer


class Trainer(ABC):
    def __init__(self, config):
        self.optimizer: Optimizer = self.create_optimizer()
        self.dataloader: IterableDataset = self.create_dataloader()
        self.model = self.load_model()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def create_optimizer(self) -> Optimizer:
        pass

    @abstractmethod
    def create_dataloader(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass


if __name__ == "__main__":
    print(torch.cuda.is_available())
