from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch import nn
from torch.utils.data import IterableDataset
from torch.optim import Optimizer


class Trainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.training_parameters = config["training_parameters"]
        self.optimizer: Optimizer = self.create_optimizer(config["optimizer"])
        self.dataloader: IterableDataset = self.create_dataloader(config["dataloader"])
        self.model: nn.Module = self.load_model(config["load_config"])
        self.logger = self.attach_logger(config["loger_config"])

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def create_optimizer(self, optimizer_config: Dict[str, Any]) -> Optimizer:
        pass

    @abstractmethod
    def create_dataloader(self, dataloader_config: Dict[str, Any]) -> IterableDataset:
        pass

    @abstractmethod
    def load_model(self, load_config: Dict[str, Any]) -> nn.Module:
        pass

    @abstractmethod
    def save_model(self, load_config: Dict[str, Any]):
        pass

    @abstractmethod
    def attach_logger(self, logger_config: Dict[str, Any]):
        pass


if __name__ == "__main__":
    print(torch.cuda.is_available())
