from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
class Trainer(ABC):
    def __init__(self, config):
        self.oprtimizer = create_optimizer()

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def create_optimizer(self, optimizer: Dict[str, Any]):
        pass

    @abstractmethod
    def create_dataloader(self):


if __name__ == "__main__":
    print(torch.cuda.is_available())