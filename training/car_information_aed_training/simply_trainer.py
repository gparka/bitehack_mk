from typing import Dict, Any

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import SGD, Optimizer
from torch.utils.data import IterableDataset
from models.CarInformationAED import CarInformationAED
from training.trainer import Trainer


class SimplyTrainer(Trainer):
    def __init__(self, model, dataloader, path_to_save):
        self.path_to_save = path_to_save
        self.model = model
        self.dataloader = dataloader
        self.epochs = 1000
        self.lr = 0.001
        self.momentum = 0.9
        self.optimizer = SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        self.loss = MSELoss()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            for step, x in enumerate(self.dataloader):
                x = x.cuda()
                encoded, decoded = self.model(x)
                loss_val = self.loss(decoded, x)
                if step == 0:
                    print(f"Epoch {epoch + 1} | loss_val {loss_val:.4f}")
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()


if __name__ == "__main__":
    # Test duuuuuuupa
    model = CarInformationAED(10)
    dataloader = [
        torch.zeros(10),
        torch.zeros(10),
        torch.zeros(10)
    ]
    trainer = SimplyTrainer(model, dataloader, ".")
    trainer.train()
