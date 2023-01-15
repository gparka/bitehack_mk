import torch
from torch.nn import MSELoss
from torch.optim import SGD

from Data.Cars.dataloader import get_dataloader, CarDataset
from models.CarInformationAED import CarInformationAED
from training.trainer import Trainer


class ResnetTrainer(Trainer):
    def __init__(self, model, aed, dataloader, path_to_save):
        self.path_to_save = path_to_save
        self.model = model
        self.aed = aed
        self.dataloader = dataloader
        self.epochs = 30
        self.lr = 0.01
        self.momentum = 0.9
        self.optimizer = SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        self.loss = MSELoss()

    def train(self):
        loss_sum = 0
        ep_cnt = 1
        loss_old = 123443212
        for epoch in range(1, self.epochs + 1):
            for step, x in enumerate(self.dataloader):
                x = x.cuda()
                x_latent = self.model(x)
                y = self.aed.decode(x_latent)
                loss_val = self.loss(y, x)
                loss_sum += loss_val.item()
                if step % 2000 == 0:
                    print(f"Epoch {epoch} | loss_val {loss_sum:.6f}")
                    torch.save(self.model.state_dict(), f"/home/sportv09/PycharmProjects/bitehack_mk/checkpoints/custom_resnet/custom_resnet_{ep_cnt}.pth")
                    if abs(loss_old/loss_sum - 1) < 0.025:
                        self.optimizer.zero_grad()
                        loss_val.backward()
                        self.optimizer.step()
                        break
                    loss_old = loss_sum
                    loss_sum = 0
                    ep_cnt += 1
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()



if __name__ == "__main__":
    #batch = 256
    #cars = CarDataset()
    #dataloader = get_dataloader(cars, batch)

    #window = 23
    #model = CarInformationAED(window)

    #trainer = SimplyTrainer(model, dataloader, ".")
    #trainer.train()
