import torch
from torch import nn
from torchvision.models import resnet
import torchvision.models


class CustomResnet(torch.nn.Module):
    def __init__(self):
        self.window = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(CustomResnet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.training = True
        for i, param in enumerate(self.resnet.parameters()):
            param.requires_grad = False
            if i == 55:
                break
        self.lin_layers = torch.nn.Sequential(
            nn.BatchNorm1d(
                1000, track_running_stats=True,
                momentum=0.04, device=self.device
            ),
            nn.ReLU(),
            nn.Linear(1000, 32),
            nn.BatchNorm1d(
                32, track_running_stats=True,
               momentum=0.04, device=self.device
            ),
            nn.ReLU(),
            nn.Linear(32, self.window)
        )

    def forward(self, x):
        xR = self.resnet(x)
        x = self.lin_layers(xR)
        return x


model = CustomResnet()
model = model.to(model.device)

upload_model = True
PATH_model = './modelAEDSTL.pth'

if upload_model:
    model.load_state_dict(torch.load(PATH_model, map_location=model.device))

if __name__ == "__main__":
    aaa = torch.zeros((5, 3, 512, 512)).to(model.device)
    y = model.forward(aaa)
    print(y.shape)
    #torch.save(model.state_dict(), PATH_model)
    print("dsdasdsads")