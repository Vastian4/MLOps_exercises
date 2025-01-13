import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d


class my_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

        #first layer
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.acti1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        #second layer
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.acti2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        #third layer
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.acti3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.maxpool3 = nn.MaxPool2d(2)

        #linear layer
        self.out = Linear(14*14*128, out_features=10, bias=False)

    
    def forward(self, x):
        
        #error messages
        if x.ndim != 4:
            raise ValueError("Input should be a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Samples should have shape [1, 28, 28]")

        x = self.acti1(self.conv1(x))
        x = self.drop1(x)
        
        x = self.acti2(self.conv2(x))
        x = self.drop2(x)
        
        x = self.acti3(self.conv3(x))
        x = self.maxpool3(self.drop3(x))
        
        x = x.reshape(x.shape[0], 14*14*128)

        x = self.out(x)

        return x

if __name__ == "__main__":
    model = my_model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")