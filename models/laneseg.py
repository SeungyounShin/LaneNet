import torch
import torch.nn as nn
from models.ERFNet import *
from models.coordconv import *

class LaneSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.coordconv1 = CoordConv(3,16,kernel_size=3,stride=1, padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = Encoder(in_channels=16)
        self.segmentation = Decoder(num_classes = 2)

    def forward(self, x):
        x = self.coordconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        encoded = self.encoder(x)
        segmentation = self.segmentation(encoded)

        return segmentation


if __name__ == "__main__":
    import time

    model = LaneSeg()
    x = torch.randn(2,3,512,256)

    startT = time.time()
    out = model(x)
    endT = time.time()


    print("forward time : ",endT - startT)
    print("output shape : ",out[0].shape,out[1].shape)
