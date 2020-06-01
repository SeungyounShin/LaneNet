import torch
import torch.nn as nn
from ERFNet import *

class LaneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.embedding = Decoder(num_classes = 2) #2d embedding
        self.segmentation = Decoder(num_classes = 2)

    def forward(self, x):
        encoded = self.encoder(x)

        embedding = self.embedding(encoded)
        segmentation = self.segmentation(encoded)

        return embedding, segmentation


if __name__ == "__main__":
    import time

    lanenet = LaneNet()
    x = torch.randn(2,3,512,256)

    startT = time.time()
    out = lanenet(x)
    endT = time.time()


    print("forward time : ",endT - startT)
    print("output shape : ",out[0].shape,out[1].shape)
