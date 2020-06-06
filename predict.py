import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataloader import TUSimple
from models.lanenet import LaneNet

#basic setting
batch_size = 1


#model load
model = torch.load("./ckpt/25.pth").cuda().eval()
model_name = model.__class__.__name__

#dataset load
dataset = TUSimple("/home/yo0n/바탕화면/TUsimple")
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
print("dataset lenght :: ",len(dataset))


for iteration,sample in enumerate(loader):
    img, binary_label , instance_label = sample
    img = img.permute(0,3,1,2).float()/255.
    binary_label = binary_label.view(-1,1,368,640).float()
    instance_label = instance_label.view(-1,1,368,640).float()
    img,binary_label,instance_label = img.cuda(),binary_label.cuda(),instance_label.cuda()

    binary_pred, instance_pred = model(img)

    binary_pred = np.argmax(F.softmax(binary_pred).squeeze().detach().cpu().numpy(), axis=0)
    instance_pred = instance_pred[0].detach().cpu().numpy()
    instance_pred = instance_pred*binary_pred

    img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.subplot(3,1,2)
    plt.imshow(binary_pred)
    plt.subplot(3,1,3)
    plt.imshow(instance_pred[0])
    plt.show()
