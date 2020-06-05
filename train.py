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
from loss.loss import *

#basic setting
num_epochs = 30
print_iter = 100
batch_size = 2
vis_result= True
validation_ratio = 0.05
startlr = 1e-4

#model load
model = LaneNet()
model_name = model.__class__.__name__

#dataset load
train_dataset = TUSimple("./tusimple_part")
dataiter,dataset_len = len(train_dataset)//batch_size,len(train_dataset)
train_len = int(dataset_len*(1-validation_ratio))
train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_len, dataset_len-train_len])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1,shuffle=True, num_workers=0)
print("trainset lenght :: ",len(train_dataset))
