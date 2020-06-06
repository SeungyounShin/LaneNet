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
print_iter = 50
batch_size = 4
vis_result= True
validation_ratio = 0.05
startlr = 1e-4

#model load
model = LaneNet().cuda()
model_name = model.__class__.__name__

#dataset load
train_dataset = TUSimple("/home/yo0n/바탕화면/TUsimple")
dataiter,dataset_len = len(train_dataset)//batch_size,len(train_dataset)
train_len = int(dataset_len*(1-validation_ratio))
train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_len, dataset_len-train_len])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1,shuffle=True, num_workers=0)
print("trainset lenght :: ",len(train_dataset))

#loss
binary_criterion  = FocalLoss( gamma=2.1)
cluster_criterion = Clustering()

#optim setting
optimizer = optim.RMSprop(model.parameters(), lr=startlr, weight_decay=1e-5, momentum=0)
scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=startlr, max_lr=startlr*3, step_size_up=2000, mode='triangular2' , gamma=0.9994,cycle_momentum=False )
opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

global_iter = 0
for epoch in range(num_epochs):
    losses = list()
    binary_losses  = list()
    cluster_losses = list()
    print("optim lr : ",optimizer.param_groups[0]['lr'])
    for iteration,sample in enumerate(train_loader):
        global_iter+=1

        img, binary_label , instance_label = sample #torch.Size([4, 368, 640, 3]) torch.Size([4, 368, 640]) torch.Size([4, 368, 640])
        img = img.permute(0,3,1,2).float()/255.
        binary_label = binary_label.view(-1,1,368,640).float()
        instance_label = instance_label.view(-1,1,368,640).float()
        img,binary_label,instance_label = img.cuda(),binary_label.cuda(),instance_label.cuda()

        binary_pred, instance_pred = model(img)

        binary_loss     = binary_criterion(binary_pred, binary_label )
        clustering_loss = cluster_criterion(instance_pred, binary_label, instance_label)

        loss =  clustering_loss + binary_loss
        losses.append(loss.item())

        if(iteration%print_iter == 0):
            print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())
            print("  avg loss :: ",sum(losses)/len(losses))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
