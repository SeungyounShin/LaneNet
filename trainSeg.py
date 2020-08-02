import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

from dataloader import *
from albumentations import *
from albumentations.pytorch.transforms import ToTensor
from models.laneseg import LaneSeg
from loss.loss import *
from loss.loss2 import *
from loss.lovasz import *

#basic setting
num_epochs = 300
print_iter = 50
batch_size = 10
vis_result= True
validation_ratio = 0.05
startlr = 1e-4
weight_decay = 5e-4

#model load
model = LaneSeg().cuda()
#model = torch.load("./ckpt/seg_model.pth").cuda()

model_name = model.__class__.__name__

#dataset load
aug = Compose([
             #HorizontalFlip(),
             OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                ISONoise(p=0.3),
                Downscale(scale_min=0.7, scale_max=0.9, interpolation=0, always_apply=False, p=0.5),
                Blur(blur_limit=3, p=0.1),], p=0.35),
             RandomBrightnessContrast(brightness_limit=(-0.3,0.4),p=0.33),
             ShiftScaleRotate(shift_limit=0.0125, scale_limit=0.1, rotate_limit=7,border_mode=cv2.BORDER_CONSTANT, p=0.2),
             OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                    ToSepia(p=0.08),], p=0.3),
             OneOf([RandomShadow(p=0.2),
                    RandomRain(blur_value=2,p=0.4),
                    RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.2, alpha_coef=0.25, p=0.2),], p=0.25),
             Cutout(num_holes=7, max_h_size=64, max_w_size=64,p=0.33),
             Resize(240,320, p=1),#Resize(368,640, p=1),
             ToTensor()
            ])


#train_dataset = KcitySeg(transform= aug)
train_dataset = Kookmin(transform= aug)
dataiter,dataset_len = len(train_dataset)//batch_size,len(train_dataset)
train_len = int(dataset_len*(1-validation_ratio))
#train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_len, dataset_len-train_len])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
#valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1,shuffle=True, num_workers=0)
print("trainset lenght :: ",len(train_dataset))

#loss
#criterion  = SoftDiceLoss()
#criterion = LovaszBSE()
#criterion = nn.CrossEntropyLoss()
criterion = SoftDiceLoss()
#criterion = FocalTversky_loss(apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,square=False,gamma=2.1)

softmax = nn.Softmax(dim=1)

#optim setting
optimizer = optim.AdamW(model.parameters(), lr=startlr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=startlr, max_lr=startlr*3, step_size_up=2000, mode='triangular2' , gamma=0.9994,cycle_momentum=False )
opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

global_iter = 0
for epoch in range(num_epochs):
    losses = list()
    print("optim lr : ",optimizer.param_groups[0]['lr'])
    for iteration,sample in enumerate(train_loader):
        global_iter+=1

        img, label = sample #torch.Size([4, 368, 640, 3]) torch.Size([4, 368, 640]) torch.Size([4, 368, 640])
        label = label.float()
        img,label = img.float().cuda(),label.cuda()

        out = model(img)

        #loss = criterion(out, label)
        loss = criterion(softmax(out), label)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(iteration%print_iter == 0):
            print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())
            print("  avg loss :: ",sum(losses)/len(losses))

            if True:
                #print( np.argmax(out[0].detach().cpu().numpy(), axis=0))
                out_plot = np.argmax(out[0].detach().cpu().numpy(), axis=0)
                img_0 = img[0].permute(1,2,0).squeeze().cpu().numpy()
                label_0 = label[0].detach().cpu().max(0)[1].numpy()

                plt.subplot(3,1,1)
                plt.imshow(img_0 )
                plt.subplot(3,1,2)
                plt.imshow(label_0)
                plt.subplot(3,1,3)
                plt.imshow(out_plot)
                plt.show(block=False)
                plt.pause(3)
                plt.close()

    scheduler.step()
    if(epoch %5 ==0):
        #torch.save(model,"./ckpt/seg_"+str(epoch)+".pth")
        pass

torch.save(model,"./ckpt/seg_model_kookmin.pth")
