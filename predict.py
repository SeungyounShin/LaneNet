import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import MeanShift,DBSCAN

from dataloader import TUSimple
from models.lanenet import LaneNet

#basic setting
batch_size = 1


#model load
model = torch.load("./ckpt/25.pth",map_location=torch.device('cpu')).eval()
model_name = model.__class__.__name__

#dataset load
dataset = TUSimple("./tusimple_part")
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
print("dataset lenght :: ",len(dataset))


for iteration,sample in enumerate(loader):
    img, binary_label , instance_label = sample
    img = img.permute(0,3,1,2).float()/255.
    binary_label = binary_label.view(-1,1,368,640).float()
    instance_label = instance_label.view(-1,1,368,640).float()
    #img,binary_label,instance_label = img.cuda(),binary_label.cuda(),instance_label.cuda()

    start =time.time()
    binary_pred, instance_pred = model(img)
    end = time.time()

    binary_pred = np.argmax(F.softmax(binary_pred).squeeze().detach().cpu().numpy(), axis=0)
    instance_pred = instance_pred.squeeze().detach().cpu().numpy()
    mask = (instance_pred * np.stack([binary_pred, binary_pred], axis=0)) #(2, 368, 640)
    lanes = instance_pred[np.stack([binary_pred, binary_pred], axis=0) > 0].reshape(2,-1).transpose(1,0)

    start_clustering = time.time()
    #clustering = DBSCAN(eps=0.5, min_samples=0).fit(lanes)
    clustering = MeanShift(n_jobs=4).fit(lanes)
    end_clustering = time.time()
    print(clustering.labels_, np.unique(clustering.labels_))
    plt.scatter(lanes[:,0], lanes[:,1] , c=clustering.labels_)
    plt.show()

    img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()

    print("infer time : ",end-start)
    print("clustering time : ",end_clustering-start_clustering)
    plt.subplot(4,1,1)
    plt.imshow(img)
    plt.subplot(4,1,2)
    plt.imshow(binary_pred)
    plt.subplot(4,1,3)
    plt.imshow(mask[0,:,:])
    plt.subplot(4,1,4)
    plt.imshow(mask[1,:,:])
    plt.show()
