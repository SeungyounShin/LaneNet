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
import hdbscan

from dataloader import TUSimple
from models.lanenet import LaneNet

#basic setting
batch_size = 1


#model load
model = torch.load("./ckpt/10.pth").eval().cuda()
model_name = model.__class__.__name__

#dataset load
dataset = TUSimple("/home/yo0n/바탕화면/TUsimple")
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
print("dataset lenght :: ",len(dataset))

hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=150,allow_single_cluster=False)
MeanShift_cluster = MeanShift(bandwidth=0.05, bin_seeding=True, min_bin_freq=80,cluster_all=False, max_iter=3)

for iteration,sample in enumerate(loader):
    img, binary_label , instance_label = sample
    img = img.permute(0,3,1,2).float()
    binary_label = binary_label.view(-1,1,368,640).float()
    instance_label = instance_label.view(-1,1,368,640).float()
    img,binary_label,instance_label = img.cuda(),binary_label.cuda(),instance_label.cuda()

    start =time.time()
    binary_pred, instance_pred = model(img)
    end = time.time()

    binary_pred = np.argmax(F.softmax(binary_pred).squeeze().detach().cpu().numpy(), axis=0)
    instance_final = np.zeros_like(binary_pred)
    embedding_dim = int(instance_pred.shape[1])
    instance_pred = instance_pred.squeeze().detach().cpu().numpy() #(embedding dim , 368, 640)
    binary_mask = np.stack([binary_pred for i in range(instance_pred.shape[0])], axis=0)
    mask = instance_pred * binary_mask #(2, 368, 640)
    lanes = instance_pred[binary_mask > 0].reshape(embedding_dim,-1).transpose(1,0)

    start_clustering = time.time()
    #clustering = DBSCAN(eps=0.25, min_samples=8,n_jobs=-1).fit(lanes)
    cluster_labels = MeanShift_cluster.fit(lanes).labels_
    #cluster_labels = hdbscan_cluster.fit_predict(lanes)
    end_clustering = time.time()

    #print(cluster_labels, np.unique(cluster_labels))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(lanes[:,0], lanes[:,1] ,lanes[:,2] , c=cluster_labels)
    #plt.show()

    instance_final[binary_pred > 0] = cluster_labels + 1
    #(4, 368, 640) (4, 368, 640) (42646,) (21323,)
    lanes_cnt = np.unique(cluster_labels+1).max()

    plt.imshow(instance_final)
    for i in range(1,lanes_cnt+1):
        tmp_lane = (instance_final == i)
        indexs = np.where(tmp_lane>0)
        ys = indexs[0]
        xs = indexs[1]
        z = np.polyfit(xs,ys,deg=2)
        p = np.poly1d(z)

        xp = np.linspace(0,640,100)

        plt.plot(xp, p(xp) , '-')

    plt.xlim(0,640)
    plt.ylim(0,360)
    plt.show()


    img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()

    print("infer time : ",end-start)
    print("clustering time : ",end_clustering-start_clustering)
    plt.subplot(3,1,1)
    plt.imshow(img + np.array([0.485, 0.456, 0.406]))
    plt.subplot(3,1,2)
    plt.imshow(binary_pred)
    plt.subplot(3,1,3)
    plt.imshow(instance_final)
    plt.show()
