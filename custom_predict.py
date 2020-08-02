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
import cv2,io

from testloader import CustomLaneTest
from models.lanenet import LaneNet

#basic setting
batch_size = 1
pathOut = 'video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(pathOut ,fourcc, 20.0, (640,368))

#model load
model = torch.load("./ckpt/10.pth").eval().cuda()
model_name = model.__class__.__name__

#dataset load

dataset =  CustomLaneTest('/home/yo0n/바탕화면/LaneNet-master/kookmin/', 15)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=0)
print("dataset lenght :: ",len(dataset))

hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=150,allow_single_cluster=False)
frame_array = list()

for iteration,sample in tqdm(enumerate(loader)):
    img = sample
    img = img.permute(0,3,1,2).float()
    img = img.cuda()

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
    #cluster_labels = MeanShift(n_jobs=4, min_bin_freq=1000 ,bin_seeding=True, max_iter=10).fit(lanes).labels_
    cluster_labels = hdbscan_cluster.fit_predict(lanes)
    end_clustering = time.time()

    instance_final[binary_pred > 0] = cluster_labels + 1
    #(4, 368, 640) (4, 368, 640) (42646,) (21323,)

    img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()

    #print("infer time : ",end-start)
    #print("clustering time : ",end_clustering-start_clustering)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.imshow(img+ np.array([0.485, 0.456, 0.406]))
    #ax.imshow(np.ma.masked_where(instance_final < 1, instance_final))
    #plt.show()

    img = img+ np.array([0.485, 0.456, 0.406])
    mask = np.stack([instance_final for i in range(3)], axis=-1)
    lanes = np.unique(instance_final).shape[0]
    img[mask >= 1] = mask[mask >= 1]/lanes/2
    frame_array.append(np.uint8(img*255.))

for i in range(len(frame_array)):
    img = cv2.resize(frame_array[i],(640,368) )
    out.write(img)
out.release()
