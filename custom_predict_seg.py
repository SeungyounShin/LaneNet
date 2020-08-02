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
pathOut = 'kookmin.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outv = cv2.VideoWriter(pathOut ,fourcc, 30.0, (320,496)) #368

#model load
model = torch.load("./ckpt/seg_model_kookmin.pth").eval().cuda()
model_name = model.__class__.__name__

#dataset load
dataset =  CustomLaneTest('/home/yo0n/바탕화면/LaneNet-master/kookmin/', 15)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=0)
print("dataset lenght :: ",len(dataset))

frame_array = list()

for iteration,sample in tqdm(enumerate(loader)):
    img = sample
    img = img.permute(0,3,1,2).float()
    img = img.cuda()
    img_h, img_w = img.shape[2], img.shape[3]

    start =time.time()
    out = model(img)
    end = time.time()

    output = np.argmax(F.softmax(out).squeeze().detach().cpu().numpy(), axis=0)
    outputRGB = np.stack([output,output,output],axis=-1)

    outputRGB[:,:,0][outputRGB[:,:,0] == 1] = 0
    outputRGB[:,:,1][outputRGB[:,:,1] == 1] = 255
    outputRGB[:,:,2][outputRGB[:,:,2] == 1] = 0


    img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()
    img = np.asarray(img, np.float64)
    outputRGB = np.asarray(outputRGB, np.float64)

    imgFinal = cv2.addWeighted(img ,1,outputRGB,0.1,0, dtype=cv2.CV_64F) #+ np.array([0.485, 0.456, 0.406])
    imgFinal = np.uint8(imgFinal*255.)
    imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)

    h,w = imgFinal.shape[0],imgFinal.shape[1]
    h /= 2
    #src =np.float32([[0,h],[604,h], [0,0], [w, 0]])
    #dst =np.float32([[285,h],[356,h], [0,0], [w, 0]])
    #src = np.float32([[262,220],[300,220],[76,346],[520,327]])
    #dst = np.float32([[0,0],[0,420],[320,0],[320,420]])
    src =np.float32([[262,220],[385,220], [71,363], [w, h]])
    dst =np.float32([[100,0],[600,0], [100,355], [600, 355]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    ROI = imgFinal[120:, ]
    warpROI = cv2.warpPerspective(ROI, M, (int(w),int(h) ))
    #plt.imshow(warpROI)
    #plt.show()
    maskROIplot = cv2.warpPerspective(np.uint8(outputRGB[:, 0:w]), M, (int(w),int(h) ))
    maskROI = cv2.warpPerspective(np.uint8(output[:, 0:w]), M, (int(w),int(h) ))

    #plt.imshow(np.uint8(maskROI))
    mh = maskROI.shape[0]
    divide = 10
    for i in range(divide):
        start = int(mh/divide*i)
        maskIndX, maskIndY = np.where(maskROI[start:int(mh/divide*(i+1)), :]==1)
        try:
            cntx,cnty = int(maskIndX.mean()), int(maskIndY.mean())
            maskROIplot = cv2.circle(maskROIplot, (cnty,start+cntx), 2, (0,0,255), -1)
        except:
            pass

        #plt.scatter(maskIndY.mean(), int(mh/divide*i)+maskIndX.mean(), s=1,c='r')
    #plt.show()


    warpimg_h = warpROI.shape[0]
    warpmask_h = maskROIplot.shape[0]

    frame = np.zeros((img_h+warpimg_h+warpmask_h, img_w, 3))
    frame[:warpimg_h,:,:] = maskROIplot/255.
    frame[warpimg_h:img_h+warpimg_h,:,:] = imgFinal/255.
    frame[img_h+warpimg_h:,:,:] = warpROI/255.

    frame_array.append(np.uint8(frame*255.))
    #print(frame_array[-1].shape)
    #plt.imshow(frame_array[-1])
    #plt.show()

for i in range(len(frame_array)):
    #img = cv2.resize(frame_array[i],(320,496) )
    outv.write(frame_array[i])
outv.release()
