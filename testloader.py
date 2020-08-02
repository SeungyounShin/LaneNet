from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os,json
import numpy as np
import cv2
from utils.customImageLoader import imload
from skimage import filters
import re

warnings.filterwarnings("ignore")

class CustomLaneTest(Dataset):
    def __init__(self, path, fps ,transform = None):
        self.path = path
        self.LINE_SIZE = 30
        self.fps = fps
        self.transform = transform
        self.images = [path + i for i in os.listdir(self.path) if i!=".DS_Store"]
        self.images.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.length = len(self.images)
        tmp = list()
        for i in range(self.length//fps):
            tmp.append(self.images[i*fps])
        self.images = tmp

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = imload(image_path) #(360, 640, 3)
        image=np.pad(image, ((4,4), (0,0), (0, 0)), 'constant')

        image = (image/255.) #- np.array([0.485, 0.456, 0.406])

        return image

if __name__=="__main__":
    import random

    dataset = CustomLaneTest('/home/yo0n/바탕화면/LaneNet-master/test_frames/')
    img = dataset[random.randint(0,len(dataset)-1)]

    print("test image shaep : ", img.shape)
    plt.imshow(img + np.array([0.485, 0.456, 0.406]))
    plt.show()
