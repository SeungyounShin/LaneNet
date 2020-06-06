from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os,json
import numpy as np
import cv2
from scipy.ndimage.morphology import grey_dilation
from scipy.interpolate import CubicSpline
#from utils.util import *
from skimage import filters

warnings.filterwarnings("ignore")

class TUSimple(Dataset):
    def __init__(self, path):
        self.path = path
        self.LINE_SIZE = 30
        sub    = [i for i in os.listdir(self.path) if i!=".DS_Store"]
        labels = [self.path + "/" + i for i in sub if i[-4:]=="json"]
        images_root_path = self.path + "/clips"
        images = list()
        self.labels = dict()
        images_folders = [self.path+"/clips/"+i for i in os.listdir(images_root_path) if i!=".DS_Store"]
        for imgs_folder in images_folders:
            for i in os.listdir(imgs_folder):
                if("DS" in i):
                    continue

                tmp_path = imgs_folder + "/" +i
                lst_of_imgs = [imgs_folder + "/" + i+"/"+j for j in os.listdir(tmp_path) if j=="20.jpg"]
                images += lst_of_imgs

        self.images = images
        for label_path in labels:
            with open(label_path,"r") as f:
                for i in f.readlines():
                    todict = json.loads(i[:-1])
                    label_img_name = todict['raw_file']
                    self.labels[label_img_name] = todict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        key_ind = image_path.split("/").index("clips")
        key_path = os.path.join( *image_path.split("/")[key_ind:])
        abs_path = self.path +"/"+os.path.join( *image_path.split("/")[key_ind:])

        label = self.labels[key_path]
        lanes_w = np.array(label['lanes'])
        lanes_h = np.array(label['h_samples'])
        lane_cnt = lanes_w.shape[0]

        image = plt.imread(image_path) #(720, 1280, 3)
        image=np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        hmap = np.zeros(image.shape[:2])

        lane_pair = list()
        point = 0
        for i in range(lane_cnt):
            mask = (lanes_w[i,:] * lanes_h) > 0
            xs = (lanes_w[i,:][mask]-8) /1280. * 640.
            ys = lanes_h[mask] /728. * 368.
            ys = np.clip(ys, 0, 639)
            for j in range(xs.shape[0]):
                try:
                    hmap[int(ys[j]), int(xs[j])] = 1
                except:
                    print(ys)
                    print(xs)
                if(j<xs.shape[0]-1):
                    cv2.line(hmap, (int(xs[j]), int(ys[j])), (int(xs[j+1]), int(ys[j+1])), (i+1, i+1, i+1),  self.LINE_SIZE//2)
                #hmap = draw_umich_gaussian(hmap, [int(xs[j]), int(ys[j])], 10)
                point+=1

        binary = np.where(hmap>0, 1, 0)
        instance = hmap

        show = False
        if show:
            plt.subplot(4,1,1)
            plt.imshow(image)
            plt.subplot(4,1,2)
            plt.imshow(image)
            plt.imshow(hmap, alpha=0.5)
            plt.subplot(4,1,3)
            plt.imshow(instance)
            plt.subplot(4,1,4)
            plt.imshow(binary)
            plt.show()

        return image, binary, instance

if __name__=="__main__":
    import random
    random.seed(a=None)
    dataset = TUSimple("/home/yo0n/바탕화면/TUsimple")
    o = dataset[random.randint(0,len(dataset)-1)]
    print(o[0].shape, o[1].shape, o[2].shape) # (368, 640) (368, 640)

    lanes = int(o[2].max())
    plt.subplot(lanes+2,1,1)
    plt.imshow(o[0])
    plt.subplot(lanes+2,1,2)
    plt.imshow(o[2])
    for i in range(lanes):
        plt.subplot(lanes+2,1,i+3)
        plt.imshow(o[2]==(i+1))
    plt.show()
