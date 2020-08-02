from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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

class TUSimpleHNet(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.LINE_SIZE = 30
        self.transform = transform
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
        #image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        image = cv2.resize(image, dsize=(128,72), interpolation=cv2.INTER_AREA)

        lane_pair = list()
        point = 0
        for i in range(lane_cnt):
            mask = (lanes_w[i,:] * lanes_h) > 0
            xs = (lanes_w[i,:][mask]-8) / 10.
            ys = lanes_h[mask] / 10.
            ys = np.clip(ys, 0, 127)
            pair = np.stack([xs,ys])
            lane_pair.append(pair)

        image = image/255. - np.array([0.485, 0.456, 0.406])

        show = False
        if show:
            plt.subplot(2,1,1)
            plt.imshow(image + np.array([0.485, 0.456, 0.406]))
            plt.subplot(2,1,2)
            plt.imshow(image + np.array([0.485, 0.456, 0.406]))
            for i in range(lane_cnt):
                plt.scatter(lane_pair[i][0,:], lane_pair[i][1,:], s=5)
            plt.show()

        return image, lane_pair

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()

if __name__=="__main__":
    import random
    from albumentations import *

    aug = Compose([
                 HorizontalFlip(),
                 OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    ISONoise(p=0.3),
                    Blur(blur_limit=3, p=0.1),], p=0.35),
                 RandomBrightnessContrast(brightness_limit=(-0.3,0.4),p=0.5),
                 ShiftScaleRotate(shift_limit=0.0125, scale_limit=0.1, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=0.2),
                 OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                        ToSepia(p=0.08),], p=0.3),
                 OneOf([RandomShadow(p=0.2),
                        RandomRain(blur_value=2,p=0.4),
                        RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.2, alpha_coef=0.25, p=0.2),], p=0.25),
                 Cutout(num_holes=5, max_h_size=40, max_w_size=40,p=0.1)
                ], p=0.5)

    random.seed(a=None)
    dataset = TUSimpleHNet("/home/yo0n/바탕화면/TUsimple", transform = aug)
    o = dataset[random.randint(0,len(dataset)-1)]
