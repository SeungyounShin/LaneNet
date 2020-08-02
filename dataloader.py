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

class TUSimple(Dataset):
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

        instance = hmap

        if self.transform:
            augmented = self.transform(image=image, mask=instance)

            image = augmented['image']
            instance = augmented['mask']


        binary = np.where(instance>0, 1, 0)

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


class KcitySeg(Dataset):
    def __init__(self, path="./Kcity", transform = None):
        self.path = path
        self.transform = transform
        self.image_root_path = self.path + "/images"
        self.label_root_path = self.path + "/labels"

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'straight' : 1, 'stop':2}

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.path+"/images/" + key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(720, 1280, 3)
        image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask  = np.zeros(image.shape[:2])

        for poly in polys:
            cls = self.classMap[poly['label']]
            pts = np.array(poly['points'])
            cv2.fillPoly(mask, np.int32([pts]) , 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        #print(image.shape, mask.shape)

        mask2 = np.zeros((2, mask.shape[1], mask.shape[2]))
        mask2[1,:,:] = mask
        mask2[0,:,:][mask2[1,:,:]!=1] = 1


        show = False
        if show:
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.subplot(2,2,2)
            plt.imshow(mask)
            plt.show()

        return image, mask2


class Kookmin(Dataset):
    def __init__(self, path="/home/yo0n/바탕화면/kookmin_Lane", transform = None):
        self.path = path
        self.transform = transform
        self.image_root_path = self.path + "/images"
        self.label_root_path = self.path + "/labels"

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'lane' : 1, 'stop':2}

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.path+"/images/" + key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(720, 1280, 3)
        #image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask  = np.zeros(image.shape[:2])

        for poly in polys:
            cls = self.classMap[poly['label']]
            pts = np.array(poly['points'])
            cv2.fillPoly(mask, np.int32([pts]) , 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        #print(image.shape, mask.shape)

        mask2 = np.zeros((2, mask.shape[1], mask.shape[2]))
        mask2[1,:,:] = mask
        mask2[0,:,:][mask2[1,:,:]!=1] = 1


        show = False
        if show:
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.subplot(2,2,2)
            plt.imshow(mask)
            plt.show()

        return image, mask2

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
    """
    dataset = TUSimple("/home/yo0n/바탕화면/TUsimple", transform = aug)
    o = dataset[random.randint(0,len(dataset)-1)]
    print(o[0].shape, o[1].shape, o[2].shape) # (368, 640) (368, 640)

    lanes = int(o[2].max())
    plt.subplot(2,1,1)
    plt.imshow(o[0] + np.array([0.485, 0.456, 0.406]))
    plt.subplot(2,1,2)
    plt.imshow(o[2])
    plt.show()
    """
    """
    #aug = RandomBrightnessContrast(brightness_limit=(-0.3,0.0),p=1)
    aug = Normalize(std=(0., 0., 0),p=1)
    augmented = aug(image=o[0], mask=o[2])

    img_aug = augmented['image']
    img_aug = o[0]/255. - (0.485, 0.456, 0.406)
    mask_aug = augmented['mask']

    visualize(img_aug, mask_aug, original_image=o[0], original_mask=o[2])
    """

    dataset = Kookmin()
    o = dataset[random.randint(0,len(dataset)-1)]
