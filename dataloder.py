from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os,json

warnings.filterwarnings("ignore")

class TUSimple(Dataset):

    def __init__(self, path):
        self.path = path
        sub    = [i for i in os.listdir(self.path) if i!=".DS_Store"]
        labels = [self.path + "/" + i for i in sub if i[-4:]=="json"]
        images_folders = [self.path + "/" + i for i in sub if i[-4:]!="json"]
        images = list()
        self.labels = dict()
        images_folders = [self.path+"/clips/"+i for i in os.listdir(images_folders[0]) if i!=".DS_Store"]
        for imgs_folder in images_folders:
            for i in os.listdir(imgs_folder):
                if("DS" in i):
                    continue
                i = imgs_folder + "/" +i
                lst_of_imgs = [imgs_folder + "/" + j for j in os.listdir(i) if j!=".DS_Store"]
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
        print(image_path.split("/")[key_ind:])

        return None


if __name__=="__main__":
    dataset = TUSimple("./tusimple_part")
    o = dataset[0]
