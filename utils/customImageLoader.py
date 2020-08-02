import PIL.Image
from PIL.ExifTags import TAGS
import numpy as np

def imload(path):
    img = PIL.Image.open(path)
    #img = img.rotate(-90)
    img = img.resize(size=(320,240))#img.resize(size=(640,360))
    return np.array(img)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    img = imload('/home/yo0n/바탕화면/LaneNet-master/test_frames/frame24.jpg')
    print(img.shape)
    plt.imshow(img)
    plt.show()
