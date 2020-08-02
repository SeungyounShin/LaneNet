import torch
import torch.nn as nn
import torch.nn.functional as F

class PerspectiveLoss(nn.Module):
    def __init__(self):
        super(PerspectiveLoss, self).__init__()


    def forward(self, pred, labels):
        #pred (B,6)
        H = torch.zeros(2,3,3)
        H[:,0,0] = pred[:,0] #a
        H[:,0,1] = pred[:,1] #b
        H[:,0,2] = pred[:,2] #c
        H[:,1,1] = pred[:,3] #d
        H[:,1,2] = pred[:,4] #e
        H[:,2,1] = pred[:,5] #f
        H[:,-1,-1] = 1

        lanes_weight = list()
        print(labels)
        for i in labels:
            i = torch.tensor(i)
            x = i[0]
            Y  = i[1]
            print(x,x.shape)
            Y = Y.repeat(3).view(3,-1)
            print(Y,Y.shape)
            print("="*30)
        return None

def test_collate(batch):
    dict = {'image':list(), 'labels':list()}
    for i in range(len(batch)):
        b = batch[i]
        dict['image'].append(torch.tensor(b[0]))
        dict['labels'].append(b[1])
    dict['image'] = torch.stack(dict['image'])
    return dict['image'], dict['labels']

if __name__=="__main__":
    import os
    import sys,random
    from torch.autograd import Variable
    sys.path.insert(0, os.path.abspath('..'))
    from dataloaderHNet import *

    criterion = PerspectiveLoss()

    dataset = TUSimpleHNet("/home/yo0n/바탕화면/TUsimple")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0, collate_fn = test_collate)

    image,lane_pair = next(iter(train_loader))

    loss = criterion(torch.randn(2,6), lane_pair)

    plt.subplot(2,1,1)
    plt.imshow(image + np.array([0.485, 0.456, 0.406]))
    plt.subplot(2,1,2)
    plt.imshow(image + np.array([0.485, 0.456, 0.406]))
    for i in range(len(lane_pair)):
        plt.scatter(lane_pair[i][0,:], lane_pair[i][1,:], s=5)
    plt.show()
