import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, num_class=4, alpha=None, gamma=5, balance_index=-1, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        if alpha:
            self.alpha = torch.tensor(alpha).view(1,-1,1,1)
        else :
            self.alpha = 1.
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p_hat, p):
        p_hat = self.softmax(p_hat)
        FL = -( (self.alpha *torch.pow((1-p_hat),self.gamma) * p * torch.log(p_hat)) + ((1-self.alpha) * torch.pow(p_hat,self.gamma) * (1-p)* torch.log(1-p_hat)) )
        return FL.mean()

class Clustering(nn.Module):
    def __init__(self, radius=0.5 ,  distance=3 ):
        super(Clustering, self).__init__()
        self.delta_v = radius
        self.delta_d = distance


    def forward(self,pred, binary_label,instance_label):
        batch_size = pred.shape[0]

        embedding_dim = pred.shape[1]

        L_var_list  = list()
        L_dist_list = list()

        for i in range(batch_size):

            C = int(instance_label[i].max())
            pred_roi = binary_label[i] * pred[i] # embedding_dim, H , W
            means = list()

            for c in range(1,C+1):
                laneMask = (instance_label[i]==c).repeat(embedding_dim,1,1) #B,1,H,W
                xi = pred_roi[laneMask].view(embedding_dim,-1)
                mu = xi.mean(dim=1).view(embedding_dim,1)
                cond = torch.where( torch.norm(mu - xi) > self.delta_v , torch.norm(mu - xi) - self.delta_v , torch.zeros_like(xi[0,:]))
                var = torch.pow(cond, 2).mean()
                L_var_list.append(var)
                means.append(mu)

            dist_tmp_loss = 0
            if C>1:
                for i in range(0,C):
                    for j in range(0,C):
                        if(i==j):
                            continue
                        dist_tmp_loss += torch.pow(F.relu(self.delta_d - torch.norm(means[i]-means[j])),2)

            L_dist_list.append(dist_tmp_loss)

        L_var =  sum(L_var_list)/len(L_var_list)
        L_dist = sum(L_dist_list)/len(L_dist_list)

        return L_var + L_dist

if __name__=="__main__":
    import os
    import sys
    from torch.autograd import Variable
    sys.path.insert(0, os.path.abspath('..'))

    from dataloder import TUSimple
    dataset = TUSimple("/Users/seungyoun/Desktop/ML/PR/LaneNet/tusimple_part")
    o = dataset[0]

    binary_label = torch.tensor(o[0]).view(1,1,368,640)
    instance_label = torch.tensor(o[1]).view(1,1,368,640)

    model_bianry_pred   = Variable(torch.randn(1, 2, 368, 640), requires_grad=True)
    model_instance_pred = Variable(torch.randn(1, 2, 368, 640) , requires_grad=True)

    focal_loss_binary = FocalLoss(num_class=2, gamma=2.1)
    cluster_loss = Clustering()

    binary_loss = focal_loss_binary(model_bianry_pred, binary_label )
    clustering_loss = cluster_loss(model_instance_pred, binary_label, instance_label)

    print("binary loss   :: ", binary_loss)
    print("instance loss :: ", clustering_loss)

    binary_loss.backward()
    clustering_loss.backward()
