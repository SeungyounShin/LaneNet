import torch
import torch.nn as nn
import torch.nn.functional as F

"""
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
        #print(p_hat.shape)
        FL = -( (self.alpha *torch.pow((1-p_hat),self.gamma) * p * torch.log(p_hat)) + ((1-self.alpha) * torch.pow(p_hat,self.gamma) * (1-p)* torch.log(1-p_hat)) )
        #print(FL)
        #print("="*50)
        return FL.mean()
"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            shape = targets.shape
            mark = (targets == 1).float().view(shape)
            nomark = (targets == 0).float().view(shape)
            logit = torch.zeros_like(targets ).repeat(1,2,1,1)
            logit[:,0,:,:] = nomark[:,0,:,:]
            logit[:,1,:,:] = mark[:,0,:,:]
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, logit, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

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
            lane_nums = [int(i) for i in torch.unique(instance_label[i])]
            pred_roi = binary_label[i] * pred[i] # embedding_dim, H , W
            means = list()

            var_tmp_loss = 0
            for c in lane_nums:
                laneMask = (instance_label[i]==c).repeat(embedding_dim,1,1) #B,1,H,W
                xi = pred_roi[laneMask].view(embedding_dim,-1)
                mu = xi.mean(dim=1).view(embedding_dim,1)
                cond = torch.where( torch.norm(mu - xi) > self.delta_v , torch.norm(mu - xi) - self.delta_v , torch.zeros_like(xi[0,:]))
                if cond.shape[0] != 0:
                    var = torch.pow(cond, 2).mean()
                    var_tmp_loss += var
                means.append(mu)
            L_var_list.append(var_tmp_loss/C)

            dist_tmp_loss = 0
            if len(means)>1:
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


    binary_label = (torch.randn(4,1,368,640) > 0).float()
    instance_label = torch.randint(0, 4, (4,1,368,640))

    model_bianry_pred   = Variable(torch.randn(4, 2, 368, 640), requires_grad=True)
    model_instance_pred = Variable(torch.randn(4, 2, 368, 640) , requires_grad=True)

    focal_loss_binary = FocalLoss(num_class=2, gamma=2.1)
    cluster_loss = Clustering()

    binary_loss = focal_loss_binary(model_bianry_pred, binary_label )
    clustering_loss = cluster_loss(model_instance_pred, binary_label, instance_label)

    print("binary loss   :: ", binary_loss)
    print("instance loss :: ", clustering_loss)

    binary_loss.backward()
    clustering_loss.backward()
