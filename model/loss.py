import torch
import torch.nn as nn
import numpy as np
from lietorch import SE3
import torch.nn.functional as F
class losses(nn.Module):
    def __init__(self,scale_af):
        super(losses, self).__init__()
        self.scale = scale_af
        self.MSELoss = nn.MSELoss()
    def rotation_loss_GEO(self,Ps, Gs):

        ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0]) 

        dP = Ps[:,jj] * Ps[:,ii].inv()
        dG = Gs[:,jj] * Gs[:,ii].inv()

        
        d = (dG * dP.inv()).log()

        tau, phi = d.split([3,3], dim=-1) 
        geodesic_loss_tr = tau.norm(dim=-1).mean()
        geodesic_loss_rot = phi.norm(dim=-1).mean()

        return geodesic_loss_rot,geodesic_loss_tr

    def translation_loss_MSE(self,pred_t,target):
        loss = self.MSELoss(pred_t,target.squeeze(1)[:,0:3])
        return loss
    def rotation_loss_MSE(self,pred_q,target):
        loss = self.MSELoss(pred_q,target.squeeze(1)[:,3:7])
        return loss

    def norm_Quaternion(self,quat):
        normalized = quat.norm(dim=-1).unsqueeze(-1)
        eps = torch.ones_like(normalized) * .01
        pred_q = quat / torch.max(normalized, eps)
        #print(torch.sum(quat.norm(dim=-1)))
        return pred_q

    def Metric_learning(self,feat1,feat2,target):

        distance = torch.sum(F.pairwise_distance(feat1, feat2, p=2)) / feat1.shape[0]
        margin_t = torch.norm(target.squeeze(1)[:,0:3])
        margin_q = torch.norm(target.squeeze(1)[:,3:7])
        margin = margin_q + margin_t

        if(self.scale * margin - distance > 0):
            loss = self.scale * margin - distance
        else:
            loss = 0
        return loss/20000


    def forward(self,feat1,feat2,pred_q,pred_t,target,Pose,loss_type):
        base_pose = np.asarray([0,0,0,0,0,0,1])
        base_pose =torch.from_numpy(base_pose)
        base_pose = base_pose.repeat(pred_t.shape[0],1).cuda().unsqueeze(1)
        prediction = torch.cat((pred_t,pred_q),1).unsqueeze(1)
        prediction = SE3(torch.cat((base_pose,prediction),1)) 
        if loss_type == 0:
            loss = self.translation_loss_MSE(pred_t,target) + self.rotation_loss_MSE(pred_q,target)
        elif loss_type == 1:
            pred_q = self.norm_Quaternion(pred_q)
            prediction = torch.cat((pred_t,pred_q),1).unsqueeze(1)
            prediction = SE3(torch.cat((base_pose,prediction),1)) 
            loss_q,loss_t = self.rotation_loss_GEO(prediction,Pose)
            loss = loss_q + loss_t
        elif loss_type == 2:
            pred_q = self.norm_Quaternion(pred_q)
            prediction = torch.cat((pred_t,pred_q),1).unsqueeze(1)
            prediction = SE3(torch.cat((base_pose,prediction),1)) 
            loss_q,loss_t = self.rotation_loss_GEO(prediction,Pose)
            loss_ML = self.Metric_learning(feat1,feat2,target)
            loss = loss_q +  loss_t + loss_ML
        return loss