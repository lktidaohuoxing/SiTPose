import sys
sys.path.insert(0,"../")
from model.dataset import RPdataset
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
import time
import os 
import argparse
from lietorch import SE3
import model.SiTPose as ST
from model.loss import losses
from scipy.spatial.transform import Rotation as R
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def calc_translation(pred_t,GT_t):
    trans_l = 0.0
    for i in range(0,pred_t.shape[0]):
        x2 = (pred_t[i][0] - GT_t[i][0]) * (pred_t[i][0] - GT_t[i][0])
        y2 = (pred_t[i][1] - GT_t[i][1]) * (pred_t[i][1] - GT_t[i][1])
        z2 = (pred_t[i][2] - GT_t[i][2]) * (pred_t[i][2] - GT_t[i][2])
        res = torch.sqrt(x2+y2+z2)
        trans_l = trans_l + res
    return trans_l.detach().cpu().numpy()

def calc_rotation(pred_q,GT_q):
    rot_l = 0.0
    for i in range(0,pred_q.shape[0]):
        R1 = R.from_quat(pred_q[i].detach().cpu().numpy())
        R2 = R.from_quat(GT_q[i].detach().cpu().numpy())
        R12 = R1.inv()*R2
        R12 = R12.as_matrix()
        theta = (np.trace(R12)-1)/2
        theta = np.clip(theta,-1,1)
        theta = np.arccos(theta) * (180/np.pi)
        rot_l = rot_l + theta
    return rot_l


def eval(args):
    #load data
    ValidationData = RPdataset(args.val_path)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size )
    print(" Validation Samples:",len(ValidationData))
    #load model
    model = ST.SiTPose_1()
    models = torch.load("./checkpoints/SiTPose.pth")
    model.load_state_dict(models['state_dict'])
    model.cuda() 
    model.eval()

    val_loss = 0.0
    trans_error = 0.0
    rotation_error = 0.0
    with torch.no_grad():
        for i, batch_value in enumerate(val_loader):
            #加载数据
            img1 = batch_value[0].float()
            img2 = batch_value[1].float()
            target = batch_value[3].float()
            pose = batch_value[4].float()
            img1,img2,target,pose = img1.cuda(),img2.cuda(),target.cuda(),pose.cuda()
            #预测
            pose = SE3(pose)
            pred_q,pred_t,feat1,feat2 = model(img1,img2)

            trans_error = trans_error + calc_translation(pred_t,target.squeeze(1)[:,0:3])
            rotation_error = rotation_error + calc_rotation(pred_q,target.squeeze(1)[:,3:7])

    print("Rotation Error:{:.4f}, Translation Error: {:.4f}".format(rotation_error/len(ValidationData),trans_error*100 / len(ValidationData)))
if __name__ == "__main__":


    validation_path = "./data/7Scenes/validation/"

    # total_iteration = 220000
    batch_size = 64
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--val_path", type=str, default=validation_path, help="path to validation imgs")
    args = parser.parse_args()
    eval(args)