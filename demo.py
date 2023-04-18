import sys
sys.path.insert(0,"../")
from model.dataset import RPdataset
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
import os 
import argparse
from lietorch import SE3
import model.SiTPose_light as ST
from model.loss import losses
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
warnings.filterwarnings('ignore')
def generate_data():
    imsize = (224,224)
    img1_path = "./demo/img1.png"
    img2_path = "./demo/img2.png"
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1,imsize)
    img2 = cv2.resize(img2,imsize)

    with open("./demo/img1_pose.txt","r") as f:

        data = f.read()
        # print(data)
        data = data.replace("\t"," ")
        data = data.replace("\n", " ")
        data = data.split(" ")
        data = list(filter(lambda x: x != "", data))
        data = [float(x) for x in data]
        T1 = np.asarray(data).reshape(4,4)
    with open("./demo/img2_pose.txt","r") as f:

        data = f.read()
        # print(data)
        data = data.replace("\t"," ")
        data = data.replace("\n", " ")
        data = data.split(" ")
        data = list(filter(lambda x: x != "", data))
        # t2 = [float(data[3]),float(data[7]),float(data[11])]
        data = [float(x) for x in data]
        T2 = np.asarray(data).reshape(4,4)
        
    print("GroundTruth t is ",(T2-T1)[0:3,3])
    relative_change = (np.linalg.inv(T1).dot(T2))
    print("GroundTruth q is " ,R.from_matrix(relative_change[0:3,0:3]).as_quat())
    target =  np.zeros((1, 7))

    datum = (img1,img2,target)
    np.save("./demo/npy/npyfile" , datum)


def eval():
    #load data
    ValidationData = RPdataset("./demo/npy/")
    val_loader = DataLoader(ValidationData, batch_size=1 )
    #load model
    model = ST.SiTPose_1()
    models = torch.load("./checkpoints/SiTPose_light.pth")
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
            print("Predict t is ",pred_t.detach().cpu().numpy())
            print("Predict q is ",pred_q.detach().cpu().numpy())
            
        

if __name__ == "__main__":

    generate_data()
    eval()
