import torch
from torch.utils.data import Dataset
import os
import numpy as np
from lietorch import SE3
class RPdataset(Dataset):
    def __init__(self,path,rho=32):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho
    
    def __getitem__(self,index):
        img1,img2,target = np.load(self.data[index], allow_pickle=True)

        img1 = (img1.astype(float) - 127.5) / 127.5
        img2 = (img2.astype(float) - 127.5) / 127.5
        img1 = np.transpose(img1,[2,0,1])
        img2 = np.transpose(img2,[2,0,1])

        images = np.stack([img1,img2])

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        images = torch.from_numpy(images)
        target = torch.from_numpy(target.astype(float))

        base_pose = np.array([0.,0.,0.,0.,0.,0.,1])
        
        poses = np.vstack([base_pose, target])
        
        poses = torch.from_numpy(poses)
        
        return img1,img2,images ,target,poses   
        
    def __len__(self):
        return len(self.data)