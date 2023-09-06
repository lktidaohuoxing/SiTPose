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
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def change_loss(epoch,style):
    if(epoch== 200 ):
        style = 1
    if(epoch== 240 ):
        style = 2    
    return style

def train(args):
    TrainingData = RPdataset(args.train_path)
    ValidationData = RPdataset(args.val_path)
    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size )
    print("Training Samples:{}, Validation Samples:{}".format(len(TrainingData),len(ValidationData)))

    model = ST.SiTPose_1()
    model.cuda()

    criterion = losses(14.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 170], gamma=0.1)
    loss_type = 0

    print("start training")
    t0 = time.time()

    best_translation = 10
    best_rotation = 10
    for epoch in range(args.epochs):
        epoch_start = time.time() 
        model.train()
        train_loss = 0.0

        for i, batch_value in enumerate(train_loader):

            img1 = batch_value[0].float()
            img2 = batch_value[1].float()
            target = batch_value[3].float()
            pose = batch_value[4].float()
            img1,img2,target,pose = img1.cuda(),img2.cuda(),target.cuda(),pose.cuda()
            optimizer.zero_grad()

            pose = SE3(pose)
            pred_q,pred_t,feat1,feat2 = model(img1,img2)
            loss = criterion(feat1,feat2,pred_q,pred_t,target,pose,loss_type)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 200 == 0 or (i+1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Mean Squared Error: {:.4f} lr={:.6f} loss_type={}".format(
                    epoch+1, args.epochs, i+1, len(train_loader), train_loss / 200, scheduler.get_lr()[0],loss_type ) ) 
                train_loss=0
        scheduler.step()

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            trans_loss = 0.0
            rotation_loss = 0.0

            for i, batch_value in enumerate(val_loader):

                img1 = batch_value[0].float()
                img2 = batch_value[1].float()
                target = batch_value[3].float()
                pose = batch_value[4].float()
                img1,img2,target,pose = img1.cuda(),img2.cuda(),target.cuda(),pose.cuda()
                pose = SE3(pose)
                pred_q,pred_t,feat1,feat2 = model(img1,img2)
                loss = criterion(feat1,feat2,pred_q,pred_t,target,pose,loss_type)
                val_loss += loss.item()
                trans_loss = trans_loss + calc_translation(pred_t,target.squeeze(1)[:,0:3])
                rotation_loss = rotation_loss + calc_rotation(pred_q,target.squeeze(1)[:,3:7])
    
            print("Validation: Epoch[{:0>3}/{:0>3}] Rotation Error:{:.4f}, Translation Error: {:.4f}, epoch time: {:.1f}s".format(
                epoch + 1, args.epochs,rotation_loss/len(ValidationData),trans_loss*100 / len(ValidationData),time.time() - epoch_start))
        f = open("./log/SiTPose.txt","a")
        if( (epoch+1) % 20 == 0):
            filename = 'SiTPose_'+str(epoch)  + '.pth'
            model_save_path = os.path.join("./checkpoints/", filename)
            state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}
            torch.save(state, model_save_path)
            f.write(60*"="+"\n" )
        f.write("{:0>3} epochs, Loss error: {:.6f}, Rot error:{:.4f}, Trans error:{:.4f}, lr={:.6f}, loss_type={}\n".format(epoch, val_loss/len(val_loader)  ,rotation_loss/len(ValidationData) ,  trans_loss*100 / len(ValidationData),scheduler.get_lr()[0],loss_type ))
        f.close()
            
if __name__ == "__main__":

    train_path = "./data/7Scenes/training/"
    validation_path = "./data/7Scenes/validation/"

    total_iteration = 220000
    batch_size = 64
    num_samples = 52000
    step_per_epoch = num_samples //  batch_size
    epochs = int(total_iteration / step_per_epoch)
    print(epochs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs")
    parser.add_argument("--train_path", type=str, default=train_path, help="path to training imgs")
    parser.add_argument("--val_path", type=str, default=validation_path, help="path to validation imgs")
    args = parser.parse_args()
    train(args)