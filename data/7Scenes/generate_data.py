from ast import Gt
from email.mime import image
import linecache
from operator import ge
import os
from turtle import pos, position
import cv2
import random
from cv2 import transform
import numpy as np
from numpy.linalg import inv
import time
from scipy.spatial.transform import Rotation as R




def generate_data(filename,save_path):
    f = open(filename)
    imsize = (224,224)
    scenes_name = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    index = 0
    if os.path.exists(save_path):
        print("save path is:",save_path)
    else:
        os.mkdir(save_path)
        print("save path is:",save_path)
    for line in f:
        line = line.replace("\n","")
        line_list = line.split(" ")

        img1_path = "../../../../../media/data/7Scenes/" + scenes_name[int(line_list[2])] + "/" +line_list[0]
        img2_path = "../../../../../media/data/7Scenes/" + scenes_name[int(line_list[2])] + "/" +line_list[1]
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.resize(img1,imsize)
        img2 = cv2.resize(img2,imsize)

        t = np.asarray([float(line_list[3]),float(line_list[4]),float(line_list[5])])
        q = np.asarray ( [ float(line_list[7]),float(line_list[8]),float(line_list[9]),float(line_list[6]) ])
        target = np.c_[np.asarray(t).reshape(1,3), np.asarray(q).reshape(1,4)]
        

        datum = (img1,img2,target)
        np.save("./"+save_path+"/"  + ('%s' % str(int(index))).zfill(6), datum)
        index = index + 1
        if(index % 200 == 0):
            print(index)
generate_data("./db_all_med_hard_train.txt","training")
print("训练集完成")
generate_data("./db_all_med_hard_valid.txt","validation")
print("测试集完成")