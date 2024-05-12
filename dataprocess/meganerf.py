# Meganerf dataset process
import os
import torch
from tqdm.auto import tqdm
import numpy as np

dataset_path = "/DATA1/zhuyunwei/OpenDataLab___Mill_19/raw/Mill_19/building-pixsfm"
train_data_path = os.path.join(dataset_path,"train")
val_data_path = os.path.join(dataset_path,"val")

def try_mkdir(path:str):
    if not os.path.exists(path):
        os.mkdir(path)

try_mkdir(os.path.join(dataset_path,"chunks"))

def readCamerasFromTransforms_MegaNerf(path, scale_factor:float):
    cam_infos = []

    cam_dir = os.path.join(path,"metadata")
    cam_list = os.listdir(cam_dir)
    # cam_list = choose_random_elements(cam_list,load_ratio) # load only part of all datasets
    for cam_id in tqdm(cam_list):
        cam_path = os.path.join(cam_dir,cam_id) 
        camera = torch.load(cam_path)
        H = camera["H"]
        W = camera["W"]

        c2w = camera["c2w"].numpy()
        c2w = np.vstack((c2w,np.array([0,0,0,1])))
        c2w[:3,3] *= scale_factor
        c2w[[0,1],:] = c2w[[1,0],:]

        # intrinsics = camera["intrinsics"].numpy() # fx,fy,cx,cy
        cam_infos.append(c2w)

    return cam_infos

cam_infos = np.array(readCamerasFromTransforms_MegaNerf(train_data_path,225))

import matplotlib.pyplot as plt
x = cam_infos[:,0,3]
y = cam_infos[:,1,3]
z = cam_infos[:,2,3]

y_vec = np.array([0,1,0,0]).reshape((4,1))
x_vec = np.array([1,0,0,0]).reshape((4,1))
z_vec = np.array([0,0,1,0]).reshape((4,1))
y_world = (cam_infos @ y_vec)[:,:,0]
x_world = (cam_infos @ x_vec)[:,:,0]
z_world = (cam_infos @ z_vec)[:,:,0]
print(x_world.shape)
print(y_world.shape)

x_cross_y = np.cross(x_world,y_world)
print(x_cross_y)
print(z_world)
# ups = ups[:,:,0]
# ups /= np.linalg.norm(ups,axis=1,keepdims=True)
# ups = ups[:,:3] * 0.5 + 0.5 

# # plt.subplot(2,2,1)
# # plt.scatter(x,y)
# # plt.subplot(2,2,2)
# plt.scatter(x,z,c=ups)
# # plt.subplot(2,2,3)
# # plt.scatter(y,z)

# plt.savefig("./result.jpg")