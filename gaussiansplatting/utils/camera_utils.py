#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from gaussiansplatting.scene.cameras import Camera, Simple_Camera
import numpy as np
from gaussiansplatting.utils.general_utils import PILtoTorch
from gaussiansplatting.utils.graphics_utils import fov2focal
import torch

WARNED = False

import math
def get_ray_directions(image_width, image_height, fov, principal_point):
    # Generate pixel grid
    u = torch.arange(0, image_width, dtype=torch.float32,device="cuda")
    v = torch.arange(0, image_height, dtype=torch.float32,device="cuda")
    u, v = torch.meshgrid(u, v)

    # Convert pixel coordinates to normalized device coordinates
    x = (u - principal_point[0]) / image_width *  2 * math.tan(fov[0] / 2) 
    y = (v - principal_point[1]) / image_height * 2 * math.tan(fov[1] / 2)

    # Compute ray directions
    ray_directions = torch.stack([x, y, torch.ones_like(x), torch.zeros_like(x)], dim=-1) # 齐次坐标
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    return ray_directions

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.mask is not None:
        resized_image_mask = PILtoTorch(cam_info.mask, resolution)
    else:
        import torch
        resized_image_mask = torch.zeros_like(resized_image_rgb,dtype=torch.bool)


    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,mask=resized_image_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

from tqdm.auto import tqdm
def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_load(cam_infos, h, w):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            Simple_Camera(colmap_id=c.uid, R=c.R, T=c.T,
                   FoVx=c.FovX, FoVy=c.FovY, h=h, w=w, qvec = c.qvec,
                   image_name=c.image_name, uid=id, data_device='cuda')
        )
    return camera_list

def cameraList_load_kitti(cam_infos):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            Simple_Camera(colmap_id=c.uid, R=c.R, T=c.T,
                   FoVx=c.FovX, FoVy=c.FovY, h=c.height, w=c.width, qvec = c.qvec,
                   image_name=c.image_name, uid=id, data_device='cuda')
        )
    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
