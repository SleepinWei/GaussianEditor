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

import os
import sys
from PIL import Image
from typing import NamedTuple
from gaussiansplatting.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from gaussiansplatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    qvec: np.array
    mask: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras_hw(cam_extrinsics, cam_intrinsics, height, width, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        origin_height = intr.height
        origin_width = intr.width
        origin_aspect = origin_height/origin_width
        aspect = height/width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        qvec = np.array(extr.qvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]/h_scale
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            if origin_aspect > aspect: # shrink height
                FovY = focal2fov(focal_length_y, origin_width * aspect)
                FovX = focal2fov(focal_length_x, origin_width)
            else: # shrink width
                FovY = focal2fov(focal_length_y, origin_height)
                FovX = focal2fov(focal_length_x, origin_height/aspect)
        elif intr.model=="SIMPLE_RADIAL": # ZYW DEBUG 可能要改，是否要 shrink height or width
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        qvec = np.array(extr.qvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        resolution_resize = 8 
        img_size = image.size
        image = image.resize((int(img_size[0] / resolution_resize),int(img_size[1] / resolution_resize)))

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, qvec=qvec,
                              image_path=image_path, image_name=image_name, width=image.width, height=image.height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo_hw(path, h, w, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_hw(cam_extrinsics=cam_extrinsics, height=h, width=w, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            qvec=np.array([0])))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
import torch
import math
from tqdm.auto import tqdm
import random 

def choose_random_elements(l : list, ratio: float):
    num_elements = int(len(l) * ratio)
    chosen_elements = random.sample(l, num_elements)
    return sorted(chosen_elements)

def readCamerasFromTransforms_MegaNerf(path, white_background, scale_factor:float, image_resize:float , load_ratio=1.0, extension=".jpg"):
    cam_infos = []

    cam_dir = os.path.join(path,"metadata")
    cam_list = os.listdir(cam_dir)
    cam_list = choose_random_elements(cam_list,load_ratio) # load only part of all datasets
    # cam_path_list = [os.path.join(path,p) for p in cam_list]
    # with open(os.path.join(path, transformsfile)) as json_file:
    for cam_id in tqdm(cam_list):
        cam_path = os.path.join(cam_dir,cam_id) 
        camera = torch.load(cam_path)
        H = camera["H"]
        W = camera["W"]
        c2w = camera["c2w"].numpy()
        c2w = np.vstack((c2w,np.array([0,0,0,1])))
        # fovx = 2 *math.atan (W /2 / fx)

        # for idx, frame in enumerate(frames):
        # cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            # c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, "rgbs", cam_id.split(".")[0] + ".jpg")

        # image_name = Path(im).stem
        image = Image.open(image_path)
        from PIL.Image import Resampling
        image_resize = 8 # ZYW DEBUG IMAGERESIZE is hard coded
        image = image.resize((int(H/image_resize),int(W/image_resize)),resample=Resampling.LANCZOS)

        intrinsics = camera["intrinsics"].numpy() # fx,fy,cx,cy
        intrinsics = intrinsics / image_resize # meganerf 这样写的
        fx = intrinsics[0]; fy=intrinsics[1]; cx = intrinsics[2]; cy = intrinsics[3]

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = 2 *math.atan (H /2 / fy)
        FovX = 2 *math.atan (W /2 / fx)
            # FovY = fovy 
            # FovX = fovx
        cam_infos.append(CameraInfo(uid=int(cam_id.split(".")[0]), R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=cam_id, width=image.size[0], height=image.size[1],
                        qvec=np.array([0])))
            
    return cam_infos

def readMegaNerfSyntheticInfo(path, white_background, eval, image_resize, extension=".png"):
    print("Reading Training Transforms")
    train_path = os.path.join(path,"train")
    train_cam_infos = readCamerasFromTransforms_MegaNerf(train_path,False,225,image_resize) # 225 is read from coordinates.pt

    print("Reading Test Transforms")
    test_path = os.path.join(path,"val")
    test_cam_infos = readCamerasFromTransforms_MegaNerf(test_path,False,225,image_resize)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

import os
import numpy as np

class KITTI360(object):
    def __init__(self, data_dir, seq=0, cam=0):

        if cam!=0:
            raise NotImplementedError('Please generate cam%d_to_world.txt at first!')
    
        # intrinsics
        calib_dir = '%s/calibration' % (data_dir)
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.sequence_name = f"2013_05_28_drive_{seq:04d}_sync"

        # camera poses 
        # sequence_dir = '%s/2013_05_28_drive_%04d_sync/' % (data_dir, seq)
        data_pose_dir = os.path.join(data_dir,"data_poses",self.sequence_name)
        self.root_dir = data_dir
        self.pose_file = os.path.join(data_pose_dir, 'cam%d_to_world.txt' % cam)
        self.image_dir = os.path.join(data_dir,"data_2d_raw",self.sequence_name, f"image_{cam:02d}", "data_rect")
        self.points_dir = os.path.join(data_dir,"data_3d_semantics",self.sequence_name,"points.ply")

        assert os.path.isfile(self.pose_file), '%s does not exist!' % self.pose_file
        assert os.path.isfile(self.intrinsic_file), '%s does not exist!' % self.intrinsic_file
        
        print('-----------------------------------------------')
        print('Loading KITTI-360, sequence %04d, camera %d' % (seq, cam))
        print('-----------------------------------------------')
        self.load_intrinsics()
        print('-----------------------------------------------')
        self.load_poses()
        print('-----------------------------------------------')
        
    def load_intrinsics(self):
        # load intrinsics
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(self.intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            if line[0] == "S_rect_00:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)

        self.K = K
        self.width = width
        self.height = height
        print ('Image size %dx%d ' % (self.width, self.height))
        print ('Intrinsics \n', self.K)

    def load_poses(self):
        # load poses of the current camera
        poses = np.loadtxt(self.pose_file)
        self.frames = poses[:,0].astype(int)
        self.poses = np.reshape(poses[:,1:], (-1, 4, 4))
        print('Number of posed frames %d' % len(self.frames))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        pose = self.poses[idx]
        basename = '%010d.png' % frame
        image_file = os.path.join(self.image_dir, basename)
        # if not os.path.isfile(image_file): 
            # print(f"{image_file} does not exist")
        return frame, pose, image_file
    
def center_crop(image, size):
    width, height = image.size
    left = (width - size[0]) // 2
    top = (height - size[1]) // 2
    right = left + size[0]
    bottom = top + size[1]
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def readCamerasFromTransforms_KITTI(kitti_data: KITTI360, cam_ids, white_background, chunk_id, load_ratio=1.0, extension=".jpg"):
    cam_infos = []

    for cam_id in tqdm(cam_ids):
        c2w = kitti_data.poses[cam_id]

        # H = kitti_data.height
        # W = kitti_data.width

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = kitti_data.frames[cam_id]
        if chunk_id == -1:
            image_path = os.path.join(kitti_data.image_dir,f"{image_name:010d}_orig.png")
            mask_path = None
        else:
            image_path = os.path.join(kitti_data.root_dir,"chunks",kitti_data.sequence_name,str(chunk_id),"rgbs",f"{image_name:010d}.png")
            mask_path = os.path.join(kitti_data.root_dir,"chunks",kitti_data.sequence_name,str(chunk_id),"rgbs",f"{image_name:010d}_mask.png")

        if not os.path.exists(image_path):
            # 有些图片被降分辨率处理了，原图为 _orig.png
            # image_path = os.path.join(kitti_data.image_dir,f"{image_name:010d}.png")
            print(f"[ERROR] {image_path} does not exist!")

        # image_name = Path(im).stem


        # DEBUG crop 后结果
        # FovX = FovY

        from PIL.Image import Resampling
        image = Image.open(image_path)
        # image = image.resize((int(W/image_resize),int(H/image_resize)),resample=Resampling.LANCZOS)
        # crop_size = min(image.width,image.height)
        # crop_size = (900 ,image.height)
        # image = center_crop(image,crop_size)

        if mask_path is None or not os.path.exists(mask_path):
            mask = None
        else:
            mask = Image.open(mask_path)
            # mask = mask.resize((int(W/image_resize),int(H/image_resize)),resample=Resampling.LANCZOS)
            # mask = center_crop(mask,crop_size)

        # im_data = np.array(image.convert("RGBA"))
        intrinsics = kitti_data.K # fx,fy,cx,cy
        # intrinsics = intrinsics # meganerf 这样写的
        fx = intrinsics[0,0]; fy=intrinsics[1,1]; cx = intrinsics[0,2]; cy = intrinsics[1,2]
        FovY = 2 *math.atan (image.height /2 / fy)
        FovX = 2 *math.atan (image.width /2 / fx)
        # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


        cam_infos.append(CameraInfo(uid=int(cam_id), R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=f"{image_name:010d}", width=image.size[0], height=image.size[1],
                        qvec=np.array([0]),mask=mask))
            
    return cam_infos

def fetchPly_KITTI(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors,normals=normals)
    
def readKITTISyntheticInfo(path, seq, eval, chunk_id=-1, extension=".png"):
    print("Reading Training Transforms")
    kitti_data: KITTI360 = KITTI360(path,seq=seq,cam=0)
    chunk_info_path = os.path.join(kitti_data.root_dir,"chunks",kitti_data.sequence_name,str(chunk_id))
    if chunk_id == -1:
        cam_ids = list(range(len(kitti_data.frames)))
    else:
        cam_ids = np.load(os.path.join(chunk_info_path,"cam_ids.npy"))

    train_cam_infos = readCamerasFromTransforms_KITTI(kitti_data=kitti_data,cam_ids=cam_ids,chunk_id=chunk_id,white_background=False) # 225 is read from coordinates.pt

    print("Reading Test Transforms: NO TEST")

    # ZYW CURRENTLY NO TEST
    # test_cam_infos = readCamerasFromTransforms_KITTI(kitti_data=kitti_data,cam_ids=,white_background=False,image_resize=image_resize)
    
    # if not eval:
        # train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if chunk_id == -1:
        ply_path = kitti_data.points_dir
    else:
        ply_path = os.path.join(chunk_info_path,"points.ply")

    if not os.path.exists(ply_path):
        assert False, f"Ply path {ply_path} does not exist"

    try:
        pcd = fetchPly_KITTI(ply_path)
    except:
        assert False, f"Ply path {ply_path} failed to fetch"

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_hw": readColmapSceneInfo_hw,
    "Blender" : readNerfSyntheticInfo,
    "MegaNerf": readMegaNerfSyntheticInfo,
    "KITTI": readKITTISyntheticInfo
}
