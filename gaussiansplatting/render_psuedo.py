import os
import torch
import random
from random import randint
import time

from utils.loss_utils import l1_loss, ssim, L_aniso
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.grad_flow import plot_grad_flow_gaussian,plot_histogram_gaussian
import numpy as np
import torchshow as ts
from torchvision.utils import save_image
from gaussiansplatting.scene.dataset_readers import KITTI360
from gaussiansplatting.utils.appearance_modeling import AppearanceCNN
from gaussiansplatting.scene.sky_model import SkyModel

def training(gs_source,dataset, opt, pipe,angle_degs, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    gaussians = GaussianModel(dataset.sh_degree,0,0,0)
    scene = Scene(dataset, gaussians,shuffle=False)
    gaussians.load_ply(gs_source)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # angle = np.pi / 3  # 60 degrees
    sky = SkyModel()
    sky.load_state_dict(torch.load(dataset.sky_source))

    for angle_deg in angle_degs:
        angle = angle_deg / 180 * np.pi
        # Define the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), 0, -np.sin(angle),0],
                                [0, 1, 0,0],
                                [np.sin(angle), 0, np.cos(angle),0],
                                [0,0,0,1]])
        rotation_matrix = torch.tensor(rotation_matrix,dtype=torch.float,device="cuda")

        viewpoint_stack = scene.getTrainCameras().copy()
        from gaussiansplatting.utils.graphics_utils import getWorld2View2
        for cam in viewpoint_stack:
            cam.R = (rotation_matrix[:3,:3].cpu().numpy() @ cam.R.T).T# R is stored transposed due to 'glm' in CUDA code
            cam.world_view_transform = cam.world_view_transform @ rotation_matrix# torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            
        with torch.no_grad():
            save_folder = os.path.join("/home/zhuyunwei/GaussianEditor/gaussiansplatting/vis/pseudo",str(int(angle_deg)))
            os.makedirs(save_folder,exist_ok=True)
            for id,viewpoint_cam in enumerate(tqdm(viewpoint_stack)):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background,sky=sky,include_feature=opt.include_feature)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                lang_feat_img = render_pkg["language_feature_image"]
                # ts.save(image,os.path.join("/home/zhuyunwei/GaussianEditor/gaussiansplatting/vis/pseudo",f"{id}_{int(angle / np.pi * 180)}.jpg"))
                save_image(image,os.path.join(save_folder,f"{id}.jpg"))
                save_image(render_pkg["normal"] / torch.norm(render_pkg["normal"],dim=0,keepdim=True) * 0.5 + 0.5 ,os.path.join(save_folder,f"{id}_normal.jpg"))
                save_image(viewpoint_cam.original_image,os.path.join(save_folder,f"{id}_gt.jpg"))
                save_image(lang_feat_img,os.path.join(save_folder,f"{id}_lang.jpg"))

                depth = render_pkg["depth_3dgs"]
                depth = 1 - torch.exp(-depth/20)
                # depth = depth / (depth.max() + 1e-5)
                import cv2
                from PIL import Image
                depth = depth.permute(1,2,0) * 255
                depth = cv2.applyColorMap(cv2.convertScaleAbs(depth.cpu().numpy()),cv2.COLORMAP_RAINBOW)
                # save_image(depth[0].repeat(3,1,1),os.path.join(save_folder,f"{id}_depth.jpg"))
                depth = Image.fromarray(depth)
                depth.save(os.path.join(save_folder,f"{id}_depth.jpg"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser,60_000)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=True)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,10_000, 20_000, 30_000,60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,3000,5000,7_000,10_000, 20_000, 30_000,40000,50000,60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1,3000,5000,7_000,10_000, 20_000, 30_000,40000,50000,60_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--angle",nargs="+", type=int,required=True)
    parser.add_argument("--gs_source",type=str,required=True)

    args = parser.parse_args(sys.argv[1:])
    training(args.gs_source,lp.extract(args), op.extract(args), pp.extract(args),args.angle, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nrendering complete.")
