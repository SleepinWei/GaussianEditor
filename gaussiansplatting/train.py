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
import torch
import random
from random import randint
import time

from utils.loss_utils import l1_loss, ssim, L_aniso,l2_loss,norm_loss,normal_loss_2DGS,get_edge_map
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.grad_flow import plot_grad_flow_gaussian,plot_histogram_gaussian

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,opt)
    gaussians = GaussianModel(dataset.sh_degree,0,0,0)
    scene = Scene(dataset, gaussians,shuffle=False,resolution_scales=[1.0,2.0])
    # appearance embedding
    # if opt.appearance_embedding:
    appearance_length = 64
    appearance_embeddings = torch.nn.Parameter(torch.randn((len(scene.getTrainCameras()),appearance_length),device="cuda"))

    # appearance_length = 64
    # TODO: extenstion loading & saving of appearance embedding & sky model
    # self.appearance_embedding = torch.nn.Parameter(torch.randn(appearance_length,device="cuda")*0.5 + 0.5)
    from gaussiansplatting.utils.appearance_modeling import AppearanceCNN
    from gaussiansplatting.scene.sky_model import SkyModel
    CNN = AppearanceCNN(upsample_num=4).to("cuda")
    sky = SkyModel()
    if pipe.enable_sky:
        if dataset.sky_source != "": 
            sky_ckpt = dataset.sky_source
        sky.load_state_dict(torch.load(sky_ckpt))

    additional = [
        {
            "params": [appearance_embeddings],
            "lr": 0.01,
            "name": "appearance",
        },
        {
            "params": list(CNN.parameters()),
            "lr": 0.01,
            "name": "cnn",
        },
        {
            "params": list(sky.parameters()),
            "lr": 0.01,
            "name": "sky",
        },
    ]
    gaussians.training_setup(opt,additional=additional)

    if checkpoint:
        if checkpoint.endswith("pth"):
            print("[info] using pth checkpoint")
            (model_params, first_iter) = torch.load(checkpoint)
            if opt.include_feature:
                first_iter = 0
            additional = []

            if pipe.enable_sky:
                if dataset.sky_source == "": 
                    sky_ckpt =  checkpoint.split(".")[0] + "_sky.pth"
                else :
                    sky_ckpt = dataset.sky_source
                sky.load_state_dict(torch.load(sky_ckpt))
                additional.append({
                    "params": list(sky.parameters()),
                    "lr": 0.01,
                    "name": "sky",
                })

            if opt.appearance_embedding:
                cnn_path = checkpoint.split(".")[0] + "_cnn.pth"
                appearance_embedding_path = checkpoint.split(".")[0] + "_embeddings.pth" 
                CNN.load_state_dict(torch.load(cnn_path))
                appearance_embeddings = torch.nn.Parameter(torch.load(appearance_embedding_path))
                additional.extend([
                {
                    "params": [appearance_embeddings],
                    "lr": 0.01,
                    "name": "appearance",
                },
                {
                    "params": list(CNN.parameters()),
                    "lr": 0.01,
                    "name": "cnn",
                },
                ])

            gaussians.restore(model_params, opt,additional=additional)
        elif checkpoint.endswith("ply"):
            print("[INFO] using ply file")
            gaussians.load_ply(checkpoint)
            gaussians.training_setup(opt,additional)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    avg_loss_for_log = 0.0
    avg_l1_loss_for_log = 0.0
    avg_regularization_for_log = 0.0
    avg_Ssim_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    random.seed(int(time.time()))

    Ld_value = torch.zeros(1, dtype=torch.float32, device="cuda")

    for iteration in range(first_iter, opt.iterations + 1):        
        Ld_value *= 0.0

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (iteration-opt.warmup_iteration) % 3200 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            # zyw: warmup
            if iteration < opt.warmup_iteration:
                viewpoint_stack = scene.getTrainCameras(scale=2.0).copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
        chosen_cam_id = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(chosen_cam_id)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,sky=sky,Ld_value=Ld_value,include_feature=opt.include_feature)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, viewspace_point_tensor, visibility_filter, radii, depth, distortion, ray_P, ray_M, blend_normal = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth_3dgs"], render_pkg["distortion"], render_pkg["ray_P"], render_pkg["ray_M"], render_pkg["normal"]
        opacity = render_pkg["opacity"]
        language_feature = render_pkg["language_feature_image"]

        if opt.include_feature:
            gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)
            Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)            
            loss = Ll1
            loss_dict = {
                "l1":Ll1,
            }
        else:
            ray_P.retain_grad()
            ray_M.retain_grad()
            depth.retain_grad()

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            ground_mask = ~viewpoint_cam.mask.cuda() # ZYW DEBUG

            newdepth = torch.clamp(depth, 0.1)
            newdepth.retain_grad()

            # fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
            # fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
            # Ln, depth_norm, loss_map = norm_loss(ray_P, ray_M, newdepth, fx, fy, viewpoint_cam.image_width, viewpoint_cam.image_height)
            # random crop 
            # if opt.random_crop == 1:
            #     crop_width = min(opt.crop_width,image.shape[2])
            #     if opt.crop_width >= image.shape[2]: 
            #         x_start = 0
            #     else:
            #         x_start = torch.randint(low=0,high=image.shape[2]-crop_width,size=(1,))
            #     crop_mask = torch.zeros_like(ground_mask,dtype=torch.bool)
            #     crop_mask[:,:,x_start:min(x_start + crop_width,image.shape[2])] = True
            #     ground_mask = torch.logical_and(ground_mask, crop_mask)

            appearance_transform = 1
            if opt.appearance_embedding:
                appearance_transform = CNN(image,appearance_embeddings[viewpoint_cam.uid,:])

            Ll1 = l1_loss(appearance_transform * image, gt_image,ground_mask)
            Ssim = ssim(image, gt_image,ground_mask)
            Lopacity = l1_loss(ground_mask[0,:,:].float(),opacity)
            # Ll1 = l2_loss(image, gt_image,ground_mask)
            Laniso = L_aniso(gaussians.get_scaling,opt.max_scale_ratio)

            gt_exp_neg_grad = get_edge_map(gt_image)
            Ln, depth_norm, loss_map = normal_loss_2DGS(ray_P, ray_M, newdepth, viewpoint_cam)
            Ld = (distortion * gt_exp_neg_grad).mean()
            
            if torch.isnan(Ln):
                print("Nan Loss")

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Ssim) \
                + opt.lambda_Laniso * Laniso + opt.lambda_Lopacity * Lopacity \
                + 0.05 * Ln + 100 * Ld
                        # Log and save
            loss_dict = {
                "l1":Ll1,
                "total_loss":loss,
                "Ln":Ln,
                "Ld":Ld,
                "ssim":Ssim,
            }
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            avg_loss_for_log += loss.item() 
            avg_l1_loss_for_log += Ll1.item() 
            if not opt.include_feature:
                avg_Ssim_for_log += Ssim.item()
                avg_regularization_for_log += Laniso.item()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if opt.appearance_embedding: 
                    torch.save(appearance_embeddings,scene.model_path + "/chkpnt" + str(iteration) + "_embeddings.pth")
                    torch.save(CNN.state_dict(),scene.model_path + "/chkpnt" + str(iteration) + "_cnn.pth")
                if pipe.enable_sky:
                    torch.save(sky.state_dict(),scene.model_path + "/chkpnt" + str(iteration) + "_sky.pth")

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    # "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Loss": f"{avg_loss_for_log / 10.0:.{7}f}",
                    "l1": f"{(1.0 - opt.lambda_dssim) * avg_l1_loss_for_log / avg_loss_for_log * 100:.{3}f}%"
                    })
                progress_bar.update(10)
                avg_loss_for_log = 0.0
                avg_l1_loss_for_log = 0.0
                avg_Ssim_for_log = 0.0
                avg_regularization_for_log = 0.0

            if  iteration % 400 == 1:
                import torchshow as ts
                if not opt.include_feature:
                    ts.save(image,f"{dataset.model_path}/vis/render.jpg") # DEBUG ZYW
                    ts.save(gt_image,f"{dataset.model_path}/vis/gt.jpg")
                    # ts.save(ground_mask.float(),f"{dataset.model_path}/vis/{iteration}_mask.jpg")
                    ts.save(render_pkg["normal"],os.path.join(f"{dataset.model_path}/vis",f"normal.jpg"))
                    ts.save(render_pkg["opacity"],os.path.join(f"{dataset.model_path}/vis",f"opacity.jpg"))
                    if opt.appearance_embedding:
                        ts.save(appearance_transform,f"{dataset.model_path}/vis/transformed.jpg")
                else:
                    ts.save(language_feature,os.path.join(f"{dataset.model_path}/vis",f"lang.jpg"))
                    ts.save(language_feature_mask.float(),os.path.join(f"{dataset.model_path}/vis",f"lang_mask.jpg"))
                    ts.save(gt_language_feature.float(),os.path.join(f"{dataset.model_path}/vis",f"lang_gt.jpg"))

                # plot_histogram_gaussian(gaussians,f"./vis_temp/gradient_histogram_{dataset.source_path.split('/')[-1]}_{iteration}.jpg") # ZYW DEBUG

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration,loss_dict, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,opt.include_feature)

            # Densification
            if not opt.include_feature and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # size_threshold = None  # ZYW DEBUG
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 1 ,opt.min_opacity, scene.cameras_extent, size_threshold)
                    # 加了一个 max densify percentage = 1
                
                if opt.opacity_reset and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                # if opt.opacity_reset and (iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if not opt.include_feature and (iteration % 2000 == 1999):
                l1_test = 0.0
                psnr_test = 0.0
                train_cams = scene.getTrainCameras().copy()
                for id,viewpoint_cam in enumerate(tqdm(train_cams)):
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background,sky=sky,include_feature=opt.include_feature)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    ts.save(image,os.path.join(f"{dataset.model_path}/train_view",f"{id}.png"))
                    ts.save(render_pkg["normal"] * 0.5 + 0.5,os.path.join(f"{dataset.model_path}/train_view",f"{id}_normal.png"))
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(train_cams)
                l1_test /= len(train_cams)
                print("\n[ITER {}] Evaluating: L1 {} PSNR {}".format(iteration, l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar('/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar('/loss_viewpoint - psnr', psnr_test, iteration)
                

def prepare_output_and_logger(args,opt):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
            # unique_str=os.getenv('OAR_JOB_ID')
        # else:
        import datetime
        unique_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")# str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    if opt.include_feature:
        args.model_path = args.model_path + "lang"
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(os.path.join(args.model_path,"vis"))
    os.makedirs(os.path.join(args.model_path,"train_view"))
    import json
    with open(os.path.join(args.model_path, "cfg_args.json"), 'w') as cfg_log_f:
        json.dump({
            "dataset":args.__dict__,
            "opt":opt.__dict__
        },cfg_log_f,indent=4)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict:dict, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for k,v in loss_dict.items():
            tb_writer.add_scalar(f'train_loss_patches/{k}', v.item(), iteration)
            # tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # for config in validation_configs:
        #     if config['cameras'] and len(config['cameras']) > 0:
        #         l1_test = 0.0
        #         psnr_test = 0.0
        #         for idx, viewpoint in enumerate(config['cameras']):
        #             image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        #             gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        #             # if tb_writer and (idx < 5):
        #                 # tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
        #                 # if iteration == testing_iterations[0]:
        #                     # tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
        #             l1_test += l1_loss(image, gt_image).mean().double()
        #             psnr_test += psnr(image, gt_image).mean().double()
        #         psnr_test /= len(config['cameras'])
        #         l1_test /= len(config['cameras'])          
        #         print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        #         if tb_writer:
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
        #             tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,5000,7_000,10_000, 20_000, 30_000,40000,50000,60_000,70000,80000,90000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,5000,7_000,10_000, 20_000, 30_000,40000,50000,60_000,70000,80000,90000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000,5000,7_000,10_000, 20_000, 30_000,40000,50000,60_000,70000,80000,90000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
