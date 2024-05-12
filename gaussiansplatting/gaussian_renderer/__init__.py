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

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussiansplatting.utils.sh_utils import eval_sh


def camera2rasterizer(viewpoint_camera, bg_color: torch.Tensor, include_feature, sh_degree: int = 0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # include_feature=include_feature
        Ld_value=Ld_value
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    include_feature =False, # ZYW: default include_feature
    scaling_modifier=1.0,
    override_color=None,
    sky=None,
    Ld_value = None
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if Ld_value is None:
        Ld_value = torch.zeros(1, 3, dtype=torch.float32, device="cuda")
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True, # ZYW DEBUG 
        # include_feature=include_feature
        Ld_value=Ld_value
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

        shs = shs.float()
    else:
        colors_precomp = override_color
    
    if include_feature:
        language_feature_precomp = pc.get_language_feature
        language_feature_precomp = language_feature_precomp / \
            (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        # language_feature_precomp = torch.sigmoid(language_feature_precomp)
    else:
        if shs is not None: # ZYW DEBUG when rendering semantic map, shs = None
            language_feature_precomp = torch.zeros(
                (shs.shape[0],3), dtype=opacity.dtype, device=opacity.device)
        else:
            language_feature_precomp = torch.zeros(
                (3), dtype=opacity.dtype, device=opacity.device)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    values = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        # language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # rendered_image, language_feature_image, depth, alpha,radii= values
    # rendered_image, radii, depth, alpha, normal, ray_P, ray_M = values
    rendered_image, radii, depth, alpha, normal, distortion, ray_P, ray_M = values
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if pipe.enable_sky and sky is not None:
        # get dirs 
        from gaussiansplatting.utils.camera_utils import get_ray_directions
        dirs = get_ray_directions(viewpoint_camera.image_width,viewpoint_camera.image_height,(viewpoint_camera.FoVx,viewpoint_camera.FoVy),(viewpoint_camera.image_width/2,viewpoint_camera.image_height/2))
        world_dirs = torch.bmm(torch.inverse(viewpoint_camera.world_view_transform).repeat(dirs.shape[0] * dirs.shape[1],1,1) , dirs.reshape((-1,4)).unsqueeze(-1)).squeeze(-1)
        # world_dirs /= torch.norm(world_dirs,dim=-1,keepdim=True)

        sky_mask = (alpha < 0.95)[0]
        sky_image = sky(world_dirs[sky_mask.flatten()][:,:3]).permute(1,0)
        # image_sky_mask = sky_mask.reshape((viewpoint_camera.image_height,viewpoint_camera.image_width))
        rendered_image[:,sky_mask] += (1-alpha[0][sky_mask]) * sky_image

        # debug 
        # import matplotlib.pyplot as plt
        # import torchshow as ts
        # plt.scatter(dirs[:,:,1].flatten().cpu().numpy(),dirs[:,:,2].flatten().cpu().numpy(),s=1)
        # plt.savefig("./vis/cam_dirs.jpg")
        # ts.save(sky_mask.float().repeat(3,1,1),"./vis/image_sky_mask.jpg")
        # ts.save(rendered_image,"./vis/image_sky.jpg")

    return {
        "render": rendered_image,
        # "language_feature_image": language_feature_image,
        "normal" : normal, 
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity" : alpha,
        "depth_3dgs": depth,
        "ray_P": ray_P,
        "ray_M": ray_M,
        "distortion": distortion
    }


# from gaussiansplatting.scene.gaussian_model import GaussianModel


def point_cloud_render(
    viewpoint_camera,
    xyz,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype,
                         requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = torch.ones_like(xyz[..., 0:1])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = torch.ones_like(xyz) * 0.005
    rotations = torch.zeros(
        [xyz.shape[0], 4], dtype=xyz.dtype, device=xyz.device)
    rotations[..., 0] = 1.0

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(
    #             -1, 3, (pc.max_sh_degree + 1) ** 2
    #         )
    #         dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
    #             pc.get_features.shape[0], 1
    #         )
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features

    #     shs = shs.float()
    # else:
    #     colors_precomp = override_color
    colors_precomp = torch.ones_like(xyz[..., 0:1]).repeat(1, 3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    rendered_image, language_feature, depth,radii = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }
