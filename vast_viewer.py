# for multiple chunks 
import time
import numpy as np
import torch
import torchvision
import rembg
from gaussiansplatting.scene.colmap_loader import qvec2rotmat
from gaussiansplatting.scene.cameras import Simple_Camera
from threestudio.utils.dpt import DPT
from torchvision.ops import masks_to_boxes
from gaussiansplatting.utils.graphics_utils import fov2focal

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kornia.geometry.quaternion import Quaternion

from threestudio.utils.typing import *
from threestudio.utils.transform import rotate_gaussians
from gaussiansplatting.gaussian_renderer import render, point_cloud_render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)
from threestudio.utils.camera import camera_ray_sample_points, project, unproject

# from threestudio.utils.dpt import DPT
# from threestudio.utils.config import parse_structured
from gaussiansplatting.scene.camera_scene import CamScene
import math

import os
import random
import ui_utils

import datetime
import subprocess
from pathlib import Path
from threestudio.utils.transform import (
    rotate_gaussians,
    translate_gaussians,
    scale_gaussians,
    default_model_mtx,
)
from gaussiansplatting.scene.dataset_readers import KITTI360
from gaussiansplatting.scene.vast_gaussian_model import VastGaussianModelKITTI


class WebUI:
    def __init__(self, cfg) -> None:
        # ZYW DEBUG
        # torch.cuda.memory._record_memory_history()

        self.gs_source = cfg.gs_source
        self.colmap_dir = None # cfg.colmap_dir
        self.port = 8084
        self.model_path = cfg.model_path
        # training cfg

        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system
        self.points3d = []
        self.gaussian = VastGaussianModelKITTI(
            "/DATA1/zhuyunwei/KITTI-360",
            seq=0,
            cam=0
        )
        # load
        # center_pos = self.gaussian.get_xyz.mean(axis=0)
        # center_pos = center_pos.cpu().detach().numpy()
        center_pos = np.array([1068.44775390625,3309.992431640625,117.1793975830078])

        # self.gaussian.max_radii2D = torch.zeros(
            # (self.gaussian.get_xyz.shape[0]), device="cuda"
        # )
        # front end related
        self.render_cameras = None

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False
        self.display_point_prompt = False

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )

        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        

        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=512
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            # specify options
            self.renderer_options = ["comp_rgb"]
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value
        @self.server.on_client_connect
        def _(client:viser.ClientHandle)->None:
            client.camera.position = center_pos
            print(f"Camera Position: {center_pos[0]},{center_pos[1]},{center_pos[2]}")

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))

        data_dir = "/DATA1/zhuyunwei/KITTI-360"
        self.kitti_data = KITTI360(data_dir,0,0)
        chunk_id = 22
        chunk_dir = f"{self.kitti_data.root_dir}/chunks/{self.kitti_data.sequence_name}/{chunk_id}"
        img_dir = f"{chunk_dir}/rgbs"
        camera_ids= np.load(os.path.join(chunk_dir,"cam_ids.npy"))

        # with torch.no_grad():
        #     self.frames = []
        #     random.seed(0)
        #     random_cam_ids = np.random.choice(camera_ids,size=50)

        #     # frame_index = random.sample(
        #         # range(0, len(self.kitti_cameras)),
        #         # min(len(self.kitti_cameras), 50),
        #     # )

        #     for i in random_cam_ids:
        #         self.make_one_camera_pose_frame(i,img_dir)

    def make_one_camera_pose_frame(self, idx,img_dir):
        frame_name, c2w, _ = self.kitti_data[idx]
        fx = self.kitti_data.K[0,0]
        fovx = np.arctan(self.kitti_data.width / 2 / fx)
        aspect = self.kitti_data.width / self.kitti_data.height

        # w2c = np.linalg.inv(c2w) 
        # T_world_camera = tf.SE3.from_rotation_and_translation(
            # tf.SO3(cam.qvec), cam.T
        # ).inverse()
        T_world_camera = tf.SE3.from_matrix(c2w)
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=True,
        )
        self.frames.append(frame)
        image = Image.open(os.path.join(img_dir,f"{frame_name:010d}.png"))
        image = image.resize((image.width//2,image.height//2))
        image = np.array(image)

        self.server.add_camera_frustum(
            f"/colmap/frustum_{idx}",
            fov = fovx,
            aspect = aspect,
            scale =1.0,
            image = image,
            wxyz = wxyz,
            position = position
        )

        # self.server.add_camera_frustum()

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps.value,
            lr_scaler = self.gs_lr_scaler.value,
            lr_final_scaler = self.gs_lr_end_scaler.value,
            color_lr_scaler = self.color_lr_scaler.value,
            opacity_lr_scaler = self.opacity_lr_scaler.value,
            scaling_lr_scaler = self.scaling_lr_scaler.value,
            rotation_lr_scaler = self.rotation_lr_scaler.value,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
    ) -> Dict[str, Any]:
        # self.gaussian.localize = local

        # ZYW: include_feature = False
        # render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor, False)
        render_pkg = self.gaussian.render(cam,self.pipe,self.background_tensor)
        import torchshow as ts
        # ts.save(render_pkg["render"],"./vis_temp/rendered.jpg")
        # ts.save(render_pkg["depth_3dgs"],"./vis_temp/depth.jpg")


        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            # render_pkg["language_feature_image"]
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        elif self.render_cameras is None: 
            self.aspect = 1.0

        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def click_cb(self, pointer):
        if self.sam_enabled.value and self.add_sam_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos)

            self.add_points3d(self.camera, click_pos)

            self.viwer_need_update = True
        elif self.draw_bbox.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.draw_flag:
                self.left_up.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                if (self.left_up.value[0] < new_value[0]) and (
                    self.left_up.value[1] < new_value[1]
                ):
                    self.right_down.value = new_value
                    self.draw_flag = True
                else:
                    self.left_up.value = new_value

    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)

        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d)

    # no longer needed since can be extracted from langsam
    # def sam_encode_all_view(self):
    #     assert hasattr(self, "sam_predictor")
    #     self.sam_features = {}
    #     # NOTE: assuming all views have the same size
    #     for id, frame in self.origin_frames.items():
    #         # TODO: check frame dtype (float32 or uint8) and device
    #         self.sam_predictor.set_image(frame)
    #         self.sam_features[id] = self.sam_predictor.features

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        # out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            out_img = output["comp_rgb"][0]
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        elif out_key == "language_feature_image": # ZYW semantic segmantation
            out_img = output["language_feature_image"].to(torch.float32).permute(1,2,0)
        elif out_key == "language_feature_mix": # TODO: 改成 button 控制
            alpha = 0.5
            _mask = output["language_feature_image"].to(torch.float32).permute(1,2,0)
            _rgb = output["comp_rgb"][0]
            out_img = alpha * _mask + (1-alpha) * _rgb
        elif out_key == "language_query":
            alpha = 0.5 
            # ZYW: text query relevency. 
            language_feat = output["language_feature_image"].permute(1,2,0)
            out_img = self.clip_encoder.get_relevancy(self.language_query.value,language_feat,0)
            # norm = torch.norm(out_img,dim=2,keepdim=True)
            # out_img /= norm
            out_img = out_img[:,:,0][...,None].repeat(1,1,3)
            # out_img = torch.pow(out_img,1/2.2)

        else: 
            print("[ERROR] Output format not supported")
            out_img = torch.ones_like(output["semantic"][0],dtype=torch.float32).repeat(1,1,3)

        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W


        self.renderer_output.options = self.renderer_options
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        # ZYW DEBUG
        gs_camera:Simple_Camera = self.camera
        if gs_camera is None:
            return

        self.gaussian.tick(gs_camera.camera_center.cpu().numpy(),0,0,0)
        output = self.render(gs_camera, sam=False)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats_grad(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras, train_frames, train_frustums):
        inpaint_2D_mask = []
        origin_frames = []

        for idx, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate.value)
            if self.fix_holes.value:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)
            train_frustums[idx].remove()
            mask_view = torch.stack([mask] * 3, dim=3)  # 1 H W C
            train_frustums[idx] = ui_utils.new_frustums(
                idx, train_frames[idx], cam, mask_view, True, self.server
            )
        return inpaint_2D_mask, origin_frames

    def add_theme(self):
        image = TitlebarImage(
            image_url_light="https://github.com/buaacyw/gaussian-editor/raw/master/static/images/logo.png",
            image_alt="GaussianEditor Logo",
            href="https://buaacyw.github.io/gaussian-editor/",
        )
        titlebar_theme = TitlebarConfig(image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=False)
    parser.add_argument("--model_path","-m",type=str,required=False) # trained checkpoint model path
    parser.add_argument("--include_feature", action="store_false")
    parser.add_argument("--vae_path",default= None,required=False) # "/home/zhuyunwei/LangSplat/model/pretrained_model/autoencoder/sofa/best_ckpt.pth") # vae_path for langsplat

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
