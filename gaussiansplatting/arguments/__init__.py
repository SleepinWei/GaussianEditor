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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.chunk_id = -1
        self.seq = 0
        self.image_resize = 1.0
        self.sky_source = ""
        self.dataset_type = ""

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.enable_sky = 0
        super().__init__(parser, "Pipeline Parameters")

 # OptimizationParams


class OptimizationParams(ParamGroup):
    def __init__(self, parser, max_steps,
                 lr_scaler=1.0,
                 lr_final_scaler=1.0,
                 color_lr_scaler=1.0,
                 opacity_lr_scaler=1.0,
                 scaling_lr_scaler=1.0,
                 rotation_lr_scaler=1.0,
                 warmup_iteration=0, include_feature=False):
        self.iterations = max_steps
        # if warmup_iteration is None:
        #     warmup_iteration = max_steps / 10
        self.warmup_iteration = warmup_iteration
        self.crop_width = 400
        self.random_crop = 1  # 1: True, 0: False
        self.opacity_reset = 1
        self.appearance_embedding = 1

        self.lr_scaler = lr_scaler
        self.lr_final_scaler = lr_final_scaler
        self.color_lr_scaler = color_lr_scaler
        self.opacity_lr_scaler = opacity_lr_scaler
        self.scaling_lr_scaler = scaling_lr_scaler
        self.rotation_lr_scaler = rotation_lr_scaler

        self.position_lr_max_steps = max_steps
        self.position_lr_delay_mult = 0.01
        self.position_lr_init = 0.00016 * self.lr_scaler
        self.position_lr_final = 0.000016 * self.lr_final_scaler
        self.feature_lr = 0.025 * self.color_lr_scaler
        self.opacity_lr = 0.05 * self.opacity_lr_scaler
        self.scaling_lr = 0.005 * self.scaling_lr_scaler
        self.rotation_lr = 0.001 * self.rotation_lr_scaler

        self.lambda_dssim = 0.2
        self.lambda_Laniso = 1.0
        self.lambda_Lopacity = 1.0

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.percent_dense = 0.01
        self.min_opacity = 0.05
        self.include_feature = include_feature
        self.max_scale_ratio = 5
        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        group = super().extract(args)

        lr_scalers = {
            "position_lr_init": "lr_scaler",
            "position_lr_final": "lr_final_scaler",
            "feature_lr": "color_lr_scaler",
            "opacity_lr": "opacity_lr_scaler",
            "scaling_lr": "scaling_lr_scaler",
            "rotation_lr": "rotation_lr_scaler",
        }

        for k, v in lr_scalers.items():
            setattr(group, k, getattr(group, k) * getattr(group, v))

        return group


# class EditOptimizationParams(ParamGroup):
#     def __init__(self, parser,):
#         self.iterations = 3_200
#         self.position_lr_init = 0.00005
#         self.position_lr_final = 0.000025
#         self.position_lr_delay_mult = 0.5
#         self.position_lr_max_steps = 30_000
#         self.feature_lr = 0.0125
#         self.opacity_lr = 0.01
#         self.scaling_lr = 0.005
#         self.rotation_lr = 0.001
#         self.percent_dense = 0.01
#         self.lambda_dssim = 0.2
#         self.densification_interval = 100
#         self.opacity_reset_interval = 3000
#         self.densify_from_iter = 500
#         self.densify_until_iter = 15_000
#         self.densify_grad_threshold = 0.0002
#         super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
