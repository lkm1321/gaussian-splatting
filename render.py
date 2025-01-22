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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def depth_to_pointcloud(
    depth_img,
    camera,
    depth_scale = 1.0 
):
    u, v = np.meshgrid(
        np.arange(camera.image_width),
        np.arange(camera.image_height)
    )

    z = depth_img / depth_scale

    x = (u - camera.c_x) * z / camera.focal_x
    y = (v - camera.c_y) * z / camera.focal_y

    return np.stack([x, y, z], axis=-1)

def depth_to_range(
    depth_img,
    camera,
    depth_scale = 1.0
):
    return np.linalg.norm(
        depth_to_pointcloud(
            depth_img,
            camera,
            depth_scale=depth_scale
        ),
        axis=-1
    )

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)

    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth")
    makedirs(render_depth_path, exist_ok=True)

    render_range_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_range")
    makedirs(render_range_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)

        rgb_img = render_pkg["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rgb_img = rgb_img[..., rgb_img.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rgb_img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        depth_img = np.reciprocal(render_pkg["depth"].detach().cpu().numpy()[0, ...])
        depth_img_mm = (1000 * depth_img).astype(np.uint16)        
        depth_img_jet = cv2.applyColorMap(cv2.normalize(depth_img_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), cv2.COLORMAP_JET)


        # save depth
        cv2.imwrite(os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"), depth_img_mm)
        cv2.imwrite(os.path.join(render_depth_path, 'jetmap_{0:05d}'.format(idx) + ".png"), depth_img_jet)

        range_img = depth_to_range(depth_img, view)
        range_img_mm = (1000 * range_img).astype(np.uint16)
        range_img_jet = cv2.applyColorMap(cv2.normalize(range_img_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), cv2.COLORMAP_JET)

        # save range
        cv2.imwrite(os.path.join(render_range_path, '{0:05d}'.format(idx) + ".png"), range_img_mm)
        cv2.imwrite(os.path.join(render_range_path, 'jetmap_{0:05d}'.format(idx) + ".png"), range_img_jet)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)