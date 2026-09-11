"""
WonderPlay run_genesis: one-click scene compositional reconstruction and simulation.
Run from repo root: python WonderPlay_new/run_genesis.py
Depth: Marigold only. Image-to-3D: InstantMesh only (in models).
Local modules (WonderPlay_new): util, models, arguments, gaussian_renderer, scene, utils,
  syncdiffusion, marigold_lcm, simulator.
"""
import gc
import os
import random
import shutil
import sys
import time
import json
import warnings
import importlib
from types import SimpleNamespace
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from random import randint

# Ensure WonderPlay_new is on path when run from repo root (e.g. python WonderPlay_new/run_genesis.py)
_WONDERPLAY_DIR = Path(__file__).resolve().parent
if str(_WONDERPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(_WONDERPLAY_DIR))

_HF_HOME = Path(os.environ.get("HF_HOME", "/root/autodl-tmp/huggingface"))
_HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", _HF_HOME / "hub"))
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HUB_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_HUB_CACHE))
# The local Mihomo proxy can reach Hugging Face directly.  hf-mirror redirects
# some Hub metadata requests in a way that huggingface_hub rejects.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import flow_to_image
from huggingface_hub import hf_hub_download
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import (
    CLIPTokenizer,
    OneFormerForUniversalSegmentation,
    OneFormerImageProcessor,
    OneFormerProcessor,
)
from kornia.morphology import dilation

# WonderPlay_new local
from util.utils import (
    sam_get_amg_kwargs,
    save_depth_map,
    prepare_scheduler,
    soft_stitching,
    crop_to_square,
    convert_pt3d_cam_to_3dgs_cam,
    load_example_yaml,
)
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from util.image_edit_inpaint import ImageEditInpaintPipeline
from util.segment_utils import create_mask_generator_repvit
from models.models import KeyframeGen, save_point_cloud_as_ply, debug_vis_func
from arguments import GSParams, CameraParams
from gaussian_renderer import (
    proj_uv,
    render,
    render_interaction,
    render_w_shift,
    render_w_shift_flow,
    render_w_shift_da,
)
from gaussian_renderer.living_world_render import (
    render_MLP,
    render_interaction_mlp,
    pre_euler_integral,
)
from hashgrid import HashEncoderMotionModel
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.loss import l1_loss, ssim
from utils.flow import visualize_flow_as_arrows, camera_traj
from utils.vace_flow import save_vace_raft_flow
from realwonder_conditions import prepare_realwonder_conditions
from realwonder_refinement import run_realwonder_refinement
from video_scene_refinement import refine_dynamic_scene_from_video
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from simulator.diff_simulator_v3 import Simulator, pt3d_to_gs

from marigold_lcm.marigold_pipeline import (
    MarigoldPipeline,
    MarigoldPipelineNormal,
    MarigoldNormalsPipeline,
)
import pyvista as pv

warnings.filterwarnings("ignore")

ENABLE_REALWONDER_REFINEMENT_BY_DEFAULT = True


def _realwonder_enabled(config):
    realwonder_config = config.get("realwonder", {}) if config is not None else {}
    return bool(
        realwonder_config.get(
            "enabled",
            ENABLE_REALWONDER_REFINEMENT_BY_DEFAULT,
        )
    )


def _video_scene_refinement_enabled(config):
    refinement_config = (
        config.get("video_scene_refinement", {}) if config is not None else {}
    )
    return bool(refinement_config.get("enabled", True))


def _realwonder_output_frame_count(config):
    realwonder_config = config.get("realwonder", {}) if config is not None else {}
    latent_frames = int(realwonder_config.get("num_output_frames", 12))
    return latent_frames * 4 - 3


xyz_scale = 1000
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
iter_number = None
kf_gen = None
annotation_image = None
input_image = None
annotation_depth = None
annotation_valid_mask = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
object_mask_selector = None
auxiliary_model_runtime = None

sim = None
movement = None
current_object_xyz = None
already_object_pts_num = 0


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def _initial_model_dir(config):
    """Return the stable, scene-level checkpoint directory.

    ``work_dir`` is the existing WonderPlay equivalent of LivingWorld's
    ``input_dir``.  Per-run outputs remain under ``kf_gen.run_dir``.
    """
    model_dir = Path(config["work_dir"]) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _cuda_tree(value):
    if torch.is_tensor(value):
        return value.to("cuda")
    if isinstance(value, dict):
        return {key: _cuda_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cuda_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cuda_tree(item) for item in value)
    return value


def _gaussian_state(gaussians):
    """Capture both trainable and frozen Gaussian partitions."""
    tensor_names = [
        "_xyz",
        "_features_dc",
        "_scaling",
        "_rotation",
        "_opacity",
        "filter_3D",
        "_xyz_prev",
        "_features_dc_prev",
        "_scaling_prev",
        "_rotation_prev",
        "_opacity_prev",
        "filter_3D_prev",
        "_scene_flow",
        "_scene_flow_prev",
        "_motion_mask",
        "_motion_mask_prev",
    ]
    state = {}
    for name in tensor_names:
        value = getattr(gaussians, name, None)
        if value is not None:
            state[name] = value.detach().cpu()
    state["visibility_filter_all"] = gaussians.visibility_filter_all.detach().cpu()
    state["is_sky_filter"] = gaussians.is_sky_filter.detach().cpu()
    state["delete_mask_all"] = gaussians.delete_mask_all.detach().cpu()
    return state


def _restore_gaussian_state(state):
    gaussians = GaussianModel(sh_degree=0)
    parameter_names = {
        "_xyz",
        "_features_dc",
        "_scaling",
        "_rotation",
        "_opacity",
    }
    for name, value in state.items():
        if name in {"visibility_filter_all", "is_sky_filter", "delete_mask_all"}:
            continue
        value = value.to("cuda")
        if name in parameter_names:
            value = nn.Parameter(value, requires_grad=True)
        setattr(gaussians, name, value)
    gaussians.visibility_filter_all = state["visibility_filter_all"].to("cuda")
    gaussians.is_sky_filter = state["is_sky_filter"].to("cuda")
    gaussians.delete_mask_all = state["delete_mask_all"].to("cuda")
    point_count = gaussians.get_xyz_all.shape[0]
    gaussians.max_radii2D = torch.zeros(point_count, device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros(point_count, device="cuda")
    gaussians.denom = torch.zeros(point_count, device="cuda")
    gaussians.active_sh_degree = gaussians.max_sh_degree
    return gaussians


def _keyframe_state(kf_gen):
    names = [
        "image_latest",
        "image_latest_init",
        "depth_latest",
        "depth_latest_init",
        "disparity_latest",
        "disparity_latest_init",
        "sky_mask_latest",
        "mask_disocclusion",
        "current_pc",
        "current_pc_sky",
        "current_pc_layer",
        "current_pc_latest",
        "current_pc_layer_latest",
        "images",
        "images_layer",
        "depths",
        "disparities",
        "masks",
        "post_masks",
    ]
    return {
        name: _cpu_tree(getattr(kf_gen, name))
        for name in names
        if getattr(kf_gen, name, None) is not None
    }


def _restore_keyframe_state(kf_gen, state):
    for name, value in state.items():
        setattr(kf_gen, name, _cuda_tree(value))


def save_initial_scene_cache(
    config,
    gaussians,
    kf_gen,
    object_infos,
    object_pts_num_list,
    gt_masks,
    ground_value,
    camera,
    particle_num_sky,
    particle_num_base,
    particle_num_object,
):
    """Save only the fixed initial scene, before user view expansion."""
    model_dir = _initial_model_dir(config)
    camera_R = torch.as_tensor(camera.R, dtype=torch.float32).cpu()
    camera_T = torch.as_tensor(camera.T, dtype=torch.float32).cpu()
    gaussians.save_ply_for_3dgs((model_dir / "finished_3dgs.ply").as_posix())
    torch.save(
        gaussians.visibility_filter_all.detach().cpu(),
        model_dir / "visibility_filter_all.pth",
    )
    torch.save(
        gaussians.is_sky_filter.detach().cpu(),
        model_dir / "is_sky_filter.pth",
    )
    torch.save(
        gaussians.delete_mask_all.detach().cpu(),
        model_dir / "delete_mask_all.pth",
    )
    torch.save(
        {
            "version": 1,
            "gaussians": _gaussian_state(gaussians),
            "keyframe": _keyframe_state(kf_gen),
            "object_infos": _cpu_tree(object_infos),
            "object_pts_num_list": list(object_pts_num_list),
            "gt_masks": _cpu_tree(gt_masks),
            "ground_value": float(ground_value),
            "particle_num_sky": int(particle_num_sky),
            "particle_num_base": int(particle_num_base),
            "particle_num_object": int(particle_num_object),
            "camera_R": camera_R,
            "camera_T": camera_T,
            "camera_FoVx": float(camera.FoVx),
            "camera_FoVy": float(camera.FoVy),
        },
        model_dir / "scene_state.pt",
    )
    return model_dir


def load_initial_scene_cache(config, kf_gen):
    """Load the fixed scene checkpoint saved before multiview interaction."""
    model_dir = _initial_model_dir(config)
    state_path = model_dir / "scene_state.pt"
    required = [
        state_path,
        model_dir / "finished_3dgs.ply",
        model_dir / "visibility_filter_all.pth",
        model_dir / "is_sky_filter.pth",
        model_dir / "delete_mask_all.pth",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "load_gen=True requires the fixed-scene checkpoint. "
            f"Missing: {missing_text}"
        )

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    gaussians = _restore_gaussian_state(state["gaussians"])
    kf_state = state.get("keyframe", {})
    _restore_keyframe_state(kf_gen, kf_state)

    camera = Camera(
        R=state["camera_R"].numpy(),
        T=state["camera_T"].numpy(),
        FoVx=float(state["camera_FoVx"]),
        FoVy=float(state["camera_FoVy"]),
        image=kf_gen.image_latest[0].detach(),
        data_device="cuda",
    )
    scene = SimpleNamespace(
        train_cameras=[camera],
        getTrainCameras=lambda: [camera],
    )
    return {
        "model_dir": model_dir,
        "gaussians": gaussians,
        "scene": scene,
        "object_infos": _cuda_tree(state["object_infos"]),
        "object_pts_num_list": list(state["object_pts_num_list"]),
        "gt_masks": _cuda_tree(state["gt_masks"]),
        "ground_value": float(state["ground_value"]),
        "particle_num_sky": int(state["particle_num_sky"]),
        "particle_num_base": int(state["particle_num_base"]),
        "particle_num_object": int(state["particle_num_object"]),
        "camera": camera,
    }


def _run_loaded_initial_scene(
    config,
    kf_gen,
    content_prompt,
    style_prompt,
    background_prompt,
):
    """Resume the fixed initial scene and enter the existing interaction loop."""
    global gaussians, opt, scene_dict, scene_name, annotation_image, input_image
    global annotation_depth, annotation_valid_mask, sim, current_object_xyz

    loaded = load_initial_scene_cache(config, kf_gen)
    gaussians = loaded["gaussians"]
    scene = loaded["scene"]
    opt = GSParams()
    kf_gen.set_current_camera(kf_gen.get_camera_at_origin())

    content_list = content_prompt.split(",")
    scene_dict = {
        "scene_name": content_list[0],
        "entities": content_list[1:],
        "style": style_prompt,
        "background": background_prompt,
    }
    scene_name = content_list[0]
    # Frontend display uses ``input_image``. Backend motion preparation uses
    # the cached post-separation/inpaint interaction condition.
    # The cached post-separation/inpaint image is the annotation surface for
    # a loaded scene.  Keep the frontend and the interaction backend on the
    # same image instead of falling back to the raw input frame.
    input_image = kf_gen.image_latest.detach().clone()
    annotation_image = input_image.detach().clone()
    annotation_depth = kf_gen.depth_latest.detach().clone()
    annotation_valid_mask = (~kf_gen.sky_mask_latest.bool()).detach().clone()

    print("SAVING 3D RESULTS")

    object_infos = loaded["object_infos"]
    simulation_steps = config["simulation_steps"]
    dt = config["dt"]
    video_gen_fps = 8
    video_gen_dt = 1 / video_gen_fps
    num_dt = int(video_gen_dt / dt)
    simulation_steps = max(1500, num_dt * simulation_steps)
    simulation_steps = 1000

    save_dir_sim = kf_gen.run_dir / "simulation"
    save_dir_sim.mkdir(parents=True, exist_ok=True)
    xyz_obj, xyz_env = gaussians._tmp_get_xyz_all_separate()
    scaling_obj, scaling_env = gaussians._tmp_get_scaling_all_separate()
    rotation_obj, rotation_env = gaussians._tmp_get_rotation_all_separate()
    features_obj, features_env = gaussians._tmp_get_features_dc_all_separate()
    opacity_obj, opacity_env = gaussians._tmp_get_opacity_all_separate()
    simulator = Simulator(
        config=config,
        obj_gaussians=object_infos,
        env_gaussians={
            "xyz": xyz_env.data.requires_grad_(False),
            "rotation": rotation_env.data.requires_grad_(False),
            "scaling": scaling_env.data.requires_grad_(False),
            "features_dc": features_env.data.requires_grad_(False),
            "opacity": opacity_env.data.requires_grad_(False),
        },
        delta_time=dt,
        save_dir=save_dir_sim,
    )
    sim = simulator
    current_object_xyz = xyz_obj.detach().clone()
    tdgs_cam = loaded["camera"]

    if config.get("motion_type") == "interaction" and config.get(
        "multiview", {}
    ).get("enabled", False):
        from multiview_controller import run_multiview_controller

        return run_multiview_controller(
            config=config,
            simulator=simulator,
            scene=scene,
            save_dir=save_dir_sim,
            simulation_steps=simulation_steps,
            video_gen_fps=video_gen_fps,
        )

    return run_interaction_pipeline(
        simulator,
        scene,
        save_dir_sim,
        config,
        simulation_steps,
        video_gen_fps,
        viewpoint_camera=kf_gen.get_camera_at_origin(),
    )


# ========== LivingWorld Environment Motion Functions ==========

def train_hashgrid(pc, model, scheduler_gamma=0.2, scheduler_step=100, iterations=100, lr=1e-2, device='cuda', freeze_mlp=False):
    """
    Train HashGrid motion model (from LivingWorld)
    """
    means3D = pc.get_xyz_all
    scene_flow = pc.get_scene_flow_all

    assert means3D.shape[0] == scene_flow.shape[0]

    pos_all = means3D.detach().clone().to(device).requires_grad_(True)
    flow_all = scene_flow.detach().clone().to(device).requires_grad_(False)

    with torch.no_grad():
        if not (hasattr(model, "center") and hasattr(model, "bound_xyz")):
            pos_min = pos_all.min(0).values
            pos_max = pos_all.max(0).values
            center = 0.5 * (pos_min + pos_max)
            half_extent = 0.5 * (pos_max - pos_min)
            bound_xyz = half_extent.clamp_min(1e-6)

            model.center = center
            model.bound_xyz = bound_xyz

    model.train()

    pos_extent = pos_all.max(0).values - pos_all.min(0).values

    if not hasattr(model, "flow_scale"):
        current_mag = flow_all.norm(dim=1).mean()
        WW_VEL_MEAN = torch.tensor([
            0.0006032090168446302,
            6.930906238267198e-05,
            8.437281394435558e-06
        ], device=scene_flow.device)

        target_mean = WW_VEL_MEAN * 10.0
        target_mag = target_mean.norm()
        model.flow_scale = (target_mag / current_mag.clamp_min(1e-12)).item()

    flow_train = flow_all * model.flow_scale

    flow_mean = flow_train.mean(0)
    flow_std  = flow_train.std(0)
    flow_abs_mean = flow_train.abs().mean(0)
    flow_max = flow_train.abs().max(0).values

    print(f"[train_hashgrid] Flow training stats:")
    print(f"  mean: {flow_mean.cpu().numpy()}")
    print(f"  std: {flow_std.cpu().numpy()}")
    print(f"  abs_mean: {flow_abs_mean.cpu().numpy()}")
    print(f"  max: {flow_max.cpu().numpy()}")
    print(f"  flow_scale: {model.flow_scale}")

    if freeze_mlp:
        for p in model.mlp.parameters():
            p.requires_grad = False
    else:
        for p in model.mlp.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    for iteration in range(iterations):
        pred_flow = model(pos_all)
        loss = ((pred_flow - flow_train) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 20 == 0 or iteration == iterations - 1:
            print(f"[train_hashgrid] iter {iteration}/{iterations}, loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    print(f"[train_hashgrid] Training completed!")
    return model


def make_final_hints_xy(hints, H, W, yflip=False, as_column_list=True, dtype=np.float32):
    """Convert hints from [[sx,sy,ex,ey], ...] to column lists (from LivingWorld)"""
    hints = np.asarray(hints)
    if hints.ndim == 2 and hints.shape[0] == 4:
        # Already in (4, N) format
        pass
    elif hints.ndim == 2 and hints.shape[1] == 4:
        # Convert from (N, 4) to (4, N)
        hints = hints.T
    else:
        raise ValueError(f"Expected hints in (4,N) or (N,4), got {hints.shape}")

    N = hints.shape[1]
    if N == 0:
        return [], [], [], []

    sx = hints[0].astype(dtype, copy=False)
    sy = hints[1].astype(dtype, copy=False)
    ex = hints[2].astype(dtype, copy=False)
    ey = hints[3].astype(dtype, copy=False)

    if yflip:
        sy = (H - 1) - sy
        ey = (H - 1) - ey

    sx = np.clip(sx, 0, W - 1)
    ex = np.clip(ex, 0, W - 1)
    sy = np.clip(sy, 0, H - 1)
    ey = np.clip(ey, 0, H - 1)

    if as_column_list:
        to_col_list = lambda v: [np.asarray([v[i]], dtype=dtype) for i in range(N)]
        return to_col_list(sx), to_col_list(sy), to_col_list(ex), to_col_list(ey)
    else:
        return sx.astype(float).tolist(), sy.astype(float).tolist(), ex.astype(float).tolist(), ey.astype(float).tolist()


def estimate_flow(frame_pil, depth, mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args):
    """Estimate 2D flow using Cinemagraphy (from LivingWorld)"""
    from thirdparty.cinemagraphy.demo import eulerian_estimation

    print(
        "[estimate_flow] Motion hints accepted:",
        len(final_hint_start_x),
        "vectors",
        list(zip(final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y)),
    )
    frame = {
        'image': frame_pil,
        'depth': depth,
        'mask': mask,
        'final_hint_start_x': final_hint_start_x,
        'final_hint_start_y': final_hint_start_y,
        'final_hint_end_x': final_hint_end_x,
        'final_hint_end_y': final_hint_end_y,
    }
    flow = eulerian_estimation(args, frame)
    magnitude = flow.norm(dim=1)
    print(
        "[estimate_flow] Motion flow stats:",
        f"active={(magnitude > 1e-4).float().mean().item():.4f}",
        f"mean={magnitude.mean().item():.6f}",
        f"max={magnitude.max().item():.6f}",
    )

    active_mask = magnitude > 1e-4

    u = flow[:, 0]
    v = flow[:, 1]

    print(
        "[estimate_flow-debug] signed flow:",
        f"u_mean={u[active_mask].mean().item():.6f}",
        f"v_mean={v[active_mask].mean().item():.6f}",
    )

    sx = int(final_hint_start_x[0][0])
    sy = int(final_hint_start_y[0][0])
    ex = float(final_hint_end_x[0][0])
    ey = float(final_hint_end_y[0][0])

    hint_u = (ex - sx) / 50.0
    hint_v = (ey - sy) / 50.0

    pred_u = flow[0, 0, sy, sx].item()
    pred_v = flow[0, 1, sy, sx].item()

    print(
        "[estimate_flow-debug] hint consistency:",
        f"hint_uv=({hint_u:.6f}, {hint_v:.6f})",
        f"pred_uv_at_hint=({pred_u:.6f}, {pred_v:.6f})",
    )
    return flow


# ========== End LivingWorld Functions ==========


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


# ========== Environment Motion Rendering Function ==========

def prepare_environment_motion_fields(
    save_dir,
    config,
    fixed_hints_override=None,
    mask_review_callback=None,
    image_override=None,
    depth_override=None,
    valid_mask_override=None,
):
    """Estimate 2D motion fields and bind them to current_pc_latest before 3DGS training."""
    env_config = config.get('environment_motion', {})
    sam_prompt = env_config.get('sam_prompt', 'water')
    fixed_hints = (
        fixed_hints_override
        if fixed_hints_override is not None
        else env_config.get('fixed_hints', [])
    )

    if len(fixed_hints) == 0:
        raise ValueError("No fixed_hints in config, cannot estimate environment motion")

    print(f"[environment_motion_prepare] Step 1: SAM3 segmentation with prompt='{sam_prompt}'")

    # Motion annotations and SAM3 must use the exact input frame shown to the
    # user.  Gaussian re-composition is deliberately excluded here.
    image_tensor = image_override if image_override is not None else kf_gen.image_latest
    image_pil = ToPILImage()(image_tensor[0].detach().cpu().clamp(0, 1))
    image_pil.save(save_dir / "motion_input.png")

    if auxiliary_model_runtime is not None:
        motion_mask_2d = auxiliary_model_runtime.segment_environment(
            image_pil, sam_prompt
        )
    else:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        state = sam3_processor.set_image(image_pil)
        prompts = [p.strip() for p in str(sam_prompt).split(",") if p.strip()]
        masks_list = []
        for prompt in prompts:
            output = sam3_processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks", None)
            if masks is None or (torch.is_tensor(masks) and masks.numel() == 0):
                continue
            masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]
            elif masks_np.ndim == 4:
                if masks_np.shape[0] == 1:
                    masks_np = masks_np[0, ...]
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0, :, :]
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                masks_list.append(masks_np.astype(bool))
        if not masks_list:
            raise RuntimeError(
                f"SAM3 returned no masks for environment_motion.sam_prompt='{sam_prompt}'. "
                "Please adjust the prompt so it selects the moving environment region."
            )
        motion_mask_2d = np.any(np.concatenate(masks_list, axis=0), axis=0)

    mask_area = motion_mask_2d.sum()
    print(
        f"[environment_motion_prepare] SAM3 result: "
        f"{mask_area} pixels ({mask_area / (512 * 512) * 100:.2f}%)"
    )
    if mask_area == 0:
        raise RuntimeError(
            f"SAM3 produced an empty mask for environment_motion.sam_prompt='{sam_prompt}'."
        )

    Image.fromarray((motion_mask_2d.astype(np.uint8) * 255)).save(save_dir / "sam3_mask.png")
    overlay = np.array(image_pil.convert("RGB"))
    overlay_mask = motion_mask_2d.astype(bool)
    overlay[overlay_mask] = (
        overlay[overlay_mask].astype(np.float32) * 0.35
        + np.array([255, 40, 40], dtype=np.float32) * 0.65
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(save_dir / "sam3_mask_overlay.png")
    if mask_review_callback is not None:
        mask_review_callback(overlay_image)

    print(f"[environment_motion_prepare] Step 2: Cinemagraphy 2D flow estimation with {len(fixed_hints)} hints")
    hints_array = np.array(fixed_hints)
    final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = make_final_hints_xy(
        hints_array, 512, 512, yflip=False, as_column_list=True
    )

    import argparse
    cinema_dir = _WONDERPLAY_DIR / "thirdparty" / "cinemagraphy"
    cinema_ckpt_dir = cinema_dir / "ckpts"
    args = argparse.Namespace()
    args.input_dir = (save_dir / "cinemagraphy").as_posix()
    Path(args.input_dir).mkdir(exist_ok=True, parents=True)
    args.distributed = False
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.save_frames = False
    args.correct_inpaint_depth = False
    args.config = (cinema_dir / "config.yaml").as_posix()
    args.ckpt_path = (cinema_ckpt_dir / "model_150000.pth").as_posix()
    args.cinema_ckpt = cinema_ckpt_dir.as_posix()
    args.no_reload = False
    args.no_load_opt = False
    args.no_load_scheduler = False
    args.ds_factor = 1.0
    args.flow_scale = 1.0
    args.eval_mode = True
    args.point_radius = 1.5
    args.vary_pts_radius = True
    args.split = "demo"
    args.scene_id = scene_name or config.get("example_name", "scene")

    motion_depth = depth_override if depth_override is not None else kf_gen.depth_latest
    point_valid_mask = (
        valid_mask_override
        if valid_mask_override is not None
        else ~kf_gen.sky_mask_latest.bool()
    )
    flow_2d = estimate_flow(
        image_pil,
        motion_depth,
        motion_mask_2d,
        final_hint_start_x,
        final_hint_start_y,
        final_hint_end_x,
        final_hint_end_y,
        args,
    )

    scene_flow, point_motion_mask = kf_gen.attach_flow_to_current_pc_latest(
        flow=flow_2d,
        motion_mask=motion_mask_2d,
        valid_mask=point_valid_mask,
        depth=motion_depth,
    )
    motion_count = point_motion_mask.sum().item()
    point_count = point_motion_mask.shape[0]
    print(
        f"[environment_motion_prepare] Bound motion to current point cloud: "
        f"{motion_count}/{point_count} points ({motion_count / max(point_count, 1) * 100:.2f}%)"
    )
    print(
        "[environment_motion_prepare] Scene flow stats:",
        f"mean={scene_flow.mean(dim=0).detach().cpu().numpy()}",
        f"std={scene_flow.std(dim=0).detach().cpu().numpy()}",
    )
    return flow_2d, motion_mask_2d


class _InteractionMotionPointCloud:
    """Minimal tensor view accepted by the existing HashGrid trainer."""

    def __init__(self, xyz, scene_flow):
        self.get_xyz_all = xyz
        self.get_scene_flow_all = scene_flow


def project_environment_motion_mask_to_final_gaussians(viewpoint_camera, motion_mask_path):
    """Project a prepared 2D environment mask onto final environment Gaussians."""
    _, env_xyz = gaussians._tmp_get_xyz_all_separate()
    env_sky_mask = gaussians.is_sky_filter[-env_xyz.shape[0]:]
    motion_mask = cv2.imread(str(motion_mask_path), cv2.IMREAD_GRAYSCALE)
    if motion_mask is None:
        raise FileNotFoundError(f"Environment motion mask not found: {motion_mask_path}")

    uv = proj_uv(env_xyz, viewpoint_camera)
    camera_rotation = torch.as_tensor(
        viewpoint_camera.R.T,
        dtype=env_xyz.dtype,
        device=env_xyz.device,
    )
    camera_translation = torch.as_tensor(
        viewpoint_camera.T,
        dtype=env_xyz.dtype,
        device=env_xyz.device,
    )
    camera_xyz = (camera_rotation @ env_xyz.T).T + camera_translation
    pixel_x = torch.floor(uv[:, 0]).long()
    pixel_y = torch.floor(uv[:, 1]).long()
    in_frame = (
        torch.isfinite(uv).all(dim=1)
        & (camera_xyz[:, 2] > 1e-6)
        & (pixel_x >= 0)
        & (pixel_x < motion_mask.shape[1])
        & (pixel_y >= 0)
        & (pixel_y < motion_mask.shape[0])
    )
    projected_motion_mask = torch.zeros_like(env_sky_mask, dtype=torch.bool)
    if in_frame.any():
        motion_mask_tensor = torch.from_numpy(motion_mask > 0).to(env_xyz.device)
        projected_motion_mask[in_frame] = motion_mask_tensor[
            pixel_y[in_frame], pixel_x[in_frame]
        ]

    environment_mask = (~env_sky_mask) & projected_motion_mask
    if environment_mask.sum().item() == 0:
        raise RuntimeError("Interaction scene contains no moving environment Gaussians")
    return env_xyz.detach(), environment_mask.detach().bool()


@torch.no_grad()
def sync_current_pc_scene_flow_to_final_environment_gaussians(
    config,
    viewpoint_camera,
    motion_mask_path,
):
    """Transfer current_pc scene flow onto the final environment Gaussians."""
    env_xyz, env_motion_mask = project_environment_motion_mask_to_final_gaussians(
        viewpoint_camera,
        motion_mask_path,
    )
    current_pc = kf_gen.get_current_pc_latest()
    source_xyz = current_pc["xyz"].to(device=env_xyz.device, dtype=env_xyz.dtype)
    source_flow = current_pc.get("scene_flow", torch.zeros_like(current_pc["xyz"]))
    source_flow = source_flow.to(device=env_xyz.device, dtype=env_xyz.dtype)
    source_motion_mask = current_pc.get(
        "motion_mask",
        torch.ones(
            source_xyz.shape[0],
            1,
            dtype=torch.bool,
            device=source_xyz.device,
        ),
    )
    source_motion_mask = source_motion_mask.to(device=env_xyz.device).bool()
    if source_motion_mask.ndim > 1:
        source_motion_mask = source_motion_mask[:, 0]

    scale = float(config.get("interaction", {}).get("xyz_scale", xyz_scale))
    source_xyz = source_xyz * scale
    source_flow = source_flow * scale

    source_valid = (
        source_motion_mask
        & torch.isfinite(source_xyz).all(dim=1)
        & torch.isfinite(source_flow).all(dim=1)
        & (source_flow.norm(dim=1) > 0)
    )
    if source_valid.sum().item() == 0:
        raise RuntimeError("No moving current_pc scene-flow points to transfer")

    source_xyz = source_xyz[source_valid]
    source_flow = source_flow[source_valid]
    target_idx = env_motion_mask.nonzero(as_tuple=False).squeeze(1)
    target_xyz = env_xyz[target_idx]

    previous_env_scene_flow = getattr(
        gaussians, "_scene_flow_prev", torch.zeros_like(env_xyz)
    )
    if previous_env_scene_flow.shape != env_xyz.shape:
        previous_env_scene_flow = torch.zeros_like(env_xyz)
    previous_env_motion_mask = getattr(
        gaussians,
        "_motion_mask_prev",
        torch.zeros(
            env_xyz.shape[0], 1, dtype=torch.bool, device=env_xyz.device
        ),
    )
    if previous_env_motion_mask.shape[0] != env_xyz.shape[0]:
        previous_env_motion_mask = torch.zeros(
            env_xyz.shape[0], 1, dtype=torch.bool, device=env_xyz.device
        )
    if previous_env_motion_mask.ndim > 1:
        previous_env_motion_mask = previous_env_motion_mask[:, 0]

    # Preserve supervision attached from older viewpoints. Points selected in
    # the current view are updated with the current nearest-neighbour flow.
    env_scene_flow = previous_env_scene_flow.detach().clone()
    chunk_size = 256
    for start in range(0, target_xyz.shape[0], chunk_size):
        end = min(start + chunk_size, target_xyz.shape[0])
        dist = torch.cdist(target_xyz[start:end].float(), source_xyz.float())
        nn_idx = dist.argmin(dim=1)
        env_scene_flow[target_idx[start:end]] = source_flow[nn_idx].to(env_xyz.dtype)

    base_scene_flow = getattr(gaussians, "_scene_flow", torch.zeros_like(gaussians._xyz))
    if base_scene_flow.shape[0] != gaussians._xyz.shape[0]:
        base_scene_flow = torch.zeros_like(gaussians._xyz)
    base_motion_mask = getattr(
        gaussians,
        "_motion_mask",
        torch.zeros(
            gaussians._xyz.shape[0],
            1,
            dtype=torch.bool,
            device=gaussians._xyz.device,
        ),
    )
    if base_motion_mask.shape[0] != gaussians._xyz.shape[0]:
        base_motion_mask = torch.zeros(
            gaussians._xyz.shape[0],
            1,
            dtype=torch.bool,
            device=gaussians._xyz.device,
        )
    elif base_motion_mask.ndim == 1:
        base_motion_mask = base_motion_mask[:, None]
    else:
        base_motion_mask = base_motion_mask[:, :1]

    gaussians._scene_flow = base_scene_flow.detach()
    gaussians._motion_mask = base_motion_mask.detach().bool()
    accumulated_env_motion_mask = previous_env_motion_mask | env_motion_mask
    gaussians._scene_flow_prev = env_scene_flow.detach()
    gaussians._motion_mask_prev = (
        accumulated_env_motion_mask[:, None].detach().bool()
    )
    gaussians._scene_flow_all = torch.cat(
        [gaussians._scene_flow, gaussians._scene_flow_prev], dim=0
    )
    gaussians._motion_mask_all = torch.cat(
        [gaussians._motion_mask, gaussians._motion_mask_prev], dim=0
    )

    moving_flow = env_scene_flow[env_motion_mask]
    print(
        "[interaction] Transferred current_pc scene flow to final environment "
        f"Gaussians: current={int(env_motion_mask.sum().item())}, "
        f"accumulated={int(accumulated_env_motion_mask.sum().item())}/"
        f"{env_motion_mask.numel()} "
        f"points, source={int(source_valid.sum().item())}, xyz_scale={scale:g}"
    )
    print(
        "[interaction-debug] transferred env flow stats:",
        f"abs_mean={moving_flow.abs().mean(dim=0).detach().cpu().numpy()}",
        f"max={moving_flow.abs().max(dim=0).values.detach().cpu().numpy()}",
    )
    return env_xyz, accumulated_env_motion_mask, env_scene_flow


def validate_interaction_config(config):
    interaction = config.get("interaction", {})
    direction = str(interaction.get("direction", "")).lower()
    if direction not in {"env2obj", "obj2env"}:
        raise ValueError(
            "interaction.direction must be either 'env2obj' or 'obj2env'"
        )
    if int(config.get("object_num", 0)) != 1:
        raise ValueError("Interaction V1 supports exactly one object")
    if len(config.get("material_types", [])) != 1:
        raise ValueError("Interaction V1 requires exactly one material type")
    if direction == "env2obj" and config["material_types"][0] != "rigid":
        raise ValueError("interaction.direction='env2obj' requires one rigid object")
    velocity_scale = float(interaction.get("velocity_scale", 1.0))
    if not np.isfinite(velocity_scale):
        raise ValueError("interaction.velocity_scale must be finite")
    return direction, velocity_scale


def train_interaction_motion_model(config, motion_model=None):
    """Train on the inpainted environment points in Gaussian world units."""
    direction = str(config.get("interaction", {}).get("direction", "")).lower()
    scale = float(config.get("interaction", {}).get("xyz_scale", xyz_scale))
    source = "current_pc_scaled"

    if direction in {"env2obj", "obj2env"} and gaussians is not None:
        try:
            _, env_xyz = gaussians._tmp_get_xyz_all_separate()
            scene_flow_all = gaussians.get_scene_flow_all.detach()
            motion_mask_all = gaussians.get_motion_mask_all.detach()
            env_count = env_xyz.shape[0]
            total_count = gaussians.get_xyz_all.shape[0]
            if scene_flow_all.shape[0] == total_count:
                env_scene_flow = scene_flow_all[-env_count:]
            elif scene_flow_all.shape[0] == env_count:
                env_scene_flow = scene_flow_all
            else:
                raise RuntimeError(
                    f"scene_flow length {scene_flow_all.shape[0]} does not match "
                    f"total={total_count} or env={env_count}"
                )
            if motion_mask_all.shape[0] == total_count:
                env_motion_mask = motion_mask_all[-env_count:]
            elif motion_mask_all.shape[0] == env_count:
                env_motion_mask = motion_mask_all
            else:
                env_motion_mask = torch.ones(
                    env_count,
                    1,
                    dtype=torch.bool,
                    device=env_xyz.device,
                )
            if env_motion_mask.ndim > 1:
                env_motion_mask = env_motion_mask[:, 0]
            env_motion_mask = env_motion_mask.bool()

            # if direction == "obj2env":
            #     if env_motion_mask.sum().item() == 0:
            #         raise RuntimeError("obj2env has no moving final environment Gaussians")
            #     xyz = env_xyz[env_motion_mask].detach()
            #     scene_flow = env_scene_flow[env_motion_mask]
            #     source = "gaussians_env_motion_mask"
            # else:
            #     xyz = env_xyz.detach()
            #     scene_flow = env_scene_flow
            #     source = "gaussians_env"

            if direction == "obj2env" and env_motion_mask.sum().item() == 0:
                raise RuntimeError("obj2env has no moving final environment Gaussians")

            xyz = env_xyz.detach()
            scene_flow = env_scene_flow
            source = "gaussians_env"
        except Exception as exc:
            print(
                "[interaction] WARNING: failed to train HashGrid from Gaussian "
                f"environment points ({exc}); falling back to scaled current_pc"
            )
            current_pc = kf_gen.get_current_pc_latest()
            xyz = current_pc["xyz"] * scale
            scene_flow = current_pc.get(
                "scene_flow",
                torch.zeros_like(current_pc["xyz"]),
            ) * scale
    else:
        current_pc = kf_gen.get_current_pc_latest()
        xyz = current_pc["xyz"] * scale
        scene_flow = current_pc.get(
            "scene_flow",
            torch.zeros_like(current_pc["xyz"]),
        ) * scale

    print(
        f"[interaction] Training HashGrid source={source}, xyz_scale={scale:g}: "
        f"xyz_abs_mean={xyz.detach().abs().mean().item():.6f}, "
        f"flow_abs_mean={scene_flow.detach().abs().mean().item():.6f}"
    )
    motion_pc = _InteractionMotionPointCloud(xyz, scene_flow)
    reusing_model = motion_model is not None
    if motion_model is None:
        motion_model = HashEncoderMotionModel().to(config["device"])
    motion_model = train_hashgrid(
        motion_pc,
        motion_model,
        iterations=100,
        freeze_mlp=reusing_model,
    )
    motion_model._interaction_train_source = source
    motion_model._interaction_train_xyz_count = int(xyz.shape[0])
    motion_model._interaction_train_xyz_mean = xyz.detach().mean(dim=0)
    return motion_model


def get_interaction_query_points(save_dir, config=None, viewpoint_camera=None):
    """Select inpainted environment points under the single object silhouette."""
    object_mask = None
    if viewpoint_camera is not None and current_object_xyz is not None:
        object_count = gaussians._xyz.shape[0]
        render_mask = torch.zeros(
            gaussians.get_xyz_all.shape[0],
            dtype=torch.bool,
            device=gaussians.get_xyz_all.device,
        )
        render_mask[:object_count] = True
        with torch.no_grad():
            object_render = render(
                viewpoint_camera,
                gaussians,
                opt,
                background,
                timestep=0,
                movement_sim=[current_object_xyz],
                render_mask=render_mask,
            )
        object_opacity = object_render["final_opacity"].detach().squeeze()
        if object_opacity.ndim != 2:
            raise RuntimeError(
                "Object opacity must reduce to a two-dimensional image"
            )
        object_mask = object_opacity.cpu().numpy() > 0.01
        Image.fromarray((object_mask.astype(np.uint8) * 255)).save(
            save_dir / "current_object_mask.png"
        )

    if object_mask is None:
        object_mask_path = kf_gen.run_dir / "segmentation" / "object_00.png"
        if not object_mask_path.exists():
            raise FileNotFoundError(f"Object mask not found: {object_mask_path}")
        object_mask = cv2.imread(
            object_mask_path.as_posix(), cv2.IMREAD_GRAYSCALE
        )
        if object_mask is None:
            raise RuntimeError(f"Failed to read object mask: {object_mask_path}")
        object_mask = object_mask > 0

    depth = (
        annotation_depth.detach()
        if annotation_depth is not None
        else kf_gen.depth_latest.detach()
    )
    if object_mask.shape != tuple(depth.shape[-2:]):
        raise RuntimeError(
            f"Object mask/depth resolution mismatch: {object_mask.shape} vs "
            f"{tuple(depth.shape[-2:])}"
        )
    base_valid = (
        annotation_valid_mask.bool()
        if annotation_valid_mask is not None
        else ~kf_gen.sky_mask_latest.bool()
    )
    inpaint_environment_mask = (
        base_valid
        & torch.isfinite(depth)
        & (depth > 1e-6)
    )
    object_mask_tensor = torch.from_numpy(object_mask).to(depth.device)[None, None]
    interaction_mask = object_mask_tensor & inpaint_environment_mask

    # update_current_pc_by_kf flattens masks in (w, h, b) order.
    valid_flat = base_valid.permute(3, 2, 0, 1).reshape(-1)
    query_flat = interaction_mask.permute(3, 2, 0, 1).reshape(-1)
    query_in_environment = query_flat[valid_flat]

    environment_xyz = kf_gen.get_current_pc_latest()["xyz"]
    if environment_xyz.shape[0] != query_in_environment.shape[0]:
        raise RuntimeError(
            "Environment point/mask count mismatch: "
            f"{environment_xyz.shape[0]} vs {query_in_environment.shape[0]}"
        )
    scale = float(
        (config or {}).get("interaction", {}).get("xyz_scale", xyz_scale)
    )
    query_points = environment_xyz[query_in_environment] * scale
    if query_points.shape[0] == 0:
        raise RuntimeError("Object mask contains no valid inpainted environment points")

    Image.fromarray(
        (interaction_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    ).save(save_dir / "interaction_query_mask.png")
    print(
        f"[interaction] HashGrid query region: {query_points.shape[0]} "
        f"environment points in 3DGS units with xyz_scale={scale:g}"
    )
    return query_points


def renderer_displacement_to_genesis(displacement):
    """Convert a PyTorch3D/3DGS displacement to Genesis axes.

    The simulator uses the same point-space mapping for positions:
    PyTorch3D/3DGS ``(x, y, z)`` -> Genesis ``(-x, z, y)``.  A displacement
    has no positional offset, so explicitly disable the simulator's optional
    z offset.
    """
    return pt3d_to_gs(displacement, no_z_offset=True)


def generate_object_motion_hints(simulation_states, viewpoint_camera):
    """Generate one image-space hint from adjacent visible object states."""
    if len(simulation_states) < 2:
        raise RuntimeError("obj2env requires at least two visible Genesis states")
    object_key = "obj_0000"
    xyz_start = simulation_states[0][object_key]["xyz"]
    center_start = xyz_start.mean(dim=0, keepdim=True)

    width = viewpoint_camera.image_width
    height = viewpoint_camera.image_height
    camera_rotation = torch.as_tensor(
        viewpoint_camera.R.T,
        dtype=center_start.dtype,
        device=center_start.device,
    )
    camera_translation = torch.as_tensor(
        viewpoint_camera.T,
        dtype=center_start.dtype,
        device=center_start.device,
    )

    def project_visible_center(center):
        uv = proj_uv(center, viewpoint_camera)[0]
        camera_center = (camera_rotation @ center.T).T + camera_translation
        visible = (
            camera_center[0, 2] > 1e-6
            and 0 <= uv[0] < width
            and 0 <= uv[1] < height
        )
        return uv, bool(torch.isfinite(uv).all() and visible)

    uv_start, start_visible = project_visible_center(center_start)
    if not start_visible:
        raise RuntimeError("obj2env start object center is outside the camera view")

    # Use the last visible state as the hint endpoint when the final
    # simulated state has already left the camera view.
    end_state_index = None
    uv_end = None
    for state_index in range(len(simulation_states) - 1, 0, -1):
        xyz_end = simulation_states[state_index][object_key]["xyz"]
        center_end = xyz_end.mean(dim=0, keepdim=True)
        candidate_uv, candidate_visible = project_visible_center(center_end)
        if candidate_visible:
            end_state_index = state_index
            uv_end = candidate_uv
            break

    if end_state_index is None:
        raise RuntimeError("obj2env end object center is outside the camera view")

    if end_state_index != len(simulation_states) - 1:
        print(
            "[interaction] Final Genesis state is outside the camera view; "
            f"using last visible state {end_state_index} as the motion hint endpoint"
        )

    hint = [
        float(uv_start[0].item()),
        float(uv_start[1].item()),
        float(uv_end[0].item()),
        float(uv_end[1].item()),
    ]
    displacement = np.asarray(hint[2:]) - np.asarray(hint[:2])
    if np.linalg.norm(displacement) < 1e-4:
        raise RuntimeError("Genesis object displacement is too small to form a motion hint")
    print(f"[interaction] Generated obj2env motion hint: {hint}")
    return [hint]
    # hints = [
    #     [311, 361, 254, 403],
    #     [369, 382, 285, 451],
    #     [431, 401, 355, 472],
    # ]

    # print(f"[interaction-debug] Using 3 baseline hints: {hints}")

    # return hints


def collect_interaction_states(
    simulator,
    save_dir,
    simulation_steps,
    num_frames,
    gaussian_vis_freq=20,
    persistent_velocity=None,
):
    """Run Genesis once and retain the visible object states for unified rendering."""
    genesis_dir = save_dir / "genesis"
    genesis_dir.mkdir(parents=True, exist_ok=True)
    states_dir = save_dir / "4d"
    states_dir.mkdir(parents=True, exist_ok=True)

    simulation_states = []
    for sid in tqdm(range(simulation_steps), desc="Interaction Genesis simulation"):
        if persistent_velocity is not None:
            simulator.set_object_linear_velocity(persistent_velocity)

        sim_out = simulator.simulate_step(
            sid,
            simulation_steps,
            genesis_dir,
            gaussian_vis_freq,
        )
        if not sim_out:
            continue

        state = {
            key: {"xyz": values["xyz"].detach().clone()}
            for key, values in sim_out.items()
        }
        simulation_states.append(state)
        np.savez(
            states_dir / f"simulated_xyz_{sid:08d}.npz",
            **{
                key: values["xyz"].detach().cpu().numpy()
                for key, values in state.items()
            },
        )
        if len(simulation_states) >= num_frames:
            if sid != simulation_steps - 1:
                simulator.cam.stop_recording(
                    save_to_filename=(genesis_dir / "render_rgb.mp4").as_posix(),
                    fps=20,
                )
            break

    if len(simulation_states) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} interaction states, got {len(simulation_states)}"
        )
    return simulation_states


def precompute_interaction_environment_positions(
    motion_model,
    num_frames,
    scale_factor,
    viewpoint_camera,
    motion_mask_path,
):
    """Integrate HashGrid displacement for environment points while keeping sky static."""
    env_xyz, environment_mask = project_environment_motion_mask_to_final_gaussians(
        viewpoint_camera,
        motion_mask_path,
    )
    environment_points = env_xyz[environment_mask]
    if environment_points.shape[0] == 0:
        raise RuntimeError("Interaction scene contains no moving environment Gaussians")
    # ============================================================
    # ADD DEBUG HERE
    # ============================================================
    with torch.no_grad():
        scene_flow_all = gaussians.get_scene_flow_all.detach()

        env_count = env_xyz.shape[0]
        if scene_flow_all.shape[0] >= env_count:
            env_scene_flow = scene_flow_all[-env_count:]
        else:
            raise RuntimeError(
                f"scene_flow count mismatch: "
                f"{scene_flow_all.shape[0]} vs env_count={env_count}"
            )

        transferred_flow = env_scene_flow[environment_mask]

        uv_before = proj_uv(environment_points, viewpoint_camera)

        uv_after_transfer = proj_uv(
            environment_points + transferred_flow,
            viewpoint_camera,
        )

        transferred_duv = uv_after_transfer - uv_before
        valid_transfer = torch.isfinite(transferred_duv).all(dim=1)

        print(
            "[interaction-debug] transferred flow screen direction:",
            f"du_mean={transferred_duv[valid_transfer, 0].mean().item():.6f}",
            f"dv_mean={transferred_duv[valid_transfer, 1].mean().item():.6f}",
        )

        predicted_flow = motion_model(environment_points)

        uv_after_hashgrid = proj_uv(
            environment_points + predicted_flow,
            viewpoint_camera,
        )

        predicted_duv = uv_after_hashgrid - uv_before
        valid_pred = torch.isfinite(predicted_duv).all(dim=1)

        print(
            "[interaction-debug] HashGrid flow screen direction:",
            f"du_mean={predicted_duv[valid_pred, 0].mean().item():.6f}",
            f"dv_mean={predicted_duv[valid_pred, 1].mean().item():.6f}",
        )

        
    train_source = getattr(motion_model, "_interaction_train_source", None)
    train_count = getattr(motion_model, "_interaction_train_xyz_count", None)
    if train_source == "gaussians_env_motion_mask" and train_count != environment_points.shape[0]:
        raise RuntimeError(
            "HashGrid Train XYZ and render query XYZ are not aligned: "
            f"train={train_count}, render={environment_points.shape[0]}"
        )
    
    print(
        f"[interaction-debug] environment_mask count = "
        f"{int(environment_mask.sum().item())} / {environment_mask.numel()}",
        flush=True,
    )

    integration_steps = 100
    smooth = (
        1.2
        / integration_steps
        * torch.tensor([0.5, 0.5, 1.3], device=env_xyz.device)
        * scale_factor
    )
    forward_positions, _ = pre_euler_integral(
        environment_points.detach(),
        motion_model,
        num_frames,
        smooth,
    )

    shift = forward_positions[-1] - forward_positions[0]

    print(
        f"[interaction-debug] env centroid shift = "
        f"{(forward_positions[-1].mean(0) - forward_positions[0].mean(0)).detach().cpu().numpy()}",
        flush=True,
    )
    print(
        f"[interaction-debug] env mean point shift = "
        f"{shift.norm(dim=1).mean().item()}",
        flush=True,
    )
    print(
        f"[interaction-debug] env max point shift = "
        f"{shift.norm(dim=1).max().item()}",
        flush=True,
    )

    return env_xyz.detach(), environment_mask, forward_positions


def interaction_rendering(
    simulation_states,
    motion_model,
    scene,
    save_dir,
    config,
    video_gen_fps,
    viewpoint_camera=None,
    status_callback=None,
    refined_preview_callback=None,
):
    """Render dynamic object/environment states and generate VACE RAFT flow."""
    global gaussians, opt, background

    interaction = config.get("interaction", {})
    scale_factor = float(
        interaction.get(
            "environment_scale_factor",
            config.get("environment_motion", {}).get("scale_factor", 1.0),
        )
    )
    num_frames = len(simulation_states)
    if viewpoint_camera is None:
        viewpoint_camera = scene.getTrainCameras().copy()[0]

    traj_dir = save_dir / "traj_00"
    output_dir = save_dir / "interaction_motion"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["frames", "masks", "depths", "flows", "flows_actual"]:
        path = traj_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    gt_image = viewpoint_camera.original_image.detach().cpu()
    ToPILImage()(gt_image).save(save_dir / "gt.png")
    with open((save_dir / "text_prompt.txt").as_posix(), "w") as file:
        file.write(config.get("text_prompt", " "))

    obj_xyz_static, env_xyz_static = gaussians._tmp_get_xyz_all_separate()
    object_count = obj_xyz_static.shape[0]
    total_count = gaussians.get_xyz_all.shape[0]
    object_render_mask = torch.zeros(
        total_count, dtype=torch.bool, device=config["device"]
    )
    object_render_mask[:object_count] = True

    frames = []
    frame_pils = []
    depth_arrays = []
    for frame_idx, state in enumerate(
        tqdm(simulation_states, desc="Unified interaction rendering")
    ):
        obj_xyz_t = state["obj_0000"]["xyz"]

        render_pkg = render_interaction_mlp(
            viewpoint_camera=viewpoint_camera,
            pc=gaussians,
            motion_model=motion_model,
            obj_xyz_t=obj_xyz_t,
            t=frame_idx,
            opt=opt,
            bg_color=background,
            render_visible=False,
            scale_factor=scale_factor,
        )
        object_pkg = render_interaction(
            viewpoint_camera=viewpoint_camera,
            pc=gaussians,
            obj_xyz_t=obj_xyz_t,
            env_xyz_t=env_xyz_static,
            opt=opt,
            bg_color=background,
            render_visible=False,
            render_mask=object_render_mask,
        )

        image = render_pkg["render"].detach().cpu().clamp(0, 1)
        image_np = (
            image.permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil.save(traj_dir / "frames" / f"frame_{frame_idx:08d}.png")
        image_pil.save(output_dir / f"frame_{frame_idx:04d}.png")
        frames.append(image_np)
        frame_pils.append(image_pil)

        depth = render_pkg["depth"].detach().cpu()
        depth_arrays.append(depth.numpy())
        ToPILImage()(depth).save(
            traj_dir / "depths" / f"depth_{frame_idx:08d}.png"
        )

        object_mask = (
            object_pkg["final_opacity"]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        object_mask = (object_mask > 0.5).astype(np.uint8) * 255
        cv2.imwrite(
            (traj_dir / "masks" / f"frame_{frame_idx:08d}.png").as_posix(),
            object_mask,
        )

    render_video_path = traj_dir / "render_video.mp4"
    imageio.mimsave(render_video_path, frames, fps=video_gen_fps)
    imageio.mimsave(
        output_dir / "interaction_motion.mp4",
        frames,
        fps=video_gen_fps,
    )
    frame_pils[0].save(
        traj_dir / "render_video.gif",
        save_all=True,
        append_images=frame_pils[1:],
        fps=video_gen_fps,
        loop=0,
    )

    render_depths = torch.from_numpy(np.concatenate(depth_arrays, axis=0)).float()
    depth_min = render_depths.min()
    depth_range = (render_depths.max() - depth_min).clamp_min(1e-12)
    render_depths = (render_depths - depth_min) ** 2 / depth_range ** 2
    depth_pils = [ToPILImage()(1 - depth) for depth in render_depths]
    imageio.mimsave(
        traj_dir / "render_depths.mp4",
        depth_pils,
        fps=video_gen_fps,
    )

    playback_states = simulation_states
    if _realwonder_enabled(config):
        raft_checkpoint = (
            _WONDERPLAY_DIR.parent
            / "VACE"
            / "models"
            / "VACE-Annotators"
            / "flow"
            / "raft-things.pth"
        )
        save_vace_raft_flow(
            video_path=render_video_path,
            traj_dir=traj_dir,
            fps=video_gen_fps,
            checkpoint_path=raft_checkpoint,
            device="cuda:0",
        )
        realwonder_conditions = prepare_realwonder_conditions(
            save_dir, config=config, traj_id=0, overwrite=True
        )
        if status_callback is not None:
            status_callback("Video Refinement.")
        if auxiliary_model_runtime is not None:
            refinement_result = auxiliary_model_runtime.run_realwonder(
                realwonder_conditions
            )
        else:
            refinement_result = run_realwonder_refinement(
                realwonder_conditions,
                config=config,
            )
        empty_cache()
        if _video_scene_refinement_enabled(config):
            if status_callback is not None:
                status_callback("Optimizing 3D scene from generated video.")
            scene_refinement_result = refine_dynamic_scene_from_video(
                target_frames_dir=refinement_result["refined_frames_dir"],
                simulation_states=simulation_states,
                motion_model=motion_model,
                gaussians=gaussians,
                viewpoint_camera=viewpoint_camera,
                opt=opt,
                background=background,
                config=config,
                output_dir=traj_dir / "video_supervised_3d",
                fps=refinement_result["fps"],
                status_callback=status_callback,
            )
            empty_cache()
            playback_states = scene_refinement_result["playback_states"]
            if status_callback is not None:
                status_callback("Optimized 3D scene ready for playback.")
        elif refined_preview_callback is not None:
            refined_preview_callback(refinement_result)
    else:
        print("[realwonder] Video refinement disabled; using coarse 3DGS preview.", flush=True)
    print(f"[interaction] Unified output saved to {traj_dir}")
    return playback_states


def run_interaction_pipeline(
    simulator,
    scene,
    save_dir,
    config,
    simulation_steps,
    video_gen_fps,
    viewpoint_camera=None,
    motion_model=None,
    status_callback=None,
    refined_preview_callback=None,
):
    global current_object_xyz
    direction, velocity_scale = validate_interaction_config(config)
    interaction = config.get("interaction", {})
    num_frames = int(config.get("interaction", {}).get("num_frames", 50))
    if _realwonder_enabled(config) and _video_scene_refinement_enabled(config):
        required_frames = _realwonder_output_frame_count(config)
        if num_frames != required_frames:
            print(
                f"[interaction] Using {required_frames} frames so the simulated 3D "
                f"sequence exactly matches the RealWonder output; configured "
                f"interaction.num_frames={num_frames}.",
                flush=True,
            )
        num_frames = required_frames
    if num_frames < 2:
        raise ValueError("interaction.num_frames must be at least 2")

    if direction == "env2obj":
        motion_model = train_interaction_motion_model(config, motion_model)
        query_points = get_interaction_query_points(
            save_dir, config, viewpoint_camera=viewpoint_camera
        )
        with torch.no_grad():
            mean_displacement = motion_model(query_points).mean(dim=0)
        renderer_velocity = velocity_scale * mean_displacement
        genesis_velocity = renderer_displacement_to_genesis(renderer_velocity)

        # Water drives boat horizontally only.
        genesis_velocity[2] = 0.0

        print(
            "[interaction] Mean environment displacement:",
            mean_displacement.detach().cpu().numpy(),
        )
        print(
            "[interaction] Scaled renderer velocity:",
            renderer_velocity.detach().cpu().numpy(),
        )
        simulator.force_function = None
        simulation_states = collect_interaction_states(
            simulator,
            save_dir,
            simulation_steps,
            num_frames,
            persistent_velocity=genesis_velocity,
        )
        current_object_xyz = (
            simulation_states[-1]["obj_0000"]["xyz"].detach().clone()
        )
    else:
        simulation_states = collect_interaction_states(
            simulator,
            save_dir,
            simulation_steps,
            num_frames,
        )
        current_object_xyz = (
            simulation_states[-1]["obj_0000"]["xyz"].detach().clone()
        )
        if viewpoint_camera is None:
            viewpoint_camera = scene.getTrainCameras().copy()[0]
        generated_hints = generate_object_motion_hints(
            simulation_states,
            viewpoint_camera,
        )
        prepare_environment_motion_fields(
            save_dir,
            config,
            fixed_hints_override=generated_hints,
        )
        sync_current_pc_scene_flow_to_final_environment_gaussians(
            config,
            viewpoint_camera,
            save_dir / "sam3_mask.png",
        )
        motion_model = train_interaction_motion_model(config, motion_model)

        env_xyz, env_motion_mask, forward_positions = precompute_interaction_environment_positions(
            motion_model,
            num_frames,
            float(
                interaction.get(
                    "environment_scale_factor",
                    config.get("environment_motion", {}).get("scale_factor", 1.0),
                )
            ),
            viewpoint_camera,
            save_dir / "sam3_mask.png",
        )
        print(
            "[interaction] HashGrid train/render XYZ aligned on final Gaussians: "
            f"{int(env_motion_mask.sum().item())}/{env_motion_mask.numel()} points"
        )

    playback_states = interaction_rendering(
        simulation_states,
        motion_model,
        scene,
        save_dir,
        config,
        video_gen_fps,
        viewpoint_camera=viewpoint_camera,
        status_callback=status_callback,
        refined_preview_callback=refined_preview_callback,
    )
    if playback_states:
        current_object_xyz = (
            playback_states[-1]["obj_0000"]["xyz"].detach().clone()
        )
    return motion_model, playback_states


def environment_motion_rendering(gaussians, scene, save_dir, config, video_gen_fps, sky_gaussians=None):
    """
    LivingWorld environment motion rendering using HashGrid MLP.
    SAM3/Cinemagraphy/unprojection are bound to the point cloud before base
    Gaussian training; this function only trains HashGrid and renders frames.
    """
    global iter_number, kf_gen, opt, background

    print("[environment_motion_rendering] Starting environment motion rendering...")

    env_config = config.get('environment_motion', {})
    scale_factor = env_config.get('scale_factor', 1.0)
    num_frames = 50

    motion_mask_all = gaussians.get_motion_mask_all
    motion_count = int(motion_mask_all.bool().sum().item())
    point_count = int(motion_mask_all.shape[0])
    print(
        f"[environment_motion_rendering] Gaussian motion mask: "
        f"{motion_count}/{point_count} points ({motion_count / max(point_count, 1) * 100:.2f}%)"
    )
    if motion_count == 0:
        print("[environment_motion_rendering] WARNING: No motion points, rendering will be static")

    print(f"[environment_motion_rendering] Step 4: Training HashGrid motion model...")
    from hashgrid import HashEncoderMotionModel
    motion_model = HashEncoderMotionModel().to('cuda')
    motion_model = train_hashgrid(gaussians, motion_model, iterations=100)

    print(f"[environment_motion_rendering] Step 5: Rendering {num_frames} frames with scale_factor={scale_factor}...")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack[0]
    if getattr(kf_gen, "images", None):
        gt_image = kf_gen.images[-1][0].detach().cpu()
    else:
        gt_image = viewpoint_cam.original_image.detach().cpu()
    ToPILImage()(gt_image).save(save_dir / "gt.png")
    text_prompt = config.get("text_prompt", " ")
    with open((save_dir / "text_prompt.txt").as_posix(), "w") as f:
        f.write(text_prompt)

    output_dir = save_dir / 'environment_motion'
    output_dir.mkdir(exist_ok=True, parents=True)
    traj_dir = save_dir / "traj_00"
    save_kwargs = [
        "frames",
        "masks",
        "depths",
        "flows",
        "flows_actual",
        "flows_arrows",
    ]
    for save_name in save_kwargs:
        save_path = traj_dir / save_name
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    static_pkg = render_MLP(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        motion_model=None,
        t=0,
        opt=opt,
        bg_color=background,
        scale_factor=scale_factor,
        render_visible=False,
    )
    ToPILImage()(static_pkg["render"].detach().cpu().clamp(0, 1)).save(
        output_dir / "debug_static_frame_0000.png"
    )

    frames = []
    frame_pils = []
    depth_arrays = []
    flow_tensors = []
    flow_arrow_pils = []
    prev_gray = None
    mask_img = None
    sam_mask_path = save_dir / "sam3_mask.png"
    if sam_mask_path.exists():
        mask_img = cv2.imread(sam_mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)

    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        render_pkg = render_MLP(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            motion_model=motion_model,
            t=frame_idx,
            opt=opt,
            bg_color=background,
            scale_factor=scale_factor,
            render_visible=False
        )

        image = render_pkg['render']
        image_np = (image.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        frames.append(image_np)
        Image.fromarray(image_np).save(output_dir / f"frame_{frame_idx:04d}.png")

        sid = frame_idx
        image_pil = Image.fromarray(image_np)
        image_pil.save(traj_dir / "frames" / f"frame_{sid:08d}.png")
        frame_pils.append(image_pil)

        depth = render_pkg.get("depth", None)
        if depth is not None:
            depth_cpu = depth.detach().cpu()
            depth_arrays.append(depth_cpu.numpy())
            ToPILImage()(depth_cpu).save(traj_dir / "depths" / f"depth_{sid:08d}.png")

        if mask_img is None:
            mask_for_frame = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        else:
            mask_for_frame = cv2.resize(
                mask_img,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imwrite((traj_dir / "masks" / f"frame_{sid:08d}.png").as_posix(), mask_for_frame)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            flow_np = np.zeros((image_np.shape[0], image_np.shape[1], 2), dtype=np.float32)
        else:
            flow_np = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            ).astype(np.float32)
        prev_gray = gray

        flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).float()
        flow_tensors.append(flow_tensor)
        flow_actual = np.clip((flow_tensor.numpy() / 512.0 + 1.0) * 0.5, 0.0, 1.0)
        np.save((traj_dir / "flows_actual" / f"flow_{sid:08d}.npy").as_posix(), flow_actual)

        flow_frame = flow_to_image(flow_tensor)
        flow_pil = ToPILImage()(flow_frame)
        flow_pil.save(traj_dir / "flows" / f"flow_{sid:08d}.png")

        flow_arrows = visualize_flow_as_arrows(flow_tensor, image.detach().cpu())
        flow_arrow_pil = ToPILImage()(flow_arrows)
        flow_arrow_pil.save(traj_dir / "flows_arrows" / f"frame_{sid:08d}.png")
        flow_arrow_pils.append(flow_arrow_pil)

    print(f"[environment_motion_rendering] Saving video...")
    imageio.mimsave(
        (output_dir / "environment_motion.mp4").as_posix(),
        frames,
        fps=video_gen_fps
    )
    frame_pils[0].save(
        (traj_dir / "render_video.gif").as_posix(),
        save_all=True,
        append_images=frame_pils[1:],
        fps=video_gen_fps,
        loop=0,
    )
    imageio.mimsave((traj_dir / "render_video.mp4").as_posix(), frame_pils, fps=video_gen_fps)

    if len(depth_arrays) > 0:
        render_depths = np.concatenate(depth_arrays, axis=0)
        render_depths = torch.from_numpy(render_depths).float().cuda()
        render_depths_min = render_depths.min()
        render_depths_max = render_depths.max()
        denom = (render_depths_max - render_depths_min).clamp_min(1e-12)
        render_depths = (render_depths - render_depths_min) ** 2 / denom ** 2
        render_depth_pils = [ToPILImage()(1 - depth) for depth in render_depths]
        imageio.mimsave(
            (traj_dir / "render_depths.mp4").as_posix(),
            render_depth_pils,
            fps=video_gen_fps,
        )
        (traj_dir / "depths_compose").mkdir(parents=True, exist_ok=True)
        for i, depth_pil in enumerate(render_depth_pils):
            depth_pil.save(traj_dir / "depths_compose" / f"frame_{i:08d}.png")

    if len(flow_tensors) > 0:
        render_flows = flow_to_image(torch.stack(flow_tensors, dim=0))
        render_flow_pils = [ToPILImage()(flow) for flow in render_flows]
        imageio.mimsave(
            (traj_dir / "render_flows.mp4").as_posix(),
            render_flow_pils,
            fps=video_gen_fps,
        )
        (traj_dir / "flows_compose").mkdir(parents=True, exist_ok=True)
        for i, flow_pil in enumerate(render_flow_pils):
            flow_pil.save(traj_dir / "flows_compose" / f"frame_{i:08d}.png")

    if len(flow_arrow_pils) > 0:
        imageio.mimsave(
            (traj_dir / "render_flows_arrows.mp4").as_posix(),
            flow_arrow_pils,
            fps=video_gen_fps,
        )

    prepare_realwonder_conditions(save_dir, config=config, traj_id=0, overwrite=True)

    print(f"[environment_motion_rendering] Completed! Saved to {output_dir}")
    print(f"  - {num_frames} frames")
    print(f"  - Video: environment_motion.mp4")
    print(f"  - Stage2-compatible trajectory: traj_00/render_video.mp4")
    print(f"  - Stage2-compatible flows: traj_00/flows_actual/*.npy")
    print(f"  - SAM3 mask: sam3_mask.png")
    return


# ========== End Environment Motion Function ==========


def _hf_snapshot_dir(repo_id, repo_type):
    """Return a locally cached Hugging Face snapshot, if one is available."""
    cache_dirs = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            cache_dirs.append(Path(value))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dirs.append(Path(hf_home) / "hub")

    cache_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    # AutoDL commonly keeps persistent model caches outside the home directory.
    cache_dirs.append(_HF_HUB_CACHE)

    repo_prefix = "datasets" if repo_type == "dataset" else "models"
    repo_dir_name = f"{repo_prefix}--{repo_id.replace('/', '--')}"
    for cache_dir in dict.fromkeys(cache_dirs):
        repo_dir = cache_dir / repo_dir_name
        ref_path = repo_dir / "refs" / "main"
        if ref_path.is_file():
            snapshot_dir = repo_dir / "snapshots" / ref_path.read_text().strip()
            if snapshot_dir.is_dir():
                return snapshot_dir
    return None


def _cached_model_source(repo_id, fallback_repo_id=None):
    candidates = [repo_id]
    if fallback_repo_id and fallback_repo_id not in candidates:
        candidates.append(fallback_repo_id)

    for candidate in candidates:
        snapshot_dir = _hf_snapshot_dir(candidate, "model")
        if snapshot_dir is not None:
            return str(snapshot_dir), True
    return repo_id, False


def _fp16_safetensors_kwargs(model_source):
    model_dir = Path(model_source)
    if not model_dir.is_dir():
        return {}

    has_fp16_vae = (
        model_dir / "vae" / "diffusion_pytorch_model.fp16.safetensors"
    ).is_file()
    has_fp16_unet = (
        model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    ).is_file()
    if has_fp16_vae and has_fp16_unet:
        return {"variant": "fp16", "use_safetensors": True}
    return {}


def load_oneformer():
    """Load OneFormer from a local cache without fetching its ADE20K metadata."""
    model_id = "shi-labs/oneformer_ade20k_swin_large"
    model_dir = _hf_snapshot_dir(model_id, "model")
    metadata_dir = _hf_snapshot_dir("shi-labs/oneformer_demo", "dataset")

    if model_dir is None or metadata_dir is None:
        processor = OneFormerProcessor.from_pretrained(model_id)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
        return processor, model

    processor_config = json.loads((model_dir / "preprocessor_config.json").read_text())
    processor_config["repo_path"] = str(metadata_dir)
    image_processor = OneFormerImageProcessor(**processor_config)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, local_files_only=True)
    processor = OneFormerProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_dir, local_files_only=True
    )
    return processor, model


def _load_generation_models(config, motion_type):
    """Load models needed only when generating a new view or scene."""
    segment_processor, segment_model = load_oneformer()
    segment_model = segment_model.to("cuda")
    if auxiliary_model_runtime is not None:
        mask_generator = auxiliary_model_runtime.mask_generator
    else:
        mask_generator = create_mask_generator_repvit()

    inpainting_backend = str(
        config.get("inpainting_backend", "stable_diffusion")
    ).lower()
    if inpainting_backend in {"image_edit", "qwen", "qwen_image_edit"}:
        if auxiliary_model_runtime is not None:
            inpainter_pipeline = auxiliary_model_runtime.inpainting_pipeline
            print("[INFO] Image-Edit is provided by the auxiliary GPU worker.")
        else:
            image_edit_checkpoint = config.get(
                "image_edit_checkpoint", "/root/autodl-tmp/huggingface/hub"
            )
            print(
                "[INFO] Loading ModelScope Qwen ImageEdit Plus inpainter from "
                f"{image_edit_checkpoint} ..."
            )
            inpainter_pipeline = ImageEditInpaintPipeline.from_pretrained(
                image_edit_checkpoint,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(config["device"])
            inpainter_pipeline.set_progress_bar_config(disable=None)
            print("[INFO] Image-Edit inpainter loaded.")
    else:
        sd_checkpoint = config["stable_diffusion_checkpoint"]
        print(f"[INFO] Loading Stable Diffusion 2 inpainter from {sd_checkpoint} ...")
        inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_checkpoint,
            safety_checker=None,
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True,
        ).to(config["device"])
        inpainter_pipeline.scheduler = DDIMScheduler.from_config(
            inpainter_pipeline.scheduler.config
        )
        inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        print("[INFO] Stable Diffusion 2 inpainter loaded.")

    depth_model = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-v1-0",
        torch_dtype=torch.bfloat16,
        cache_dir=str(_HF_HUB_CACHE),
    ).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(
        depth_model.scheduler.config
    )
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normals_checkpoint, normals_are_local = _cached_model_source(
        "prs-eth/marigold-normals-v0-1"
    )
    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        normals_checkpoint,
        torch_dtype=torch.bfloat16,
        cache_dir=str(_HF_HUB_CACHE),
        local_files_only=normals_are_local,
        variant="fp16",
        use_safetensors=True,
    ).to(config["device"])

    if motion_type == "environment":
        print("[INFO] Skipping mvdiffusion loading (not needed for environment motion)")
        mvdiffusion = None
    else:
        mvdiffusion = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
            cache_dir=str(_HF_HUB_CACHE),
        ).to(config["device"])
        mvdiffusion.scheduler = EulerAncestralDiscreteScheduler.from_config(
            mvdiffusion.scheduler.config, timestep_spacing="trailing"
        )
        print("Loading custom white-background unet ...")
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
            cache_dir=str(_HF_HUB_CACHE),
        )
        state_dict = torch.load(unet_ckpt_path, map_location="cpu")
        mvdiffusion.unet.load_state_dict(state_dict, strict=True)

    return {
        "segment_processor": segment_processor,
        "segment_model": segment_model,
        "mask_generator": mask_generator,
        "inpainter_pipeline": inpainter_pipeline,
        "depth_model": depth_model,
        "normal_estimator": normal_estimator,
        "mvdiffusion": mvdiffusion,
    }


def run(config, dt_string=None):
    global view_matrix, scene_name, kf_gen, gaussians, opt, background, scene_dict, style_prompt, pt_gen
    global sim, movement, already_object_pts_num, current_object_xyz
    global annotation_image, input_image, annotation_depth, annotation_valid_mask
    already_object_pts_num = 0
    current_object_xyz = None
    annotation_image = None
    input_image = None
    annotation_depth = None
    annotation_valid_mask = None
    ###### ------------------ Load modules ------------------ ######

    seeding(config["seed"])
    example = config["example_name"]
    motion_type = config.get("motion_type", "object")
    if motion_type == "interaction":
        validate_interaction_config(config)

    interactive_object_selection = (
        object_mask_selector is not None and bool(config.get("gen_layer", False))
    )
    use_loaded_scene = bool(config.get("load_gen", False))

    if use_loaded_scene:
        generation_models = {
            "segment_processor": None,
            "segment_model": None,
            "mask_generator": None,
            "inpainter_pipeline": None,
            "depth_model": None,
            "normal_estimator": None,
            "mvdiffusion": None,
        }
        print("[cache] Deferring generation-model loading until a new view is requested.")
    else:
        generation_models = _load_generation_models(config, motion_type)

    rotation_path = config["rotation_path"][: config["num_scenes"]]
    assert len(rotation_path) == config["num_scenes"]

    segment_processor = generation_models["segment_processor"]
    segment_model = generation_models["segment_model"]
    mask_generator = generation_models["mask_generator"]
    inpainter_pipeline = generation_models["inpainter_pipeline"]
    depth_model = generation_models["depth_model"]
    normal_estimator = generation_models["normal_estimator"]
    mvdiffusion = generation_models["mvdiffusion"]

    # sam_model = sam_model_registry['vit_l'](checkpoint="/viscam/projects/wonder_dy/zzli/ckpts/sam_vit_l_0b3195.pth")
    # _ = sam_model.to(device=config["device"])
    # output_mode = "binary_mask"
    # amg_kwargs = sam_get_amg_kwargs()
    # sam_generator = SamPredictor(sam_model)

    # Resolve all paths relative to repo root (so running from repo root works regardless of cwd)
    _repo_root = Path(__file__).resolve().parent.parent
    _examples_dir = _repo_root / "examples" / "imgs" / config["example_name"]
    config["examples_dir"] = str(_examples_dir.resolve())
    config["sky_image_dir"] = str(_examples_dir.resolve())
    if "runs_dir" in config and not Path(config["runs_dir"]).is_absolute():
        config["runs_dir"] = str((_repo_root / config["runs_dir"]).resolve())
    if "work_dir" in config and not Path(config["work_dir"]).is_absolute():
        config["work_dir"] = str((_repo_root / config["work_dir"]).resolve())

    print(
        "###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######"
    )
    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=inpainter_pipeline,
        mask_generator=mask_generator,
        depth_model=depth_model,
        segment_model=segment_model,
        segment_processor=segment_processor,
        normal_estimator=normal_estimator,
        rotation_path=rotation_path,
        inpainting_resolution=config["inpainting_resolution_gen"],
        mvdiffusion=mvdiffusion,
        sam_model=None,
        load_instantmesh=not use_loaded_scene,
        dt_string=dt_string,
    ).to(config["device"])
    kf_gen.object_mask_selector = object_mask_selector
    kf_gen.defer_object_selection_until_expansion = use_loaded_scene and interactive_object_selection

    def ensure_generation_models():
        nonlocal generation_models
        required_models_ready = (
            kf_gen.segment_model is not None
            and kf_gen.mask_generator is not None
            and kf_gen.inpainting_pipeline is not None
            and kf_gen.depth_model is not None
            and kf_gen.normal_estimator is not None
            and kf_gen.mvdiffusion is not None
        )
        if required_models_ready:
            return
        generation_models = _load_generation_models(config, motion_type)
        kf_gen.segment_processor = generation_models["segment_processor"]
        kf_gen.segment_model = generation_models["segment_model"]
        kf_gen.mask_generator = generation_models["mask_generator"]
        kf_gen.inpainting_pipeline = generation_models["inpainter_pipeline"]
        kf_gen.depth_model = generation_models["depth_model"]
        kf_gen.normal_estimator = generation_models["normal_estimator"]
        kf_gen.mvdiffusion = generation_models["mvdiffusion"]
        kf_gen._load_instantmesh()
        print("[cache] Deferred generation models are ready.")

    kf_gen.ensure_generation_models = ensure_generation_models

    prompt_data = None
    prompt_yaml = _repo_root / "examples" / "examples.yaml"
    if prompt_yaml.exists():
        prompt_data = load_example_yaml(config["example_name"], prompt_yaml)

    content_prompt = config.get("content_prompt", None)
    if content_prompt is None or str(content_prompt).strip() == "":
        content_prompt = prompt_data.get("content_prompt") if prompt_data else None
    if content_prompt is None or str(content_prompt).strip() == "":
        raise ValueError(
            "Missing scene content prompt. Add content_prompt to the scene "
            "config or add the example to examples/examples.yaml."
        )

    style_prompt = config.get("style_prompt", None)
    if style_prompt is None or str(style_prompt).strip() == "":
        style_prompt = (
            prompt_data.get("style_prompt")
            if prompt_data
            else "DSLR 35mm landscape"
        )
    adaptive_negative_prompt = config.get("negative_prompt", "")
    if adaptive_negative_prompt is None or str(adaptive_negative_prompt).strip() == "":
        adaptive_negative_prompt = (
            prompt_data.get("negative_prompt", "") if prompt_data else ""
        )
    background_prompt = config.get("background", None)
    if background_prompt is None and prompt_data:
        background_prompt = prompt_data.get("background", None)
    control_text = config.get(
        "control_text",
        prompt_data.get("control_text", None) if prompt_data else None,
    )
    outdoor = config.get(
        "outdoor",
        prompt_data.get("outdoor", False) if prompt_data else False,
    )
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    # Resolve image path: config image_filepath, or data_path/example_name/input_image
    if config.get("image_filepath"):
        _img_path = Path(config["image_filepath"])
    else:
        _img_path = Path(config.get("data_path", "examples/imgs")) / config["example_name"] / config.get("input_image", "image.png")
    if not _img_path.is_absolute():
        _img_path = (_repo_root / _img_path).resolve()
    if not _img_path.exists():
        raise FileNotFoundError(f"Input image not found: {_img_path}")
    start_keyframe = Image.open(str(_img_path)).convert("RGB")
    start_keyframe = crop_to_square(start_keyframe)
    start_keyframe = start_keyframe.resize((512, 512))
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config["device"])
    # Keep the raw input frame as the immutable image used for the first
    # motion annotation.  Recomposition below may update image_latest.
    input_image = kf_gen.image_latest.detach().clone()
    annotation_image = input_image.detach().clone()

    if use_loaded_scene:
        print("[cache] Loading fixed initial scene; skipping scene reconstruction.")
        return _run_loaded_initial_scene(
            config=config,
            kf_gen=kf_gen,
            content_prompt=content_prompt,
            style_prompt=style_prompt,
            background_prompt=background_prompt,
        )

    # always check if sky image and sky pointcloud are existed or not, if not then generate
    example_name = config["example_name"]

    '''
    if exist an in-painted image, use that for sky mask for sky generation, should generally be better
    if the objects occluded the sky, we should use the in-painted image for sky mask
    if the objects occluded other layers, the in-painted image doesn't effect the sky mask
    '''
    _base_layer_path = Path(config["examples_dir"]) / f"{example_name}_base_layer.png"
    if _base_layer_path.exists():
        for_sky_image = Image.open(str(_base_layer_path)).convert("RGB")
        for_sky_image = crop_to_square(for_sky_image)
        for_sky_image = for_sky_image.resize((512, 512))
        for_sky_image = ToTensor()(for_sky_image).unsqueeze(0).to(config["device"])
        sky_mask = kf_gen.generate_sky_mask(input_image = for_sky_image).float()
    else:
        sky_mask = kf_gen.generate_sky_mask().float()
        for_sky_image = kf_gen.image_latest

    _sky_dir = Path(config["sky_image_dir"])
    config["gen_sky_image"] = not (
        (_sky_dir / "sky_0.png").exists() and (_sky_dir / "sky_1.png").exists()
    )
    config["gen_sky"] = not (_sky_dir / "finished_3dgs_sky_tanh.ply").exists()
    syncdiffusion_model = None
    if config["gen_sky_image"]:
        print("[INFO] Sky panorama cache is incomplete; loading SyncDiffusion.")
        syncdiffusion_model = SyncDiffusion(config["device"], sd_version="2.0-inpaint")
    else:
        print("[cache] Reusing sky panorama images; SyncDiffusion is not loaded.")

    content_list = content_prompt.split(",")
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {
        "scene_name": scene_name,
        "entities": entities,
        "style": style_prompt,
        "background": background_prompt,
    }
    kf_gen.generate_sky_pointcloud( # the image is actually not used here, just the sky_mask
        syncdiffusion_model,
        image=for_sky_image,
        mask=sky_mask,
        gen_sky=config["gen_sky_image"],
        style=style_prompt,
    )

    kf_gen.recompose_image_latest_and_set_current_pc(
        use_cached_assets=True, scene_name=scene_name
    )
    annotation_image = kf_gen.image_latest.detach().clone()
    annotation_depth = kf_gen.depth_latest.detach().clone()
    annotation_valid_mask = (~kf_gen.sky_mask_latest.bool()).detach().clone()

    kf_gen.increment_kf_idx()
    ###### ------------------ Main loop ------------------ ######

    if config["gen_sky"]:
        traindatas = kf_gen.convert_to_3dgs_traindata(
            xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False
        )
        if config["gen_layer"]:
            traindata, traindata_sky, traindata_layer = traindatas
        else:
            traindata, traindata_sky = traindatas
        gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
        opt = GSParams()
        opt.max_screen_size = (
            100  # Sky is supposed to be big; set a high max screen size
        )
        opt.scene_extent = 1.5  # Sky is supposed to be big; set a high scene extent
        opt.densify_from_iter = 200  # Need to do some densify
        opt.prune_from_iter = 200  # Don't prune for sky because sky 3DGS are supposed to be big; prevent it by setting a high prune iter
        opt.densify_grad_threshold = (
            1.0  # Do not need to densify; Set a high threshold to prevent densifying
        )
        opt.iterations = 399  # More iterations than 100 needed for sky
        scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_sky"
        train_gaussian(gaussians, scene, opt, save_dir, initialize_scaling=False)
        gaussians.save_ply_with_filter(
            str(Path(config["sky_image_dir"]) / "finished_3dgs_sky_tanh.ply")
        )
    else:
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(
            str(Path(config["sky_image_dir"]) / "finished_3dgs_sky_tanh.ply")
        )  # pure sky
    sky_gaussians = gaussians

    gaussians.visibility_filter_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    gaussians.is_sky_filter = torch.ones(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    gaussians.delete_mask_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    opt = GSParams()

    particle_num_sky = gaussians.get_xyz_all.shape[0]
    particle_num_base = 0
    particle_num_object = 0

    save_dir_sim = None
    if motion_type == "environment":
        save_dir_sim = kf_gen.run_dir / "simulation"
        save_dir_sim.mkdir(parents=True, exist_ok=True)
        prepare_environment_motion_fields(save_dir_sim, config)
    elif motion_type == "interaction":
        save_dir_sim = kf_gen.run_dir / "simulation"
        save_dir_sim.mkdir(parents=True, exist_ok=True)
        interaction_direction, _ = validate_interaction_config(config)
        if interaction_direction == "env2obj" and not config.get("multiview", {}).get("enabled", False):
            prepare_environment_motion_fields(save_dir_sim, config)

    ### First scene 3DGS
    if config["gen_layer"]:
        # traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
        traindata_list, traindata_layer = (
            kf_gen.convert_to_3dgs_traindata_list_latest_layer(xyz_scale=xyz_scale)
        )

        if isinstance(traindata_layer, list) and len(traindata_layer) == 2:
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer[0], gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_layer{0:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)  # Base layer training

            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer[1], gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_layer{0:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)  # Base layer training

            particle_num_base = gaussians.get_xyz_all.shape[0] - particle_num_sky

            traindata_layer = traindata_layer[1]
        else:
            if traindata_layer['pcd_points'].shape[1] > 0:  # [3, N]
                # some scenes, there are no points of base_layer
                gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
                scene = Scene(traindata_layer, gaussians, opt)
                dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
                save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_layer{0:02d}"
                train_gaussian(gaussians, scene, opt, save_dir)  # Base layer training
                particle_num_base = gaussians.get_xyz_all.shape[0] - particle_num_sky
    else:
        traindata_layer = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=xyz_scale, use_no_loss_mask=False
        )
        traindata_layer.setdefault("ground_value", 0.0)
        if traindata_layer["pcd_points"].shape[1] > 0:
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_layer{0:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)
            particle_num_base = gaussians.get_xyz_all.shape[0] - particle_num_sky
        traindata_list = []

    environment_scene = scene if "scene" in locals() else Scene(traindata_layer, gaussians, opt)

    gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
    i = 0
    # Boundary rules are only used by the physics simulator. Environment-only
    # motion exits below before simulator construction.
    if motion_type != "environment":
        config["boundary_rules"] = {"y_min": traindata_layer["ground_value"]}

    total_object_pts_num = 0
    object_pts_num_list = []
    gt_masks = []
    faces = []
    object_meshes_paths = []
    object_meshes_translations = []

    for train_data in traindata_list:
        no_loss_mask = train_data["frames"][0]["no_loss_mask"]
        gt_mask = 1.0 - no_loss_mask
        gt_mask = gt_mask.cuda()
        gt_masks.append(gt_mask)

        scene = Scene(train_data, gaussians, opt, is_obj=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene{i:02d}"
        train_gaussian(
            gaussians, scene, opt, save_dir, initialize_scaling=False, obj=True
        )

        object_pts_num = (
            gaussians._tmp_get_xyz_all_separate()[0].shape[0] - total_object_pts_num
        )
        total_object_pts_num += object_pts_num
        object_pts_num_list.append(object_pts_num)
        if motion_type != "environment":
            faces.append(train_data["faces"])
            object_meshes_paths.append(train_data["mesh_path"])
            object_meshes_translations.append(train_data["mesh_translation"])
        print(f"Object points number: {object_pts_num}")

    # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    # i = 0
    # save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i:02d}"

    # # here train the boat gaussian
    # train_gaussian(gaussians, scene, opt, save_dir, debug=True)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
        kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale
    )
    gaussians.set_inscreen_points_to_visible(tdgs_cam)

    particle_num_object = gaussians.get_xyz_all.shape[0] - particle_num_sky - particle_num_base

    # here the boat gaussians are ready
    save_dir_3d = kf_gen.run_dir / "3d_results"
    save_dir_3d.mkdir(parents=True, exist_ok=True)

    # ---------- Environment motion exit after layer/object reconstruction ----------
    # Environment motion still needs object 3DGS when gen_layer=True so the object
    # remains visible as a static layer while the environment/base points move.
    if motion_type == "environment":
        print("=" * 60)
        print("Environment Motion (LivingWorld HashGrid) - skipping physics only")
        print("=" * 60)
        print(
            "Environment reconstruction summary: "
            f"sky={particle_num_sky}, base={particle_num_base}, "
            f"object={particle_num_object}"
        )
        if save_dir_sim is None:
            save_dir_sim = kf_gen.run_dir / "simulation"
            save_dir_sim.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply_for_3dgs((save_dir_3d / "gaussians.ply").as_posix())
        environment_motion_rendering(
            gaussians,
            environment_scene,
            save_dir_sim,
            config,
            video_gen_fps=8,
            sky_gaussians=sky_gaussians,
        )
        print("Scene compositional reconstruction and environment motion finished.")
        return
    # ---------- End environment motion exit ----------

    # sim = 'debug'
    if sim is None:
        xyz_obj, xyz_env = (
            gaussians._tmp_get_xyz_all_separate()
        )  # here xyz is boat, xyz_prev is all other
        scaling_obj, scaling_env = gaussians._tmp_get_scaling_all_separate()
        rotation_obj, rotation_env = gaussians._tmp_get_rotation_all_separate()
        opacity_obj, opacity_env = gaussians._tmp_get_opacity_all_separate()
        features_dc_obj, features_dc_env = gaussians._tmp_get_features_dc_all_separate()

        xyz_max_obj = xyz_obj.max(dim=0)[0]
        xyz_min_obj = xyz_obj.min(dim=0)[0]
        y_mean = ((xyz_max_obj[1] + xyz_min_obj[1]) / 2).item()
        z_mean = ((xyz_max_obj[2] + xyz_min_obj[2]) / 2).item()
        x_mean = ((xyz_max_obj[0] + xyz_min_obj[0]) / 2).item()
        x_left_point = x_mean + (xyz_max_obj[0] - x_mean).item() * 5.0
        x_right_point = x_mean + (xyz_min_obj[0] - x_mean).item() * 5.0
        print(f"x_left_point: {x_left_point}, x_right_point: {x_right_point}")
        print(f"y_mean: {y_mean}, z_mean: {z_mean}")

        object_infos = []
        total_object_pts_num = 0
        for iid in range(len(object_pts_num_list)):
            object_infos.append(
                {
                    "xyz": xyz_obj.data.requires_grad_(False)[
                        total_object_pts_num : total_object_pts_num
                        + object_pts_num_list[iid]
                    ],
                    "rotation": rotation_obj.data.requires_grad_(False)[
                        total_object_pts_num : total_object_pts_num
                        + object_pts_num_list[iid]
                    ],
                    "scaling": scaling_obj.data.requires_grad_(False)[
                        total_object_pts_num : total_object_pts_num
                        + object_pts_num_list[iid]
                    ],
                    "features_dc": features_dc_obj.data.requires_grad_(False)[
                        total_object_pts_num : total_object_pts_num
                        + object_pts_num_list[iid]
                    ],
                    "opacity": opacity_obj.data.requires_grad_(False)[
                        total_object_pts_num : total_object_pts_num
                        + object_pts_num_list[iid]
                    ],
                    "faces": faces[iid],
                    'mesh_path': object_meshes_paths[iid],
                    'translation': object_meshes_translations[iid],
                }
            )
            total_object_pts_num += object_pts_num_list[iid]

        # duplicate the object info for the second object
        # object_infos.append({
        #     'xyz': xyz_obj.data.requires_grad_(False) + torch.tensor([-0.4, 0., 0], device=config['device']),
        #     'rotation': rotation_obj.data.requires_grad_(False),
        #     'scaling': scaling_obj.data.requires_grad_(False),
        #     'features_dc': features_dc_obj.data.requires_grad_(False),
        #     'opacity': opacity_obj.data.requires_grad_(False),
        # })

        simulation_steps = config["simulation_steps"]
        dt = config["dt"]
        video_gen_fps = 8
        video_gen_dt = 1 / video_gen_fps
        num_dt = int(video_gen_dt / dt)
        simulation_steps = num_dt * simulation_steps

        if simulation_steps < 1500:
            simulation_steps = 1500
        simulation_steps = 1000

        save_dir_sim = kf_gen.run_dir / "simulation"
        save_dir_sim.mkdir(parents=True, exist_ok=True)

        # setup simulation
        sim = Simulator(
            config=config,
            obj_gaussians=object_infos,
            env_gaussians={
                "xyz": xyz_env.data.requires_grad_(False),
                "rotation": rotation_env.data.requires_grad_(False),
                "scaling": scaling_env.data.requires_grad_(False),
                "features_dc": features_dc_env.data.requires_grad_(False),
                "opacity": opacity_env.data.requires_grad_(False),
            },
            delta_time=dt,
            save_dir=save_dir_sim,
        )

        save_initial_scene_cache(
            config=config,
            gaussians=gaussians,
            kf_gen=kf_gen,
            object_infos=object_infos,
            object_pts_num_list=object_pts_num_list,
            gt_masks=gt_masks,
            ground_value=traindata_layer["ground_value"],
            camera=tdgs_cam,
            particle_num_sky=particle_num_sky,
            particle_num_base=particle_num_base,
            particle_num_object=particle_num_object,
        )
        print(
            f"[cache] Saved fixed initial scene checkpoint to "
            f"{_initial_model_dir(config)}"
        )

        print("SAVING 3D RESULTS")
        save_dir_3d = kf_gen.run_dir / "3d_results"
        save_dir_3d.mkdir(parents=True, exist_ok=True)

        input_view_R = tdgs_cam.R.reshape(-1).tolist()
        input_view_T = tdgs_cam.T.reshape(-1).tolist()
        bbox_min = sim.obj_valid_min.reshape(-1).tolist()
        bbox_max = sim.obj_valid_max.reshape(-1).tolist()
        save_info = {
            "particle_num_object": particle_num_object,
            "particle_num_base": particle_num_base,
            "particle_num_sky": particle_num_sky,
            "width": 512,
            "height": 512,
            "position": input_view_T,
            "rotation": input_view_R,
            "fy": tdgs_cam.focal_y,
            "fx": tdgs_cam.focal_x,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }

        for tmp_id in range(len(object_pts_num_list)):
            save_info.update({f"obj_{tmp_id:04d}": object_pts_num_list[tmp_id]})
        
        save_info.update({'material_types': list(config['material_types'])})
        
        gaussians.save_ply_for_3dgs((save_dir_3d / "gaussians.ply").as_posix())
        with open((save_dir_3d / "info.json").as_posix(), "w") as f:
            json.dump(save_info, f, indent=4)
        f.close()
        # train_simulation(sim, scene, save_dir_sim)

        if motion_type == "interaction" and config.get("multiview", {}).get("enabled", False):
            from multiview_controller import run_multiview_controller

            run_multiview_controller(
                config=config,
                simulator=sim,
                scene=scene,
                save_dir=save_dir_sim,
                simulation_steps=simulation_steps,
                video_gen_fps=video_gen_fps,
            )
        elif motion_type == "interaction":
            print("=" * 60)
            print("Environment-Object Interaction")
            print("=" * 60)
            run_interaction_pipeline(
                sim,
                scene,
                save_dir_sim,
                config,
                simulation_steps,
                video_gen_fps,
            )
        else:
            print("=" * 60)
            print("Object Motion (WonderPlay Genesis)")
            print("=" * 60)
            simulation_efficient(
                sim,
                scene,
                save_dir_sim,
                config,
                gt_masks,
                object_pts_num_list,
                simulation_steps,
                video_gen_fps,
            )

    print("Scene compositional reconstruction and simulation finished.")


def simulation_efficient(simulator, scene, save_dir, config, gt_masks, object_pts_num_list, simulation_steps, video_gen_fps):
    global iter_number, kf_gen, gaussians, opt, background, view_matrix_wonder
    global movement

    """
    No training of simulation, just simulate N steps, render each step once it's finished so less memory consumption
    """

    if "text_prompt" not in config:
        print("No text prompt in config, using a void text prompt for now, please add it in the text_prompt.txt later")
        text_prompt = " "
    else:
        text_prompt = config["text_prompt"]

    with open((save_dir / "text_prompt.txt").as_posix(), "w") as f:
        f.write(text_prompt)
    f.close()

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack[0]
    gt_image = viewpoint_cam.original_image.cuda()
    ToPILImage()(gt_image).save(save_dir / f"gt.png")

    # object id mask for drag anything
    if 'drag_anything_object' not in config:
        print("No specific object id for drag anything, using the first object")
        da_id = 0
    else:
        print(f"Using object id {config['drag_anything_object']} for drag anything")
        da_id = config['drag_anything_object']

    # Single fixed camera (no movement) - only first trajectory
    number_traj = 1

    # for the gaussian renderings, we'll have outputs in:
    # frames / masks / depths / flows / flows_actual / flows_arrows for different camera trajectories
    save_infos = dict()
    save_kwargs = [
        "frames",
        "masks",
        "depths",
        "flows",
        "flows_actual",
        "flows_arrows",
    ]
    for tid in range(number_traj):
        save_traj_dir = save_dir / f"traj_{tid:02d}"
        save_traj_dir.mkdir(parents=True, exist_ok=True)

        traj_save_infos = dict()

        for save_name in save_kwargs:
            (save_traj_dir / save_name).mkdir(parents=True, exist_ok=True)
            traj_save_infos.update({
                save_name: {
                    "path": save_traj_dir / save_name,
                    "data": []
                }
            })
        
        save_infos.update({
            f"traj_{tid:02d}": traj_save_infos,
            f"traj_{tid:02d}_dir": save_traj_dir
        })

    save_genesis_subdir = save_dir / "genesis"
    save_genesis_subdir.mkdir(parents=True, exist_ok=True)

    save_4d_subdir = save_dir / "4d"
    save_4d_subdir.mkdir(parents=True, exist_ok=True)

    render_videos = []
    render_cv2 = []
    render_masks = []
    render_videos_side = []
    render_depths = []
    render_flows = []
    render_flows_arrows = []
    # mask for foreground
    # foreground_num = gaussians._opacity.shape[0] * len(list(sim_out[0].keys()))
    foreground_num = gaussians._opacity.shape[0]
    background_num = gaussians._opacity_prev.shape[0]
    render_mask = torch.zeros(foreground_num + background_num).to(config["device"])
    render_mask[:foreground_num] = 1
    render_mask = render_mask.bool()

    prev_sim_out_step = None
    prev_sim_out_step_cams = [None for _ in range(number_traj)]

    force_function_config = config["force_function"]
    force_function_module, force_function_func_name = force_function_config.rsplit(
        ".", 1
    )
    force_function_module = importlib.import_module(force_function_module)
    force_function = getattr(force_function_module, force_function_func_name)

    # gaussian_vis_freq = num_dt
    # gaussian_vis_freq = 50
    gaussian_vis_freq = 20

    for sid in tqdm(range(simulation_steps)):
        sim_out = simulator.simulate_step(sid, simulation_steps, save_genesis_subdir, gaussian_vis_freq)

        if sid % gaussian_vis_freq == 0:
            # in num_dt steps, record the output frame information
            sim_out_step = sim_out
            tid = 0  # single fixed camera (no movement)
            # fixed camera at origin
            x_value, y_value, z_value = 0.0, 0.0, 0.0
            view_matrix_wonder_here = [
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, 1, 0,
                -x_value, -y_value, z_value, 1,
            ]

            tdgs_cam_noisy = convert_pt3d_cam_to_3dgs_cam(
                kf_gen.get_camera_by_js_view_matrix(
                    view_matrix_wonder_here, xyz_scale=xyz_scale
                ),
                xyz_scale=xyz_scale,
            )

            # save the camera trajectory sequence in the corresponding folder
            cam_save_path = save_dir / f"traj_{tid:02d}" / "cams"
            if not os.path.exists(cam_save_path.as_posix()):
                cam_save_path.mkdir(parents=True, exist_ok=True)
            cam_info_save_path = cam_save_path / f"frame_{sid:08d}.txt"
            if isinstance(view_matrix_wonder_here, np.ndarray):
                view_matrix_wonder_here = view_matrix_wonder_here.tolist()
            np.savetxt(cam_info_save_path.as_posix(), view_matrix_wonder_here + [xyz_scale])

            # save the simulated xyz sequence once per step
            simulated_xyz = dict()
            for obj_name in sim_out_step.keys():
                simulated_xyz[obj_name] = sim_out_step[obj_name]["xyz"].detach().cpu().numpy()
            np.savez(save_4d_subdir / f"simulated_xyz_{sid:08d}.npz", **simulated_xyz)

            if prev_sim_out_step is None:
                prev_sim_out_step = {k: {kk: vv.clone().detach() for kk, vv in v.items()} for k, v in sim_out_step.items()}
            if prev_sim_out_step_cams[tid] is None:
                prev_sim_out_step_cams[tid] = tdgs_cam_noisy

            with torch.no_grad():
                render_pkg = render_w_shift_flow(
                    tdgs_cam_noisy,
                    sim_out_step,
                    gaussians,
                    opt,
                    background,
                    render_visible=True,
                    object_pts_num_list=object_pts_num_list,
                    prev_step=prev_sim_out_step,
                    prev_step_cam=prev_sim_out_step_cams[tid],
                )
                render_pkg_foreground = render_w_shift(
                    tdgs_cam_noisy,
                    sim_out_step,
                    gaussians,
                    opt,
                    background,
                    render_visible=True,
                    render_mask=render_mask,
                    object_pts_num_list=object_pts_num_list,
                )
                num_objects = len(sim_out_step.keys())
                render_mask_da = []
                for obj_id in range(num_objects):
                    obj_pts_num = sim_out_step[f"obj_{obj_id:04d}"]["xyz"].shape[0]
                    if obj_id == da_id:
                        render_mask_da_this = torch.ones(obj_pts_num).to(config["device"])
                    else:
                        render_mask_da_this = torch.zeros(obj_pts_num).to(config["device"])
                    render_mask_da.append(render_mask_da_this)
                render_mask_da.append(torch.zeros(background_num).to(config["device"]))
                render_mask_da = torch.cat(render_mask_da, dim=0)
                render_mask_da = render_mask_da.bool()
                render_pkg_da = render_w_shift_da(
                    tdgs_cam_noisy,
                    sim_out_step,
                    gaussians,
                    opt,
                    background,
                    render_visible=True,
                    render_mask=render_mask_da,
                    object_pts_num_list=object_pts_num_list,
                )
                # render masks for each object
                for obj_id in range(num_objects):
                    obj_pts_num = sim_out_step[f"obj_{obj_id:04d}"]["xyz"].shape[0]
                    render_mask_obj_this = []
                    for obj_id_this in range(num_objects):
                        if obj_id_this == obj_id:
                            render_mask_obj_this.append(torch.ones(obj_pts_num).to(config["device"]))
                        else:
                            render_mask_obj_this.append(torch.zeros(obj_pts_num).to(config["device"]))
                    render_mask_obj_this.append(torch.zeros(background_num).to(config["device"]))
                    render_mask_obj_this = torch.cat(render_mask_obj_this, dim=0)
                    render_mask_obj_this = render_mask_obj_this.bool()
                    render_pkg_obj = render_w_shift_da(
                        tdgs_cam_noisy,
                        sim_out_step,
                        gaussians,
                        opt,
                        background,
                        render_visible=True,
                        render_mask=render_mask_obj_this,
                        object_pts_num_list=object_pts_num_list,
                    )
                    mask_obj = render_pkg_obj["final_opacity"]
                    mask_obj = mask_obj.detach().cpu().permute(1, 2, 0).numpy()
                    mask_obj = (mask_obj > 0.).astype(np.uint8) * 255
                    mask_obj_path = save_infos[f"traj_{tid:02d}"]["masks"]["path"].as_posix()
                    mask_obj_path = mask_obj_path.replace("masks", f"masks_obj_{obj_id:04d}")
                    os.makedirs(mask_obj_path, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(mask_obj_path, f"frame_{sid:08d}.png"), mask_obj
                    )

            # frame output
            image = render_pkg["render"]
            image_pil = ToPILImage()(image)
            image_pil.save((save_infos[f"traj_{tid:02d}"]["frames"]["path"] / f"frame_{sid:08d}.png").as_posix())
            save_infos[f"traj_{tid:02d}"]["frames"]["data"].append(image_pil)
            foreground_mask = render_pkg_foreground["final_opacity"]
            foreground_mask = foreground_mask.detach().cpu().permute(1, 2, 0).numpy()
            foreground_mask = (foreground_mask > 0.5).astype(np.uint8) * 255
            cv2.imwrite(
                (save_infos[f"traj_{tid:02d}"]["masks"]["path"] / f"frame_{sid:08d}.png").as_posix(), foreground_mask
            )
            save_infos[f"traj_{tid:02d}"]["masks"]["data"].append(foreground_mask)
            foreground_mask_da = render_pkg_da["final_opacity"]
            foreground_mask_da = foreground_mask_da.detach().cpu().permute(1, 2, 0).numpy()
            foreground_mask_da = (foreground_mask_da > 0.5).astype(np.uint8) * 255
            masks_da_path = save_infos[f"traj_{tid:02d}"]["masks"]["path"].as_posix()
            masks_da_path = masks_da_path.replace("masks", "masks_da")
            os.makedirs(masks_da_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(masks_da_path, f"frame_{sid:08d}.png"), foreground_mask_da
            )
            depth = render_pkg["depth"]
            save_infos[f"traj_{tid:02d}"]["depths"]["data"].append(depth.cpu().numpy())
            depth_pil = ToPILImage()(depth)
            depth_pil.save((save_infos[f"traj_{tid:02d}"]["depths"]["path"] / f"depth_{sid:08d}.png").as_posix())
            flow = render_pkg["optical_flow"]
            flow = flow[:2, :, :]
            flow_actual = flow / 512.0
            flow_actual = (1 + flow_actual) * 0.5
            flow_actual = flow_actual.clamp(0, 1).detach().cpu().numpy()
            np.save((save_infos[f"traj_{tid:02d}"]["flows_actual"]["path"] / f"flow_{sid:08d}.npy").as_posix(), flow_actual)
            flow_arrows = visualize_flow_as_arrows(flow, image)
            save_infos[f"traj_{tid:02d}"]["flows_arrows"]["data"].append(flow_arrows)
            flow_frame = flow_to_image(flow)
            flow_pil = ToPILImage()(flow_frame)
            flow_pil.save((save_infos[f"traj_{tid:02d}"]["flows"]["path"] / f"flow_{sid:08d}.png").as_posix())
            save_infos[f"traj_{tid:02d}"]["flows"]["data"].append(flow)

            if sid == 0:
                first_frame_image = image.clone().detach()
                ToPILImage()(first_frame_image).save(save_dir / f"first_frame.png")
                all_object_masks = torch.zeros_like(gt_masks[0]).bool()
                for i in range(len(gt_masks)):
                    all_object_masks = torch.logical_or(
                        all_object_masks, gt_masks[i].bool()
                    )
                all_object_masks = all_object_masks.int()
                compose_image = gt_image * all_object_masks + image * (
                    1 - all_object_masks
                )
                ToPILImage()(compose_image).save(save_dir / f"compose_first_frame.png")

            del render_pkg
            del render_pkg_foreground
            prev_sim_out_step_cams[tid] = tdgs_cam_noisy

            del prev_sim_out_step
            prev_sim_out_step = {k: {kk: vv.clone().detach() for kk, vv in v.items()} for k, v in sim_out_step.items()}
            del sim_out_step

            if len(save_infos["traj_00"]["frames"]["data"]) == 50:
                break
    
    # Save masks as video
    genesis_images = sorted(list(save_genesis_subdir.glob("*_[0-9][0-9][0-9][0-9].png")))
    if len(genesis_images) > 0:
        genesis_frames = []
        for genesis_image in genesis_images:
            genesis_frame = cv2.imread(str(genesis_image))
            genesis_frames.append(genesis_frame)
        imageio.mimsave(
            (save_dir / "genesis_video.mp4").as_posix(),
            genesis_frames,
            fps=video_gen_fps
        )
    
    for tid in range(number_traj):
        traj_save_infos = save_infos[f"traj_{tid:02d}"]
        traj_save_dir = save_infos[f"traj_{tid:02d}_dir"]

        render_videos = traj_save_infos["frames"]["data"]
        render_masks = traj_save_infos["masks"]["data"]
        render_depths = traj_save_infos["depths"]["data"]
        render_flows = traj_save_infos["flows"]["data"]
        render_flows_arrows = traj_save_infos["flows_arrows"]["data"]
    
        render_videos[0].save(
            (traj_save_dir / "render_video.gif").as_posix(),
            save_all=True,
            append_images=render_videos[1:],
            fps=video_gen_fps,
            loop=0,
        )
        imageio.mimsave(
            (traj_save_dir / "render_video.mp4").as_posix(), render_videos, fps=video_gen_fps
        )

        render_depths = np.concatenate(render_depths, axis=0) # [T, H, W]
        render_depths = torch.from_numpy(render_depths).float().cuda()
        render_depths_min = render_depths.min()
        render_depths_max = render_depths.max()
        render_depths = (render_depths - render_depths_min) ** 2 / (render_depths_max - render_depths_min) ** 2
        render_depths = [ToPILImage()(1 - depth) for depth in render_depths]
        imageio.mimsave(
            (traj_save_dir / "render_depths.mp4").as_posix(), render_depths, fps=video_gen_fps
        )
        (traj_save_dir / "depths_compose").mkdir(parents=True, exist_ok=True)
        for i in range(len(render_depths)):
            render_depths[i].save((traj_save_dir / "depths_compose" / f"frame_{i:08d}.png").as_posix())

        render_flows = torch.stack(render_flows, dim=0)
        render_flows = flow_to_image(render_flows)
        render_flows = [ToPILImage()(flow) for flow in render_flows]
        imageio.mimsave(
            (traj_save_dir / "render_flows.mp4").as_posix(), render_flows, fps=video_gen_fps
        )
        (traj_save_dir / "flows_compose").mkdir(parents=True, exist_ok=True)
        for i in range(len(render_flows)):
            render_flows[i].save((traj_save_dir / "flows_compose" / f"frame_{i:08d}.png").as_posix())
    
        render_flows_arrows = torch.stack(render_flows_arrows, dim=0)
        render_flows_arrows = [ToPILImage()(flow_arrow) for flow_arrow in render_flows_arrows]
        imageio.mimsave(
            (traj_save_dir / "render_flows_arrows.mp4").as_posix(), render_flows_arrows, fps=video_gen_fps
        )
        (traj_save_dir / "flows_arrows").mkdir(parents=True, exist_ok=True)
        for i in range(len(render_flows_arrows)):
            render_flows_arrows[i].save((traj_save_dir / "flows_arrows" / f"frame_{i:08d}.png").as_posix())
        
    # simulator.init_simulation_list()
    print("simulation efficient finish")
    return None


def train_gaussian(
    gaussians: GaussianModel,
    scene: Scene,
    opt: GSParams,
    save_dir: Path,
    initialize_scaling=True,
    obj=False,
):
    global iter_number, view_matrix, already_object_pts_num
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(
        cameras=trainCameras, initialize_scaling=initialize_scaling
    )

    if obj:
        opt.densify_from_iter = opt.iterations + 10
        iterable_gauss = range(0, 1)  # TODO: debug how to use mask supervision

    for iteration in iterable_gauss:
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # import pdb; pdb.set_trace()
        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # if iteration % 1000 == 0 or iteration == opt.iterations:
        #     ToPILImage()(image).save(save_dir / f'it{iteration:04d}_rendered.png')
        #     ToPILImage()(gt_image).save(save_dir / f'it{iteration:04d}_gt.png')

        if obj:
            no_loss_mask = scene.traindata["frames"][0]["no_loss_mask"]
            # no_loss_mask = None
            gt_mask = 1.0 - no_loss_mask
            gt_mask = gt_mask.cuda()

            n_trainable = gaussians.get_xyz.shape[0] - already_object_pts_num
            n_all = viewspace_point_tensor.shape[0]
            if n_trainable <= 0 or already_object_pts_num + n_trainable > n_all:
                raise RuntimeError(
                    "Invalid object Gaussian range for mask supervision: "
                    f"start={already_object_pts_num}, count={n_trainable}, total={n_all}"
                )
            foreground_mask = torch.zeros(
                n_all,
                dtype=torch.bool,
                device=viewspace_point_tensor.device,
            )
            foreground_mask[
                already_object_pts_num : (already_object_pts_num + n_trainable)
            ] = True

            render_foreground_pkg = render(
                viewpoint_cam, gaussians, opt, background, render_mask=foreground_mask
            )
            output_mask = render_foreground_pkg["final_opacity"]

            loss = ((output_mask - gt_mask) ** 2).mean()
        else:
            no_loss_mask = None

            Ll1 = l1_loss(image, gt_image, no_loss_mask=no_loss_mask)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt_image, no_loss_mask=no_loss_mask)
            )
        if iteration == opt.iterations:
            # if iteration % 5 == 0 or iteration == 1:
            time.sleep(0.1)
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                    kf_gen.get_camera_by_js_view_matrix(
                        view_matrix, xyz_scale=xyz_scale
                    ),
                    xyz_scale=xyz_scale,
                )
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                image = render_pkg["render"]
            # (optional: could save debug render here)
        loss.backward()
        if iteration == opt.iterations:
            print(f"Final loss: {loss.item()}")

        if not obj:
            # Use variables that related to the trainable GS
            n_trainable = gaussians.get_xyz.shape[0]
            viewspace_point_tensor_grad, visibility_filter, radii = (
                viewspace_point_tensor.grad[:n_trainable],
                visibility_filter[:n_trainable],
                radii[:n_trainable],
            )

            with torch.no_grad():
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_grad, visibility_filter
                    )

                    if (
                        iteration >= opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        max_screen_size = (
                            opt.max_screen_size
                            if iteration >= opt.prune_from_iter
                            else None
                        )
                        camera_height = 0.0003 * xyz_scale
                        scene_extent = (
                            camera_height * 2
                            if opt.scene_extent is None
                            else opt.scene_extent
                        )
                        opacity_lowest = 0.05
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opacity_lowest,
                            scene_extent,
                            max_screen_size,
                        )
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    # if (iteration % opt.opacity_reset_interval == 0
                    #     or (opt.white_background and iteration == opt.densify_from_iter)
                    # ):
                    #     gaussians.reset_opacity()

                # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                #     if iteration < opt.iterations - 100:
                #         # don't update in the end of training
                #         gaussians.compute_3D_filter(cameras=trainCameras)

        else:
            if iteration >= 80 and iteration % 100 == 0:
                opacity_lowest = 0.05
                prune_mask = (gaussians.get_opacity < opacity_lowest).squeeze()
                gaussians.prune_points(prune_mask)
                gaussians.compute_3D_filter(cameras=trainCameras)
            pass
        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    if obj:
        already_object_pts_num += n_trainable


if __name__ == "__main__":
    parser = ArgumentParser(description="WonderPlay: scene compositional reconstruction and simulation")
    parser.add_argument(
        "--config",
        default="examples/configs/venice.yaml",
        help="Path to config YAML (default: examples/configs/venice.yaml)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional string for run directory naming (overrides current time if provided).",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root (parent of WonderPlay_new when running from there)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = OmegaConf.load(str(config_path))

    # Ensure required config keys with defaults for one-click venice run
    OmegaConf.set_struct(config, False)
    if "runs_dir" not in config:
        config["runs_dir"] = config.get("work_dir", "3d_result/wonderplay/venice")
    if "num_scenes" not in config:
        config["num_scenes"] = 1
    if "rotation_path" not in config:
        config["rotation_path"] = [0]  # single scene
    # Keep this key in the structured config so physics branches can update it
    # with the automatically estimated ground height.
    if "boundary_rules" not in config:
        config["boundary_rules"] = {}
    if "force_function" not in config and "force_function_name" in config:
        config["force_function"] = f"simulator.genesis_functions.{config['force_function_name']}"
    if "use_gpt" not in config:
        config["use_gpt"] = False
    if "debug" not in config:
        config["debug"] = False
    OmegaConf.set_struct(config, True)

    run(config, dt_string=args.prefix)
