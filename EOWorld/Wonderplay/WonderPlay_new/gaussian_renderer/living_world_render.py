"""
LivingWorld environment motion rendering functions
Copied from LivingWorld/gaussian_renderer/__init__.py
"""

import torch
import math
import time
import numpy as np
from flow_viz import flow_uv_to_colors
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer


def flow_to_rgb(flow, clip=None):
    """
    flow: [H,W,2]  (u,v)
    return: [H,W,3] in [0,1]
    """
    u, v = flow[...,0], flow[...,1]
    mag = torch.sqrt(u*u + v*v)
    ang = torch.atan2(v, u)
    hue = (ang + torch.pi) / (2*torch.pi)
    sat = torch.ones_like(hue)

    if clip is None:
        clip = torch.quantile(mag, 0.95).clamp(min=1e-8)
    val = (mag / clip).clamp(0, 1)

    h6 = (hue * 6) % 6
    c  = val * sat
    x  = c * (1 - torch.abs(h6 % 2 - 1))
    z  = torch.zeros_like(c)

    r = torch.where((0<=h6)&(h6<1), c, torch.where((1<=h6)&(h6<2), x, torch.where((2<=h6)&(h6<3), z, torch.where((3<=h6)&(h6<4), z, torch.where((4<=h6)&(h6<5), x, c)))))
    g = torch.where((0<=h6)&(h6<1), x, torch.where((1<=h6)&(h6<2), c, torch.where((2<=h6)&(h6<3), c, torch.where((3<=h6)&(h6<4), x, torch.where((4<=h6)&(h6<5), z, z)))))
    b = torch.where((0<=h6)&(h6<1), z, torch.where((1<=h6)&(h6<2), z, torch.where((2<=h6)&(h6<3), x, torch.where((3<=h6)&(h6<4), c, torch.where((4<=h6)&(h6<5), c, x)))))
    m = val - c
    rgb = torch.stack([r+m, g+m, b+m], dim=-1).clamp(0,1)
    return rgb


def pre_euler_integral(xyz, model, T, smooth):
    if (model is None) or (not hasattr(model, "forward")):
        raise ValueError("motion_model not ready")

    dev, dt = xyz.device, xyz.dtype

    smooth = smooth.to(dev, dtype=dt).view(1, -1)
    f = torch.empty((T, *xyz.shape), device=dev, dtype=dt); f[0]=xyz
    b = torch.empty((T, *xyz.shape), device=dev, dtype=dt); b[0]=xyz

    with torch.no_grad():
        for i in range(1, T):
            v  = model(f[i-1])
            if v.shape != f[i-1].shape:
                raise RuntimeError(f"model out {v.shape} != {f[i-1].shape}")
            f[i] = f[i-1] + v * smooth

            vb = model(b[i-1])
            if vb.shape != b[i-1].shape:
                raise RuntimeError(f"model out {vb.shape} != {b[i-1].shape}")
            b[i] = b[i-1] - vb * smooth
    return f, b


def _check(*tensors):
    n = None
    for name, t in tensors:
        if t is None: continue
        assert t.is_cuda, f"{name} device"
        assert torch.isfinite(t).all(), f"{name} has NaN/Inf"
        if n is None: n = t.shape[0]
        else: assert t.shape[0]==n, f"len mismatch: {name}"
    return n


def render_MLP(viewpoint_camera, pc, motion_model, t, opt, bg_color: torch.Tensor,
               scaling_modifier=1.0, override_color=None, flow_render=False,
               render_visible=False, exclude_sky=False, sam_mask=None, scale_factor=2.0):
    """
    Render the scene with MLP motion model (LivingWorld).

    Background tensor (bg_color) must be on GPU!
    """

    if not hasattr(render_MLP, "_cache"):
        render_MLP._cache = {
            "key": None,
            "f_pos": None,
            "b_pos": None,
        }

    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

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
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scene_flow = pc.get_scene_flow_all.detach()
    motion_mask = pc.get_motion_mask_all.detach()

    means3D = pc.get_xyz_all
    means2D = screenspace_points
    opacity = pc.get_opacity_all

    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            colors_precomp = pc.color_activation(shs_view)
        else:
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = pc.visibility_filter_all & ~pc.delete_mask_all
    else:
        visibility_filter_all = ~pc.delete_mask_all

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter

    def dup(x):
        return None if x is None else torch.cat([x, x], dim=0)

    if motion_model is None:
        means3D = means3D[visibility_filter_all]
        means2D = means2D[visibility_filter_all]
        shs = None if shs is None else shs[visibility_filter_all]
        colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
        scene_flow = scene_flow[visibility_filter_all]
        motion_mask = motion_mask[visibility_filter_all]
        opacity = opacity[visibility_filter_all]
        scales = scales[visibility_filter_all]
        rotations = rotations[visibility_filter_all]
        cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    else:
        T=100

        means3D = means3D[visibility_filter_all]

        motion_mask = motion_mask[visibility_filter_all][:, 0].bool()
        motion_pts = means3D[motion_mask]

        ax_len = torch.max(motion_pts, dim=0)[0] - torch.min(motion_pts, dim=0)[0]

        smooth = (1.2 / T * torch.tensor([0.5,0.5,2.3], device=means3D.device)) * scale_factor

        torch.cuda.synchronize()
        t0 = time.time()

        cache_key = (
            T,
            float(scale_factor),
            int(motion_pts.shape[0]),
            float(smooth[0].item()), float(smooth[1].item()), float(smooth[2].item()),
            float(motion_pts[:,0].mean().item()), float(motion_pts[:,1].mean().item()), float(motion_pts[:,2].mean().item()),
            float(motion_pts[:,0].std().item()),  float(motion_pts[:,1].std().item()),  float(motion_pts[:,2].std().item()),
        )

        c = render_MLP._cache
        if c["key"] != cache_key:
            torch.cuda.synchronize()
            t0 = time.time()

            f_pos, b_pos = pre_euler_integral(motion_pts.detach(), motion_model, T+1, smooth)

            torch.cuda.synchronize()
            t1 = time.time()
            print(f"[timing] pre_euler_integral (recompute) {(t1-t0)*1000:.2f} ms")

            c["key"] = cache_key
            c["f_pos"] = f_pos
            c["b_pos"] = b_pos
        else:
            f_pos, b_pos = c["f_pos"], c["b_pos"]

        torch.cuda.synchronize()
        t1 = time.time()

        b_idx = T - t if T is not None else t
        pos_f = f_pos[t]
        pos_b = b_pos[b_idx]

        means3D_f = means3D.clone()
        means3D_b = means3D.clone()
        means3D_o = means3D.clone()

        moving_idx = motion_mask.nonzero(as_tuple=False).squeeze()
        means3D_f[moving_idx] = pos_f
        means3D_b[moving_idx] = pos_b

        forward = means3D_f-means3D_o
        backward = means3D_b-means3D_o

        alpha = t / T

        w_f = 1 - alpha
        w_b = alpha

        opacity_base  = opacity[visibility_filter_all]
        opacity = torch.cat([opacity_base * w_f, opacity_base * w_b], dim=0)

        means3D = torch.cat([means3D_f, means3D_b], dim=0)

        means2D = dup(means2D[visibility_filter_all])
        shs = dup(None if shs is None else shs[visibility_filter_all])
        colors_precomp = dup(None if colors_precomp is None else colors_precomp[visibility_filter_all])

        scales = dup(scales[visibility_filter_all])
        rotations = dup(rotations[visibility_filter_all])
        cov3D_precomp = dup(None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all])

        _check(
            ("means3D", means3D), ("means2D", means2D),
            ("opacity", opacity), ("scales", scales),
            ("rot", rotations), ("cov", cov3D_precomp),
            ("shs", shs), ("colors", colors_precomp),
        )

        if flow_render:
            flow2d_f = forward[:, :2].detach().cpu().numpy()
            u_f = -flow2d_f[:, 0]
            v_f = -flow2d_f[:, 1]
            color_rgb_f = flow_uv_to_colors(u_f, v_f, convert_to_bgr=False)
            color_rgb_f = (color_rgb_f / 255.0).astype(np.float32)
            color_rgb_tensor_f = torch.from_numpy(color_rgb_f).to(scene_flow.device)

            zero_forward = color_rgb_tensor_f

            flow2d_b = backward[:, :2].detach().cpu().numpy()
            u_b = -flow2d_b[:, 0]
            v_b = -flow2d_b[:, 1]
            color_rgb_b = flow_uv_to_colors(u_b, v_b, convert_to_bgr=False)
            color_rgb_b = (color_rgb_b / 255.0).astype(np.float32)
            color_rgb_tensor_b = torch.from_numpy(color_rgb_b).to(scene_flow.device)

            zero_backward = color_rgb_tensor_b

            color_rgb = torch.cat([zero_forward, zero_backward], dim=0)
            colors_precomp = color_rgb

    # WonderPlay rasterizer expects feats3D (Nx20) and delta (Nx3)
    # Create zero tensors to avoid illegal memory access in C++ kernel
    N = means3D.shape[0]
    dummy_feats3D = torch.zeros((N, 20), device=means3D.device, dtype=means3D.dtype)
    dummy_delta = torch.zeros((N, 3), device=means3D.device, dtype=means3D.dtype)

    rendered_image, radii, feats, depth, flow = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        feats3D = dummy_feats3D,
        delta = dummy_delta)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,}
