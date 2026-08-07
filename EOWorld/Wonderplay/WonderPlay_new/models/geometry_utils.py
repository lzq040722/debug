"""Geometry, camera, and image utilities for WonderPlay."""
import torch
import cv2
from kornia.geometry import PinholeCamera


def debug_vis_func(inp, name="tmp"):
    inp = inp.clone()
    if len(inp.shape) == 4:
        inp = inp[0]
    if len(inp.shape) == 3:
        if inp.shape[0] == 1 or inp.shape[0] == 3:
            inp = inp.permute(1, 2, 0)
            if inp.shape[-1] == 3:
                inp = inp[:, :, [2, 1, 0]]
    if inp.max() <= 1.1:
        inp = inp * 255
    inp = inp.detach().cpu().numpy().astype("uint8")
    cv2.imwrite(f"{name}.png", inp)


def get_extrinsics(camera):
    extrinsics = torch.cat([camera.R[0], camera.T.T], dim=1)
    padding = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)
    return torch.cat([extrinsics, padding], dim=0)


def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    assert points.dim() == 2 and points.size(1) == 3, "Points should be [N, 3]."
    if colors is not None:
        assert colors.dim() == 2 and colors.size(1) == 3 and points.size(0) == colors.size(0)
    header = [
        "ply", "format ascii 1.0", f"element vertex {points.size(0)}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"
            if colors is not None:
                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"
            f.write(line + "\n")


def convert_pytorch3d_kornia(camera, focal_length, size=512, update_intrinsics_parameters=None, new_size=512):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    pt3d_to_kornia = torch.diag(torch.tensor([-1.0, -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d
    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = K[0, 1, 2] = size // 2
    K[0, 0, 0] = K[0, 1, 1] = focal_length
    if update_intrinsics_parameters is not None:
        u0, v0, w_crop, h_crop, p_left, p_right, p_up, p_down, scale = update_intrinsics_parameters
        K[0, 0, 2] = (K[0, 0, 2] - u0 + p_left) * scale
        K[0, 1, 2] = (K[0, 1, 2] - v0 + p_up) * scale
        K[0, 0, 0] = K[0, 0, 0] * scale
        K[0, 1, 1] = K[0, 1, 1] * scale
        return PinholeCamera(K, extrinsics, torch.tensor([new_size], device="cuda"), torch.tensor([new_size], device="cuda"))
    return PinholeCamera(K, extrinsics, h, w)


def inpaint_cv2(rendered_image, mask_diff):
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype("uint8")
    mask_cv2 = (mask_diff[0, 0].cpu().numpy() * 255).astype("uint8")
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    return torch.from_numpy(inpainting).permute(2, 0, 1).float().unsqueeze(0) / 255
