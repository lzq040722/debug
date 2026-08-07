"""Point and mesh renderer utilities for WonderPlay."""
from typing import List, Optional, Tuple, Union, Sequence, NamedTuple

import torch
import torch.nn as nn
from pytorch3d.renderer import hard_rgb_blend
from pytorch3d.renderer.points.compositor import _add_background_color_to_images

BG_COLOR = (1, 0, 0)


class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(
        self,
        point_clouds,
        return_z=False,
        return_bg_mask=False,
        return_fragment_idx=False,
        **kwargs,
    ) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0
        images = self.compositor(
            fragment_idx,
            fragments.zbuf.permute(0, 3, 1, 2),
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        images = images.permute(0, 2, 3, 1)
        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())
        if len(ret) == 1:
            ret = images
        return ret


class SoftmaxImportanceCompositor(torch.nn.Module):
    def __init__(
        self,
        background_color: Optional[Union[Tuple, List, torch.Tensor]] = None,
        softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        background_color = kwargs.get("background_color", self.background_color)
        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = -1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)
        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(
            ptclds.shape[0],
            fragments.shape[0],
            fragments.shape[1],
            fragments.shape[2],
            fragments.shape[3],
        )
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


class MyBlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (0.0, 0.0, 0.0)


class HardShader(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = (
            blend_params if blend_params is not None else MyBlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            raise ValueError("Cameras must be specified at init or in forward.")
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)
