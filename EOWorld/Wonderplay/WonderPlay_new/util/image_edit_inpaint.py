from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


class ImageEditInpaintPipeline:
    """Small adapter that makes Qwen Image-Edit look like the old inpaint pipe."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = "cuda"

    @classmethod
    def from_pretrained(
        cls,
        model_path="/root/autodl-tmp/huggingface/hub",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        **kwargs,
    ):
        from modelscope import QwenImageEditPlusPipeline

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        return cls(pipeline)

    def to(self, device):
        self.device = device
        self.pipeline.to(device)
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(*args, **kwargs)

    @staticmethod
    def _masked_rgba(image, mask_image):
        image = image.convert("RGBA")
        mask = np.array(mask_image.convert("L"))
        alpha = np.full(mask.shape, 255, dtype=np.uint8)
        alpha[mask > 0] = 0
        image.putalpha(Image.fromarray(alpha))
        return image

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image=None,
        mask_image=None,
        num_inference_steps=40,
        guidance_scale=4.0,
        generator=None,
        **kwargs,
    ):
        if image is None or mask_image is None:
            raise ValueError("ImageEditInpaintPipeline requires image and mask_image.")

        if prompt:
            edit_prompt = (
                "Remove the object in the transparent masked area and fill it with "
                f"realistic background content matching: {prompt.strip()}."
            )
        else:
            edit_prompt = "Fill the transparent masked area with realistic background."
        if negative_prompt:
            edit_prompt = f"{edit_prompt}. Do not include: {negative_prompt}."

        edit_image = self._masked_rgba(image, mask_image)
        if generator is None:
            generator = torch.manual_seed(torch.initial_seed())

        inputs = {
            "image": [edit_image],
            "prompt": edit_prompt,
            "generator": generator,
            "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
            "negative_prompt": negative_prompt or " ",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }
        with torch.inference_mode():
            output = self.pipeline(**inputs)

        output_image = output.images[0].convert("RGB")
        output_tensor = ToTensor()(output_image).to(self.device)
        output_tensor = output_tensor * 2 - 1
        return SimpleNamespace(images=[output_tensor])
