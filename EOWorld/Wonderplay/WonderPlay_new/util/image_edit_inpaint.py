from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import ToTensor


class ImageEditInpaintPipeline:
    """Adapter that lets Qwen ImageEdit fit WonderPlay's inpainter API.

    WonderPlay's inpainting call site still passes ``mask_image`` for the
    Stable Diffusion path.  Qwen ImageEdit Plus performs better here as a
    mask-free image edit, so this adapter accepts that argument for API
    compatibility but intentionally does not forward it.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = "cuda"
        self.uses_image_edit = True

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

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image=None,
        mask_image=None,
        num_inference_steps=40,
        guidance_scale=3.0,
        height=None,
        width=None,
        generator=None,
        image_edit_seed=None,
        **kwargs,
    ):
        if image is None:
            raise ValueError("ImageEditInpaintPipeline requires an image.")

        edit_prompt = prompt.strip() if prompt else ""

        resampling = getattr(Image, "Resampling", Image)
        target_size = (
            (width, height)
            if (width is not None and height is not None)
            else image.size
        )
        edit_image = image.convert("RGB").resize(target_size, resampling.LANCZOS)
        if generator is None:
            seed = torch.initial_seed() if image_edit_seed is None else int(image_edit_seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        inputs = {
            "image": edit_image,
            "prompt": edit_prompt,
            "generator": generator,
            "height": target_size[1],
            "width": target_size[0],
            "true_cfg_scale": kwargs.get("true_cfg_scale", 7.0),
            "negative_prompt": negative_prompt or " ",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }
        with torch.inference_mode():
            output = self.pipeline(**inputs)

        output_image = output.images[0]
        if isinstance(output_image, torch.Tensor):
            output_tensor = output_image.detach().to(self.device)
            if output_tensor.ndim == 4:
                output_tensor = output_tensor[0]
            output_tensor = output_tensor.clamp(0, 1)
        else:
            output_tensor = ToTensor()(output_image.convert("RGB")).to(self.device)
        output_tensor = output_tensor * 2 - 1
        return SimpleNamespace(images=[output_tensor])
