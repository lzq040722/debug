import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


model_path = "/root/autodl-tmp/huggingface/hub/models--sd2-community--stable-diffusion-2-inpainting/snapshots/5f74973cbb64c8568780732c17f43eb269d63a0d"

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("pipeline loaded")

pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)


# 原图
image = Image.open(
    "examples/imgs/venice/image.png"
).convert("RGB")

# mask：
# 白色区域 = 船的位置，要重绘
# 黑色区域 = 其他部分，不修改
mask = Image.open(
    "3d_result/wonderplay/venice/Gen-11-09_14-53-17/segmentation/object_00.png"
).convert("L")


prompt = (
    "Remove the boat from the river and naturally fill the masked area "
)


inputs = {
    "image": image,
    "mask_image": mask,
    "prompt": prompt,

    "generator": torch.Generator(device="cuda").manual_seed(11),

    "true_cfg_scale": 1.0,
    "negative_prompt": " ",

    "num_inference_steps": 40,

    # 局部区域修改强度
    "strength":1.0,

    "guidance_scale": 3.0,
    "num_images_per_prompt": 1,
}


with torch.inference_mode():
    output = pipeline(**inputs)

output_image = output.images[0]

output_image.save("Temp/output_image_edit_mask_SD2I.png")

print(
    "image saved at",
    os.path.abspath("Temp/output_image_edit_mask_SD2I.png")
)