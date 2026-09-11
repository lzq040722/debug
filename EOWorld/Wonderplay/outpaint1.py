import os
import torch
from PIL import Image
from modelscope import QwenImageEditPlusPipeline


model_path = "/root/autodl-tmp/huggingface/hub"

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("pipeline loaded")

pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)


# 原图
image = Image.open(
    "3d_result/wonderplay/venice/Gen-02-09_21-09-32/multiview_000/generation_condition.png"
).convert("RGB")
# image = Image.open(
#     "3d_result/wonderplay/venice/Gen-11-09_14-53-17/segmentation/foreground_removed_hole.png"
# ).convert("RGB")


prompt = (
    "Outpaint the missing area as a seamless continuation of the existing scene. Match the perspective, composition, lighting, color tone, texture, and visual style of the original image. Make the water surface especially continuous across the mask boundary, with matching reflections, ripples, brightness, and shadow direction. Preserve the visible content and blend the completed region naturally."
)

# prompt = ("Remove the boat on the river.")

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.Generator(device="cuda").manual_seed(0),
    "height" : 512,
    "width": 512,

    "true_cfg_scale": 7.0,
    "negative_prompt": "visible seam, hard edge, mismatched lighting, inconsistent reflections, broken water ripples, color shift, blurry artifacts, distorted buildings, text, watermark",

    "num_inference_steps": 30,

    "guidance_scale": 3.0,
    "num_images_per_prompt": 1,
}

# inputs = {
#     "image": image,
#     "prompt": prompt,
#     "generator": torch.Generator(device="cuda").manual_seed(1),
#     "height" : 512,
#     "width": 512,

#     "true_cfg_scale": 7.0,
#     "num_inference_steps": 40,
#     "num_images_per_prompt": 1,
# }


with torch.inference_mode():
    output = pipeline(**inputs)

output_image = output.images[0]

output_image.save("Temp/outpaint.png")

print(
    "image saved at",
    os.path.abspath("Temp/outpaint.png")
)
