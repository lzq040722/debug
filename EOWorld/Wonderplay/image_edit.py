import gc
import time
import torch
from PIL import Image
from modelscope import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "/root/autodl-tmp/huggingface/hub",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
print("pipeline loaded")

pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image1 = Image.open(
    "3d_result/wonderplay/venice/Gen-13-08_22-07-45/segmentation/foreground_removed_hole.png"
)
prompt = "Remove the boat from the river."

iteration = 0
try:
    while True:
        inputs = {
            "image": [image1],
            "prompt": prompt,
            "generator": torch.Generator(device="cuda").manual_seed(iteration),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            _ = output.images[0]

        del output
        gc.collect()
        torch.cuda.empty_cache()

        iteration += 1
        print(f"iteration {iteration} done")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("stopped")

# import os
# import torch
# from PIL import Image
# from diffusers import StableDiffusionInpaintPipeline


# model_path = "/root/autodl-tmp/huggingface/hub/models--sd2-community--stable-diffusion-2-inpainting/snapshots/5f74973cbb64c8568780732c17f43eb269d63a0d"

# pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     local_files_only=True,
# )

# print("pipeline loaded")

# pipeline.to("cuda")
# pipeline.set_progress_bar_config(disable=None)


# # 原图
# image = Image.open(
#     "examples/imgs/venice/image.png"
# ).convert("RGB")

# # mask：
# # 白色区域 = 船的位置，要重绘
# # 黑色区域 = 其他部分，不修改
# mask = Image.open(
#     "3d_result/wonderplay/venice/Gen-13-08_22-07-45/segmentation/sam_mask_03.png"
# ).convert("L")


# prompt = (
#     "Remove the boat from the river and naturally fill the masked area "
# )


# inputs = {
#     "image": image,
#     "mask_image": mask,
#     "prompt": prompt,

#     "generator": torch.Generator(device="cuda").manual_seed(0),

#     "true_cfg_scale": 7.0,
#     "negative_prompt": " ",

#     "num_inference_steps": 40,

#     # 局部区域修改强度
#     "strength": 0.95,

#     "guidance_scale": 3.0,
#     "num_images_per_prompt": 1,
# }


# with torch.inference_mode():
#     output = pipeline(**inputs)

# output_image = output.images[0]

# output_image.save("output_image_edit_mask_SD2I.png")

# print(
#     "image saved at",
#     os.path.abspath("output_image_edit_mask.png")
# )


# import os
# import torch
# from PIL import Image
# from diffusers import QwenImageEditInpaintPipeline


# model_path = "/root/autodl-tmp/huggingface/hub"

# pipeline = QwenImageEditInpaintPipeline.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     local_files_only=True,
# )

# print("pipeline loaded")

# pipeline.to("cuda")
# pipeline.set_progress_bar_config(disable=None)


# # 原图
# image = Image.open(
#     "examples/imgs/venice/image.png"
# ).convert("RGB")

# # mask：
# # 白色区域 = 船的位置，要重绘
# # 黑色区域 = 其他部分，不修改
# mask = Image.open(
#     "3d_result/wonderplay/venice/Gen-13-08_22-07-45/segmentation/sam_mask_03.png"
# ).convert("L")


# prompt = (
#     "Remove the boat from the river "
# )


# inputs = {
#     "image": image,
#     "mask_image": mask,
#     "prompt": prompt,

#     "generator": torch.Generator(device="cuda").manual_seed(0),

#     "true_cfg_scale": 7.0,
#     "negative_prompt": " ",

#     "num_inference_steps": 40,

#     # 局部区域修改强度
#     "strength": 0.95,

#     "guidance_scale": 3.0,
#     "num_images_per_prompt": 1,
# }


# with torch.inference_mode():
#     output = pipeline(**inputs)

# output_image = output.images[0]

# output_image.save("output_image_edit_mask.png")

# print(
#     "image saved at",
#     os.path.abspath("output_image_edit_mask.png")
# )
