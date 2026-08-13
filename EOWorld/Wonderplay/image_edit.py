import os
import torch
from PIL import Image
from modelscope import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("/root/autodl-tmp/huggingface/hub", torch_dtype=torch.bfloat16,local_files_only = True)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("examples/imgs/venice/image.png")
prompt = "Remove the boat on the river."
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_2511.png")
    print("image saved at", os.path.abspath("output_image_edit_2511.png"))
