import base64
import binascii
from urllib.request import urlopen
from openai import OpenAI

client = OpenAI(
    api_key="sk-12ec9931a13f5b055ab4cb1b4b7db227af4d967cbe62492ae0db68825b217261",
    base_url="https://api.cisct.xyz/v1"
)

with open("3d_result/wonderplay/venice/Gen-02-09_21-09-32/multiview_000/generation_condition.png", "rb") as img:
    response = client.images.edit(
        model="gpt-image-2",
        image=img,
        prompt="Outpaint the missing area as a seamless continuation of the existing scene. Match the perspective, composition, lighting, color tone, texture, and visual style of the original image. Make the water surface especially continuous across the mask boundary, with matching reflections, ripples, brightness, and shadow direction. Preserve the visible content and blend the completed region naturally.",
        size="512x512",
        quality="high"
    )

item = response.data[0]
b64 = getattr(item, "b64_json", None)
image_url = getattr(item, "url", None)

if b64:
    # 兼容纯 base64 或 data:image/...;base64,... 两种形式。
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("接口返回了无效的 b64_json") from exc
elif image_url:
    with urlopen(image_url) as image_response:
        image_bytes = image_response.read()
else:
    raise RuntimeError(
        f"接口没有返回图片内容。响应项字段: {getattr(item, 'model_fields_set', None)}; "
        f"完整响应: {response}"
    )

with open("output.png", "wb") as f:
    f.write(image_bytes)

print("图像已保存到 output.png")
