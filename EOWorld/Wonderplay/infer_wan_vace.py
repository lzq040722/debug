#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_WAN_CODE = Path("/root/autodl-tmp/Wan2.1")
DEFAULT_CKPT = Path("/root/autodl-tmp/huggingface/Wan2.1-VACE-14B")
DEFAULT_OUT = Path("/root/autodl-tmp/huggingface/wan_vace_outputs")
DEFAULT_PROMPT = (
    "A cinematic close-up video of a young woman in a red spring festival "
    "outfit smiling beside a friendly green cartoon snake, warm lantern light, "
    "festive atmosphere, detailed, smooth camera motion."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Wan2.1 VACE inference wrapper")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--wan_code", default=str(DEFAULT_WAN_CODE))
    parser.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run Wan2.1/generate.py.")
    parser.add_argument("--out", default=str(DEFAULT_OUT / "wan_vace_test.mp4"))
    parser.add_argument("--size", default="832*480", help="Use 832*480 first; 1280*720 needs much more VRAM/time.")
    parser.add_argument("--frame_num", type=int, default=81, help="Must be 4n+1.")
    parser.add_argument("--steps", type=int, default=20, help="Use 20 for a quick test; 50 for quality.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=16.0)
    parser.add_argument("--src_ref_images", default=None, help="Comma-separated reference image paths.")
    parser.add_argument("--src_video", default=None)
    parser.add_argument("--src_mask", default=None)
    parser.add_argument("--no_t5_cpu", action="store_true", help="Keep T5 on GPU instead of CPU.")
    parser.add_argument("--no_offload", action="store_true", help="Disable model offload.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--check_deps", action="store_true", help="Only check basic runtime dependencies.")
    return parser.parse_args()


def require_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    args = parse_args()
    wan_code = Path(args.wan_code).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_path = Path(args.out).resolve()

    require_path(wan_code / "generate.py", "Wan2.1 generate.py")
    require_path(ckpt_dir / "diffusion_pytorch_model.safetensors.index.json", "VACE checkpoint index")
    require_path(ckpt_dir / "Wan2.1_VAE.pth", "Wan VAE")
    require_path(ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth", "Wan T5")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dep_check = [
        args.python,
        "-c",
        (
            "import torch, diffusers, transformers, tokenizers, accelerate, cv2, "
            "easydict, ftfy, imageio; "
            "print('python ok'); "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
        ),
    ]
    dep_result = subprocess.run(dep_check, cwd=str(wan_code), text=True)
    if args.check_deps:
        return
    if dep_result.returncode != 0:
        raise RuntimeError(
            "Dependency check failed. Try installing Wan2.1 requirements in the selected Python env."
        )

    ref_images = args.src_ref_images
    if ref_images is None:
        ref_images = f"{wan_code / 'examples' / 'girl.png'},{wan_code / 'examples' / 'snake.png'}"

    cmd = [
        args.python,
        str(wan_code / "generate.py"),
        "--task",
        "vace-14B",
        "--size",
        args.size,
        "--ckpt_dir",
        str(ckpt_dir),
        "--prompt",
        args.prompt,
        "--src_ref_images",
        ref_images,
        "--frame_num",
        str(args.frame_num),
        "--sample_steps",
        str(args.steps),
        "--sample_shift",
        str(args.shift),
        "--sample_guide_scale",
        str(args.guide_scale),
        "--base_seed",
        str(args.seed),
        "--save_file",
        str(out_path),
    ]

    if args.src_video:
        cmd.extend(["--src_video", args.src_video])
    if args.src_mask:
        cmd.extend(["--src_mask", args.src_mask])
    if not args.no_t5_cpu:
        cmd.append("--t5_cpu")
    if not args.no_offload:
        cmd.extend(["--offload_model", "True"])

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("Running:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=str(wan_code), env=env, check=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
