#!/usr/bin/env python3
"""Convert WonderPlay simulation outputs into a RealWonder inference sample."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_REALWONDER_ROOT = Path("/root/autodl-tmp/RealWonder")
DEFAULT_CHECKPOINT = Path(
    "ckpts/Realwonder-Distilled-AR-I2V-Flow/"
    "sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/"
    "step=000800.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RealWonder final_sim directory directly from WonderPlay "
            "flows_actual, RGB frames, gt.png, and text_prompt.txt."
        )
    )
    parser.add_argument(
        "--simulation_dir",
        type=Path,
        required=True,
        help="WonderPlay simulation directory containing gt.png and traj_XX.",
    )
    parser.add_argument("--traj_id", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination RealWonder final_sim directory.",
    )
    parser.add_argument(
        "--realwonder_root", type=Path, default=DEFAULT_REALWONDER_ROOT
    )
    parser.add_argument(
        "--crop_start",
        type=int,
        default=176,
        help="Top of the 480-pixel crop after resizing 512x512 to 832x832.",
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=12,
        help="Number of output latent frames; must be divisible by 3.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        nargs="+",
        default=[800, 600, 400, 200],
    )
    parser.add_argument(
        "--flow_format",
        choices=("auto", "normalized", "pixels"),
        default="auto",
        help="WonderPlay flows are normally normalized with 0.5 meaning zero motion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate frames and noises.npy when outputs already exist.",
    )
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="Run RealWonder infer_sim.py after preparing the sample.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Absolute path or path relative to --realwonder_root.",
    )
    parser.add_argument(
        "--output_video", type=Path, default=None
    )
    parser.add_argument("--eval_degradation", type=float, default=0.5)
    parser.add_argument("--local_attn_size", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_output_frames <= 0 or args.num_output_frames % 3 != 0:
        raise ValueError("--num_output_frames must be positive and divisible by 3")
    if not 0 <= args.crop_start <= 352:
        raise ValueError("--crop_start must be in [0, 352]")
    if not 0 <= args.eval_degradation <= 1:
        raise ValueError("--eval_degradation must be in [0, 1]")
    if not args.realwonder_root.joinpath("infer_sim.py").exists():
        raise FileNotFoundError(
            f"RealWonder was not found at {args.realwonder_root}"
        )


def resize_and_crop(image_path: Path, crop_start: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.size != (512, 512):
        print(f"Warning: expected 512x512, got {image.size} for {image_path}")
    image = image.resize((832, 832), Image.Resampling.BILINEAR)
    return image.crop((0, crop_start, 832, crop_start + 480))


def prepare_rgb_inputs(args: argparse.Namespace, traj_dir: Path) -> int:
    output_frames = args.output_dir / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)

    gt_path = args.simulation_dir / "gt.png"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing first frame: {gt_path}")
    resize_and_crop(gt_path, args.crop_start).save(
        args.output_dir / "resized_input_image.png"
    )

    frame_paths = sorted((traj_dir / "frames").glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames found in {traj_dir / 'frames'}")

    if args.overwrite:
        for old_frame in output_frames.glob("frame_*.png"):
            old_frame.unlink()

    for index, frame_path in enumerate(frame_paths):
        destination = output_frames / f"frame_{index:04d}.png"
        if args.overwrite or not destination.exists():
            resize_and_crop(frame_path, args.crop_start).save(destination)

    prompt_source = args.simulation_dir / "text_prompt.txt"
    if prompt_source.exists():
        prompt = prompt_source.read_text(encoding="utf-8").strip()
    else:
        prompt = "A realistic video preserving the original scene appearance."
        print(f"Warning: {prompt_source} is missing; using a neutral prompt")
    (args.output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    config = (
        f"num_output_frames: {args.num_output_frames}\n"
        f"denoising_step_list: {args.denoising_steps}\n"
        "mask_dropin_step: -1\n"
    )
    (args.output_dir / "config.yaml").write_text(config, encoding="utf-8")
    return len(frame_paths)


def load_pixel_flows(
    flow_dir: Path, flow_format: str
) -> tuple[list[np.ndarray], str]:
    flow_paths = sorted(flow_dir.glob("flow_*.npy"))
    if len(flow_paths) < 2:
        raise FileNotFoundError(f"At least two flow files are required in {flow_dir}")

    raw_flows = [np.load(path).astype(np.float32) for path in flow_paths]
    first_shape = raw_flows[0].shape
    if first_shape[0] != 2:
        raise ValueError(f"Expected flow shape [2,H,W], got {first_shape}")
    if any(flow.shape != first_shape for flow in raw_flows):
        raise ValueError("All flow arrays must have the same shape")

    detected_format = flow_format
    if flow_format == "auto":
        sample_min = min(float(flow.min()) for flow in raw_flows[:5])
        sample_max = max(float(flow.max()) for flow in raw_flows[:5])
        detected_format = (
            "normalized"
            if sample_min >= -0.05 and sample_max <= 1.05
            else "pixels"
        )

    if detected_format == "normalized":
        height, width = first_shape[1:]
        for flow in raw_flows:
            flow[0] = (flow[0] * 2.0 - 1.0) * width
            flow[1] = (flow[1] * 2.0 - 1.0) * height

    # flow_00000000 is the identity flow. The NoiseWarper creates the first
    # Gaussian noise frame itself, then applies one flow per transition.
    return raw_flows[1:], detected_format


def generate_structured_noise(
    args: argparse.Namespace, traj_dir: Path
) -> tuple[int, tuple[int, ...]]:
    noise_path = args.output_dir / "noises.npy"
    if noise_path.exists() and not args.overwrite:
        noise = np.load(noise_path, mmap_mode="r")
        expected_spatial_channels = (60, 104, 32)
        if tuple(noise.shape[1:]) != expected_spatial_channels:
            raise ValueError(
                f"Existing {noise_path} has shape {noise.shape}; expected "
                f"[T,{expected_spatial_channels[0]},{expected_spatial_channels[1]},"
                f"{expected_spatial_channels[2]}]. Use --overwrite."
            )
        print(f"Reusing existing structured noise: {noise_path}")
        return int(noise.shape[0]), tuple(noise.shape)

    flows, detected_format = load_pixel_flows(
        traj_dir / "flows_actual", args.flow_format
    )
    print(
        f"Loaded {len(flows) + 1} flow files; using {len(flows)} transitions "
        f"as {detected_format} flow"
    )

    realwonder_root = args.realwonder_root.resolve()
    sys.path.insert(0, str(realwonder_root))
    from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper

    NoiseWarper().process(
        flows,
        str(args.output_dir),
        input_flow=True,
        crop_start=args.crop_start,
        debug=False,
    )

    noise = np.load(noise_path, mmap_mode="r")
    expected_spatial_channels = (60, 104, 32)
    if tuple(noise.shape[1:]) != expected_spatial_channels:
        raise RuntimeError(
            f"Generated noise has shape {noise.shape}; expected "
            f"[T,{expected_spatial_channels[0]},{expected_spatial_channels[1]},32]"
        )
    return len(flows) + 1, tuple(noise.shape)


def run_inference(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = args.realwonder_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing RealWonder checkpoint: {checkpoint}")

    output_video = args.output_video or (args.output_dir / "realwonder_output.mp4")
    output_video = output_video.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(args.realwonder_root / "infer_sim.py"),
        "--checkpoint_path",
        str(checkpoint.resolve()),
        "--sim_data_path",
        str(args.output_dir.resolve()),
        "--output_path",
        str(output_video),
        "--eval_degradation",
        str(args.eval_degradation),
        "--local_attn_size",
        str(args.local_attn_size),
        "--seed",
        str(args.seed),
    ]
    if args.use_ema:
        command.append("--use_ema")

    print("Running RealWonder inference:")
    print(" ".join(command))
    subprocess.run(command, cwd=args.realwonder_root, check=True)
    return output_video


def main() -> None:
    args = parse_args()
    args.simulation_dir = args.simulation_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.realwonder_root = args.realwonder_root.resolve()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    traj_dir = args.simulation_dir / f"traj_{args.traj_id:02d}"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing trajectory directory: {traj_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_count = prepare_rgb_inputs(args, traj_dir)
    noise_count, noise_shape = generate_structured_noise(args, traj_dir)

    print("\nRealWonder input is ready")
    print(f"  directory: {args.output_dir}")
    print(f"  RGB frames: {rgb_count}")
    print(f"  noise frames: {noise_count}")
    print(f"  noises.npy: {noise_shape}")

    if args.run_inference:
        output_video = run_inference(args)
        print(f"  output video: {output_video}")
    else:
        print("Add --run_inference to generate the final video in the same command.")


if __name__ == "__main__":
    main()
