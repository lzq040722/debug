"""Run RealWonder video refinement from WonderPlay-native condition paths."""

from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image


_LOCAL_REALWONDER_CODE = Path(__file__).resolve().parent / "realwonder"
_DEFAULT_REALWONDER_ROOT = Path(
    os.environ.get("REALWONDER_ROOT", "/root/autodl-tmp/RealWonder")
)
_DEFAULT_CHECKPOINT = (
    Path("ckpts")
    / "Realwonder-Distilled-AR-I2V-Flow"
    / "sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4"
    / "step=000800.pt"
)


@contextmanager
def _realwonder_import_context(realwonder_root: Path):
    code_root = str(_LOCAL_REALWONDER_CODE)
    inserted = False
    old_cwd = os.getcwd()
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
        inserted = True
    os.chdir(realwonder_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(code_root)
            except ValueError:
                pass


def _get_config_value(config, section, key, default):
    if config is None:
        return default
    values = config.get(section, {})
    if values is None:
        return default
    return values.get(key, default)


def _load_sim_frames(frames_dir: Path, height: int, width: int):
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    frames = []
    for frame_path in frame_files:
        image = Image.open(frame_path).convert("RGB").resize((width, height))
        array = np.array(image, dtype=np.float32) / 127.5 - 1.0
        frames.append(torch.from_numpy(array))
    return torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous().unsqueeze(0)


def _resolve_checkpoint(realwonder_root: Path, checkpoint):
    checkpoint_path = Path(checkpoint or _DEFAULT_CHECKPOINT)
    if not checkpoint_path.is_absolute():
        checkpoint_path = realwonder_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing RealWonder checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_realwonder_refinement(
    conditions: dict,
    config=None,
    output_video_path: Path | None = None,
    refined_frames_dir: Path | None = None,
) -> dict:
    """Generate refined video/frames using RealWonder's SDEdit pipeline."""
    traj_dir = Path(conditions["traj_dir"])
    realwonder_root = Path(
        _get_config_value(config, "realwonder", "root", _DEFAULT_REALWONDER_ROOT)
    ).resolve()
    if not realwonder_root.exists():
        raise FileNotFoundError(f"RealWonder root not found: {realwonder_root}")

    height = int(_get_config_value(config, "realwonder", "height", 512))
    width = int(_get_config_value(config, "realwonder", "width", 512))
    num_output_frames = int(
        _get_config_value(config, "realwonder", "num_output_frames", 12)
    )
    denoising_step_list = list(
        _get_config_value(
            config, "realwonder", "denoising_step_list", [800, 600, 400, 200]
        )
    )
    mask_dropin_step = int(
        _get_config_value(config, "realwonder", "mask_dropin_step", -1)
    )
    eval_degradation = float(
        _get_config_value(config, "realwonder", "eval_degradation", 0.5)
    )
    local_attn_size = int(
        _get_config_value(config, "realwonder", "local_attn_size", 21)
    )
    default_seed = config.get("seed", 42) if config is not None else 42
    seed = int(_get_config_value(config, "realwonder", "seed", default_seed))
    use_ema = bool(_get_config_value(config, "realwonder", "use_ema", False))
    fps = int(_get_config_value(config, "realwonder", "fps", 8))
    checkpoint = _resolve_checkpoint(
        realwonder_root,
        _get_config_value(config, "realwonder", "checkpoint", _DEFAULT_CHECKPOINT),
    )

    output_video_path = Path(output_video_path or traj_dir / "realwonder_refined.mp4")
    refined_frames_dir = Path(refined_frames_dir or traj_dir / "refined_frames")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if refined_frames_dir.exists():
        shutil.rmtree(refined_frames_dir)
    refined_frames_dir.mkdir(parents=True, exist_ok=True)

    with _realwonder_import_context(realwonder_root):
        from vidgen import (
            CausalInferencePipelineSDEdit,
            DynamicSwapInstaller,
            WanImageEncoder,
            WanVideoUnit_ImageEmbedderCLIP,
            WanVideoUnit_ImageEmbedderVAE,
            WanVideoVAE,
            apply_config_overrides,
            get_cuda_free_memory_gb,
            gpu,
            load_first_frame,
            load_noise,
            set_seed,
        )

        device = torch.device("cuda")
        set_seed(seed)
        low_memory = get_cuda_free_memory_gb(gpu) < 40

        rw_config = OmegaConf.create(
            {
                "independent_first_frame": False,
                "warp_denoising_step": True,
                "context_noise": 0,
                "causal": True,
                "i2v": True,
                "i2v_flow": True,
                "height": height,
                "width": width,
                "num_frame_per_block": 3,
                "denoising_step_list": denoising_step_list,
                "mask_dropin_step": mask_dropin_step,
                "model_kwargs": {
                    "sink_size": 1,
                    "local_attn_size": local_attn_size,
                    "timestep_shift": 5.0,
                },
            }
        )
        rw_config = apply_config_overrides(rw_config, [], verbose=False)

        pipeline = CausalInferencePipelineSDEdit(rw_config, device=device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        key = "generator_ema" if use_ema else "generator"
        gen_state_dict = state_dict[key]
        try:
            pipeline.generator.load_state_dict(gen_state_dict)
        except RuntimeError:
            gen_state_dict = {
                key.replace("._fsdp_wrapped_module", ""): value
                for key, value in gen_state_dict.items()
            }
            pipeline.generator.load_state_dict(gen_state_dict)

        pipeline = pipeline.to(dtype=torch.bfloat16)
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        else:
            pipeline.text_encoder.to(device=gpu)
        pipeline.generator.to(device=gpu)
        pipeline.vae.to(device=gpu)

        pipeline.processor_dtype = torch.float32
        pipeline.processor_device = gpu
        pipeline.processor_vae = WanVideoVAE().to(
            device=pipeline.processor_device,
            dtype=pipeline.processor_dtype,
        )
        pipeline.processor_ienc = WanImageEncoder().to(
            device=pipeline.processor_device,
            dtype=pipeline.processor_dtype,
        )
        pipeline.processor_vae.requires_grad_(False)
        pipeline.processor_ienc.requires_grad_(False)

        for parameter in pipeline.processor_vae.parameters():
            parameter.data = parameter.data.to(dtype=pipeline.processor_dtype)
        for buffer in pipeline.processor_vae.buffers():
            buffer.data = buffer.data.to(dtype=pipeline.processor_dtype)

        pipeline.processors = [
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]

        noise_data = load_noise(
            noise_path=str(conditions["noise_path"]),
            target_frames=num_output_frames,
            channel_dim=16,
            downsample_mode="nearest",
            eval_degradation=eval_degradation,
        )
        input_image = load_first_frame(
            str(conditions["first_frame_path"]), height=height, width=width
        )
        sim_frames = _load_sim_frames(conditions["frames_dir"], height, width)
        with open(conditions["prompt_path"], "r", encoding="utf-8") as prompt_file:
            prompt = prompt_file.read().strip()

        structured_noise = noise_data["structured_noise"].unsqueeze(0).to(
            device=device, dtype=torch.bfloat16
        )
        structured_noise_sde = noise_data.get("structured_noise_sde")
        if structured_noise_sde is not None:
            structured_noise_sde = structured_noise_sde.unsqueeze(0).to(
                device=device, dtype=torch.bfloat16
            )

        sim_latent = None
        if pipeline.sdedit:
            sim_frames_device = sim_frames.to(device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                sim_latent = pipeline.vae.encode_to_latent(sim_frames_device)
            sim_latent = sim_latent.to(device=device, dtype=torch.bfloat16)
            if sim_latent.shape[1] > num_output_frames:
                sim_latent = sim_latent[:, :num_output_frames]
            elif sim_latent.shape[1] < num_output_frames:
                pad_size = num_output_frames - sim_latent.shape[1]
                sim_latent = torch.cat(
                    [sim_latent, sim_latent[:, -1:].repeat(1, pad_size, 1, 1, 1)],
                    dim=1,
                )

        batch = {
            "input_image": input_image.unsqueeze(0),
            "end_image": None,
            "height": height,
            "width": width,
            "num_frames": num_output_frames * 4 - 3,
        }

        print("[realwonder] Video Refinement.", flush=True)
        with torch.inference_mode():
            video, _ = pipeline.inference(
                noise=structured_noise,
                text_prompts=[prompt],
                return_latents=True,
                batch_sample=batch,
                sim_latent=sim_latent,
                sim_masks=None,
                sim_franka_masks=None,
                low_memory=low_memory,
                device=device,
                structured_noise_sde=structured_noise_sde,
            )
        video = rearrange(video, "b t c h w -> b t h w c").cpu()
        video = (255.0 * video[0]).to(torch.uint8).numpy()

        if hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()

    imageio.mimwrite(output_video_path, video, fps=fps)
    for frame_idx, frame in enumerate(video):
        Image.fromarray(frame).save(
            refined_frames_dir / f"frame_{frame_idx:08d}.png"
        )

    print(f"[realwonder] Refined video saved to {output_video_path}", flush=True)
    print(f"[realwonder] Refined frames saved to {refined_frames_dir}", flush=True)
    return {
        "output_video_path": output_video_path,
        "refined_frames_dir": refined_frames_dir,
        "fps": fps,
    }


class PersistentRealWonderRuntime:
    """Load RealWonder once and reuse it for all auxiliary-worker requests."""

    def __init__(self, config=None):
        self.config = config or {}
        self.pipeline = None
        self.modules = None
        self.realwonder_root = Path(
            _get_config_value(self.config, "realwonder", "root", _DEFAULT_REALWONDER_ROOT)
        ).resolve()

    def load(self):
        if self.pipeline is not None:
            return self
        if not self.realwonder_root.exists():
            raise FileNotFoundError(f"RealWonder root not found: {self.realwonder_root}")

        checkpoint = _resolve_checkpoint(
            self.realwonder_root,
            _get_config_value(self.config, "realwonder", "checkpoint", _DEFAULT_CHECKPOINT),
        )
        local_attn_size = int(
            _get_config_value(self.config, "realwonder", "local_attn_size", 21)
        )
        denoising_step_list = list(
            _get_config_value(
                self.config, "realwonder", "denoising_step_list", [800, 600, 400, 200]
            )
        )
        mask_dropin_step = int(
            _get_config_value(self.config, "realwonder", "mask_dropin_step", -1)
        )
        use_ema = bool(_get_config_value(self.config, "realwonder", "use_ema", False))

        with _realwonder_import_context(self.realwonder_root):
            from vidgen import (
                CausalInferencePipelineSDEdit,
                DynamicSwapInstaller,
                WanImageEncoder,
                WanVideoUnit_ImageEmbedderCLIP,
                WanVideoUnit_ImageEmbedderVAE,
                WanVideoVAE,
                apply_config_overrides,
                get_cuda_free_memory_gb,
                gpu,
                load_first_frame,
                load_noise,
                set_seed,
            )

            device = torch.device("cuda")
            rw_config = OmegaConf.create(
                {
                    "independent_first_frame": False,
                    "warp_denoising_step": True,
                    "context_noise": 0,
                    "causal": True,
                    "i2v": True,
                    "i2v_flow": True,
                    "height": int(_get_config_value(self.config, "realwonder", "height", 512)),
                    "width": int(_get_config_value(self.config, "realwonder", "width", 512)),
                    "num_frame_per_block": 3,
                    "denoising_step_list": denoising_step_list,
                    "mask_dropin_step": mask_dropin_step,
                    "model_kwargs": {
                        "sink_size": 1,
                        "local_attn_size": local_attn_size,
                        "timestep_shift": 5.0,
                    },
                }
            )
            rw_config = apply_config_overrides(rw_config, [], verbose=False)
            pipeline = CausalInferencePipelineSDEdit(rw_config, device=device)
            state_dict = torch.load(checkpoint, map_location="cpu")
            key = "generator_ema" if use_ema else "generator"
            gen_state_dict = state_dict[key]
            try:
                pipeline.generator.load_state_dict(gen_state_dict)
            except RuntimeError:
                pipeline.generator.load_state_dict(
                    {
                        name.replace("._fsdp_wrapped_module", ""): value
                        for name, value in gen_state_dict.items()
                    }
                )
            del state_dict, gen_state_dict

            pipeline = pipeline.to(dtype=torch.bfloat16)
            low_memory = get_cuda_free_memory_gb(gpu) < 40
            if low_memory:
                DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
            else:
                pipeline.text_encoder.to(device=gpu)
            pipeline.generator.to(device=gpu)
            pipeline.vae.to(device=gpu)
            pipeline.processor_dtype = torch.float32
            pipeline.processor_device = gpu
            pipeline.processor_vae = WanVideoVAE().to(device=gpu, dtype=torch.float32)
            pipeline.processor_ienc = WanImageEncoder().to(device=gpu, dtype=torch.float32)
            pipeline.processor_vae.requires_grad_(False)
            pipeline.processor_ienc.requires_grad_(False)
            for parameter in pipeline.processor_vae.parameters():
                parameter.data = parameter.data.to(dtype=torch.float32)
            for buffer in pipeline.processor_vae.buffers():
                buffer.data = buffer.data.to(dtype=torch.float32)
            pipeline.processors = [
                WanVideoUnit_ImageEmbedderVAE(),
                WanVideoUnit_ImageEmbedderCLIP(),
            ]

        self.pipeline = pipeline
        self.modules = {
            "gpu": gpu,
            "load_first_frame": load_first_frame,
            "load_noise": load_noise,
            "set_seed": set_seed,
            "low_memory": low_memory,
        }
        return self

    def run(self, conditions, output_video_path=None, refined_frames_dir=None):
        self.load()
        conditions = {key: Path(value) for key, value in conditions.items()}
        traj_dir = conditions["traj_dir"]
        height = int(_get_config_value(self.config, "realwonder", "height", 512))
        width = int(_get_config_value(self.config, "realwonder", "width", 512))
        num_output_frames = int(
            _get_config_value(self.config, "realwonder", "num_output_frames", 12)
        )
        eval_degradation = float(
            _get_config_value(self.config, "realwonder", "eval_degradation", 0.5)
        )
        fps = int(_get_config_value(self.config, "realwonder", "fps", 8))
        seed = int(
            _get_config_value(
                self.config, "realwonder", "seed", self.config.get("seed", 42)
            )
        )
        output_video_path = Path(output_video_path or traj_dir / "realwonder_refined.mp4")
        refined_frames_dir = Path(refined_frames_dir or traj_dir / "refined_frames")
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        if refined_frames_dir.exists():
            shutil.rmtree(refined_frames_dir)
        refined_frames_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self.pipeline
        device = torch.device("cuda")
        gpu = self.modules["gpu"]
        self.modules["set_seed"](seed)
        with _realwonder_import_context(self.realwonder_root):
            noise_data = self.modules["load_noise"](
                noise_path=str(conditions["noise_path"]),
                target_frames=num_output_frames,
                channel_dim=16,
                downsample_mode="nearest",
                eval_degradation=eval_degradation,
            )
            input_image = self.modules["load_first_frame"](
                str(conditions["first_frame_path"]), height=height, width=width
            )
            sim_frames = _load_sim_frames(conditions["frames_dir"], height, width)
            prompt = conditions["prompt_path"].read_text(encoding="utf-8").strip()
            structured_noise = noise_data["structured_noise"].unsqueeze(0).to(
                device=device, dtype=torch.bfloat16
            )
            structured_noise_sde = noise_data.get("structured_noise_sde")
            if structured_noise_sde is not None:
                structured_noise_sde = structured_noise_sde.unsqueeze(0).to(
                    device=device, dtype=torch.bfloat16
                )
            sim_latent = None
            if pipeline.sdedit:
                with torch.inference_mode():
                    sim_latent = pipeline.vae.encode_to_latent(
                        sim_frames.to(device=device, dtype=torch.bfloat16)
                    )
                if sim_latent.shape[1] > num_output_frames:
                    sim_latent = sim_latent[:, :num_output_frames]
                elif sim_latent.shape[1] < num_output_frames:
                    pad_size = num_output_frames - sim_latent.shape[1]
                    sim_latent = torch.cat(
                        [sim_latent, sim_latent[:, -1:].repeat(1, pad_size, 1, 1, 1)],
                        dim=1,
                    )
            batch = {
                "input_image": input_image.unsqueeze(0),
                "end_image": None,
                "height": height,
                "width": width,
                "num_frames": num_output_frames * 4 - 3,
            }
            with torch.inference_mode():
                video, _ = pipeline.inference(
                    noise=structured_noise,
                    text_prompts=[prompt],
                    return_latents=True,
                    batch_sample=batch,
                    sim_latent=sim_latent,
                    sim_masks=None,
                    sim_franka_masks=None,
                    low_memory=self.modules["low_memory"],
                    device=device,
                    structured_noise_sde=structured_noise_sde,
                )
            video = rearrange(video, "b t c h w -> b t h w c").cpu()
            video = (255.0 * video[0]).to(torch.uint8).numpy()
            if hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()

        imageio.mimwrite(output_video_path, video, fps=fps)
        for frame_idx, frame in enumerate(video):
            Image.fromarray(frame).save(refined_frames_dir / f"frame_{frame_idx:08d}.png")
        return {
            "output_video_path": str(output_video_path),
            "refined_frames_dir": str(refined_frames_dir),
            "fps": fps,
        }
