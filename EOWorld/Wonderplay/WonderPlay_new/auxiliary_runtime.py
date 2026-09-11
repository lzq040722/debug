"""Persistent auxiliary-model process for optional two-GPU execution."""

from __future__ import annotations

import gc
import io
import multiprocessing as mp
import os
import sys
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


_ROOT = Path(__file__).resolve().parent.parent
_WONDERPLAY = Path(__file__).resolve().parent
_REPViT_SAM = _ROOT / "submodules" / "RepViT" / "sam"
_REPViT_CHECKPOINT = _ROOT / "submodules" / "RepViT" / "checkpoints" / "repvit_sam.pt"


def _png_bytes(image):
    if torch.is_tensor(image):
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 4:
            tensor = tensor[0]
        image = ToPILImage()(tensor)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _open_image(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


class _WorkerModels:
    def __init__(self, config, gpu_index):
        self.config = config
        self.gpu_index = int(gpu_index)
        self.repvit_model = None
        self.mask_generator = None
        self.object_predictor = None
        self.image_edit = None
        self.sam3_processor = None
        self.realwonder = None
        self.lock = threading.RLock()

    def _clear_cuda(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _drop_repvit(self):
        with self.lock:
            self.object_predictor = None
            self.mask_generator = None
            self.repvit_model = None
        self._clear_cuda()

    def load_repvit(self):
        with self.lock:
            if self.mask_generator is not None:
                return
            if str(_REPViT_SAM) not in sys.path:
                sys.path.insert(0, str(_REPViT_SAM))
            from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

            model = sam_model_registry["repvit"](checkpoint=str(_REPViT_CHECKPOINT))
            model = model.to("cuda").eval()
            self.repvit_model = model
            self.mask_generator = SamAutomaticMaskGenerator(
                model=model,
                points_per_side=16,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.9,
            )
            self.object_predictor = SamPredictor(model)

    def load_image_edit(self):
        with self.lock:
            if self.image_edit is not None:
                return
            from util.image_edit_inpaint import ImageEditInpaintPipeline

            checkpoint = self.config.get(
                "image_edit_checkpoint", "/root/autodl-tmp/huggingface/hub"
            )
            self.image_edit = ImageEditInpaintPipeline.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to("cuda")

    def load_sam3(self):
        with self.lock:
            if self.sam3_processor is not None:
                return
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.sam3_processor = Sam3Processor(build_sam3_image_model(device="cuda"))

    def load_realwonder(self):
        with self.lock:
            if self.realwonder is not None:
                return
            from realwonder_refinement import PersistentRealWonderRuntime

            runtime = PersistentRealWonderRuntime(self.config)
            runtime.load()
            self.realwonder = runtime

    def _preload(self):
        # CUDA's current device is thread-local.  The worker process sets its
        # device in _worker_main(), but this preload function runs in a newly
        # created thread, whose default device is cuda:0.  Bind it explicitly
        # before any module imports or .to("cuda") calls.
        torch.cuda.set_device(self.gpu_index)
        print(
            f"[aux-worker] Preload thread bound to "
            f"cuda:{torch.cuda.current_device()} (pid={os.getpid()}).",
            flush=True,
        )
        # RepViT is needed first for object selection. ImageEdit follows the
        # selection, then SAM3 and RealWonder are used by motion processing.
        for loader in (
            self.load_repvit,
            self.load_image_edit,
            self.load_sam3,
            self.load_realwonder,
        ):
            try:
                loader()
                allocated_gib = torch.cuda.memory_allocated(self.gpu_index) / 1024**3
                reserved_gib = torch.cuda.memory_reserved(self.gpu_index) / 1024**3
                print(
                    f"[aux-worker] Preloaded {loader.__name__.removeprefix('load_')} "
                    f"on cuda:{self.gpu_index}; allocated={allocated_gib:.2f} GiB, "
                    f"reserved={reserved_gib:.2f} GiB.",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as exc:
                # RepViT is small and can run lazily on GPU0. Releasing it is
                # the least disruptive way to give the large models more room.
                if loader.__name__ != "load_repvit" and self.mask_generator is not None:
                    print(
                        f"[aux-worker] {loader.__name__} ran out of memory; "
                        "moving RepViT-SAM fallback to GPU0 and retrying.",
                        flush=True,
                    )
                    self._drop_repvit()
                    try:
                        loader()
                        continue
                    except Exception as retry_exc:
                        exc = retry_exc
                print(
                    f"[aux-worker] Preload skipped: {loader.__name__}: {exc}",
                    flush=True,
                )
                self._clear_cuda()
            except Exception as exc:
                print(f"[aux-worker] Preload skipped: {loader.__name__}: {exc}", flush=True)
                self._clear_cuda()

    def automatic_masks(self, image):
        self.load_repvit()
        with self.lock:
            return self.mask_generator.generate(np.asarray(image))

    def set_object_image(self, image):
        self.load_repvit()
        with self.lock:
            self.object_predictor.set_image(np.asarray(image))

    def segment_object(self, points, labels):
        self.load_repvit()
        with self.lock:
            masks, scores, _ = self.object_predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
        return np.asarray(masks[int(np.argmax(scores))]).astype(bool)

    def segment_environment(self, image, prompt):
        self.load_sam3()
        with self.lock:
            state = self.sam3_processor.set_image(image)
            masks_list = []
            for item in (part.strip() for part in str(prompt).split(",")):
                if not item:
                    continue
                output = self.sam3_processor.set_text_prompt(state=state, prompt=item)
                masks = output.get("masks")
                if masks is None:
                    continue
                masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                masks = np.squeeze(masks)
                if masks.ndim == 2:
                    masks = masks[None]
                if masks.ndim == 3 and masks.shape[0]:
                    masks_list.append(masks.astype(bool))
            if not masks_list:
                raise RuntimeError(f"SAM3 returned no masks for '{prompt}'")
            return np.any(np.concatenate(masks_list, axis=0), axis=0)

    def inpaint(self, request):
        self.load_image_edit()
        with self.lock:
            result = self.image_edit(
                prompt=request["prompt"],
                negative_prompt=request.get("negative_prompt"),
                image=_open_image(request["image"]),
                num_inference_steps=request["num_inference_steps"],
                guidance_scale=request["guidance_scale"],
                height=request.get("height"),
                width=request.get("width"),
                true_cfg_scale=request.get("true_cfg_scale", 7.0),
                strength=request.get("strength", 0.95),
                image_edit_seed=request.get("image_edit_seed"),
            )
        tensor = (result.images[0].detach().cpu() / 2 + 0.5).clamp(0, 1)
        return _png_bytes(tensor)

    def refine_video(self, request):
        self.load_realwonder()
        with self.lock:
            return self.realwonder.run(
                request["conditions"],
                output_video_path=request.get("output_video_path"),
                refined_frames_dir=request.get("refined_frames_dir"),
            )


def _worker_main(connection, gpu_index, config):
    try:
        torch.cuda.set_device(int(gpu_index))
        models = _WorkerModels(config, gpu_index)
        # Acknowledge device initialization before loading weights so GPU0 can
        # start the normal pipeline while GPU1 preloads independently.
        connection.send({"ok": True})
        threading.Thread(target=models._preload, daemon=True).start()

        while True:
            request = connection.recv()
            operation = request.get("operation")
            if operation == "shutdown":
                connection.send({"ok": True})
                return
            try:
                if operation == "automatic_masks":
                    result = models.automatic_masks(_open_image(request["image"]))
                elif operation == "set_object_image":
                    models.set_object_image(_open_image(request["image"]))
                    result = None
                elif operation == "segment_object":
                    result = models.segment_object(request["points"], request["labels"])
                elif operation == "segment_environment":
                    result = models.segment_environment(
                        _open_image(request["image"]), request["prompt"]
                    )
                elif operation == "inpaint":
                    result = models.inpaint(request)
                elif operation == "refine_video":
                    result = models.refine_video(request)
                elif operation == "status":
                    result = {
                        "repvit": models.mask_generator is not None,
                        "image_edit": models.image_edit is not None,
                        "sam3": models.sam3_processor is not None,
                        "realwonder": models.realwonder is not None,
                    }
                else:
                    raise ValueError(f"Unknown auxiliary operation: {operation}")
                connection.send({"ok": True, "result": result})
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    models._clear_cuda()
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"Auxiliary worker startup failed: {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass


class RemoteMaskGenerator:
    def __init__(self, runtime):
        self.runtime = runtime
        self.predictor = None

    def generate(self, image):
        try:
            return self.runtime._rpc("automatic_masks", image=_png_bytes(image))
        except Exception as exc:
            print(
                f"[aux-runtime] RepViT-SAM is falling back to GPU0: {exc}",
                flush=True,
            )
            return self.runtime._local_mask_generator().generate(np.asarray(image))


class RemoteImageEditPipeline:
    uses_image_edit = True

    def __init__(self, runtime):
        self.runtime = runtime

    def to(self, device):
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        return None

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image=None,
        num_inference_steps=40,
        guidance_scale=3.0,
        height=None,
        width=None,
        image_edit_seed=None,
        **kwargs,
    ):
        result = self.runtime._rpc(
            "inpaint",
            image=_png_bytes(image),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            image_edit_seed=image_edit_seed,
            true_cfg_scale=kwargs.get("true_cfg_scale", 7.0),
            strength=kwargs.get("strength", 0.95),
        )
        tensor = ToTensor()(_open_image(result)) * 2 - 1
        return SimpleNamespace(images=[tensor])


class AuxiliaryModelRuntime:
    def __init__(self, config, gpu_index=1):
        self.config = config
        self.gpu_index = int(gpu_index)
        self._lock = threading.Lock()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(child, self.gpu_index, config),
            daemon=True,
        )
        self._process.start()
        if not parent.poll(180):
            raise TimeoutError("Auxiliary GPU worker did not initialize within 180 seconds")
        ready = parent.recv()
        if not ready.get("ok"):
            raise RuntimeError(ready.get("error", "Auxiliary GPU worker failed"))
        self.mask_generator = RemoteMaskGenerator(self)
        self.inpainting_pipeline = RemoteImageEditPipeline(self)
        self._local_predictor = None
        self._fallback_mask_generator = None

    def _local_mask_generator(self):
        if self._fallback_mask_generator is None:
            if str(_REPViT_SAM) not in sys.path:
                sys.path.insert(0, str(_REPViT_SAM))
            from util.segment_utils import create_mask_generator_repvit

            torch.cuda.set_device(0)
            self._fallback_mask_generator = create_mask_generator_repvit()
        return self._fallback_mask_generator

    def _rpc(self, operation, timeout=None, **payload):
        with self._lock:
            if not self._process.is_alive():
                raise RuntimeError("Auxiliary GPU worker is not running")
            self._connection.send({"operation": operation, **payload})
            if timeout is not None and not self._connection.poll(timeout):
                raise TimeoutError(f"Auxiliary operation timed out: {operation}")
            response = self._connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", f"Auxiliary operation failed: {operation}"))
        return response.get("result")

    def set_object_image(self, image, local_predictor=None):
        self._local_object_image = np.asarray(image.convert("RGB"))
        self._local_predictor = local_predictor
        try:
            self._rpc("set_object_image", image=_png_bytes(image))
        except Exception as exc:
            if local_predictor is None:
                print(
                    f"[aux-runtime] RepViT-SAM is falling back to GPU0: {exc}",
                    flush=True,
                )
                local_predictor = self._local_mask_generator().predictor
                self._local_predictor = local_predictor
            local_predictor.set_image(self._local_object_image)

    def segment_object(self, points, labels, local_predictor=None):
        predictor = local_predictor or self._local_predictor
        try:
            return self._rpc("segment_object", points=points, labels=labels)
        except Exception as exc:
            if predictor is None:
                print(
                    f"[aux-runtime] RepViT-SAM is falling back to GPU0: {exc}",
                    flush=True,
                )
                predictor = self._local_mask_generator().predictor
                self._local_predictor = predictor
                image = getattr(self, "_local_object_image", None)
                if image is None:
                    raise RuntimeError(
                        "Object image is unavailable for the GPU0 SAM fallback"
                    ) from exc
                predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
            return masks[int(np.argmax(scores))]

    def segment_environment(self, image, prompt):
        return self._rpc(
            "segment_environment", image=_png_bytes(image), prompt=str(prompt)
        )

    def run_realwonder(self, conditions, output_video_path=None, refined_frames_dir=None):
        return self._rpc(
            "refine_video",
            conditions={key: str(value) for key, value in conditions.items()},
            output_video_path=str(output_video_path) if output_video_path else None,
            refined_frames_dir=str(refined_frames_dir) if refined_frames_dir else None,
        )

    def close(self):
        if self._process.is_alive():
            try:
                self._rpc("shutdown", timeout=10)
            except Exception:
                pass
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


def build_auxiliary_runtime(config):
    values = config.get("runtime", {}) or {}
    mode = str(values.get("mode", "auto")).lower()
    if mode not in {"auto", "single", "dual"}:
        raise ValueError("runtime.mode must be one of: auto, single, dual")
    if mode == "single" or torch.cuda.device_count() < 2:
        if mode == "dual":
            print("[aux-runtime] Only one CUDA device is visible; using single-GPU mode.", flush=True)
        return None
    gpu_index = int(values.get("auxiliary_gpu", 1))
    try:
        try:
            from omegaconf import OmegaConf

            worker_config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            worker_config = dict(config)
        runtime = AuxiliaryModelRuntime(worker_config, gpu_index=gpu_index)
        print(
            f"[aux-runtime] Persistent model worker started on cuda:{gpu_index}; "
            "model preloading continues in parallel.",
            flush=True,
        )
        return runtime
    except Exception as exc:
        if mode == "dual" and not bool(values.get("fallback_to_single", True)):
            raise
        print(f"[aux-runtime] Falling back to single-GPU mode: {exc}", flush=True)
        return None
