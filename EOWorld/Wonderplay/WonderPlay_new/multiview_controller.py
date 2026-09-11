"""LivingWorld-style orchestration for WonderPlay interaction scenes.

The reconstruction, inpainting, motion binding, HashGrid training, and
interaction rendering remain in :mod:`run_genesis`.  This module only joins
those existing operations into the interactive multi-view loop.
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from kornia.morphology import dilation


def _save_tensor_image(image, path):
    tensor = image.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        array = tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
    elif tensor.ndim == 2:
        array = tensor.mul(255).round().byte().numpy()
    else:
        array = tensor.squeeze().mul(255).round().byte().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


class PromptManager:
    """Keep LivingWorld prompt progression at the orchestration boundary."""

    def __init__(self, genesis, config, emit, consume_scene_prompt):
        self.genesis = genesis
        self.config = config
        self.scene_dict = genesis.scene_dict
        self.style_prompt = genesis.style_prompt
        self.use_gpt = bool(config.get("use_gpt", False))
        self.fixed_prompt = str(config.get("prompt", "") or "").strip()
        self.emit = emit
        self.consume_scene_prompt = consume_scene_prompt

        self.generator = None
        if self.use_gpt:
            # GPT and its optional NLP dependencies are not part of the normal
            # initial-scene path.  Import them only when GPT is explicitly on.
            from util.chatGPT4 import TextpromptGen

            self.generator = TextpromptGen(
                genesis.kf_gen.run_dir,
                isinstance(config.get("control_text", None), list),
            )

    @staticmethod
    def _format_prompt(style, entities, background=None, scene_name=None):
        """Format an edit prompt without importing optional GPT/NLP packages."""
        if background is not None:
            if isinstance(background, list):
                background = background[0]
            prompt = (
                f"Style: {style}. Entities: {', '.join(map(str, entities))}. "
                f"Background: {background}"
            )
        else:
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            entity_text = ", ".join(map(str, entities))
            prompt = f"Style: {style}. {scene_name} with {entity_text}"
        print("PROMPT TEXT: ", prompt)
        return prompt

    def next_prompt(self):
        user_scene_name = self.consume_scene_prompt()
        changed_by_user = user_scene_name is not None
        if changed_by_user:
            self.scene_dict["scene_name"] = user_scene_name

        scene_name = self.scene_dict["scene_name"]
        if isinstance(scene_name, list):
            scene_name = scene_name[0]
        self.emit("scene-prompt", scene_name)
        if self.fixed_prompt:
            print("PROMPT TEXT: ", self.fixed_prompt)
            return self.fixed_prompt

        if self.use_gpt:
            self.scene_dict = self.generator.wonder_next_scene(
                scene_name=self.scene_dict["scene_name"],
                entities=self.scene_dict["entities"],
                style=self.style_prompt,
                background=self.scene_dict["background"],
                change_scene_name_by_user=changed_by_user,
            )
            self.genesis.scene_dict = self.scene_dict

        scene_name = self.scene_dict["scene_name"]
        if isinstance(scene_name, list):
            scene_name = scene_name[0]
        self.emit("scene-prompt", scene_name)

        if self.generator is not None:
            prompt = self.generator.generate_prompt(
                style=self.style_prompt,
                entities=self.scene_dict["entities"],
                background=self.scene_dict["background"],
                scene_name=scene_name,
            )
        else:
            prompt = self._format_prompt(
                style=self.style_prompt,
                entities=self.scene_dict["entities"],
                background=self.scene_dict["background"],
                scene_name=scene_name,
            )
        return prompt


class ExpansionManager:
    """Run one LivingWorld expansion and append only environment Gaussians."""

    def __init__(self, genesis, config, emit):
        self.genesis = genesis
        self.config = config
        self.emit = emit
        self.view_index = 0
        self.preserve_sky_once = False

    def _emit(self, message):
        self.emit("server-state", message)

    def _render_masks(self, tdgs_camera, object_xyz=None):
        g = self.genesis
        dynamic_kwargs = {}
        if object_xyz is not None:
            dynamic_kwargs = {
                "timestep": 0,
                "movement_sim": [object_xyz],
            }
        elif g.current_object_xyz is not None:
            dynamic_kwargs = {
                "timestep": 0,
                "movement_sim": [g.current_object_xyz],
            }
        with torch.no_grad():
            render_pkg = g.render(
                tdgs_camera,
                g.gaussians,
                g.opt,
                g.background,
                **dynamic_kwargs,
            )
            render_pkg_nosky = g.render(
                tdgs_camera,
                g.gaussians,
                g.opt,
                g.background,
                exclude_sky=True,
                **dynamic_kwargs,
            )

        side_sky_height = 128
        sky_cond_width = 40
        opacity_nosky = render_pkg_nosky["final_opacity"]
        opacity_full = render_pkg["final_opacity"]
        if opacity_nosky.ndim == 3:
            opacity_nosky = opacity_nosky.unsqueeze(0)
        if opacity_full.ndim == 3:
            opacity_full = opacity_full.unsqueeze(0)
        if opacity_nosky.ndim != 4 or opacity_full.ndim != 4:
            raise RuntimeError(
                "Renderer opacity must have shape [1,H,W] or [1,1,H,W]"
            )
        inpaint_mask_0p5_nosky = opacity_nosky < 0.6
        inpaint_mask_0p0_nosky = opacity_nosky < 0.01
        inpaint_mask_0p5 = opacity_full < 0.6
        inpaint_mask_0p0 = opacity_full < 0.01
        mask_using_full_render = torch.zeros_like(
            inpaint_mask_0p5, dtype=render_pkg["render"].dtype
        )
        if self.preserve_sky_once:
            mask_using_full_render[:, :, :side_sky_height, :] = 1
        else:
            foreground_cols = (
                torch.sum(~inpaint_mask_0p5_nosky[:, 0], dim=1) > 150
            )
            foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]
            if foreground_cols_idx.numel() > 0:
                min_index = foreground_cols_idx.min().item()
                max_index = foreground_cols_idx.max().item()
                mask_using_full_render[:, :, :, min_index : max_index + 1] = 1
            mask_using_full_render[:, :, :sky_cond_width, :] = 1
            mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
            mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        mask_using_nosky_render = 1.0 - mask_using_full_render

        condition_image = (
            render_pkg_nosky["render"] * mask_using_nosky_render
            + render_pkg["render"] * mask_using_full_render
        )
        fill_mask = (
            inpaint_mask_0p5_nosky * mask_using_nosky_render
            + inpaint_mask_0p5 * mask_using_full_render
        )
        outpaint_mask = (
            inpaint_mask_0p0_nosky * mask_using_nosky_render
            + inpaint_mask_0p0 * mask_using_full_render
        )
        outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
        return render_pkg, condition_image, fill_mask, outpaint_mask

    def _selected_object_mask_for_view(self, viewpoint_camera, object_xyz):
        """Project the frontend-selected object into the current view.

        The selection remains the source of truth; rendering the object-only
        Gaussian subset merely accounts for camera motion during multiview
        expansion instead of copying the original 2D mask at the wrong pixels.
        """
        g = self.genesis
        if g.kf_gen.selected_object_masks is None:
            return None
        if object_xyz is None:
            object_xyz = g.current_object_xyz
        if object_xyz is None:
            return None
        object_xyz_static, _ = g.gaussians._tmp_get_xyz_all_separate()
        object_count = object_xyz_static.shape[0]
        total_count = g.gaussians.get_xyz_all.shape[0]
        render_mask = torch.zeros(
            total_count, dtype=torch.bool, device=self.config["device"]
        )
        render_mask[:object_count] = True
        with torch.no_grad():
            object_render = g.render(
                viewpoint_camera,
                g.gaussians,
                g.opt,
                g.background,
                timestep=0,
                movement_sim=[object_xyz],
                render_mask=render_mask,
            )
        opacity = object_render["final_opacity"].detach().squeeze()
        if opacity.ndim != 2:
            raise RuntimeError("Projected object opacity must be a 2D image")
        return [opacity.cpu().numpy() > 0.01]

    def expand(self, viewpoint_camera, prompt, scene_name=None, object_xyz=None):
        g = self.genesis
        kf_gen = g.kf_gen
        self._emit("Generating new scene...")
        expansion_dir = Path(kf_gen.run_dir) / f"multiview_{self.view_index:03d}"
        expansion_dir.mkdir(parents=True, exist_ok=True)

        tdgs_camera = g.convert_pt3d_cam_to_3dgs_cam(
            viewpoint_camera, xyz_scale=g.xyz_scale
        )
        render_pkg, condition_image, fill_mask, outpaint_mask = self._render_masks(
            tdgs_camera,
            object_xyz=object_xyz,
        )
        _save_tensor_image(condition_image, expansion_dir / "generation_condition.png")
        _save_tensor_image(fill_mask.float(), expansion_dir / "generation_fill_mask.png")
        _save_tensor_image(outpaint_mask.float(), expansion_dir / "generation_hole_mask.png")

        # FrameSyn.inpaint already dispatches to WonderPlay ImageEdit.
        kf_gen.inpaint(
            condition_image,
            inpaint_mask=outpaint_mask,
            fill_mask=fill_mask,
            inpainting_prompt=prompt,
            negative_prompt=self.config.get("negative_prompt", None),
            mask_strategy=np.max,
            diffusion_steps=int(
                self.config.get("multiview", {}).get("inpainting_steps", 40)
            ),
            inpaint_context=f"multiview expansion/outpaint view={self.view_index:03d}",
        )
        target_size = condition_image.shape[-2:]
        if kf_gen.image_latest.shape[-2:] != target_size:
            kf_gen.image_latest = F.interpolate(
                kf_gen.image_latest,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        _save_tensor_image(kf_gen.image_latest, expansion_dir / "generated_outpaint.png")
        if kf_gen.post_mask_latest.shape[-2:] != target_size:
            kf_gen.post_mask_latest = F.interpolate(
                kf_gen.post_mask_latest.float(),
                size=target_size,
                mode="nearest",
            )

        sem_seg = kf_gen.update_sky_mask()
        recomposed = g.soft_stitching(
            render_pkg["render"].unsqueeze(0),
            kf_gen.image_latest,
            kf_gen.sky_mask_latest,
        )
        depth_should_be = render_pkg["median_depth"][0:1].unsqueeze(0) / g.xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (
            depth_should_be > 0.001
        )
        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (
            depth_should_be_ground < 0.006 * 0.8
        )
        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(
            mask_to_align_depth, depth_should_be, depth_should_be_ground
        )
        kf_gen.get_depth(
            kf_gen.image_latest,
            target_depth=depth_should_be_joint,
            mask_align=joint_mask,
            archive_output=True,
            diffusion_steps=30,
            guidance_steps=8,
        )
        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
        kf_gen.image_latest = recomposed

        valid_mask = outpaint_mask.bool() & ~kf_gen.sky_mask_latest.bool()
        if valid_mask.sum().item() == 0:
            raise RuntimeError("Current viewpoint produced no valid expansion pixels")

        kf_gen.set_current_camera(viewpoint_camera, archive_camera=True)
        if self.config.get("gen_layer", False):
            selected_view_masks = self._selected_object_mask_for_view(
                viewpoint_camera, object_xyz
            )
            if (
                selected_view_masks is None
                and getattr(kf_gen, "defer_object_selection_until_expansion", False)
                and getattr(kf_gen, "object_mask_selector", None) is not None
            ):
                selected_view_masks = kf_gen.object_mask_selector(
                    Image.fromarray(
                        (kf_gen.image_latest[0].detach().cpu().clamp(0, 1)
                         .permute(1, 2, 0).mul(255).round().byte().numpy())
                    ),
                    local_predictor=getattr(kf_gen.mask_generator, "predictor", None),
                )
                if not selected_view_masks:
                    raise RuntimeError("Interactive object selection returned no masks")
                normalized_masks = []
                for selected_mask in selected_view_masks:
                    selected_mask = np.asarray(selected_mask).astype(bool)
                    if selected_mask.shape != (512, 512):
                        selected_mask = cv2.resize(
                            selected_mask.astype(np.uint8),
                            (512, 512),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    normalized_masks.append(selected_mask)
                kf_gen.selected_object_masks = normalized_masks
                kf_gen.defer_object_selection_until_expansion = False
            kf_gen.generate_layer(
                pred_semantic_map=sem_seg,
                foreground_ids=self.config.get("foreground_ids", [4, 76, 83, 87]),
                ground_ids=self.config.get(
                    "ground_ids", ["3", "6", "9", "11", "13", "26", "29", "46", "52", "128"]
                ),
                scene_name=scene_name,
                use_precomputed_assets=False,
                object_masks_override=selected_view_masks,
            )
            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = (
                ~kf_gen.mask_disocclusion.bool()
                & (depth_should_be < 0.006 * 0.8)
            )
            mask_to_farther_depth = (
                kf_gen.mask_disocclusion.bool()
                & (depth_should_be < 0.006 * 0.8)
            )
            with torch.no_grad():
                kf_gen.get_depth(
                    kf_gen.image_latest,
                    target_depth=depth_should_be,
                    mask_align=mask_to_align_depth,
                    mask_farther=mask_to_farther_depth,
                    archive_output=True,
                    diffusion_steps=30,
                    guidance_steps=8,
                )
            kf_gen.refine_disp_with_segments(
                no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                existing_mask=(
                    ~kf_gen.mask_disocclusion.bool().squeeze().cpu().numpy()
                ),
                existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy(),
            )
            wrong_depth_mask = kf_gen.depth_latest < kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = (
                kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            )
            kf_gen.depth_latest = (
                kf_gen.mask_disocclusion * kf_gen.depth_latest
                + (1 - kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            )
            kf_gen.update_sky_mask()
            valid_mask = outpaint_mask.bool() & ~kf_gen.sky_mask_latest.bool()

        # Frontend arrows are collected on input_image's 512x512 grid, while
        # SAM3/Cinemagraphy operate on this post-separation interaction
        # condition. The two images must stay size-aligned, not pixel-identical.
        g.annotation_image = kf_gen.image_latest.detach().clone()
        g.annotation_depth = kf_gen.depth_latest.detach().clone()
        g.annotation_valid_mask = valid_mask.detach().clone()
        kf_gen.update_current_pc_by_kf(
            valid_mask=valid_mask,
            image=kf_gen.image_latest,
            depth=kf_gen.depth_latest,
            camera=viewpoint_camera,
        )
        if self.config.get("gen_layer", False):
            # Keep LivingWorld's two point-cloud paths: the inpainted base is
            # the environment layer, while the original disoccluding pixels
            # populate the foreground-layer cache used by layered traindata.
            # Only the base traindata is appended to WonderPlay's environment
            # Gaussians below, so separately managed dynamic objects are not
            # duplicated.
            foreground_valid_mask = (
                kf_gen.mask_disocclusion.bool() & outpaint_mask.bool()
            )
            kf_gen.update_current_pc_by_kf(
                valid_mask=foreground_valid_mask,
                image=kf_gen.image_latest_init,
                depth=kf_gen.depth_latest_init,
                camera=viewpoint_camera,
                gen_layer=True,
            )
        else:
            kf_gen.image_latest_init = kf_gen.image_latest
        kf_gen.archive_latest(idx=kf_gen.kf_idx)

        if self.config.get("gen_layer", False):
            _, traindata = kf_gen.convert_to_3dgs_traindata_latest_layer(
                xyz_scale=g.xyz_scale
            )
        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(
                xyz_scale=g.xyz_scale,
                use_no_loss_mask=False,
            )
        if traindata["pcd_points"].shape[1] == 0:
            raise RuntimeError("Expansion generated an empty point cloud")

        temp_gaussians = g.GaussianModel(
            sh_degree=0,
            previous_gaussian=g.gaussians,
        )
        temp_scene = g.Scene(traindata, temp_gaussians, g.opt)
        expansion_opt = copy.deepcopy(g.opt)
        expansion_opt.iterations = int(
            self.config.get("multiview", {}).get(
                "expansion_iterations", 100
            )
        )
        g.train_gaussian(
            temp_gaussians,
            temp_scene,
            expansion_opt,
            expansion_dir,
            initialize_scaling=False,
        )

        before = g.gaussians.get_xyz_all.shape[0]
        g.gaussians.append_environment_from_gaussian(temp_gaussians)
        g.gaussians.set_inscreen_points_to_visible(tdgs_camera)
        after = g.gaussians.get_xyz_all.shape[0]
        object_count = g.gaussians._tmp_get_xyz_all_separate()[0].shape[0]
        environment_count = g.gaussians._tmp_get_xyz_all_separate()[1].shape[0]
        if after - before != temp_gaussians._xyz.shape[0]:
            raise RuntimeError("Environment Gaussian append count mismatch")
        if object_count == 0 or environment_count == 0:
            raise RuntimeError("Gaussian object/environment split is invalid")

        Image.fromarray(
            (valid_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(expansion_dir / "expansion_mask.png")
        kf_gen.increment_kf_idx()
        self._emit(
            "Expansion finished: "
            f"object={object_count}, environment={environment_count}, added={after - before}"
        )
        self.view_index += 1
        self.preserve_sky_once = False
        return expansion_dir


class InteractionManager:
    """Call the existing WonderPlay V1 interaction pipeline."""

    def __init__(
        self,
        genesis,
        config,
        simulator,
        scene,
        simulation_steps,
        video_gen_fps,
        emit,
        review_motion_mask,
        set_interaction_preview,
        set_refined_preview,
    ):
        self.genesis = genesis
        self.config = config
        self.simulator = simulator
        self.scene = scene
        self.simulation_steps = simulation_steps
        self.video_gen_fps = video_gen_fps
        self.emit = emit
        self.review_motion_mask = review_motion_mask
        self.set_interaction_preview = set_interaction_preview
        self.set_refined_preview = set_refined_preview
        self.motion_model = None

    def run(self, viewpoint_camera, render_camera, save_dir, hints):
        g = self.genesis
        g.kf_gen.set_current_camera(viewpoint_camera)
        interaction = self.config.get("interaction", {})
        direction = str(interaction.get("direction", "")).lower()
        if direction == "env2obj":
            if hints is None or len(hints) == 0:
                raise ValueError("env2obj requires at least one motion annotation")
            g.prepare_environment_motion_fields(
                save_dir,
                self.config,
                fixed_hints_override=hints,
                mask_review_callback=self.review_motion_mask,
                image_override=g.annotation_image,
                depth_override=g.annotation_depth,
                valid_mask_override=g.annotation_valid_mask,
            )
            g.sync_current_pc_scene_flow_to_final_environment_gaussians(
                self.config,
                render_camera,
                save_dir / "sam3_mask.png",
            )

        self.motion_model, simulation_states = g.run_interaction_pipeline(
            self.simulator,
            self.scene,
            save_dir,
            self.config,
            self.simulation_steps,
            self.video_gen_fps,
            viewpoint_camera=render_camera,
            motion_model=self.motion_model,
            status_callback=lambda message: self.emit("server-state", message),
            refined_preview_callback=lambda result: self.set_refined_preview(
                result["refined_frames_dir"], result["fps"]
            ),
        )
        self.set_interaction_preview(self.motion_model, simulation_states)
        self.emit("server-state", "Interaction finished for current view.")


class MultiViewController:
    """Drive the initial-view annotation and subsequent view expansions."""

    def __init__(
        self,
        genesis,
        config,
        simulator,
        scene,
        save_dir,
        simulation_steps,
        video_gen_fps,
        annotation_event,
        expansion_event,
        get_annotations,
        get_view_matrix,
        consume_expansion_request,
        consume_scene_prompt,
        set_command_handler,
        set_pipeline_phase,
        set_annotation_frame,
        set_annotation_live_preview,
        reset_annotations,
        review_motion_mask,
        set_interaction_preview,
        set_refined_preview,
        runtime_lock,
        emit,
    ):
        self.genesis = genesis
        self.config = config
        self.simulator = simulator
        self.scene = scene
        self.save_dir = Path(save_dir)
        self.simulation_steps = simulation_steps
        self.video_gen_fps = video_gen_fps
        self.annotation_event = annotation_event
        self.expansion_event = expansion_event
        self.get_annotations = get_annotations
        self.get_view_matrix = get_view_matrix
        self.consume_expansion_request = consume_expansion_request
        self.runtime_lock = runtime_lock
        self.emit = emit
        self.set_pipeline_phase = set_pipeline_phase
        self.set_annotation_frame = set_annotation_frame
        self.set_annotation_live_preview = set_annotation_live_preview
        self.reset_annotations = reset_annotations
        self._gaussian_snapshot = None
        self.prompt_manager = PromptManager(
            genesis, config, emit, consume_scene_prompt
        )
        self.expansion_manager = ExpansionManager(genesis, config, emit)
        self.interaction_manager = InteractionManager(
            genesis,
            config,
            simulator,
            scene,
            simulation_steps,
            video_gen_fps,
            emit,
            review_motion_mask,
            set_interaction_preview,
            set_refined_preview,
        )
        set_command_handler(self._handle_command)

    def _handle_command(self, command, payload=None):
        if command == "fill_hole":
            self.expansion_manager.preserve_sky_once = True
            self.emit("server-state", "Sky-preserving expansion enabled once.")
            return
        with self.runtime_lock:
            if command == "undo":
                if self._gaussian_snapshot is None:
                    self.emit("server-state", "Nothing to undo.")
                    return
                self.genesis.gaussians = self._gaussian_snapshot
                self._gaussian_snapshot = None
                self.emit("server-state", "Latest Gaussian expansion undone.")
            elif command == "delete":
                camera = self.genesis.kf_gen.get_camera_by_js_view_matrix(
                    payload or self.get_view_matrix(),
                    xyz_scale=self.genesis.xyz_scale,
                )
                tdgs_camera = self.genesis.convert_pt3d_cam_to_3dgs_cam(
                    camera, xyz_scale=self.genesis.xyz_scale
                )
                self.genesis.gaussians.delete_points(tdgs_camera)
                self.emit("server-state", "Visible non-sky points deleted.")
            elif command == "save":
                model_dir = self.genesis.kf_gen.run_dir / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                self.genesis.gaussians.save_ply_for_3dgs(
                    (model_dir / "finished_3dgs.ply").as_posix()
                )
                torch.save(
                    self.genesis.gaussians.visibility_filter_all,
                    model_dir / "visibility_filter_all.pth",
                )
                torch.save(
                    self.genesis.gaussians.is_sky_filter,
                    model_dir / "is_sky_filter.pth",
                )
                torch.save(
                    self.genesis.gaussians.delete_mask_all,
                    model_dir / "delete_mask_all.pth",
                )
                if self.interaction_manager.motion_model is not None:
                    torch.save(
                        self.interaction_manager.motion_model,
                        model_dir / "motion_model.pth",
                    )
                metadata = {
                    "object_gaussians": int(self.genesis.gaussians._xyz.shape[0]),
                    "environment_gaussians": int(
                        self.genesis.gaussians._xyz_prev.shape[0]
                    ),
                }
                (model_dir / "multiview_state.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                self.emit("server-state", f"Scene saved to {model_dir}")

    def _current_camera(self, view_matrix=None):
        if view_matrix is None:
            view_matrix = self.get_view_matrix()
        self.genesis.view_matrix_wonder = view_matrix
        return self.genesis.kf_gen.get_camera_by_js_view_matrix(
            view_matrix,
            xyz_scale=self.genesis.xyz_scale,
        )

    def _wait_for_annotation(self, message, live_preview=False):
        self.annotation_event.clear()
        self.reset_annotations()
        self.set_annotation_live_preview(live_preview)
        preparing_message = (
            "Preparing current view..." if live_preview else "Preparing input image..."
        )
        self.set_pipeline_phase("PREPARING_ANNOTATION", preparing_message)
        if not live_preview:
            # The raw input is only the first annotation surface.  After an
            # expansion, the browser must keep and then resume the scene at the
            # submitted camera pose instead of jumping back to input_image.
            display_image = getattr(self.genesis, "input_image", None)
            if display_image is None:
                display_image = self.genesis.annotation_image
            self.set_annotation_frame(display_image)
        self.set_pipeline_phase("WAITING_ANNOTATION", message)
        self.annotation_event.wait()
        annotations = self.get_annotations()
        print(
            f"[multiview] Confirmation received with "
            f"{int(annotations.shape[1])} motion arrow(s).",
            flush=True,
        )
        return annotations

    def _run_interaction(self, camera, hints, index):
        save_dir = self.save_dir / f"view_{index:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        render_camera = self.genesis.convert_pt3d_cam_to_3dgs_cam(
            camera,
            xyz_scale=self.genesis.xyz_scale,
        )
        original_image = self.genesis.annotation_image
        if index == 0 and getattr(self.genesis, "input_image", None) is not None:
            original_image = self.genesis.input_image
        render_camera.original_image = original_image[0].detach()
        self.interaction_manager.run(camera, render_camera, save_dir, hints)

    def run(self):
        initial_scene_name = self.genesis.scene_dict.get("scene_name", "")
        if isinstance(initial_scene_name, list):
            initial_scene_name = initial_scene_name[0]
        self.emit("scene-prompt", initial_scene_name)
        initial_hints = self._wait_for_annotation(
            "Waiting for OK...",
            live_preview=False,
        )
        # The first annotation frame is the raw input image captured at the
        # origin camera, regardless of browser camera movement while waiting.
        initial_camera = self.genesis.kf_gen.get_camera_at_origin()
        with self.runtime_lock:
            self._run_interaction(initial_camera, initial_hints, 0)

        while True:
            self.expansion_event.clear()
            self.set_pipeline_phase(
                "WAITING_EXPANSION", "Waiting to generate new scenes..."
            )
            self.expansion_event.wait()
            if self.config.get("multiview", {}).get("stop", False):
                return

            expansion_request = self.consume_expansion_request()
            camera = self._current_camera(expansion_request.get("view_matrix"))
            expansion_object_xyz = expansion_request.get("object_xyz")
            if expansion_object_xyz is not None:
                self.genesis.current_object_xyz = expansion_object_xyz.detach().clone()
            prompt = self.prompt_manager.next_prompt()
            scene_name = self.prompt_manager.scene_dict.get("scene_name", "")
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            with self.runtime_lock:
                ensure_models = getattr(
                    self.genesis.kf_gen, "ensure_generation_models", None
                )
                if ensure_models is not None:
                    ensure_models()
                self._gaussian_snapshot = copy.deepcopy(self.genesis.gaussians)
                expansion_dir = self.expansion_manager.expand(
                    camera,
                    prompt,
                    scene_name=scene_name,
                    object_xyz=expansion_object_xyz,
                )
                if not getattr(
                    self.simulator, "supports_runtime_environment_append", False
                ):
                    self.emit(
                        "server-state",
                        "Expansion is render-only for physics: this Genesis build "
                        "cannot add fixed particles after Scene.build().",
                    )
            hints = self._wait_for_annotation(
                "Waiting for OK...",
                live_preview=True,
            )
            with self.runtime_lock:
                self._run_interaction(camera, hints, self.expansion_manager.view_index)
            self.emit("server-state", f"View saved in {expansion_dir}")


def run_multiview_controller(
    config,
    simulator,
    scene,
    save_dir,
    simulation_steps,
    video_gen_fps,
):
    """Entry called by ``run_genesis.run`` after initial reconstruction."""
    import run_genesis as genesis
    from run_multiview_interaction import get_runtime_hooks

    hooks = get_runtime_hooks()
    controller = MultiViewController(
        genesis=genesis,
        config=config,
        simulator=simulator,
        scene=scene,
        save_dir=save_dir,
        simulation_steps=simulation_steps,
        video_gen_fps=video_gen_fps,
        annotation_event=hooks["annotation_event"],
        expansion_event=hooks["expansion_event"],
        get_annotations=hooks["get_annotations"],
        get_view_matrix=hooks["get_view_matrix"],
        consume_expansion_request=hooks["consume_expansion_request"],
        consume_scene_prompt=hooks["consume_scene_prompt"],
        set_command_handler=hooks["set_command_handler"],
        set_pipeline_phase=hooks["set_pipeline_phase"],
        set_annotation_frame=hooks["set_annotation_frame"],
        set_annotation_live_preview=hooks["set_annotation_live_preview"],
        reset_annotations=hooks["reset_annotations"],
        review_motion_mask=hooks["review_motion_mask"],
        set_interaction_preview=hooks["set_interaction_preview"],
        set_refined_preview=hooks["set_refined_preview"],
        runtime_lock=hooks["runtime_lock"],
        emit=hooks["emit"],
    )
    return controller.run()
