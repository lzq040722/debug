"""VACE-compatible RAFT optical-flow export for interaction rendering."""

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image


def save_vace_raft_flow(video_path, traj_dir, fps, checkpoint_path, device="cuda"):
    """Read the final video, run VACE's RAFT forward, and save Stage 2 flow."""
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame[..., ::-1])
    capture.release()

    if len(frames) < 2:
        raise RuntimeError(
            f"VACE RAFT requires at least two frames in the final video: {video_path}"
        )

    try:
        from raft import RAFT
        from raft.utils import flow_viz
        from raft.utils.utils import InputPadder
    except ImportError as exc:
        raise RuntimeError(
            "VACE RAFT is not installed. Install the VACE 'raft' dependency before "
            "running interaction rendering."
        ) from exc

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VACE RAFT checkpoint not found: {checkpoint_path}")

    traj_dir = Path(traj_dir)
    flows_dir = traj_dir / "flows"
    flows_actual_dir = traj_dir / "flows_actual"
    flows_dir.mkdir(parents=True, exist_ok=True)
    flows_actual_dir.mkdir(parents=True, exist_ok=True)

    raft_args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = RAFT(raft_args)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(
        {key.replace("module.", ""): value for key, value in state_dict.items()}
    )
    model = model.to(device).eval()

    frame_tensors = [
        torch.from_numpy(np.asarray(frame).astype(np.uint8))
        .permute(2, 0, 1)
        .float()[None]
        .to(device)
        for frame in frames
    ]

    raw_flows = []
    flow_visualizations = []
    with torch.no_grad():
        for image1, image2 in zip(frame_tensors[:-1], frame_tensors[1:]):
            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)
            _, flow_up = model(
                image1_padded,
                image2_padded,
                iters=20,
                test_mode=True,
            )
            flow_hwc = flow_up[0].permute(1, 2, 0).cpu().numpy()
            raw_flows.append(flow_hwc)
            flow_visualizations.append(flow_viz.flow_to_image(flow_hwc))

    # Match VACE FlowVisAnnotator: duplicate the first pair for frame zero.
    raw_flows = raw_flows[:1] + raw_flows
    flow_visualizations = flow_visualizations[:1] + flow_visualizations

    for frame_idx, (flow_hwc, flow_vis) in enumerate(
        zip(raw_flows, flow_visualizations)
    ):
        flow_chw = np.transpose(flow_hwc, (2, 0, 1)).astype(np.float32)
        flow_actual = np.clip((flow_chw / 512.0 + 1.0) * 0.5, 0.0, 1.0)
        np.save(flows_actual_dir / f"flow_{frame_idx:08d}.npy", flow_actual)
        Image.fromarray(flow_vis).save(flows_dir / f"flow_{frame_idx:08d}.png")

    imageio.mimsave(
        traj_dir / "render_flows.mp4",
        flow_visualizations,
        fps=fps,
    )
    return raw_flows, flow_visualizations
