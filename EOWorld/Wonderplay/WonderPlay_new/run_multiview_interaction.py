"""Interactive LivingWorld-style expansion + WonderPlay interaction entry."""

import io
import sys
import threading
import time
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import cv2
import torch
from flask import Flask, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from omegaconf import OmegaConf
from PIL import Image

# This file is normally launched as ``__main__``.  The multiview controller
# imports it by its module name, so make both import paths share one module
# instance and therefore one set of events and pipeline state.
sys.modules["run_multiview_interaction"] = sys.modules[__name__]

import run_genesis as genesis


_SPLAT_DIR = Path(__file__).resolve().parent / "splat-main"
app = Flask(
    __name__,
    static_folder=str(_SPLAT_DIR),
    static_url_path="",
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

_runtime_lock = threading.Lock()
_annotation_event = threading.Event()
_mask_review_event = threading.Event()
_object_selection_event = threading.Event()
_expansion_event = threading.Event()
_stop_event = threading.Event()
_record_lock = threading.Lock()
_frame_lock = threading.Lock()
_client_lock = threading.Lock()
_client_ids = set()
_annotation_lock = threading.Lock()
_object_selection_lock = threading.Lock()
_phase_lock = threading.Lock()
_view_lock = threading.Lock()
_playback_lock = threading.Lock()
_pipeline_phase = "INITIALIZING"
_phase_message = "Pipeline is initializing."
_view_matrix = list(genesis.view_matrix_wonder)
_expansion_request = None
_last_emitted_frame_info = None
_last_preview_frame_bytes = None
_last_preview_frame_info = None
_fixed_view_matrix = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.2, 0.5, 1],
    ],
    dtype=float,
)
_theta = np.radians(-3)
_fixed_view_matrix = (
    _fixed_view_matrix
    @ np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(_theta), -np.sin(_theta), 0],
            [0, np.sin(_theta), np.cos(_theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
).flatten().tolist()
_sam_enabled = False
_sam_prompt = "water"
_scale_factor = 1.0
_clicks = []
_confirmed_clicks = []
_annotation_frame_bytes = None
_preview_frame_bytes = None
_pending_scene_prompt = None
_command_handler = None
_capture_active = False
_capture_writer = None
_capture_path = None
_capture_frame_count = 0
_interaction_motion_model = None
_interaction_simulation_states = []
_interaction_frame_index = 0
_interaction_last_emit = 0.0
_interaction_fps = 8.0
_refined_preview_frames = []
_refined_frame_index = 0
_annotation_live_preview = False
_object_selection_image = None
_object_selection_predictor = None
_object_selection_points = []
_object_selection_labels = []
_object_selection_mask = None


@app.route("/")
def index():
    return app.send_static_file("index_stream.html")


@app.route("/annotation-frame.png")
def annotation_frame():
    """Serve the current annotation image without requiring Socket.IO."""
    with _frame_lock:
        frame_bytes = _annotation_frame_bytes
    if frame_bytes is None:
        return Response(status=404)
    return Response(
        frame_bytes,
        mimetype="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _emit(event, payload):
    with _client_lock:
        client_ids = tuple(_client_ids)
    for client_id in client_ids:
        socketio.emit(event, payload, room=client_id)


def _set_pipeline_phase(phase, message):
    global _pipeline_phase, _phase_message
    with _phase_lock:
        _pipeline_phase = str(phase)
        _phase_message = str(message)
    print(f"[multiview] phase={_pipeline_phase}: {_phase_message}", flush=True)
    _emit(
        "pipeline-phase",
        {"phase": _pipeline_phase, "message": _phase_message},
    )
    _emit("server-state", _phase_message)


def _get_pipeline_phase():
    with _phase_lock:
        return _pipeline_phase, _phase_message


def _image_to_png_bytes(image):
    if torch.is_tensor(image):
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 4:
            tensor = tensor[0]
        array = (
            tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
        )
        image = Image.fromarray(array, mode="RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_jpeg_bytes(image):
    if torch.is_tensor(image):
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 4:
            tensor = tensor[0]
        array = (
            tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
        )
        image = Image.fromarray(array, mode="RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _set_annotation_frame(image):
    global _annotation_frame_bytes, _preview_frame_bytes
    frame_bytes = _image_to_png_bytes(image)
    preview_frame_bytes = _image_to_jpeg_bytes(image)
    with _frame_lock:
        _annotation_frame_bytes = frame_bytes
        _preview_frame_bytes = preview_frame_bytes
        _emit("annotation-frame", _annotation_frame_bytes)


def _reset_annotations():
    with _annotation_lock:
        _clicks.clear()
        _confirmed_clicks.clear()


def _object_mask_overlay(image, mask, points, labels):
    array = np.asarray(image.convert("RGB")).copy()
    if mask is not None:
        selected = np.asarray(mask).astype(bool)
        array[selected] = (
            array[selected].astype(np.float32) * 0.35
            + np.array([30, 144, 255], dtype=np.float32) * 0.65
        ).astype(np.uint8)
    for (x, y), label in zip(points, labels):
        color = (30, 220, 80) if label == 1 else (235, 50, 65)
        cv2.circle(array, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(array, (int(x), int(y)), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(array)


def _select_object_mask(image, local_predictor=None):
    """Block the reconstruction at its existing mask-selection point."""
    global _object_selection_image, _object_selection_predictor
    global _object_selection_mask

    image = image.convert("RGB")
    with _object_selection_lock:
        _object_selection_image = image.copy()
        _object_selection_predictor = local_predictor
        _object_selection_points.clear()
        _object_selection_labels.clear()
        _object_selection_mask = None
    _object_selection_event.clear()
    _set_annotation_frame(image)
    _set_pipeline_phase(
        "OBJECT_SELECTION",
        "Click the moving object. Shift-click excludes background; then Confirm.",
    )

    runtime = getattr(genesis, "auxiliary_model_runtime", None)
    if runtime is not None:
        runtime.set_object_image(image, local_predictor=local_predictor)
    elif local_predictor is not None:
        local_predictor.set_image(np.asarray(image))

    _object_selection_event.wait()
    with _object_selection_lock:
        if _object_selection_mask is None:
            raise RuntimeError("Object selection was confirmed without a valid mask")
        selected_mask = np.asarray(_object_selection_mask).astype(bool).copy()
    _set_pipeline_phase("PROCESSING", "Object mask confirmed. Reconstructing scene...")
    return [selected_mask]


def _review_motion_mask(image):
    # LivingWorld proceeds directly from SAM3 segmentation to flow
    # estimation.  Keep this callback for the existing controller hook, but
    # do not replace the input frame or wait for a second user confirmation.
    _set_pipeline_phase(
        "PROCESSING",
        "SAM3 mask generated. Computing motion flow...",
    )


def _current_view_matrix():
    with _view_lock:
        return list(_view_matrix)


def _normalise_view_matrix_payload(data):
    if isinstance(data, dict):
        matrix = (
            data.get("view_matrix")
            or data.get("viewMatrix")
            or data.get("matrix")
        )
    else:
        matrix = data
    if matrix is None:
        raise ValueError("Missing view matrix")
    matrix = [float(value) for value in matrix]
    if len(matrix) != 16:
        raise ValueError(f"Expected 16 view-matrix values, got {len(matrix)}")
    return matrix


def _displayed_frame_from_payload(data):
    if not isinstance(data, dict):
        return None
    frame = data.get("displayed_frame") or data.get("displayedFrame")
    return frame if isinstance(frame, dict) else None


def _snapshot_playback_object(displayed_frame=None):
    global _last_emitted_frame_info
    with _playback_lock:
        states = list(_interaction_simulation_states)
        if not states:
            return {
                "object_xyz": None,
                "frame_index": None,
                "frame_source": None,
            }

        frame_info = displayed_frame or _last_emitted_frame_info or {}
        try:
            frame_index = int(frame_info.get("frame_index"))
        except (TypeError, ValueError):
            frame_index = (_interaction_frame_index - 1) % len(states)
        frame_index = max(0, min(len(states) - 1, frame_index))
        object_xyz = states[frame_index]["obj_0000"]["xyz"].detach().clone()
        return {
            "object_xyz": object_xyz,
            "frame_index": frame_index,
            "frame_source": frame_info.get("source"),
        }


def _store_expansion_request(view_matrix, displayed_frame=None):
    global _view_matrix, _expansion_request
    playback_snapshot = _snapshot_playback_object(displayed_frame)
    request_payload = {
        "view_matrix": list(view_matrix),
        **playback_snapshot,
    }
    with _view_lock:
        _view_matrix = list(view_matrix)
        _expansion_request = request_payload
    if request_payload["object_xyz"] is not None:
        genesis.current_object_xyz = request_payload["object_xyz"].detach().clone()
        print(
            "[multiview] Expansion locked to displayed playback "
            f"frame={request_payload['frame_index']} "
            f"source={request_payload['frame_source']}",
            flush=True,
        )


def _consume_expansion_request():
    global _expansion_request
    with _view_lock:
        request_payload = _expansion_request
        _expansion_request = None
        fallback_view_matrix = list(_view_matrix)
    if request_payload is None:
        return {
            "view_matrix": fallback_view_matrix,
            "object_xyz": None,
            "frame_index": None,
            "frame_source": None,
        }
    return request_payload


def _clicks_to_hints(clicks):
    if not clicks:
        return np.empty((4, 0), dtype=int)
    points = list(clicks)
    if len(points) % 2 == 1:
        points = points[1:]
    if not points:
        return np.empty((4, 0), dtype=int)
    pairs = np.asarray(points, dtype=int).reshape(-1, 2, 2)
    return np.stack(
        [pairs[:, 0, 0], pairs[:, 0, 1], pairs[:, 1, 0], pairs[:, 1, 1]],
        axis=0,
    )


def _points_from_arrow_payload(data):
    if not isinstance(data, dict) or "arrows" not in data:
        return None
    points = []
    for arrow in data.get("arrows", []):
        if not isinstance(arrow, dict):
            raise ValueError("Each motion arrow must be an object")
        start, end = arrow.get("start"), arrow.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ValueError("Each motion arrow requires start and end points")
        for point in (start, end):
            x = max(0, min(511, int(round(float(point["x"])))))
            y = max(0, min(511, int(round(float(point["y"])))))
            points.append((x, y))
    return points


def _get_annotations():
    with _annotation_lock:
        confirmed = list(_confirmed_clicks)
    return _clicks_to_hints(confirmed)


def _consume_scene_prompt():
    global _pending_scene_prompt
    prompt = _pending_scene_prompt
    _pending_scene_prompt = None
    return prompt


def _set_command_handler(handler):
    global _command_handler
    _command_handler = handler


def _set_interaction_preview(motion_model, simulation_states):
    global _interaction_motion_model, _interaction_simulation_states
    global _interaction_frame_index, _interaction_last_emit, _annotation_live_preview
    global _interaction_fps
    with _playback_lock:
        _interaction_motion_model = motion_model
        _interaction_simulation_states = list(simulation_states or [])
        if _interaction_simulation_states:
            refinement = _interaction_simulation_states[0].get("_video_refinement")
            if refinement is not None:
                _interaction_fps = float(refinement.get("fps", _interaction_fps))
        _interaction_frame_index = 0
        _interaction_last_emit = 0.0
    print(
        "[multiview] Interaction preview ready: "
        f"{len(_interaction_simulation_states)} frame state(s).",
        flush=True,
    )


def _set_refined_preview(frames_dir, fps=8):
    global _refined_preview_frames, _refined_frame_index, _interaction_last_emit
    global _interaction_fps
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    refined_frames = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.uint8)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        refined_frames.append((image_np, buffer.getvalue()))
    if not refined_frames:
        raise FileNotFoundError(f"No refined frame_*.png files found in {frames_dir}")
    _refined_preview_frames = refined_frames
    _refined_frame_index = 0
    _interaction_last_emit = 0.0
    _interaction_fps = float(fps)
    print(
        f"[multiview] Refined preview ready: {len(refined_frames)} frame(s) at {fps} fps.",
        flush=True,
    )


def _clear_interaction_preview():
    global _refined_preview_frames, _refined_frame_index
    global _interaction_motion_model, _interaction_simulation_states
    global _interaction_frame_index, _interaction_last_emit, _annotation_live_preview
    global _last_emitted_frame_info
    with _playback_lock:
        _interaction_motion_model = None
        _interaction_simulation_states = []
        _interaction_frame_index = 0
        _interaction_last_emit = 0.0
        _refined_preview_frames = []
        _refined_frame_index = 0
        _annotation_live_preview = False
        _last_emitted_frame_info = None


def _emit_frame(frame_bytes, source, frame_index=None):
    global _last_emitted_frame_info, _last_preview_frame_bytes
    global _last_preview_frame_info
    frame_info = {"source": source, "frame_index": frame_index}
    with _playback_lock:
        _last_emitted_frame_info = frame_info
        _last_preview_frame_bytes = frame_bytes
        _last_preview_frame_info = frame_info
    _emit("frame-info", frame_info)
    _emit("frame", frame_bytes)


def _set_annotation_live_preview(enabled):
    global _annotation_live_preview
    _annotation_live_preview = bool(enabled)


def _dispatch_command(command, payload=None):
    if _command_handler is None:
        emit("server-state", "Pipeline is still initializing.", room=request.sid)
        return
    try:
        _command_handler(command, payload)
    except Exception as exc:
        emit("server-state", f"{command} failed: {exc}", room=request.sid)


def get_runtime_hooks():
    """Return callbacks consumed by ``multiview_controller``."""
    return {
        "annotation_event": _annotation_event,
        "mask_review_event": _mask_review_event,
        "expansion_event": _expansion_event,
        "get_annotations": _get_annotations,
        "get_view_matrix": _current_view_matrix,
        "consume_expansion_request": _consume_expansion_request,
        "consume_scene_prompt": _consume_scene_prompt,
        "set_command_handler": _set_command_handler,
        "set_pipeline_phase": _set_pipeline_phase,
        "set_annotation_frame": _set_annotation_frame,
        "set_annotation_live_preview": _set_annotation_live_preview,
        "reset_annotations": _reset_annotations,
        "set_interaction_preview": _set_interaction_preview,
        "set_refined_preview": _set_refined_preview,
        "review_motion_mask": _review_motion_mask,
        "runtime_lock": _runtime_lock,
        "emit": _emit,
    }


@socketio.on("connect")
def _handle_connect():
    with _client_lock:
        _client_ids.add(request.sid)
    emit("scale-state", {"value": _scale_factor}, room=request.sid)
    if isinstance(genesis.scene_dict, dict):
        emit("scene-prompt", genesis.scene_dict.get("scene_name", ""), room=request.sid)
    # A reconnect must restore the active phase so the relevant controls are
    # enabled again.  A generic "connected" message loses that information.
    phase, phase_message = _get_pipeline_phase()
    show_annotation_frame = (
        phase == "INITIALIZING"
        or phase == "OBJECT_SELECTION"
        or (
            phase in {"PREPARING_ANNOTATION", "WAITING_ANNOTATION"}
            and not _annotation_live_preview
        )
        or phase in {"PREPARING_MASK_REVIEW", "MASK_REVIEW"}
    )
    with _frame_lock:
        annotation_frame_bytes = _annotation_frame_bytes
    with _playback_lock:
        preview_frame_bytes = _last_preview_frame_bytes
        preview_frame_info = _last_preview_frame_info
    if show_annotation_frame and annotation_frame_bytes is not None:
        emit("annotation-frame", annotation_frame_bytes, room=request.sid)
    elif preview_frame_bytes is not None:
        # Reconnects during PROCESSING or WAITING_EXPANSION should restore the
        # last scene frame, never the original input image.
        emit("frame-info", preview_frame_info, room=request.sid)
        emit("frame", preview_frame_bytes, room=request.sid)
    emit(
        "pipeline-phase",
        {"phase": phase, "message": phase_message},
        room=request.sid,
    )
    emit("server-state", phase_message, room=request.sid)


@socketio.on("disconnect")
def _handle_disconnect():
    with _client_lock:
        _client_ids.discard(request.sid)


@socketio.on("start")
def _handle_start(data=None):
    phase, phase_message = _get_pipeline_phase()
    if phase == "OBJECT_SELECTION":
        return handle_object_selection({"action": "confirm"})
    if phase in {"MASK_REVIEW", "WAITING_ANNOTATION"}:
        return handle_ok_start(data)
    emit("server-state", phase_message, room=request.sid)


@socketio.on("render-pose")
def handle_render_pose(data):
    global _view_matrix
    try:
        view_matrix = _normalise_view_matrix_payload(data)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
    with _view_lock:
        _view_matrix = list(view_matrix)
    genesis.view_matrix_wonder = list(view_matrix)
    return {"ok": True}


@socketio.on("gen")
def handle_gen(data):
    phase, phase_message = _get_pipeline_phase()
    if phase != "WAITING_EXPANSION":
        return {"ok": False, "message": f"Generate rejected: {phase_message}"}
    try:
        view_matrix = _normalise_view_matrix_payload(data)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
    _store_expansion_request(view_matrix, _displayed_frame_from_payload(data))
    genesis.view_matrix = list(view_matrix)
    _clear_interaction_preview()
    _set_pipeline_phase("PROCESSING", "Generating new scene...")
    _expansion_event.set()
    return {"ok": True, "message": "New viewpoint requested."}


@socketio.on("scene-prompt")
def handle_new_prompt(data):
    global _pending_scene_prompt
    if not isinstance(data, str):
        return
    _pending_scene_prompt = data
    genesis.scene_name = data
    if isinstance(genesis.scene_dict, dict):
        genesis.scene_dict["scene_name"] = data


@socketio.on("object-selection")
def handle_object_selection(msg):
    global _object_selection_mask

    phase, phase_message = _get_pipeline_phase()
    if phase != "OBJECT_SELECTION":
        return {"ok": False, "message": f"Object selection rejected: {phase_message}"}
    msg = msg or {}
    action = str(msg.get("action", "point")).lower()

    if action == "confirm":
        with _object_selection_lock:
            valid = _object_selection_mask is not None and bool(
                np.asarray(_object_selection_mask).any()
            )
        if not valid:
            return {"ok": False, "message": "Click the object before confirming."}
        _object_selection_event.set()
        return {"ok": True, "message": "Object mask confirmed."}

    with _object_selection_lock:
        if action == "clear":
            _object_selection_points.clear()
            _object_selection_labels.clear()
            _object_selection_mask = None
        elif action == "undo":
            if _object_selection_points:
                _object_selection_points.pop()
                _object_selection_labels.pop()
            _object_selection_mask = None
        elif action == "point":
            xy = msg.get("xy")
            if xy is None or len(xy) != 2:
                return {"ok": False, "message": "Object selection point is missing."}
            width, height = msg.get("size", [512, 512])
            source_width, source_height = _object_selection_image.size
            x = int(round(float(xy[0]) * source_width / max(1, int(width))))
            y = int(round(float(xy[1]) * source_height / max(1, int(height))))
            x = max(0, min(source_width - 1, x))
            y = max(0, min(source_height - 1, y))
            _object_selection_points.append((x, y))
            _object_selection_labels.append(0 if int(msg.get("label", 1)) == 0 else 1)
        else:
            return {"ok": False, "message": f"Unknown object-selection action: {action}"}
        points = list(_object_selection_points)
        labels = list(_object_selection_labels)
        image = _object_selection_image.copy()
        predictor = _object_selection_predictor

    if points:
        try:
            runtime = getattr(genesis, "auxiliary_model_runtime", None)
            if runtime is not None:
                mask = runtime.segment_object(points, labels, local_predictor=predictor)
            elif predictor is not None:
                masks, scores, _ = predictor.predict(
                    point_coords=np.asarray(points, dtype=np.float32),
                    point_labels=np.asarray(labels, dtype=np.int32),
                    multimask_output=True,
                )
                mask = masks[int(np.argmax(scores))]
            else:
                raise RuntimeError("SAM predictor is not available")
        except Exception as exc:
            return {"ok": False, "message": f"Object segmentation failed: {exc}"}
        with _object_selection_lock:
            if (
                points != _object_selection_points
                or labels != _object_selection_labels
            ):
                return {
                    "ok": True,
                    "message": "A newer object-selection update is being processed.",
                }
            _object_selection_mask = np.asarray(mask).astype(bool)
            mask = _object_selection_mask.copy()
    else:
        mask = None

    _set_annotation_frame(_object_mask_overlay(image, mask, points, labels))
    return {
        "ok": True,
        "message": f"Object mask updated with {len(points)} point(s).",
    }


@socketio.on("sam-toggle")
def on_sam_toggle(msg):
    global _sam_enabled
    _sam_enabled = bool(msg.get("enabled", False))
    print(
        f"[multiview] SAM {'ON' if _sam_enabled else 'OFF'}",
        flush=True,
    )
    emit("server-state", f"SAM {'ON' if _sam_enabled else 'OFF'}", room=request.sid)


@socketio.on("sam-click")
def on_sam_click(msg):
    if not _sam_enabled:
        print(
            "[multiview] SAM click rejected: motion annotation is disabled",
            flush=True,
        )
        return {"ok": False, "reason": "Motion annotation is disabled"}
    size = msg.get("size", [512, 512])
    width, height = int(size[0]), int(size[1])
    xy = msg.get("xy")
    if xy is None:
        uv = msg.get("uv")
        if uv is None:
            return {"ok": False, "reason": "Missing xy/uv"}
        x = round(float(uv[0]) * (width - 1))
        y = round(float(uv[1]) * (height - 1))
    else:
        x, y = int(xy[0]), int(xy[1])
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    with _annotation_lock:
        if not _clicks or abs(_clicks[-1][0] - x) > 4 or abs(_clicks[-1][1] - y) > 4:
            _clicks.append((x, y))
            if len(_clicks) > 100:
                del _clicks[:-100]
        click_count = len(_clicks)
    print(
        f"[multiview] SAM click accepted: ({x}, {y}), "
        f"buffered_points={click_count}",
        flush=True,
    )
    return {"ok": True}


@socketio.on("sam-clear")
def on_sam_clear():
    _reset_annotations()
    emit("server-state", "Motion annotations cleared.", room=request.sid)
    return {"ok": True}


@socketio.on("ok-start")
def handle_ok_start(data=None):
    phase, phase_message = _get_pipeline_phase()
    print(
        f"[multiview] Confirm request received: phase={phase}, "
        f"event_data={'present' if data is not None else 'none'}",
        flush=True,
    )
    if phase == "OBJECT_SELECTION":
        return handle_object_selection({"action": "confirm"})
    if phase == "MASK_REVIEW":
        message = "SAM3 mask confirmed. Computing motion flow..."
        _set_pipeline_phase("PROCESSING", message)
        _mask_review_event.set()
        return {"ok": True, "arrow_count": 0, "message": message}

    if phase != "WAITING_ANNOTATION":
        message = f"Confirm rejected: pipeline state is '{phase_message}'."
        print(f"[multiview] {message}", flush=True)
        return {"ok": False, "message": message}

    try:
        payload_points = _points_from_arrow_payload(data)
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid motion arrows: {exc}"}
    with _annotation_lock:
        # LivingWorld fixes the points already received by sam-click when
        # OK is pressed.  An empty client payload must not erase that buffer.
        buffered_points = list(_clicks)
        if payload_points:
            confirmed_points = payload_points
        else:
            confirmed_points = buffered_points
        _confirmed_clicks[:] = confirmed_points
        arrow_count = len(_confirmed_clicks) // 2
    print(
        f"[multiview] Confirm snapshot: buffered_points={len(buffered_points)}, "
        f"payload_points={len(payload_points or [])}, "
        f"confirmed_points={len(_confirmed_clicks)}",
        flush=True,
    )
    direction = str(
        genesis_runtime_config.get("interaction", {}).get("direction", "")
    ).lower()
    if direction == "env2obj" and arrow_count == 0:
        message = (
            "Confirm rejected: no motion arrows reached the server. "
            "Enable Motion annotation and draw at least one arrow."
        )
        print(f"[multiview] {message}", flush=True)
        return {"ok": False, "arrow_count": 0, "message": message}

    _set_pipeline_phase("PROCESSING", "Motion annotation received. Processing...")
    _annotation_event.set()
    message = f"OK clicked. Motion annotation confirmed: {arrow_count} arrow(s)."
    print(f"[multiview] {message}", flush=True)
    _emit("server-state", message)
    return {
        "ok": True,
        "arrow_count": arrow_count,
        "message": message,
    }


@socketio.on("set-sam-prompt")
def on_set_sam_prompt(msg):
    global _sam_prompt
    _sam_prompt = str(msg.get("value", _sam_prompt))
    if "environment_motion" in genesis_runtime_config:
        genesis_runtime_config["environment_motion"]["sam_prompt"] = _sam_prompt
    message = f"SAM3 mask prompt set to '{_sam_prompt}'."
    emit("server-state", message, room=request.sid)
    return {"ok": True, "value": _sam_prompt, "message": message}


@socketio.on("set-scale")
def on_set_scale(msg):
    global _scale_factor
    _scale_factor = max(0.0, float(msg.get("value", _scale_factor)))
    if "interaction" in genesis_runtime_config:
        genesis_runtime_config["interaction"]["environment_scale_factor"] = _scale_factor
    emit("scale-state", {"value": _scale_factor}, room=request.sid)


@socketio.on("undo")
def handle_undo():
    _dispatch_command("undo")


@socketio.on("save")
def handle_save():
    _dispatch_command("save")


@socketio.on("fill_hole")
def handle_fill_hole():
    _dispatch_command("fill_hole")


@socketio.on("delete")
def handle_delete(data):
    _dispatch_command("delete", list(data))


@socketio.on("capture")
def handle_capture():
    global _capture_active, _capture_writer, _capture_path, _capture_frame_count
    if genesis.kf_gen is None:
        emit("server-state", "Pipeline is still initializing.", room=request.sid)
        return
    with _record_lock:
        if _capture_writer is not None:
            _capture_writer.release()
        capture_dir = genesis.kf_gen.run_dir / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        _capture_path = capture_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        _capture_writer = None
        _capture_frame_count = 0
        _capture_active = True
    emit("server-state", "Recording preview frames.", room=request.sid)


@socketio.on("stop")
def handle_stop():
    global _capture_active, _capture_writer, _capture_path, _capture_frame_count
    with _record_lock:
        was_active = _capture_active
        _capture_active = False
        if _capture_writer is not None:
            _capture_writer.release()
        video_path = _capture_path
        frame_count = _capture_frame_count
        _capture_writer = None
        _capture_path = None
        _capture_frame_count = 0
    if not was_active:
        emit("server-state", "No preview recording is active.", room=request.sid)
        return
    if frame_count == 0:
        emit("server-state", "Recording stopped without frames.", room=request.sid)
        return
    emit("server-state", f"Recording saved to {video_path}", room=request.sid)


def _interaction_scale_factor():
    interaction = genesis_runtime_config.get("interaction", {})
    environment_motion = genesis_runtime_config.get("environment_motion", {})
    return float(
        interaction.get(
            "environment_scale_factor",
            environment_motion.get("scale_factor", _scale_factor),
        )
    )


def _render_interaction_frame(
    tdgs_camera,
    frame_index,
    state=None,
    motion_model=None,
):
    if state is None:
        state = _interaction_simulation_states[frame_index]
    if motion_model is None:
        motion_model = _interaction_motion_model
    obj_xyz_t = state["obj_0000"]["xyz"]
    refinement = state.get("_video_refinement")
    render_timestep = frame_index
    override_color = None
    if refinement is not None:
        render_timestep = int(refinement["source_state_index"])
        object_features = refinement["object_features"].to(
            device=obj_xyz_t.device, dtype=genesis.gaussians.get_xyz_all.dtype
        )
        background_features = refinement["background_features"].to(
            device=obj_xyz_t.device, dtype=genesis.gaussians.get_xyz_all.dtype
        )
        feature_logits = torch.cat([object_features, background_features], dim=0)
        override_color = genesis.gaussians.color_activation(feature_logits)
    render_pkg = genesis.render_interaction_mlp(
        viewpoint_camera=tdgs_camera,
        pc=genesis.gaussians,
        motion_model=motion_model,
        obj_xyz_t=obj_xyz_t,
        t=render_timestep,
        opt=genesis.opt,
        bg_color=genesis.background,
        override_color=override_color,
        render_visible=False,
        scale_factor=_interaction_scale_factor(),
    )
    image = render_pkg["render"].detach().cpu().clamp(0, 1)
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image_np).save(buffer, format="JPEG", quality=90)
    return image_np, buffer.getvalue()


def _render_static_frame(tdgs_camera):
    dynamic_kwargs = {}
    if genesis.current_object_xyz is not None:
        dynamic_kwargs = {
            "timestep": 0,
            "movement_sim": [genesis.current_object_xyz],
        }
    render_pkg = genesis.render(
        tdgs_camera,
        genesis.gaussians,
        genesis.opt,
        genesis.background,
        **dynamic_kwargs,
    )
    image = render_pkg["render"].detach().cpu().clamp(0, 1)
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image_np).save(buffer, format="JPEG", quality=90)
    return image_np, buffer.getvalue()


def _fixed_overview_camera():
    fixed_camera = genesis.kf_gen.get_camera_by_js_view_matrix(
        _fixed_view_matrix,
        xyz_scale=genesis.xyz_scale,
        big_view=True,
    )
    fixed_tdgs_camera = genesis.convert_pt3d_cam_to_3dgs_cam(
        fixed_camera, xyz_scale=genesis.xyz_scale
    )
    fixed_tdgs_camera.image_width = 1536
    return fixed_tdgs_camera


def _record_preview_frame(image_np):
    global _capture_writer, _capture_frame_count
    with _record_lock:
        if not _capture_active:
            return
        if _capture_writer is None:
            height, width = image_np.shape[:2]
            _capture_writer = cv2.VideoWriter(
                _capture_path.as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (width, height),
            )
            if not _capture_writer.isOpened():
                raise RuntimeError("Failed to open preview MP4 writer")
        _capture_writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        _capture_frame_count += 1


def render_current_scene():
    """Render browser previews from the in-memory scene, like LivingWorld."""
    global _interaction_frame_index, _interaction_last_emit, _refined_frame_index
    while not _stop_event.is_set():
        try:
            phase, _ = _get_pipeline_phase()
            show_annotation_frame = (
                phase == "INITIALIZING"
                or phase == "OBJECT_SELECTION"
                or (
                    phase in {"PREPARING_ANNOTATION", "WAITING_ANNOTATION"}
                    and not _annotation_live_preview
                )
                or phase in {"PREPARING_MASK_REVIEW", "MASK_REVIEW"}
                or genesis.kf_gen is None
                or genesis.gaussians is None
            )
            if show_annotation_frame:
                with _frame_lock:
                    preview_frame_bytes = _preview_frame_bytes
                if preview_frame_bytes is not None:
                    _emit_frame(preview_frame_bytes, "annotation")
                time.sleep(0.05)
                continue
            if genesis.kf_gen is None or genesis.gaussians is None:
                time.sleep(0.1)
                continue

            now = time.monotonic()
            if now - _interaction_last_emit < 1.0 / _interaction_fps:
                time.sleep(0.01)
                continue

            refined_preview = None
            with _playback_lock:
                if _refined_preview_frames:
                    frame_index = _refined_frame_index % len(
                        _refined_preview_frames
                    )
                    refined_preview = (
                        frame_index,
                        _refined_preview_frames[frame_index],
                    )
                    _refined_frame_index = (frame_index + 1) % len(
                        _refined_preview_frames
                    )
            if refined_preview is not None:
                frame_index, (image_np, frame_bytes) = refined_preview
                _record_preview_frame(image_np)
                _emit_frame(frame_bytes, "refined", frame_index)
                _interaction_last_emit = now
                time.sleep(0.03)
                continue

            with _runtime_lock, torch.no_grad():
                camera = genesis.kf_gen.get_camera_by_js_view_matrix(
                    _current_view_matrix(), xyz_scale=genesis.xyz_scale
                )
                tdgs_camera = genesis.convert_pt3d_cam_to_3dgs_cam(
                    camera, xyz_scale=genesis.xyz_scale
                )

                interaction_preview = None
                with _playback_lock:
                    if (
                        _interaction_motion_model is not None
                        and _interaction_simulation_states
                    ):
                        frame_index = _interaction_frame_index % len(
                            _interaction_simulation_states
                        )
                        interaction_preview = (
                            frame_index,
                            _interaction_simulation_states[frame_index],
                            _interaction_motion_model,
                        )
                        _interaction_frame_index = (frame_index + 1) % len(
                            _interaction_simulation_states
                        )

                if interaction_preview is not None:
                    frame_index, interaction_state, motion_model = (
                        interaction_preview
                    )
                    image_np, rendered_frame = _render_interaction_frame(
                        tdgs_camera,
                        frame_index,
                        state=interaction_state,
                        motion_model=motion_model,
                    )

                    fixed_tdgs_camera = _fixed_overview_camera()
                    _, rendered_viz = _render_interaction_frame(
                        fixed_tdgs_camera,
                        frame_index,
                        state=interaction_state,
                        motion_model=motion_model,
                    )
                else:
                    image_np, rendered_frame = _render_static_frame(tdgs_camera)
                    _, rendered_viz = _render_static_frame(_fixed_overview_camera())

                _record_preview_frame(image_np)
                phase, _ = _get_pipeline_phase()
                can_emit_annotation_preview = (
                    phase == "WAITING_ANNOTATION" and _annotation_live_preview
                )
                if (
                    can_emit_annotation_preview
                    or phase not in {"PREPARING_ANNOTATION", "WAITING_ANNOTATION"}
                ):
                    has_interaction_preview = interaction_preview is not None
                    frame_source = "interaction" if has_interaction_preview else "static"
                    emitted_frame_index = frame_index if has_interaction_preview else None
                    _emit_frame(rendered_frame, frame_source, emitted_frame_index)
                    _emit("viz", rendered_viz)
                    _interaction_last_emit = now
        except Exception as exc:
            _emit("server-state", f"Preview render skipped: {exc}")
            time.sleep(0.2)
            continue
        time.sleep(0.03)


def start_server(port):
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)


genesis_runtime_config = {}


def run(config, prefix=None, port=5000):
    global genesis_runtime_config, _capture_active, _capture_writer
    global _pipeline_phase, _phase_message, _scale_factor, _sam_prompt
    global _interaction_motion_model, _interaction_simulation_states
    global _interaction_frame_index, _interaction_last_emit, _annotation_live_preview
    global _expansion_request, _last_emitted_frame_info
    global _last_preview_frame_bytes, _last_preview_frame_info
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    genesis_runtime_config = config
    config["multiview"]["enabled"] = True
    config["multiview"]["stop"] = False
    _stop_event.clear()
    _annotation_event.clear()
    _mask_review_event.clear()
    _object_selection_event.clear()
    _expansion_event.clear()
    _reset_annotations()
    _interaction_motion_model = None
    _interaction_simulation_states = []
    _interaction_frame_index = 0
    _interaction_last_emit = 0.0
    _annotation_live_preview = False
    _expansion_request = None
    _last_emitted_frame_info = None
    _last_preview_frame_bytes = None
    _last_preview_frame_info = None
    _pipeline_phase = "INITIALIZING"
    _phase_message = "Pipeline is initializing."
    _sam_prompt = str(config.get("environment_motion", {}).get("sam_prompt", "water"))
    _scale_factor = float(
        config.get("interaction", {}).get("environment_scale_factor", 10.0)
    )

    repo_root = Path(__file__).resolve().parent.parent
    configured_image_path = config.get("image_filepath", None)
    if configured_image_path:
        image_path = Path(configured_image_path)
    else:
        image_path = (
            Path(config.get("data_path", "examples/imgs"))
            / config["example_name"]
            / config.get("input_image", "image.png")
        )
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    initial_image = Image.open(image_path).convert("RGB")
    side = min(initial_image.size)
    left = (initial_image.width - side) // 2
    top = (initial_image.height - side) // 2
    initial_image = initial_image.crop((left, top, left + side, top + side)).resize(
        (512, 512)
    )
    _set_annotation_frame(initial_image)

    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    render_thread = threading.Thread(target=render_current_scene, daemon=True)
    render_thread.start()

    from auxiliary_runtime import build_auxiliary_runtime

    auxiliary_runtime = build_auxiliary_runtime(config)
    try:
        genesis.auxiliary_model_runtime = auxiliary_runtime
        genesis.object_mask_selector = _select_object_mask
        genesis.run(config, dt_string=prefix)
    finally:
        genesis.object_mask_selector = None
        genesis.auxiliary_model_runtime = None
        if auxiliary_runtime is not None:
            auxiliary_runtime.close()
        _stop_event.set()
        with _record_lock:
            _capture_active = False
            if _capture_writer is not None:
                _capture_writer.release()
                _capture_writer = None


def main():
    parser = ArgumentParser(description="WonderPlay multi-view interaction")
    parser.add_argument("--config", default="examples/configs/venice_I.yaml")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = OmegaConf.load(str(config_path))

    OmegaConf.set_struct(config, False)
    if "runs_dir" not in config:
        config["runs_dir"] = config.get("work_dir", "3d_result/wonderplay")
    if "num_scenes" not in config:
        config["num_scenes"] = 1
    if "rotation_path" not in config:
        config["rotation_path"] = [0]
    if "boundary_rules" not in config:
        config["boundary_rules"] = {}
    if "use_gpt" not in config:
        config["use_gpt"] = False
    if "load_gen" not in config:
        config["load_gen"] = False
    if "multiview" not in config:
        config["multiview"] = {}
    if "enabled" not in config["multiview"]:
        config["multiview"]["enabled"] = True
    if "stop" not in config["multiview"]:
        config["multiview"]["stop"] = False
    if "expansion_iterations" not in config["multiview"]:
        config["multiview"]["expansion_iterations"] = 100
    if "environment_motion" not in config:
        config["environment_motion"] = {}
    if "sam_prompt" not in config["environment_motion"]:
        config["environment_motion"]["sam_prompt"] = "water"
    if "force_function" not in config and "force_function_name" in config:
        config["force_function"] = (
            f"simulator.genesis_functions.{config['force_function_name']}"
        )
    OmegaConf.set_struct(config, True)
    run(config, prefix=args.prefix, port=args.port)


if __name__ == "__main__":
    main()
