#Ryan Burgert 2024

#Setup:
#    Run this in a Jupyter Notebook on a computer with at least one GPU
#        `sudo apt install ffmpeg git`
#        `pip install rp`
#    The first time you run this it might be a bit slow (it will download necessary models)
#    The `rp` package will take care of installing the rest of the python packages for you

import rp
import shutil
import os
import sys
from pathlib import Path
import copy

_HF_HOME = "/root/autodl-tmp/huggingface"
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# Ensure repository root (containing `video_models`) is on sys.path so that
# running this script from the repo root works (e.g. `python WonderPlay_new/run_video_model.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

rp.r._pip_import_autoyes=True #Automatically install missing packages

rp.pip_import('fire')
rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource

import video_models.noise_warp as nw
from video_models.cut_and_drag_inference_with_masked_sdedit import get_pipe, load_sample_cartridge, run_pipe
import fire
import argparse
import numpy as np

import cv2
import numpy as np

def resize_flow(flow, new_size=(720, 720)):
    """
    Resize a flow field of shape (2, H, W) to (2, new_H, new_W).

    Parameters:
        flow (numpy.ndarray): Flow data of shape (2, H, W).
        new_size (tuple): Desired output size (new_H, new_W).

    Returns:
        numpy.ndarray: Resized flow of shape (2, new_H, new_W).
    """
    resized_flow = np.zeros((2, new_size[0], new_size[1]), dtype=flow.dtype)

    for i in range(2):  # Resize each flow channel separately
        resized_flow[i] = cv2.resize(flow[i], new_size, interpolation=cv2.INTER_LINEAR)

    return resized_flow

def main_warp_noise(args):
    """
    Takes a video URL or filepath and an output folder path
    It then resizes that video to height=480, width=720, 49 frames (CogVidX's dimensions)
    Then it calculates warped noise at latent resolution (i.e. 1/8 of the width and height) with 16 channels
    It saves that warped noise, optical flows, and related preview videos and images to the output folder
    The main file you need is <output_folder>/noises.npy which is the gaussian noises in (H,W,C) form
    """

    # should be output_video/simulation_name/traj_name
    output_folder_base = args.output_folder
    # should be simulation_name/simulation/
    input_folder = args.input_folder
    
    # Extract simulation_name and simulation_id from input folder path
    path_parts = input_folder.rstrip('/').split('/')
    if len(path_parts) < 4:
        raise ValueError(f"Input folder path {input_folder} does not have enough components to extract simulation name and ID")
    
    simulation_name = path_parts[-3]  # Third last component
    simulation_id = path_parts[-2]    # Second last component
    
    print(f"Extracted simulation name: {simulation_name}")
    print(f"Extracted simulation ID: {simulation_id}")

    traj_str = f"traj_{args.traj_id:02d}"
    traj_dir = os.path.join(input_folder, traj_str)
    if not os.path.exists(traj_dir):
        raise ValueError(f"The given input_folder={repr(input_folder)} does not contain a trajectory with id={args.traj_id}")
    
    output_folder = os.path.join(output_folder_base, simulation_name, simulation_id, traj_str)
    if os.path.exists(output_folder):
        if os.path.exists(os.path.join(output_folder, "noises.npy")):
            video_src = os.path.join(traj_dir, "render_video.mp4")
            video_dst = os.path.join(output_folder, "input.mp4")
            if not os.path.exists(video_dst) and os.path.exists(video_src):
                shutil.copy2(video_src, video_dst)
            print("The output folder already exists and contains noises.npy, will directly use it and continue")
            return
        else:
            print("The output folder already exists, will delete it and continue")
            shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    crop_start = args.crop_start
    first_frame = os.path.join(input_folder, 'gt.png')
    if first_frame is not None:
        first_frame = rp.load_image(first_frame)
        first_frame = rp.resize_image_to_hold(first_frame, height=480, width=720)
        first_frame = first_frame[crop_start:crop_start + 480, :, :]
        first_frame = rp.as_numpy_array(first_frame)
        rp.save_image(first_frame, rp.path_join(output_folder, f'{simulation_name}.png'))
        print("Resized and cropped first frame saved to", rp.path_join(output_folder, f'{simulation_name}.png'))

    FLOW = 2 ** 3
    LATENT = 8

    input_flow = False
    video = os.path.join(traj_dir, 'flows_actual')
    if rp.is_a_folder(video):
        FRAME = 1
        FLOW = 2 ** 2
        frame_flows = sorted([os.path.join(video, f) for f in os.listdir(video) if f.endswith('.npy')])
        frame_flows = [np.load(flow_file) for flow_file in frame_flows]
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i] * 2 * 512 - 512
        
        print("Number of frames for optical flow:", len(frame_flows))
        print("Shape of each frame:", frame_flows[0].shape)

        frame_flows = frame_flows[1:49] # rp.resize_list(frame_flows, length=48)
        for i in range(len(frame_flows)):
            frame_flows[i] = resize_flow(frame_flows[i])
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i] * (720 / 512)
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i][:, crop_start:crop_start + 480, :]

        frame_flows = rp.as_numpy_array(frame_flows)
        print("Shape of optical flow:", frame_flows.shape)
        video = frame_flows
        input_flow = True
        
    elif rp.is_video_file(video):
        FRAME = 2**-1
        video=rp.load_video(video)
        #Preprocess the video
        video=rp.resize_list(video,length=49) #Stretch or squash video to 49 frames (CogVideoX's length)
        video=rp.resize_images_to_hold(video,height=480,width=720)
        for i in range(len(video)):
            video[i] = video[i][crop_start:crop_start + 480, :, :]
        video=rp.as_numpy_array(video)
    else:
        raise ValueError(f"The given video={repr(video)} is not a optical flow folder or a video file")

    #See this function's docstring for more information!
    output = nw.get_noise_from_video(
        video,
        remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
        input_flow=input_flow,
        noise_channels=16,
        output_folder=output_folder,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor= round(FRAME * FLOW) * LATENT,
        device="cuda"
    )

    print("Noise shape:"  ,output.numpy_noises.shape)
    print("Output folder:",output.output_folder)

    # Create symbolic links to video and masks in output folder
    
    # Link to render_video.mp4
    video_src = os.path.join(traj_dir, "render_video.mp4")
    video_dst = os.path.join(output_folder, "input.mp4")
    shutil.copy2(video_src, video_dst)

    # # Link to masks folder
    # masks_src = os.path.join(traj_dir, "masks")
    # masks_dst = os.path.join(output_folder, "masks")
    # os.symlink(masks_src, masks_dst)

    return


def main_cut_and_drag(args):
    output_folder_base = args.output_folder
    input_folder = args.input_folder
    path_parts = input_folder.rstrip('/').split('/')
    
    simulation_name = path_parts[-3]  # Third last component
    simulation_id = path_parts[-2]    # Second last component

    traj_str = f"traj_{args.traj_id:02d}"
    output_folder = os.path.join(output_folder_base, simulation_name, simulation_id, traj_str)

    crop_start = args.crop_start
    num_inference_steps = args.num_inference_steps
    degradation = args.degradation

    # Load text prompt from file
    if args.text_prompt is not None:
        prompt = args.text_prompt
    else:
        prompt_path = os.path.join(input_folder, "text_prompt.txt")
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
    f.close()
    print("Loaded text prompt:", prompt)

    model_name = "I2V5B_final_i38800_nearest_lora_weights"
    low_vram = True
    device = "cuda"
    pipe = get_pipe(model_name=model_name, device=device, low_vram=low_vram)

    sdedit_strengths = args.sdedit_strengths
    mask_strength_gaps = [-1]  # no mask use
    print("Running with sdedit_strength:", sdedit_strengths)
    print("Running with mask_strength_gap:", mask_strength_gaps)
    
    input_image = os.path.join(output_folder, f"{simulation_name}.png")
    
    for sdedit_strength in sdedit_strengths:
        for mask_strength_gap in mask_strength_gaps:
            
            if mask_strength_gap == -1:
                mask_sdedit_strength = -1
                exp_output_folder = os.path.join(output_folder, f"sdedit_{sdedit_strength:.03f}", f"without_mask")
            else:
                mask_sdedit_strength = sdedit_strength - mask_strength_gap
                exp_output_folder = os.path.join(output_folder, f"sdedit_{sdedit_strength:.03f}", f"mask_{mask_sdedit_strength:.03f}")
            print(f"Running with sdedit_strength={sdedit_strength} and mask_sdedit_strength={mask_sdedit_strength}")
            os.makedirs(exp_output_folder, exist_ok=True)
            output_mp4_path = os.path.join(exp_output_folder, f"output.mp4")

            if os.path.exists(output_mp4_path):
                continue

            sample_path = output_folder
            noise_downtemp_interp = "nearest"
            image = os.path.join(output_folder, f"{simulation_name}.png")
            guidance_scale = 6
            mask_strength = mask_sdedit_strength

            cartridge_kwargs = rp.broadcast_kwargs(
                rp.gather_vars(
                    "sample_path",
                    "degradation",
                    "noise_downtemp_interp",
                    "image",
                    "prompt",
                    "num_inference_steps",
                    "guidance_scale",
                    "sdedit_strength",
                    "mask_strength",
                    "crop_start",
                    # "v2v_strength",
                )
            )

            rp.fansi_print("cartridge_kwargs:", "cyan", "bold")
            print(
                rp.indentify(
                    rp.with_line_numbers(
                        rp.fansi_pygments(
                            rp.autoformat_json(cartridge_kwargs),
                            "json",
                        ),
                        align=True,
                    )
                ),
            )
            cartridges = rp.load_files(lambda x:load_sample_cartridge(**x, model_name=model_name), cartridge_kwargs, show_progress='eta:Loading Cartridges')
            for cartridge in cartridges:
                pipe_out = run_pipe(
                    pipe=pipe,
                    cartridge=cartridge,
                    output_mp4_path=output_mp4_path,
                )
                # print("dry run one time")

def main():
    parser = argparse.ArgumentParser(description='Process video and generate warped noise.')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to input folder, for the entire simulation output')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Path to output folder')
    parser.add_argument('--traj_id', type=int, nargs='+', default=[0],
                      help='Trajectory id(s). Can be a single int or list of ints (optional)')
    parser.add_argument('--crop_start', type=int, default=120,
                      help='Where to start cropping (default: 120)')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                      help='Number of inference steps (default: 25)')
    parser.add_argument('--degradation', type=float, default=0.4,
                      help='Degradation level (default: 0.4)')
    parser.add_argument('--sdedit_strengths', type=float, nargs='+', default=[0.75, 0.8, 0.85, 0.9],
                      help='SDEdit strength value(s). Can be a single float or list of floats.')
    parser.add_argument('--mask_strength_gaps', type=float, nargs='+', default=[-1],
                      help='Mask strength value(s). Can be a single float or list of floats.')
    parser.add_argument('--text_prompt', type=str, default=None,
                      help='Override text prompt for the video generation')
    args = parser.parse_args()

    if isinstance(args.traj_id, int) or (isinstance(args.traj_id, list) and len(args.traj_id) == 1):
        # Convert traj_id to int if it's a single-element list
        args_new = copy.deepcopy(args)
        args_new.traj_id = args.traj_id[0] if isinstance(args.traj_id, list) else args.traj_id
        main_warp_noise(args_new)
        main_cut_and_drag(args_new)
    
    else:
        for traj_id in args.traj_id:
            args_new = copy.deepcopy(args)
            args_new.traj_id = traj_id
            print(f"Running for traj_id={traj_id}")
            main_warp_noise(args_new)
            main_cut_and_drag(args_new)

if __name__ == "__main__":
    main()
