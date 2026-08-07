import argparse
import os.path
import json
import sys

# The upstream cinemagraphy modules use absolute ``lib.*`` imports.
sys.path.insert(0, os.path.dirname(__file__))

import imageio
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode
from tqdm import tqdm
from PIL import Image

from thirdparty.cinemagraphy.lib.config import load_config
from thirdparty.cinemagraphy.lib.utils.general_utils import *
from thirdparty.cinemagraphy.lib.model.inpaint.model import SpaceTimeAnimationModel
from thirdparty.cinemagraphy.lib.utils.render_utils import *
from thirdparty.cinemagraphy.lib.model.motion.motion_model import SPADEUnetMaskMotion
from thirdparty.cinemagraphy.lib.model.motion.sync_batchnorm import convert_model
from thirdparty.cinemagraphy.lib.renderer import ImgRenderer
from thirdparty.cinemagraphy.lib.model.inpaint.inpainter import Inpainter
from thirdparty.cinemagraphy.lib.utils.data_utils import resize_img
# from third_party.DPT.run_monodepth import run_dpt

def save_tensor_img(tensor, path):
    # tensor: [H,W] 또는 [1,H,W]
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]

    # 0~1로 정규화 (시각화용)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def generate_mask_hints_from_user(args, config, frame):
    # json_file = os.path.join(args.input_dir, 'image.json')
    # mask_file = os.path.join(args.input_dir, 'image_json', 'mask.png')

    # mask
    # mask = imageio.imread(mask_file)
    mask = np.array(frame['mask'])

    # print("\n===== [BEFORE RESIZE] =====")
    # print("mask shape:", mask.shape)

    # print("raw start_x:", frame['final_hint_start_x'])
    # print("raw start_y:", frame['final_hint_start_y'])
    # print("raw end_x:", frame['final_hint_end_x'])
    # print("raw end_y:", frame['final_hint_end_y'])
    
    height, width = mask.shape[0], mask.shape[1]

    # hints
    # hint_y = []
    # hint_x = []
    # hint_motion = []

    # data = json.load(open(json_file))
    # for shape in data['shapes']:
    #     if shape['label'].startswith('hint'):
    #         start, end = np.array(shape["points"])
    #         hint_x.append(int(start[0]))
    #         hint_y.append(int(start[1]))
    #         hint_motion.append((end - start) / 50.)
    
    frame_hint_y = frame['final_hint_start_y']
    frame_hint_x = frame['final_hint_start_x']
    # print("frame_hint_x:", frame_hint_x)
    # print("frame_hint_y:", frame_hint_y)
  

    # hint_y = [int(val[0]) for val in frame_hint_y]
    # hint_x = [int(val[0]) for val in frame_hint_x]

    hint_x = []
    hint_y = []

    for x,y in zip(frame_hint_x, frame_hint_y):
        if 0 <= x[0] < 512 and 0 <= y[0] < 512:
            hint_x.append(int(x[0]))
            hint_y.append(int(y[0]))
    
    if len(hint_x) == 0:
        hint_x.append(0)
        hint_y.append(0)
    
    hint_motion = []

    for idx in range(len(hint_x)):
        end_x = frame['final_hint_end_x'][idx] - frame['final_hint_start_x'][idx]
        end_y = frame['final_hint_end_y'][idx] - frame['final_hint_start_y'][idx] 
        hint_motion.append(np.array([end_x, end_y]).flatten() / 50.)
    

    hint_y = torch.tensor(hint_y)
    hint_x = torch.tensor(hint_x)
    hint_motion = torch.tensor(np.array(hint_motion)).permute(1, 0)[None]

    # print("filtered hint_x:", hint_x)
    # print("filtered hint_y:", hint_y)
    # print("hint_motion (before blur):", hint_motion.shape)

    max_hint = hint_motion.shape[-1]
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    xs = xs.view(1, 1, width).repeat(1, height, 1)
    ys = ys.view(1, height, 1).repeat(1, 1, width)
    xys = torch.cat((xs, ys), 1).view(2, -1)

    dense_motion = torch.zeros(1, 2, height * width)
    dense_motion_norm = torch.zeros(dense_motion.shape).view(1, 2, -1)

    # print("dense_motion 512 shape:", dense_motion.shape)
    # print("dense_motion stats ch0:", 
    #     float(dense_motion[0,0].min()),
    #     float(dense_motion[0,0].max()),
    #     float(dense_motion[0,0].mean()))
    # print("dense_motion stats ch1:", 
    #     float(dense_motion[0,1].min()),
    #     float(dense_motion[0,1].max()),
    #     float(dense_motion[0,1].mean()))

    sigma = np.random.randint(height // (max_hint * 2), height // (max_hint / 2))
    hint_y = hint_y.long()
    hint_x = hint_x.long()
    for i_hint in range(max_hint):
        dist = ((xys - xys.view(2, height, width)[:, hint_y[i_hint], hint_x[i_hint]].unsqueeze(
            1)) ** 2).sum(0, True).sqrt()
        weight = (-(dist / sigma) ** 2).exp().unsqueeze(0)
        dense_motion += weight * hint_motion[:, :, i_hint].unsqueeze(2)
        dense_motion_norm += weight
    dense_motion_norm[dense_motion_norm == 0.0] = 1.0
    dense_motion = dense_motion / dense_motion_norm
    dense_motion = dense_motion.view(1, 2, height, width) * torch.tensor(mask).bool()

    hint = dense_motion
    hint_scale = [config['W'] / width, config['W'] / height]
    hint = hint * torch.FloatTensor(hint_scale).view(1, 2, 1, 1)
    hint = F.interpolate(hint, (config['W'], config['W']), mode='bilinear', align_corners=False)
    mask = F.interpolate(torch.tensor(mask[None, None]).bool().float(), (config['W'], config['W']), mode='area')

    # print("\n===== [AFTER RESIZE to 768] =====")
    # print("hint shape:", hint.shape)
    # print("mask shape:", mask.shape)

    # print("hint stats ch0:",
    #     float(hint[0,0].min()),
    #     float(hint[0,0].max()),
    #     float(hint[0,0].mean()))
    # print("hint stats ch1:",
    #     float(hint[0,1].min()),
    #     float(hint[0,1].max()),
    #     float(hint[0,1].mean()))
    
    # dbg_dir = os.path.join('/home/kbkong/mhj/3D-MOM_concat/demo/cloud8/MOM/bo_Flow_viz')
    # os.makedirs(dbg_dir, exist_ok=True)
    
    # save_tensor_img(mask[0,0], os.path.join(dbg_dir, "mask.png"))

    return mask, hint


def get_input_data(args, config, frame, ds_factor=1):
    motion_input_transform = Compose(
        [
            torchvision.transforms.Resize((config['motionH'], config['motionW']),
                                          InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    to_tensor = ToTensor()

    # try:
    #     img_file = os.path.join(args.input_dir, 'image.png')
    #     motion_rgb = Image.open(img_file)
    # except:
    #     img_file = os.path.join(args.input_dir, 'image.jpg')
    #     motion_rgb = Image.open(img_file)
    
    motion_rgb = frame['image']
    motion_rgb = motion_input_transform(motion_rgb)
    mask, hints = generate_mask_hints_from_user(args, config, frame)

    # dpt_out_dir = os.path.join(video_out_folder, 'dpt_depth')
    np_frame = np.array(frame['image'])

    # src_img = imageio.imread(img_file) / 255.
    
    src_img = np_frame / 255.
    src_img = resize_img(src_img, ds_factor)

    h, w = src_img.shape[:2]

    src_depth = frame['depth']
    if not torch.is_tensor(src_depth):
        src_depth = torch.as_tensor(src_depth)
    src_depth = src_depth.detach().float().cpu()
    if src_depth.ndim == 2:
        src_depth = src_depth[None, None]
    elif src_depth.ndim == 3:
        src_depth = src_depth[None]
    if src_depth.ndim != 4 or src_depth.shape[1] != 1:
        raise ValueError(f"Expected depth with shape [1, 1, H, W], got {tuple(src_depth.shape)}")
    if src_depth.shape[-2:] != (h, w):
        src_depth = F.interpolate(src_depth, size=(h, w), mode='bilinear', align_corners=False)
    # LivingWorld normalizes MoGe depth to a small scene-local range. The
    # upstream cinemagraphy depth layering expects metric-like values > 1e-2.
    valid_depth = src_depth[src_depth > 1e-6]
    if valid_depth.numel() == 0:
        raise ValueError('Motion estimation received an empty depth map')
    src_depth = src_depth / valid_depth.median().clamp_min(1e-6)
    src_depth = src_depth.clamp_min(1.001e-2)

    # dpt_model_path = 'ckpts/dpt_hybrid-midas-501f0c75.pt'
    # run_dpt(input_path=args.input_dir, output_path=dpt_out_dir, model_path=dpt_model_path, optimize=False)
    # disp_file = os.path.join(dpt_out_dir, 'image.png')

    # src_disp = imageio.imread(disp_file) / 65535.
    # src_disp = remove_noise_in_dpt_disparity(src_disp)

    # src_depth = 1. / np.maximum(src_disp, 1e-6)

    # src_depth = resize_img(src_depth, ds_factor)

    intrinsic = np.array([[max(h, w), 0, w // 2],
                          [0, max(h, w), h // 2],
                          [0, 0, 1]])

    pose = np.eye(4)

    # print("===== get_input_data debug =====")
    # print("orig PIL size (W,H):", frame['image'].size)

    # print("motion_rgbs resized to:", (config['motionW'], config['motionH']))
    # print("motion_rgbs tensor:", motion_rgb.shape)  # [3, motionH, motionW]

    # print("src_img size (H,W):", h, w)

    # # mask/hints 형태 확인
    # if torch.is_tensor(mask):
    #     print("mask type tensor shape:", mask.shape)
    # else:
    #     print("mask type:", type(mask))

    # if torch.is_tensor(hints):
    #     print("hints type tensor shape:", hints.shape)
    # else:
    #     print("hints type:", type(hints))

    # # 값 범위 체크 (mask가 float일 때 0~1인지)
    # if torch.is_tensor(mask):
    #     print("mask min/max/sum:", mask.min().item(), mask.max().item(), mask.sum().item())

    return {
        'motion_rgbs': motion_rgb[None, ...],
        'src_img': to_tensor(src_img).float()[None],
        'src_depth': src_depth,
        'hints': hints[0],
        'mask': mask[0],
        'intrinsic': torch.from_numpy(intrinsic).float()[None],
        'pose': torch.from_numpy(pose).float()[None],
        'scale_shift': torch.tensor([1., 0.]).float()[None],
        # 'src_rgb_file': [img_file],
    }


def eulerian_estimation(args, frame):
    device = "cuda:{}".format(args.local_rank)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES 마스킹 후 첫 GPU
    # device = str(device)

    # set up folder
    # video_out_folder = os.path.join(args.input_dir, 'output')
    # os.makedirs(video_out_folder, exist_ok=True)

    # load config
    check_file(args.config)
    config = load_config(args.config)
    # print('config loaded from {}'.format(args.config))

    data = get_input_data(args, config['data'], frame, ds_factor=1.0)
    torch.cuda.empty_cache()

    # print("===== AFTER get_input_data =====")
    # for k, v in data.items():
    #     if torch.is_tensor(v):
    #         print(k, "tensor shape:", v.shape)
    #     elif hasattr(v, "size"):
    #         print(k, "size:", v.size)
    #     else:
    #         print(k, type(v))

    """ Load model """
    model = SpaceTimeAnimationModel(args, config)
    if model.start_step == 0:
        raise Exception('no pretrained model found! please check the model path.')

    scene_flow_estimator = SPADEUnetMaskMotion(config['generator']).to(device)
    scene_flow_estimator = convert_model(scene_flow_estimator)
    scene_flow_estimator_weight = torch.load(os.path.join(args.cinema_ckpt, 'sceneflow_model.pth'),
                                             map_location=torch.device(device))
    scene_flow_estimator.load_state_dict(scene_flow_estimator_weight['netG'])
    inpainter = Inpainter(device=device, ckpt_dir=args.cinema_ckpt)
    renderer = ImgRenderer(args, config, model, scene_flow_estimator, inpainter, device)

    """ render """
    model.switch_to_eval()
    with torch.no_grad():
        renderer.process_data(data)
        _, flow, *_ = renderer.compute_flow_and_inpaint()
    return flow

    '''
        num_frames = [60, 60, 60, 90]
        video_paths = ['up-down', 'zoom-in', 'side', 'circle']
        Ts = [
            define_camera_path(num_frames[0], 0., -0.08, 0., path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[1], 0., 0., -0.24, path_type='straight-line', return_t_only=True),
            define_camera_path(num_frames[2], -0.09, 0, -0, path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[3], -0.04, -0.04, -0.09, path_type='circle', return_t_only=True),
        ]
        crop = 32
        kernel = torch.ones(5, 5, device=device)
        

        for j, T in enumerate(Ts):
            T = torch.from_numpy(T).float().to(renderer.device)
            time_steps = range(0, num_frames[j])
            start_index = torch.tensor([0]).to(device)
            end_index = torch.tensor([num_frames[j] - 1]).to(device)
            frames = []
            R_list = []
            t_list = []
            for middle_index, t_step in tqdm(enumerate(time_steps), total=len(time_steps), ncols=150,
                                             desc='generating video of {} camera trajectory'.format(video_paths[j])):
                middle_index = torch.tensor([middle_index]).to(device)
                time = ((middle_index.float() - start_index.float()).float() / (
                        end_index.float() - start_index.float() + 1.0).float()).item()

                flow_f = renderer.euler_integration(flow, middle_index.long() - start_index.long())
                flow_b = renderer.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
                flow_f = flow_f.permute(0, 2, 3, 1)
                flow_b = flow_b.permute(0, 2, 3, 1)

                _, all_pts_f, _, all_rgbas_f, _, all_feats_f, \
                    all_masks_f, all_optical_flow_f = \
                    renderer.compute_scene_flow_for_motion(coord, torch.inverse(renderer.pose), renderer.src_img,
                                                           rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                           mask_layers_src, flow_f, kernel, with_inpainted=True)
                _, all_pts_b, _, all_rgbas_b, _, all_feats_b, \
                    all_masks_b, all_optical_flow_b = \
                    renderer.compute_scene_flow_for_motion(coord, torch.inverse(renderer.pose), renderer.src_img,
                                                           rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                           mask_layers_src, flow_b, kernel, with_inpainted=True)

                all_pts_flowed = torch.cat(all_pts_f + all_pts_b)
                all_rgbas_flowed = torch.cat(all_rgbas_f + all_rgbas_b)
                all_feats_flowed = torch.cat(all_feats_f + all_feats_b)
                all_masks = torch.cat(all_masks_f + all_masks_b)
                all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)
                num_pts_2 = sum([len(x) for x in all_pts_b])
                all_side_ids[-num_pts_2:] = 1

                pred_img, _, meta, R_out, t_out = renderer.render_pcd(all_pts_flowed,
                                                        all_rgbas_flowed,
                                                        all_feats_flowed,
                                                        all_masks, all_side_ids,
                                                        t=T[middle_index.item()],
                                                        time=time,
                                                        t_step=t_step,
                                                        path_type=video_paths[j])

                frame = (255. * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                # frame = frame[crop:-crop, crop:-crop]
                frames.append(frame)
                R_list.append(R_out)
                t_list.append(t_out)

            video_out_file = os.path.join(video_out_folder,
                                          f'{video_paths[j]}_flow_scale={args.flow_scale}.mp4')

            imageio.mimwrite(video_out_file, frames, fps=25, quality=8)
            R_list_path = os.path.join(video_out_folder,f'{video_paths[j]}_R_list')
            t_list_path = os.path.join(video_out_folder,f'{video_paths[j]}_t_list')

            # """ Extrinsic parameter save! """
            # torch.save(R_list,R_list_path)
            # torch.save(t_list,t_list_path)
        print(f'space-time videos have been saved in {video_out_folder}.')
        '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ########## general ##########
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('--input_dir', type=str, help='input folder that contains src images', required=True)
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')

    parser.add_argument('--save_frames', action='store_true', help='if save frames')
    parser.add_argument('--correct_inpaint_depth', action='store_true',
                        help='use this option to correct the depth of inpainting area to prevent occlusion')
    parser.add_argument("--flow_scale", type=float, default=1.0,
                        help='flow scale that used to control the speed of fluid')
    parser.add_argument("--ds_factor", type=float, default=1.0,
                        help='downsample factor for the input images')

    ########## checkpoints ##########
    parser.add_argument("--ckpt_path", type=str, default='',
                        help='specific weights file to reload')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')
    args = parser.parse_args()

    render(args)
