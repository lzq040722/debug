1. stage 2 输入：
input_folder 3d_result/wonderplay/venice/example/simulation
gt.png：裁剪/resize 成 720x480，保存为 video model 的首帧输入图。
traj_00/flows_actual/*.npy：最关键，用来生成 warped noise，也就是 noises.npy
traj_00/render_video.mp4：复制成 input.mp4，作为参考视频/输入视频保存
text_prompt.txt 或 --text_prompt：传给视频模型的 prompt

2. stage 2 输入给视频模型的条件有 ：prompt + refer image + 光流/条件生成的wraped noise + render_video ，render_video 编码成 VAE letent，再和wraped noise 混合。 当时视频生成模型可以使用mask但是没有使用。

3. Go-with-the-Flow （CVPR 2025）把随机噪声换成一种根据 optical flow 扭曲过的 warped noise。这个 noise 本身带有运动结构，比如相机怎么动、物体怎么移动。

4.
src_video 选择光流可视化视频：
python Wan2.1/generate.py \
  --task vace-14B \
  --size  1280*720 \
  --ckpt_dir /root/autodl-tmp/huggingface/Wan2.1-VACE-14B \
  --prompt "A boat resting on a Venetian river, passively moved by the flowing water, gently swaying and bobbing with soft waves, water ripples spreading around the boat, natural fluid motion, stable camera, realistic lighting, high quality." \
  --src_video 3d_result/wonderplay/venice/Gen-08-08_11-18-55/vace_flow/src_video-flow.mp4 \
  --src_ref_images /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/simulation/gt.png \
  --frame_num 49 \
  --save_file /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/test_flow_vace_h0.55.mp4 \
  --init_video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/simulation/traj_00/render_video_high_quality.mp4 \
  --sdedit_strength 0.55 \

src_video 选择render运动视频：
  python Wan2.1/generate.py \
  --task vace-14B \
  --size 832*480 \
  --ckpt_dir /root/autodl-tmp/huggingface/Wan2.1-VACE-14B \
  --prompt "A boat on the river" \
  --src_video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/simulation/traj_00/render_video.mp4 \
  --src_ref_images /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/simulation/gt.png \
  --frame_num 49 \
  --save_file /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-08-08_11-18-55/test_flow_vace.mp4 \

5. Go-with-the-flow 是为了能够让视频模型更好地理解运动，让随机噪声按照光流运动。SDEidt 将参考视频进行VAE 得到的latent与wraped noise混合，保持原来的外观和结构。

6. 确认一下VACE 是如何得到光流信息的。现在可以尝试的方案：
a. 利用VACE 的官方处理流程（VACE preprocess / FlowVisAnnotator / RAFT）将render_video.mp4转换成光流条件视频；
b. 把VACE 官方得到光流信息的流程加入到现有的方法当中。（如何将二维光流信息可视化成RGB视频）

7. DDIM 主要解决如何更加高效地去噪，减少了推理步数，学习如何去噪；flow matching 主要学习在目前概率空间的位置出发，怎么移动才能有目前的noise分布到最终的data分布，学习一个速度场；

8. cd /root/autodl-tmp/EOWorld/Wonderplay/VACE

python vace/vace_preproccess.py \
  --task flow \
  --video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/traj_00/render_video.mp4 \
  --pre_save_dir /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/vace_flow

python Wan2.1/generate.py \
  --task vace-14B \
  --size  832*480 \
  --ckpt_dir /root/autodl-tmp/huggingface/Wan2.1-VACE-14B \
  --prompt "A boat on the river" \
  --src_video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/vace_flow/src_video-flow.mp4 \
  --src_ref_images 3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/gt.png \
  --frame_num 49 \
  --save_file 3d_result/wonderplay/venice/Gen-18-08_10-44-28/test_flow_vace_wan2.1.mp4 \
  --init_video 3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/traj_00/render_video.mp4 \
  --sdedit_strength 0.5 \


ffmpeg -y \
  -framerate 8 \
  -pattern_type glob \
  -i '3d_result/wonderplay/venice/Gen-18-08_10-44-28/simulation/traj_00/frames/frame_*.png' \
  -vf "scale=832:480:force_original_aspect_ratio=decrease,pad=832:480:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 \
  -preset slow \
  -crf 8 \
  -pix_fmt yuv420p \
  3d_result/wonderplay/venice_o/Gen-07-08_21-49-29/simulation/traj_00/render_video_high_quality.mp4

查看视频分辨率：
python - <<'PY'
import imageio.v3 as iio

video = "3d_result/wonderplay/venice_o/Gen-07-08_21-49-29/simulation/traj_00/render_video_high_quality.mp4"

frame = next(iio.imiter(video))
h, w = frame.shape[:2]
print(f"{w}x{h}")
PY

转换成realwonder 所需要格式：
CUDA_VISIBLE_DEVICES=0 python prepare_realwonder_input.py \
  --simulation_dir 3d_result/wonderplay/venice/Gen-25-08_19-20-57/simulation/view_000 \
  --traj_id 0 \
  --output_dir /root/autodl-tmp/RealWonder/input_data/venice_obj2env512*512/final_sim \
  --num_output_frames 12 \
  --flow_format normalized \
  --overwrite

跑realwonder 视频生成模型：
CUDA_VISIBLE_DEVICES=0 python infer_sim.py \
  --checkpoint_path 'ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt' \
  --sim_data_path input_data/venice_obj2env512*512/final_sim \
  --output_path input_data/vencie_obj2env512*512/final_sim/realwonder_output.mp4 \
  --eval_degradation 0.5 \
  --local_attn_size 21 \
  --seed 42

final_hint_start_x-3d [array([232.], dtype=float32), array([313.], dtype=float32), array([385.], dtype=float32)]
final_hint_end_x-3d [array([295.], dtype=float32), array([377.], dtype=float32), array([445.], dtype=float32)]
final_hint_start_y-3d [array([384.], dtype=float32), array([375.], dtype=float32), array([360.], dtype=float32)]
final_hint_end_y-3d [array([473.], dtype=float32), array([449.], dtype=float32), array([433.], dtype=float32)]

9. SD-Inpaint 的作用是生成一张没有被前景物体遮挡的keyframe/baselayer，基于这个思想在环境运动分支里边应该移除前景物体，然后补全整个环境，再把整个作用力施加在环境中。

10. Git 上传教程
git add.
git commit -m ' ' 
git push

git 第一次设置仓库地址：
git remote add origin 仓库地址

git 修改仓库地址：
git remote set-url origin 仓库地址

git branch : 查看当前分支

git branch -r ： 查看远程分支

git branch -vv : 查看本地分支绑定了哪个远程分支

git push origin 本地分支(main) ：远程分支 

11. stresss test:
/root/autodl-tmp/LivingWorld/scripts/gpu_stress_every_30min.sh

12. run_genesis.py (929行)
renderer_velocity = velocity_scale * mean_displacement

genesis_velocity = renderer_displacement_to_genesis(
    renderer_velocity
)

# Water drives boat horizontally only.
genesis_velocity[2] = 0.0

living_world_render.py(534)

f_pos, b_pos = pre_euler_integral(
    motion_pts.detach(),
    motion_model,
    T + 1,
    smooth,
)

# Keep the water surface at its original height.
f_pos[..., 1] = motion_pts[None, :, 1]
b_pos[..., 1] = motion_pts[None, :, 1]

501行修改 smooth 参数 2.3-->0.5

13. 8/24 交流内容：
a. 完整的多视角交互逻辑
b. 可以选用更少的3DGS进行模拟，最后交给视频优化就好

9/2 本周待完成任务：
1. 实现可交互、增量式场景扩展
2. 使用双卡运行，将部分模型（如Image-edit）并行加载到另一张卡，需要使用时直接给input，以减少加载模型所浪费的时间


[prepare_realwonder_input.py](/root/autodl-tmp/EOWorld/Wonderplay/prepare_realwonder_input.py)  我现在是使用这个脚本将目前产生的文件转换成realwonder所需要的格式，然后到对应的项目 /root/autodl-tmp/RealWonder 里边去在运行视频生成的部分，但是这样会显得很麻烦， 我需要你帮我把 RealWonderplay中有关视频生成的代码迁移过来。

14. ssh -N -o ExitOnForwardFailure=yes -L 17778:127.0.0.1:7778 -p 40031 root@10.130.129.33

15. 现在我们需要做以下修改：
1) 我希望用户能够通过在前端点击那个物体进行交互，就加在选择运动的环境（water） 那一步之前，不再使用后端object_split_mask_sam_ids进行分割。增加更多的灵活性
2) 为了进一步减少运行时间，我希望用两张卡来完成这个项目，一张卡用来加载正常的pipeline，另一张卡可以提前加载SAM、SAM3以及视频优化模型，Image_edit等大模型，这样等需要的条件产生之后我就可以直接输入到这么模型中进行处理，而不需要因为加载这些模型浪费大量的时间。
3) 当然，我希望在完成双卡运行的基础上可以保留单卡运行的机制。 
4) 请你对我以上两个方案进行理解，然后规划执行方案。