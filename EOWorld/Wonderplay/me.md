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
  --video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-14-08_20-43-06/simulation/traj_00/render_video.mp4 \
  --pre_save_dir /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/venice/Gen-14-08_20-43-06/simulation/vace_flow

python Wan2.1.backup_20260810_215736/generate.py \
  --task vace-14B \
  --size  832*480 \
  --ckpt_dir /root/autodl-tmp/huggingface/Wan2.1-VACE-14B \
  --prompt "river water flowing steadily downstream" \
  --src_video /root/autodl-tmp/EOWorld/Wonderplay/3d_result/wonderplay/alpine/Gen-11-08_23-26-15/simulation/vace_flow/src_video-flow.mp4 \
  --src_ref_images 3d_result/wonderplay/alpine/Gen-11-08_23-26-15/simulation/gt.png \
  --frame_num 49 \
  --save_file 3d_result/wonderplay/alpine/Gen-11-08_23-26-15/test_flow_vace_wan2.1.mp4 \
  --init_video3d_result/wonderplay/alpine/Gen-11-08_23-26-15/simulation/traj_00/render_video.mp4 \
  --sdedit_strength 0.5 \


ffmpeg -y \
  -framerate 8 \
  -pattern_type glob \
  -i '3d_result/wonderplay/venice/Gen-14-08_20-43-06/simulation/traj_00/frames/frame_*.png' \
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
  --simulation_dir 3d_result/wonderplay/venice/Gen-14-08_20-43-06/simulation \
  --traj_id 0 \
  --output_dir /root/autodl-tmp/RealWonder/input_data/venice/final_sim \
  --num_output_frames 12 \
  --flow_format normalized \
  --overwrite

跑realwonder 视频生成模型：
CUDA_VISIBLE_DEVICES=0 python infer_sim.py \
  --checkpoint_path 'ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt' \
  --sim_data_path input_data/venice/final_sim \
  --output_path input_data/venice/final_sim/realwonder_output2.mp4 \
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