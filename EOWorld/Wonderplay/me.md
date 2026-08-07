1. stage 2 输入；--input_folder 3d_result/wonderplay/venice/example/simulation
gt.png：裁剪/resize 成 720x480，保存为 video model 的首帧输入图。
traj_00/flows_actual/*.npy：最关键，用来生成 warped noise，也就是 noises.npy
traj_00/render_video.mp4：复制成 input.mp4，作为参考视频/输入视频保存
text_prompt.txt 或 --text_prompt：传给视频模型的 prompt

