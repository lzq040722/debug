export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_HUB_OFFLINE=1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/myenv/wp   

#stage 1
python WonderPlay_new/run_genesis.py --config examples/configs/venice_I.yaml --prefix obj2env
#stage 2
python WonderPlay_new/run_video_model.py --input_folder 3d_result/wonderplay/venice/smooth_0.5/simulation --output_folder 3d_result/wonderplay/venice/smooth_0.5/output_video --sdedit_strengths 0.85