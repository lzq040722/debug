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
python WonderPlay_new/run_genesis.py --config examples/configs/alpine.yaml --prefix example
#stage 2
python WonderPlay_new/run_video_model.py --input_folder 3d_result/wonderplay/alpine/Gen-11-08_23-26-15/simulation --output_folder 3d_result/wonderplay/alpine/Gen-11-08_23-26-15/output_video --sdedit_strengths 0.85