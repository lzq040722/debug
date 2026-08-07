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
