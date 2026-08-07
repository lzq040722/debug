# WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions

<div align="center">

[![a](https://img.shields.io/badge/Website-WonderPlay-blue)](https://kyleleey.github.io/WonderPlay/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18151-red)](https://arxiv.org/abs/2505.18151)
[![twitter](https://img.shields.io/twitter/url?label=TL:DR&url=https%3A%2F%2Ftwitter.com%)](https://x.com/Koven_Yu/status/1938034899765903620)
</div>

<p align="center">
  <img src="assets/github_teaser.gif"
       alt="Demo"
       width="1000"
       style="max-width: 100%; height: auto;" />
</p>


## Abstract

WonderPlay is a novel framework integrating physics simulation with video generation for generating action-conditioned dynamic 3D scenes from a single image. While prior works are restricted to rigid body or simple elastic dynamics, WonderPlay features a hybrid generative simulator to synthesize a wide range of 3D dynamics. The hybrid generative simulator first uses a physics solver to simulate coarse 3D dynamics, which subsequently conditions a video generator to produce a video with finer, more realistic motion. The generated video is then used to update the simulated dynamic 3D scene, closing the loop between the physics solver and the video generator. This approach enables intuitive user control to be combined with the accurate dynamics of physics-based simulators and the expressivity of diffusion-based video generators. Experimental results demonstrate that WonderPlay enables users to interact with various scenes of diverse content, including cloth, sand, snow, liquid, smoke, elastic, and rigid bodies -- all using a single image input.

## News

Please also check our latest follow-up work [PerpetualWonder](https://johnzhan2023.github.io/PerpetualWonder/), build upon same motivation of combining physical simulation and video generation, but with more consistent 4D quality and long-horizon actions.


## Getting Started

For the installation to be done correctly, please proceed only with CUDA-compatible GPU available. It requires 80GB GPU memory to run.

**Tested Environment:**
- PyTorch: 2.7.1+cu126
- CUDA: 12.6

> **NVIDIA Blackwell GPUs (including RTX PRO 6000 Blackwell):** the cu126
> build does not contain `sm_120` kernels. Install the CUDA 12.8 build before
> installing the remaining requirements:
> ```bash
> pip install --upgrade --force-reinstall torch==2.7.1 torchvision==0.22.1 \\
>   --index-url https://download.pytorch.org/whl/cu128
> ```

### Installation

Create and activate the main environment:

```bash
conda create -n wp python=3.10
conda activate wp
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install submodules:

```bash
cd submodules/depth_diff_gaussian_rasterization_min
pip install -e . --no-build-isolation
cd ../diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ../..
```

Install Genesis (a specific version with manual modifications):

```bash
cd Genesis
pip install -e .
cd ..
```

Install simpe_knn:
```bash
git submodule update --init --recursive submodules/simple_knn
cd submodules/simple_knn
pip install -e . --no-build-isolation
cd ../..
```

Install RepViT
```bash
git submodule update --init --recursive submodules/RepViT
mkdir -p submodules/RepViT/checkpoints
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt -P submodules/RepViT/checkpoints/
cd submodules/RepViT/sam
pip install -e .
cd ../../..
```

Install Pytorch3D
```bash
pip install -U fvcore
pip install iopath
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Install Instantmesh-related
```bash
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
pip install xatlas
```

### Run examples

We currently provide one example cases: **venice**. To run on one case, the pipeline is consist of two main stages:

#### Stage 1: Scene Reconstruction + Physical Simulation

This stage we perform scene compositional reconstruction and run physical simulation:

```bash
conda activate wp
python WonderPlay_new/run_genesis.py --config examples/configs/venice.yaml --prefix example
```

#### Stage 2: Video model refinement

After completing the scene reconstruction, and simulation, we condition an existing flow-conditioned video model with simulation intermediate results:

```bash
python WonderPlay_new/run_video_model.py --input_folder 3d_result/wonderplay/venice/example/simulation --output_folder 3d_result/wonderplay/venice/example/output_video --sdedit_strengths 0.85
```

### What's in one example?

To add a new example, follow these steps:

1. **Configuration file**: under `examples/configs/` we provide the config file for running scene reconstruction and simulation.

2. **Intermediate Results**: under `examples/imgs/` we also provide the intermediate results generated in the scene reconstruction stage, and the reconstruction pipeline will directly leverage these results if exist, including inpainted image, object mask, inpainted sky image and gaussians, etc.

## Acknowledgement

Our code references and builds upon the following open-source projects:

- [HUGS](https://github.com/hyzhou404/HUGS): Optical flow rendering from gaussians
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis): Multi-physics simulation framework
- [simple_knn](https://github.com/shuyueW1991/simple_knn_illustration): For efficient cuda initialization
- [Go-With-The-Flow](https://github.com/GoWithTheFlowPaper/gowiththeflowpaper.github.io?tab=readme-ov-file): Optical-flow conditioned video generation model

We are grateful to the authors and contributors of these projects for their valuable work.
