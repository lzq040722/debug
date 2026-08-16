# Env–Obj Interaction V1 方案

## 1. 目标

在当前 `eoworld` 代码的基础上，以**最小改动**实现 Environment 与 Object 之间的运动传播。

当前已有两条独立运动分支：

- `motion_type = environment`
  - Environment 通过 CinemaGraphy → 3D Scene Flow → HashGrid 产生运动；
  - Object 保持静止。
- `motion_type = object`
  - Object 通过 Genesis 产生物理运动；
  - Environment 保持静止。

V1 增加 `interaction` 模式，但仍然将交互拆成两个相互独立的单向过程：

```text
Env → Obj:
Environment 是运动源
Object 初始静止
Environment Motion → Object Motion

Obj → Env:
Object 是运动源
Environment 初始静止
Object Motion → Environment Motion
```

V1 不考虑：

- Env 与 Obj 同时双向耦合；
- Force / Acceleration 建模；
- Torque / Rotation；
- 作用力与反作用力；
- 空间衰减；
- 时间衰减；
- Wake / Ripple propagation；
- Env Gaussian 的额外状态更新机制；
- 大规模重构当前 Reconstruction Pipeline。

核心目标仅为：

\[
\text{Source Motion}
\rightarrow
\text{Motion Transfer}
\rightarrow
\text{Receiver Motion}
\]

---

## 2. 场景重建

场景重建部分保持当前 WonderPlay 代码不变，继续得到：

```text
Input Image
    ↓
Existing WonderPlay Reconstruction
    ↓
Sky + Environment/Base + Object
```

当前代码中已经能够获得：

- Sky Gaussians；
- Environment / Base Gaussians；
- Object Gaussians；
- Object Mesh。

因此 V1 不额外设计新的 `scene_state` 数据结构，仅复用已有数据。

---

# 3. Env → Obj

以 Venice 场景为例：

```text
Water:
已有环境运动

Boat:
初始速度 = 0

Water Motion
    ↓
Boat Motion
```

## 3.1 Environment Motion

Environment Motion 完全复用当前 `environment` 分支：

```text
Manual Motion Hints
        ↓
CinemaGraphy
        ↓
Dense 2D Flow
        ↓
Depth Unprojection
        ↓
3D Scene Flow
        ↓
绑定 Environment Point / Gaussian
        ↓
HashGrid
```

最终得到连续的 Environment Motion Field：

\[
v_{\mathrm{env}}(x)=f_\theta(x)
\]

HashGrid 负责根据任意有效的 Environment 3D 位置查询对应运动速度。

## 3.2 Env–Obj Interaction Region

V1 不再额外设计复杂的 3D contact detection。

交互区域直接利用 **Object 与 Inpaint 后 Environment 在原始图像坐标系中的重合关系**确定。

基本依据：

1. 原始图像中存在 Object；
2. Inpainting 将 Object 移除，并补全其后方的 Environment；
3. Object 与 Inpaint 后补全出的 Environment 都对应同一张原始图像坐标系；
4. 因此在 Object 与 Environment 实际接触/遮挡的位置，两者相对于整张图像的像素坐标 `(u,v)` 会发生重合。

因此可以利用：

```text
Object projected pixels
          ∩
Inpainted Environment pixels
          ↓
Candidate Interaction Region
```

记 Object 的图像区域为：

\[
M_{\mathrm{obj}}
\]

Inpaint 后对应的 Environment 区域为：

\[
M_{\mathrm{env}}
\]

则交互候选区域：

\[
M_{\mathrm{int}}
=
M_{\mathrm{obj}}
\cap
M_{\mathrm{env}}
\]

这里的核心不是比较两个独立图像，而是利用它们都位于**同一个原始图像坐标系**这一事实，通过相同的 `(u,v)` 建立 Object 与 Inpaint Environment 的对应关系。

然后根据这些 `(u,v)` 找到对应的 Environment Point / Gaussian，作为 HashGrid 的查询位置。

## 3.3 Environment Velocity Query

对于 Interaction Region 中对应的 Environment 3D 点：

\[
p_1,p_2,\ldots,p_N
\]

利用已有 HashGrid：

\[
v_i=f_\theta(p_i)
\]

获得每个交互位置的 Environment velocity。

第一版直接求平均：

\[
\bar v_{\mathrm{env}}
=
\frac{1}{N}
\sum_{i=1}^{N}v_i
\]

不引入额外的空间权重、距离衰减或复杂聚合策略。

## 3.4 Velocity Transfer

V1 不将 Environment velocity 转换成 Force。

直接进行：

```text
Average Environment Velocity
            ↓
      Velocity Transfer
            ↓
       Object Velocity
```

即：

\[
v_{\mathrm{obj}}
=
\alpha \bar v_{\mathrm{env}}
\]

其中 `α` 仅作为运动尺度控制参数，用于匹配 HashGrid Motion 与 Genesis Object Motion 的尺度。

不考虑：

- Force；
- Mass；
- Acceleration；
- Drag coefficient；
- Torque；
- Rotation。

Object 初始速度：

\[
v_{\mathrm{obj}}^0=0
\]

之后 Object 的平移运动由 Environment 传递得到的 velocity 驱动。

## 3.5 Env → Obj Pipeline

```text
Existing Environment Motion
        ↓
CinemaGraphy
        ↓
3D Scene Flow
        ↓
HashGrid
        ↓
Obj / Inpaint-Env (u,v) Overlap
        ↓
Interaction Region
        ↓
Query HashGrid Velocity
        ↓
Average Velocity
        ↓
Velocity Transfer
        ↓
Genesis Object Motion
```

---

# 4. Obj → Env

Obj → Env 与 Env → Obj 是一个独立的单向传播过程。

假设：

```text
Object:
已有 Genesis Motion

Environment:
初始静止

Object Motion
    ↓
Environment Motion
```

V1 不通过 Force，也不直接给 Environment Gaussian 赋统一速度。

核心思路是：

> **Object 提供 Motion Hints，然后完全复用现有 CinemaGraphy → Scene Flow → HashGrid 路线。**

## 4.1 Object Motion

继续使用当前 `object` 分支中的 Genesis：

```text
Object
   ↓
Genesis
   ↓
Object Trajectory
```

通过 Object 在相邻时刻的位置变化获得其运动方向和运动幅度。

## 4.2 Object-generated Motion Hints

当前 Environment 分支中的 CinemaGraphy 使用人工指定的：

```yaml
fixed_hints:
  [start_x, start_y, end_x, end_y]
```

在 Obj → Env 中，仅修改 Motion Hint 的来源：

```text
Original:
Manual Motion Hints

Obj → Env:
Object Motion
    ↓
Automatically Generated Motion Hints
```

即根据 Object 的运动，在图像平面中生成：

```text
(start_x, start_y)
        →
(end_x, end_y)
```

Motion Hint 的方向和位移来源于 Object Motion。

V1 不额外增加复杂的传播模型。

## 4.3 CinemaGraphy Motion Propagation

得到 Object-generated Motion Hints 后，后续直接复用当前 Environment Motion Pipeline：

```text
Object Motion
        ↓
Object-generated Motion Hints
        ↓
CinemaGraphy
        ↓
Dense 2D Environment Flow
        ↓
Depth Unprojection
        ↓
3D Scene Flow
        ↓
HashGrid
        ↓
Environment Motion
```

CinemaGraphy 输出的是 spatially-varying dense flow，而不是给整个 Environment 一个全局统一速度。

因此 Environment 中不同位置可以具有不同运动：

\[
v_{\mathrm{env}}(x_1)
\neq
v_{\mathrm{env}}(x_2)
\]

第一版先直接观察 CinemaGraphy 根据 Object Motion Hints 产生的传播效果。

暂时不额外加入：

- spatial decay；
- temporal decay；
- wake propagation；
- residual field。

如果 V1 效果能够证明 Object Motion Hint 可以合理驱动周围 Environment，再进一步讨论衰减机制。

## 4.4 Obj → Env Pipeline

```text
Genesis Object Motion
        ↓
Object Displacement
        ↓
Generate Motion Hints
        ↓
CinemaGraphy
        ↓
Dense 2D Flow
        ↓
Depth Unprojection
        ↓
3D Scene Flow
        ↓
HashGrid
        ↓
Environment Motion
```

相比当前 Environment Branch，主要变化只有：

```text
Manual Motion Hints
        ↓
Object-generated Motion Hints
```

其余 CinemaGraphy、3D Scene Flow、HashGrid 尽量直接复用。

---

# 5. Unified Renderer

Interaction Mode 需要同时渲染：

```text
Dynamic Object
+
Dynamic Environment
+
Static Sky
```

因此新增：

```python
render_interaction(...)
```

但不修改现有：

```python
render_MLP(...)
render_w_shift_flow(...)
```

以保证原有 `environment` 和 `object` baseline 不受影响。

Unified Renderer 的核心输入为：

```text
obj_xyz_t
env_xyz_t
sky_xyz
```

然后：

```text
Obj Gaussians at t
       +
Env Gaussians at t
       +
Static Sky Gaussians
       ↓
Concatenate
       ↓
Gaussian Rasterizer
       ↓
RGB / Depth / Mask
```

V1 中 Renderer 只负责根据当前动态 Gaussian state 生成视频帧，**不再负责 Optical Flow 的计算**。

---

# 6. Optical Flow：统一采用 VACE `--task flow`

所有 Interaction 输出视频的 Optical Flow **不再使用 projection-based Gaussian flow，也不再使用原 Environment Branch 中的 Farneback**。

统一复用：

```text
VACE/vace/vace_preproccess.py
    --task flow
```

对应的现有 VACE 流程。

## 6.1 VACE Flow Pipeline

`--task flow` 在 VACE 中对应：

```text
video_flow_anno
    ↓
FlowVisAnnotator
    ↓
RAFT
```

预训练模型：

```text
models/VACE-Annotators/flow/raft-things.pth
```

输入为 Unified Renderer 生成的 RGB 视频帧。

对于相邻两帧：

```text
Frame t-1
Frame t
    ↓
RAFT
    ↓
Optical Flow
```

RAFT 推理沿用 VACE 原有实现：

```python
flow_low, flow_up = model(
    image1,
    image2,
    iters=20,
    test_mode=True
)
```

得到 `flow_up`，随后继续沿用 VACE 的：

```python
flow_viz.flow_to_image(flow_up)
```

生成 Optical Flow visualization。

## 6.2 V1 中的统一规则

以后无论：

```text
Environment Motion
Object Motion
Env → Obj Interaction
Obj → Env Interaction
```

只要需要生成最终 Optical Flow，都统一执行：

```text
Rendered RGB Video
        ↓
VACE Flow Preprocess
        ↓
RAFT Optical Flow
        ↓
Flow Output
```

因此 Unified Renderer 不再额外实现：

\[
\pi(x_t)-\pi(x_{t-1})
\]

这套 projection-based Optical Flow。

也不再使用 OpenCV Farneback。

原则上直接将：

```text
VACE/vace/vace_preproccess.py
VACE/vace/annotators/flow.py
```

中与 `--task flow` 对应的原有代码逻辑复用到当前 pipeline。

如果后续阶段既需要 raw flow 数值又需要可视化视频，则仍然使用同一 RAFT forward 结果：

```text
flow_up      → raw Optical Flow
flow_up_vis  → Flow Visualization
```

这里只是输出格式不同，Optical Flow 的估计算法保持完全一致。

---

# 7. 最终 V1 Overall Pipeline

```text
                    Existing Reconstruction
                            ↓
                    Sky + Env + Obj
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
      Env → Obj                           Obj → Env
          │                                   │
Existing Environment Motion             Genesis Obj Motion
          │                                   │
      CinemaGraphy                         Obj Motion
          │                                   │
     3D Scene Flow                  Generate Motion Hints
          │                                   │
       HashGrid                         CinemaGraphy
          │                                   │
Obj / Inpaint-Env UV Overlap             2D Dense Flow
          │                                   │
 Interaction Region                     3D Scene Flow
          │                                   │
 Query HashGrid                            HashGrid
          │                                   │
 Average Velocity                         Env Motion
          │                                   │
 Velocity Transfer                            │
          │                                   │
      Obj Motion                               │
          └─────────────────┬─────────────────┘
                            │
                            ▼
                    Unified Renderer
                            │
                  RGB / Depth / Mask
                            │
                            ▼
                VACE `--task flow`
                            │
                            ▼
                         RAFT
                            │
                            ▼
                     Optical Flow
```

---

# 8. V1 最小代码改动

## 8.1 `run_genesis.py`

增加 Interaction Mode 的控制逻辑，使现有：

```text
Environment Motion Module
Object Genesis Module
```

可以在 Interaction Pipeline 中复用。

不重构现有 Reconstruction。

## 8.2 Env → Obj

增加：

```text
Obj 与 Inpaint Environment 的 (u,v) 对应
        ↓
Interaction Region
        ↓
HashGrid Velocity Query
        ↓
Average Velocity
        ↓
Object Velocity Transfer
```

不增加 Force / Rotation 等模块。

## 8.3 Obj → Env

当前：

```text
fixed_hints
    ↓
CinemaGraphy
```

修改为：

```text
Object Motion
    ↓
Generate Motion Hints
    ↓
CinemaGraphy
```

后续 2D Flow → 3D Scene Flow → HashGrid 尽量保持原代码。

## 8.4 Unified Renderer

新增：

```python
render_interaction(...)
```

负责：

```text
Dynamic Obj
+
Dynamic Env
+
Static Sky
→ RGB / Depth / Mask
```

不在 Renderer 内计算 Optical Flow。

## 8.5 Optical Flow

删除 Interaction Mode 中额外设计的：

```text
projection-based optical flow
Farneback optical flow
```

统一复用：

```text
VACE/vace/vace_preproccess.py --task flow
```

对应的 RAFT Flow Pipeline。

---

# 9. V1 暂不处理的问题

以下问题全部放到 V1 之后：

- Obj → Env 的空间衰减；
- 接触区域外 Environment 的运动衰减策略；
- temporal decay；
- wake / ripple；
- simultaneous Env ↔ Obj coupling；
- force-based physics coupling；
- torque / rotation；
- Environment Gaussian 的额外动态状态管理；
- Reconstruction Pipeline 重构。

V1 只验证三个核心问题：

1. **Env Motion 能否通过 HashGrid velocity 合理传递给静止 Obj；**
2. **Obj Motion 能否转换成 CinemaGraphy Motion Hints，并进一步产生合理 Env Motion；**
3. **两类动态 Gaussian 能否通过 Unified Renderer 共同渲染，并统一通过 VACE/RAFT 得到 Optical Flow。**
