# 2026-05-11: Paper-style ManipTrans reset on arm-mounted Kuka+Sharpa setup

## 目标
实现 ManipTrans Stage-2 reset（random init from any trajectory frame + 物体/手warm-start）
让 policy 能完整跟踪 OakInk b7853 的 24 帧 hand-object trajectory，超过 G8 baseline。

## 当前最好结果（SOTA）：G8 — 不能完整跟踪
- Setup: 仅 frame 0 cold start (`randomStateInit=False`, `useRetargetDofInit=False`), 没有 IK warm-start
- Final ep 10000: `mean_successes=2.13`, `sub_goal_idx_max=11/24`, `sub_goal_idx_mean=2.92`
- 行为：学会了**抓取+举起 ~10cm**；frame 4 以后的横向移动（物体 y 方向 30cm）没学会
- 物理上轨迹 24 帧只覆盖前 11 帧的最深 reach，平均只到第 3 帧

## Paper-style 实验全部 underperform G8
| Run | 改动累积 | mean_successes | sub_goal_idx_max | 失败原因 |
|-----|----------|----------------|------------------|----------|
| G10 | random init [0,T-1], 全 warm-start | 0.65 | 23 | spawn 时 84% 物体掉到桌下（kuka workspace 边界帧 14-23 IK error 大） |
| G11 | + reachable_frames filter (IK err<1cm 的 [0,14]) | 1.0 (plateau) | 16 | spawn 50% penetration；其余靠 "免费 spawn-success" 卡在 1.0 |
| G12 | + curriculum init [0,6] | 0.42 | 7 | 集中在 grasp phase 反而 penetration 升到 83% |
| G13 | + DOF relax 0.9 给手指 10% clearance | 1.0 | 16 | penetration 略降到 0.43 但学习 plateau 不变 |
| G14 | + sub_goal_idx = seq_idx + 1 (消除"免费 spawn success") | 0.91 | 16 | 没了免费 success 但 dense gradient 不足，policy 不学前进 |
| G15 | + tolerance 2.5cm | 0.0 | 15 | 紧 tolerance 完全 bootstrap 失败 |
| G16 | + tolerance 4cm | 0.0 | 15 | 同上 |
| G17 | + dense keypoint reward + closest-tracker reset, tol 2.5cm | 0.0 | 15 | dense reward delta-based, 起步阶段没增量信号 |
| G18 | tol 5cm + dense | 0.0 | 15 | 同上 |
| G19 | 加载 G8 checkpoint 做 fine-tune | crash | - | rl_games 的 sigma 参数 bug 阻塞 checkpoint resume |

## 累积发现的 5 个根因

### 1. Kuka workspace 边界 [已修]
trajectory frame 14-23 要求 wrist 深入 -y 方向，超过 kuka 工作空间。
IK 把 wrist 在 z 方向掉 1-4cm → 手指穿桌面。
**修复**：`trajectory.reachable_frames` mask（IK err < 1cm 才能 spawn），把 random init 限制到 [0, 14]。

### 2. Tolerance / trajectory motion 量级不匹配 [新发现]
- per-frame 物体 motion 平均 1.6cm，最大 3.4cm
- 默认 `successTolerance=0.075m × keypointScale=1.5 = 11.25cm tolerance`
- → tolerance / motion = 7 倍
- → policy 把物体放在某帧位置不动，能同时满足 3-4 个相邻帧的 success 条件
- **后果**：G8 的 `mean_successes=2.13` 大部分是这种"重叠 free success"，不是真正的 trajectory tracking
- **真实意义**：success 应该意味着"前进了 1 帧"，但当前 tolerance 让 success 意味着"在大致区域里"

### 3. Free spawn-success bug [已修]
我之前实现 paper-style reset 时把 sub_goal_idx 设到 seq_idx，等于"goal=当前spawn状态" → spawn 后等几步 dwell 立即 success → policy 不需要任何动作就拿 1000 reward。
**修复**：sub_goal_idx = seq_idx + 1，强制 policy 真正移动物体 ~1.6cm 才能 success。
但单独这个修复不够（G14 仍 plateau 1.0）。

### 4. Object slip during settle [未解决]
warm-start 把 dex hand DOFs 设到闭抓 pose，但 IK 误差让物体相对手指偏移。PhysX 接触力 → 物体被弹开。
DOF relax 0.9 给 finger clearance 反而 grasp 变松 → 物体 fall。
两难：紧 grasp = penetration，松 grasp = slip。
观测：spawn 后几步内 `d_obj_pos` 漂到 7cm，`d_obj_rot` 漂到 44°。

### 5. 紧 tolerance 无法 bootstrap [新发现]
- 默认 11.25cm tolerance + 弱 R_imit (~0.1) + sparse success bonus 1000
- tolerance 收紧到 ≤5cm 后，untrained policy 完全摸不到 success → 0 reward → 学不动
- dense keypoint reward 是 delta-based（"比 historical min 更近才有 reward"），不是绝对距离 → 起步无信号

## 与 paper 的根本差异 [汇总]

| # | 维度 | Paper | 我们 | 影响 |
|---|------|-------|------|------|
| 1 | 运动学 | floating wrist (free 6-DoF), IK error = 0 | kuka 7-DoF, IK err 0-6cm | paper warm-start 物体绝不 slip; 我们必 slip |
| 2 | Velocity init | spawn 带 trajectory velocity | spawn 全 zero | paper 物体已朝下一帧动；我们 frozen |
| 3 | Reward 结构 | 几乎纯 dense R_imit (13 components) | 弱 R_imit + sparse success bonus 1000 | paper 每 timestep 有梯度；我们大部分时候只能从 sparse bonus 学 |
| 4 | Goal 推进 | `progress_buf` 每步 +1 (time-driven) | `sub_goal_idx` 仅 success 后 +1 (success-driven) | paper policy 必须跟上 trajectory pace；我们可以躺平 |
| 5 | Episode 终止 | trajectory 走完 / R_imit 过差 | episodeLength + object_z_low + hand_far + dropped | 我们额外终止条件让 spawn slip 立即 reset |
| 6 | Observation | 可能含 future trajectory window + velocities | SimToolReal default + hand goal | paper 有 lookahead |

**最关键差异是 #3 + #4**：paper 的 reward + goal 机制本质上是 "continuous trajectory imitation" — policy 必须按时间轴跟随 trajectory，不跟就被惩罚。我们的机制本质上是 "goal-reaching" — 给个 goal，到了换下个，可以无限磨蹭。warm-start 后 spawn 已在 goal 附近，所以 policy 立刻拿 success+1000 然后没动力前进。

## 下一步计划：tracking branch

新建 `tracking` 分支。除了 reset 保留 ManipTrans-style（random init + warm-start），其他**全部按 paper formulation 改**：
- 改成 time-driven progress_buf advancement（不是 success-driven）
- 改成 dense R_imit dominant，去掉 sparse success bonus（或 weight 极小）
- spawn 时给 trajectory velocity init
- episode 主要靠 progress_buf 满或 R_imit 太差结束

目标：测试是否 paper formulation + 我们的 hardware (kuka+Sharpa) 能完整跟踪 24 帧 trajectory。
