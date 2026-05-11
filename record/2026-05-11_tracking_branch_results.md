# 2026-05-11: tracking branch (paper-style training) 训练结果汇总

## 目标
在 `tracking` 分支上实现"除了 kuka 机械臂以外，其他全部按 ManipTrans paper 来"的训练 setup，验证能否完整跟踪 OakInk b7853 trajectory。

## 实验序列

### G20: 初版 paper formulation
- **Setup**: 60Hz 物理 / 3Hz trajectory（K=20 hold）, paper-only reward (12 项), paper-only reset, time-driven goal
- **Result**: `mean_successes=0`, `d_obj_pos=17cm`, plateau within 50 epochs
- **诊断**: trajectory rate (3Hz) 跟物理 rate (60Hz) 错配 20× → goal 飞太快 policy 跟不上

### G21: K=20 hold + boost imit weight 50×
- **Setup**: G20 + `imitRewardScale.enable=50`, failed_execute disabled
- **Result**: `mean_successes=0`, `d_obj_pos=23cm`, plateau
- **诊断**: 即使 reward 放大 50 倍，绝对距离 23cm 时 `exp(-80×0.23) ≈ 10⁻⁸` 已经饱和到 0

### G22: G21 + wide bandwidth `r_obj_pos_wide`
- **Setup**: G21 + 加 `5.0 * exp(-5 × d_obj_pos)` 让远距离也有梯度
- **Result**: `mean_successes=0`, `d_obj_pos=23cm`, plateau (5.0 wide reward 也没驱动 policy)
- **诊断**: 单单加 wide reward 不够；可能 SimToolReal sparse 1000 success bonus 残留干扰

### G23: G22 + Codex round-1 audit fixes
应用 Codex 6 个 review 项的前 4 个：
- IK_delta consistency: reward 中 target shifted by ik_delta
- 每 env 的 phase counter（不是 global frame count）
- `paperOnlyReward=True` 完全跳过 SimToolReal sparse bonus
- `reward_shaper.scale_value=1.0`（不是 0.01）

- **Setup**: 上述 + tolerance 5cm + dense reward
- **Result**: **第一次有进展** — `d_obj_pos=7cm`（vs G22 的 23cm）, `r_imit=4.0`（vs G22 的 0.45）
- **诊断**: IK consistency 修复 + bypass super 是核心收益。但 envs 平均只活 8-15 步（failed_execute 杀的太早），`sub_goal_idx_mean=8` 没在 chain

### G24: G23 + 关掉 failed_execute (finger threshold = 9.99m)
- **Setup**: G23 + envs 不会因为手指偏离 goal 被早期 terminate
- **Result**: `sub_goal_idx_mean=11.5`, `sub_goal_idx_max=20`（约 trajectory 87%）, `d_obj_pos=9.5cm`
- **诊断**: envs 现在能活满 episode，sub_goal 自然爬升到 20。但物体跟踪精度下降（9.5cm vs G23 的 7cm），`mean_successes` 仍接近 0

### G25: 完整 Tier 0（60Hz 真数据 + velocities）
按 Codex round-2 audit consensus 完整实施：
- **数据**: prep 60Hz `oakink_b7853_60hz_kuka_ik_with_vel.json` (np.gradient + gaussian σ=2 paper recipe)
- **Velocity init at reset**: object/dex_dof/kuka_dof velocities 从 trajectory 注入
- **5 paper velocity reward terms**: r_eef_vel/r_eef_ang_vel/r_joints_vel/r_obj_vel/r_obj_ang_vel
- **Paper target obs (K=1)**: 310-dim future window (delta_wrist/joints/obj × pose+vel+ang_vel + delta_quat + obj_to_joints)
- **subGoalAdvanceInterval=1** (paper-exact，goal 每物理步前进，跟 60Hz 数据 rate 一致)
- **删除非 paper hacks**: r_obj_pos_wide, r_eef_pos_wide

- **Setup**: 上述完整 paper-style，failed_execute enabled (10cm threshold), episodeLength=80
- **Result**: `mean_successes=0`, `d_obj_pos=9.6cm`, `r_imit=0.46`, **从 epoch 21 到 epoch 100 完全没动**
- **诊断**: failed_execute 还是杀得太早 → policy 没有时间学

### G26: G25 + failed_execute=False, episodeLength=100
- **Setup**: G25 + envs 跑满 100 步不被早 terminate
- **Result @ ep 504**: `mean_successes=0`, `d_obj_pos=9.5cm`, `d_wrist_pos=64cm`, `r_imit=0.47`, `sub_goal_idx_mean=21.7`, `sub_goal_idx_max=36`
- **关键 metric 从 ep 21 到 ep 504 几乎完全不变** — policy 完全没在学

## 综合诊断

### Tracking 分支整体没 work，最大的 plateau metric:
- `d_obj_pos = 9.5cm`（目标 <5cm）
- `d_wrist_pos = 64cm`（机械臂 wrist 离 trajectory 期望位置 64cm，policy 没学怎么开 kuka arm）
- `r_imit = 0.47`（满分 ~13，只拿到 3.5%）

### Reward 信号 vanishing
- `5.0 × exp(-80 × 0.095) = 0.003` — 物体位置 reward 接近 0
- `0.1 × exp(-40 × 0.64) = 7e-12` — wrist 位置 reward 完全是 0
- 总 reward = `r_imit ≈ 0.47` 几乎是常数 → PPO gradient 等同噪声

### 怀疑的根因（按可能性排序）
1. **kuka 机械臂瓶颈**：64cm wrist 偏差暗示 RL 完全没学怎么协调 7-DoF arm joints。Paper 用浮动 wrist 直接 set，无 IK，无 RL 学习负担。**用户提议去掉 kuka 做 ablation**。
2. **Reward shape 不适合远距离**：`exp(-80×d)` 在 d>5cm 立刻饱和到 0，policy 一旦超出近距离就拿不到 gradient
3. **代码 bug 可能性**：64cm wrist 偏差不应该出现在 warm-start 后的第 1 步内；可能 IK warm-start 没生效，或 ik_delta 计算错了

## Git artifacts

- 分支: `tracking`
- 主要 commits:
  - `d28717b` "tracking branch: paper-style training formulation"
  - `6aeaeb0` "maniptrans-style implement"
- 数据: `data/trajectories/oakink_b7853_60hz_kuka_ik_with_vel.json`（60Hz with velocities）
- Prep script: `tools/prep_60hz_with_velocities.py`（np.gradient + gaussian σ=2 paper recipe）

## 下一步: 去掉 kuka 机械臂做 ablation

按用户指示，将 setup 完全改为 ManipTrans 浮动 wrist (Sharpa only, no kuka arm)。如果浮动 wrist setup 也学不会跟踪，则可以排除"机械臂是瓶颈"的假设，转去仔细比对实现细节（reward 公式、reset 顺序、obs 内容）跟 ManipTrans 是否一致。
