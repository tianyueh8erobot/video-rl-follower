# Auto Review Loop — ManipTrans × SimToolReal hybrid reward

## Round 1 (2026-05-10, REVIEWER_DIFFICULTY=medium)

### Assessment (Summary)
- Score: 4/10
- Verdict: not ready
- Key criticisms: 1 critical init crash; 3 high-severity (clip-z mutation, dexhand_actors fallback, silent dof-shape truncation); 2 medium (link-fallback wrist-proxy bias, missing-coverage misdocumentation + no NaN guard); 1 low (header inconsistency).

### Reviewer Raw Response
<details>
<summary>Codex review verbatim (gpt-5.4 high reasoning)</summary>

A=PASS, B=FAIL (clip-z mutates frame), C=PASS, D=PASS, E=FAIL (dexhand_actors not on parent), F=PASS, G=PASS, H=PASS, I=FAIL (no dof_pos shape check + sample lacks dex), J=FAIL (19.5% missing not 12%; partial NaN safety).

7 issues:
1. CRITICAL: `self.sub_goal_idx` used before allocated (env.py:264 calls _set_hand_goal_from_trajectory before line 272 alloc).
2. HIGH: `_clip_goal_z()` mutates trajectory pose after copy (env.py:533-534) → frame alignment broken.
3. HIGH: `_setup_imit_link_indices` reads non-existent `self.dexhand_actors` (env.py:477) → falls back to actor 0.
4. HIGH: silent `dex_dof_pos` shape truncation via min() (env.py:589); sample example_g0.json has no dex.
5. MEDIUM: missing-link wrist fallback (env.py:489) → biases reward, not no-op.
6. MEDIUM: 12% comment is wrong (actual 19.5%); no nan_to_num before reward write.
7. LOW: header mentions obj-vel penalty but cfg sets scales to 0.

Overall: 4/10 — not ready.
</details>

### Actions Taken (all 7 fixes applied)
1. Moved `self.sub_goal_idx` allocation BEFORE first `_set_hand_goal_from_trajectory` call.
2. Removed `_clip_goal_z` from fixed-trajectory path (commented why: trajectory was preprocessed offline with table_z).
3. Replaced `dexhand_actors[0]` access with `self.rigid_body_name_to_idx["robot/<name>"]` (built by SimToolReal at init); raises hard error if any of 28 Sharpa body_names is missing.
4. Added 2 hard assertions for `dex_dof_pos`: (a) raise if None when useRetargetDofInit=True at init, (b) raise if shape mismatch instead of silent truncation in reset_idx.
5. (Folded into #3 — missing-link fallback now raises rather than substitutes wrist proxy.)
6. Updated docstring: 9 retained terms / 14 paper terms = 5.15/6.40 ≈ 80.5% coverage; added `torch.nan_to_num` on r_imit and on combined rew_buf.
7. Fixed file header docstring: clarified that R_obj_vel_pen scales default to 0 in cfg (anti-yeet is opt-in).

### Status
- continuing to round 2 to verify fixes.

## Round 2 (2026-05-10, REVIEWER_DIFFICULTY=medium)

### Assessment (Summary)
- Score: 6/10
- Verdict: not ready
- Round-1 fixes 1-5 + 7 verified PASS. Fix 6 (NaN guard) flagged FAIL. Two NEW issues: O (phase mismatch — sub_goal_idx advances 1 step late, policy sees stale goal), N (nan_to_num masks bugs).

### Reviewer Raw Response
<details>
<summary>Codex round-2 verbatim (gpt-5.4 high reasoning)</summary>

- HIGH: sub_goal_idx still advances one step too late. compute_kuka_reward + populate_obs run first; sub_goal_idx advance happens next pre_physics_step via _reset_target. Policy sees stale _wrist_goal/_fingertip_goal_local for one full control step after every success.
- MEDIUM: nan_to_num on r_imit and combined reward MASKS bugs (corrupted trajectory, sim explosion). Should raise/assert or be behind explicit recovery flag.
- Round 1 fixes 1-5, 7 all PASS. K, L, M PASS.

Score: 6/10. Verdict: not ready.
</details>

### Actions Taken
1. (Issue O) Restructured advancement:
   - `_reset_target` is now PURE COPY (both call sites just copy from sub_goal_idx)
   - The `+1 mod T` advance moved into `compute_kuka_reward` AT THE END (after R_imit, after writing rew_buf, before populate_obs runs in the same post_physics_step).
   - Refresh `_wrist_goal`, `_fingertip_goal_local`, `goal_states` for success envs immediately after advance.
2. (Issue N) Removed both `torch.nan_to_num` calls. Replaced with strict assert that raises RuntimeError listing offending env ids + sub_goal_idx + likely causes. Added `cfg.env.allowNonfiniteReward` escape hatch (debug-only).

### Status
- continuing to round 3 to verify both new fixes.

## Round 3 (2026-05-10, REVIEWER_DIFFICULTY=medium)

### Assessment (Summary)
- Score: 7/10
- Verdict: **almost ready** ← STOP CONDITION (score ≥ 6 AND verdict contains "almost")
- O fix verified PASS (correct call order). N fix verified partially (only checks r_imit, not combined). 2 follow-up items.

### Reviewer Raw Response
<details>
<summary>Codex round-3 verbatim (gpt-5.4 high reasoning)</summary>

3 issues:
- HIGH: non-finite assert only checks r_imit, not the parent's rew_buf. Sim explosion → NaN in parent reward terms (norm-of-velocity etc.) still slips through. Fix: extend assert to cover combined reward.
- MEDIUM: subclass _reset_target stopped writing root_state_tensor[goal_object_indices] (which the parent did to update the visible goal actor). Reward/obs stay aligned (both read goal_states), but visible goal object can drift stale across sub-goal advances. Affects video/debugging only.
- MEDIUM: sub_goal_idx and derived hand-goal caches are authoritative state but not serialized. SimToolReal checkpoints don't save/restore them. Resumed runs could use wrong hand target mid-episode.

Other checks: O fix call order PASS, downstream successes consumers PASS, episode-reset overlap PASS, raw_imit_* logging PASS.

Score: 7/10. Verdict: almost ready.
</details>

### Actions Taken
1. (Round-3 HIGH) Extended non-finite guard to cover BOTH r_imit AND the combined reward (rew_buf). Single assert loop with descriptive error message. allowNonfiniteReward escape hatch covers both.
2. (Round-3 MEDIUM #1, GOAL ACTOR) Documented as known limitation — visible goal actor stays at its last set_actor_root_state pose. Not a training-correctness bug. To fix later: copy SimToolReal's `root_state_tensor[self.goal_object_indices]` write block into our `_reset_target`. Defer to follow-up because (a) the user did not ask for visualization-tracking goal actor, (b) policy reward + obs are correct, (c) requires careful indexing into root_state_tensor that risks new bugs.
3. (Round-3 MEDIUM #2, CHECKPOINT) Documented as known limitation — sub_goal_idx is not yet serialized. Not currently used (user is at v1 training, no checkpoint resume yet). To fix: override SimToolReal's save/load_states to include `sub_goal_idx`, `_wrist_goal`, `_fingertip_goal_local`. Defer to follow-up.

### Convergence Decision

The skill's stop condition is `score >= 6 AND verdict contains "ready" or "almost"`. Round 3 result: 7/10 + "almost ready" → STOP.

The final 1 cheap fix (non-finite guard extension) was applied. The 2 deferred items (visible-goal-actor refresh, checkpoint serialization) are documented above and tracked as follow-up work in the codebase comments.

### Status
- LOOP CONVERGED.

## Method Description

**Final reward design** (file: `isaacgymenvs/tasks/video_rl_follower/env.py`):

```
r_total = R_imit  +  R_goal_sparse  +  R_act_penalty  +  R_obj_vel_pen
        ─────────  ───────────────   ────────────────   ─────────────
        ManipTrans  SimToolReal       SimToolReal        SimToolReal
        Stage-1     sparse bonus      action reg         (off by default)
        9 of 14     keypoint+dwell                       opt-in anti-yeet
        ≈ 80.5%     +tolerance
        coverage    curriculum
```

**R_imit** (paper coefficients, dexhandimitator.py:1115-1172):
- 0.1·exp(-40·d_eef_pos)  +  0.6·exp(-1·θ_eef_rot)
- 0.9·exp(-100·d_thumb_tip) + 0.8·exp(-90·d_index_tip) + 0.75·exp(-80·d_middle_tip) + 0.6·exp(-60·d_ring_tip) + 0.6·exp(-60·d_pinky_tip)
- 0.5·exp(-50·d_level_1) + 0.3·exp(-40·d_level_2)
- (vel + power terms NOT YET implemented; coverage 5.15/6.40 ≈ 80.5%)

MANO 21-keypoint targets are expanded to 27 per-Sharpa-link targets via the `to_hand` mapping (paper-style; sharpa.py hand2dex_mapping).

**R_goal_sparse** (SimToolReal, super().compute_kuka_reward):
- bonus_rew = 1{near_goal_steps ≥ N} · reach_goal_bonus
- near_goal = (keypoints_max_dist ≤ tolerance(t))
- tolerance shrinks via curriculum

**Reset** (ManipTrans Stage-2, our reset_idx override):
- seq_idx ~ Uniform[0, T)  if randomStateInit else 0
- sub_goal_idx[env]  = seq_idx                  ★ goal aligned to reset frame
- arm_hand_dof_pos[env, 7:7+22] = opt_dof_pos[seq_idx]   warm-start hand
- arm DOFs left at SimToolReal-randomized default (no IK to opt_wrist_pos)

**Sub-goal advancement** (compute_kuka_reward, end of method):
- After R_imit + reward write, for envs in is_success:
  - sub_goal_idx[env] = (sub_goal_idx[env] + 1) % T
  - Refresh _wrist_goal, _fingertip_goal_local, goal_states for those envs
- This happens BEFORE populate_obs runs in the same post_physics_step, so the
  policy's next obs already reflects the new goal.
- _reset_target (called next pre_physics_step) is a pure COPY (no advance).

**Strict-correctness assertions**:
- _setup_imit_link_indices raises if any of 28 Sharpa body_names is missing.
- __init__ raises if useRetargetDofInit=True and dex_dof_pos is None.
- reset_idx raises if dex_dof_pos.shape[1] != num_hand_dofs.
- compute_kuka_reward raises if r_imit OR combined reward is non-finite.
- All can be relaxed via cfg flags for debug.

**Time axis** (single source of truth):
- sub_goal_idx[env] is the sole indexer for both R_imit AND R_goal_sparse.
- successes[env] is decoupled from goal indexing and is now a pure stats counter.

