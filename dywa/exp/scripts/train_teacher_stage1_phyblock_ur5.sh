#!/bin/bash
cd dywa/exp/train

# Train teacher on PhyBlock OBJ blocks with UR5.

name='teacher-stage1-phyblock-ur5'
root="${HOME}/DyWA/output/dywa"

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

# 关键修改：
# 1. ++env.task.nearest_induce=true: 开启引导奖励，解决 Reward 负数问题
# 2. ++env.task.regularize_coef=0.01: 防止动作过大
# 3. ++use_nvdr_record_viewer=true: 开启视觉录制 (保存到 record/ 目录)

PYTORCH_JIT=0 python3 train_ppo_arm.py \
  +platform=debug \
  +env=phyblock_ur5 \
  +run=icra_ours_abs_rel_ur5 \
  ++env.seed=56081 \
  ++tag=policy \
  ++global_device=cuda:0 \
  ++path.root="${root}/${name}/" \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" \
  ++env.task.nearest_induce=true \
  ++env.task.regularize_coef=0.01
