#!/bin/bash
cd dywa/exp/train

# 用 Franka 机械臂在 PhyBlock 积木上训练 teacher（stage 1）

name='phyblock_franka'
root="/home/user/DyWA/output/dywa"

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

# Usage:
#   bash dywa/exp/scripts/train_teacher_stage1_phyblock_franka.sh
#
# 接着之前的 run 继续训练：取消下面两行的注释并改成要恢复的 run（如 run-001）
# ++path.key=run-001
# ++load_ckpt=/home/user/DyWA/output/dywa/teacher-stage1-phyblock-franka/run-001/ckpt

ROOT=/home/user/DyWA/output/dywa/teacher-stage1-phyblock-franka/
RESUME_RUN="run-001"   # 例如 run-001 则从该 run 的 last.ckpt 恢复

PYTORCH_JIT=0 python3 train_ppo_arm.py \
  +platform=debug \
  +env=phyblock_franka \
  +run=icra_ours_abs_rel \
  ++env.seed=56081 \
  ++tag=policy \
  ++global_device=cuda:2 \
  ++path.root="${ROOT}" \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" \
  $([ -n "${RESUME_RUN}" ] && echo "++path.key=${RESUME_RUN} ++load_ckpt=${ROOT}${RESUME_RUN}/ckpt")

