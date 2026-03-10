#!/bin/bash
cd dywa/exp/train

# Train teacher on PhyBlock OBJ blocks with UR5 (Stage 2: Finetuning with Domain Randomization).

name='teacher-stage2-phyblock-ur5'
root="/home/lixin/DyWA/output/dywa"

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

# Default to the successful Stage 1 checkpoint
POLICY_CKPT=${1:-"${root}/teacher-stage1-phyblock-ur5/run-013/ckpt/"}

PYTORCH_JIT=0 python3 train_ppo_arm.py \
  +platform=debug \
  +env=phyblock_ur5 \
  +run=icra_ours_abs_rel_ur5 \
  ++env.seed=56081 \
  ++tag=finetune \
  ++global_device=cuda:0 \
  ++path.root="${root}/${name}/" \
  ++env.num_env=1024 \
  ++is_phase2=true \
  ++phase2.min_reset_to_update=16384 \
  ++agent.train.lr=5e-6 \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" \
  ++load_ckpt="${POLICY_CKPT}"
