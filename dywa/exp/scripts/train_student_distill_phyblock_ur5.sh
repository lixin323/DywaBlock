#!/bin/bash
cd dywa/exp/train

# Distill teacher policy to student policy (Stage 3).

name='student-distill-phyblock-ur5'
root="${HOME}/DyWA/output/dywa"

# NOTE: Update this path after Stage 2 finishes!
# Defaulting to a hypothetical Stage 2 run-000 for now.
TEACHER_CKPT=${1:-"${root}/teacher-stage2-phyblock-ur5/run-000/ckpt/"}

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

PYTORCH_JIT=0 python3 train_rma.py \
  +platform=debug \
  +env=phyblock_ur5 \
  +run=icra_ours_abs_rel_ur5 \
  +student=dywa_id_sconv_his5_lr6e4_film_scale \
  ++name="${name}" \
  ++path.root="${root}/${name}/" \
  ++env.num_env=1024 \
  ++global_device=cuda:0 \
  ++student.norm="ln" \
  ++add_teacher_state=1 \
  ++student.state_keys=\[\'rel_goal\',\'hand_state\',\'robot_state\',\'previous_action\'\] \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" \
  ++load_ckpt="${TEACHER_CKPT}"
