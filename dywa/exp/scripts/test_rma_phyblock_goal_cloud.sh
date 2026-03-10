#!/usr/bin/env bash

# 测试 RMA 学生策略，启用 PhyBlock 的 goal_cloud（initial 模式）

GPU=${1:-0}

NAME='phyblock_abs_goal_1view_with_goal_cloud'
ROOT="/home/user/DyWA/output/eval_phyblock_unseen_obj"

STUDENT_CKPT="/home/user/DyWA/Dywa_abs_1view/ckpt/last.ckpt"
TEACHER_CKPT_DIR="/home/user/DyWA/Dywa_abs_1view/ckpt"

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_JIT=0

# 与现有训练脚本保持一致：先进入 dywa/exp/train，再运行 test_rma.py
cd dywa/exp/train

python3 test_rma.py \
  +platform=debug \
  +env=phyblock_abs_goal_1view \
  +run=teacher_base_phyblock_franka \
  +student=dywa/base \
  ++name="${NAME}" \
  ++path.root="${ROOT}" \
  ++env.num_env=60 \
  ++global_device=cuda:0 \
  +load_student="${STUDENT_CKPT}" \
  ++load_ckpt="${TEACHER_CKPT_DIR}" \
  ++train_student_policy=true \
  ++use_icp_obs=false \
  ++plot_pc=0 \
  ++dagger_train_env.anneal_step=1 \
  ++add_teacher_state=1 \
  ++student.decoder.film_mlp=1 \
  ++monitor.num_env_record=60 \
  ++env.single_object_scene.mode=valid \
  ++env.single_object_scene.filter_file=null \
  ++log_categorical_results=true \
  ++use_goal_cloud=true \
  ++goal_cloud_type='initial'

