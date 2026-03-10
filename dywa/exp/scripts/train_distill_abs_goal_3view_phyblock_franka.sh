#!/bin/bash
cd dywa/exp/train

# Franka + PhyBlock：teacher -> student 蒸馏（abs_goal 3view）

name='phyblock_abs_goal_3view'
root="/home/user/DyWA/output/dywa"

GPU=${1:-0}

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

PYTORCH_JIT=0 python3 train_rma.py \
 +platform=debug \
 +env=phyblock_abs_goal_3view \
 +run=teacher_base_phyblock_franka \
 +student=dywa/base \
 ++name="$name" \
 ++path.root="${root}/${name}" \
 ++env.num_env=1024 \
 ++global_device=cuda:${GPU} \
 ++student.norm="ln" \
 ++add_teacher_state=1

