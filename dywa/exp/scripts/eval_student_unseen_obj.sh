#!/bin/bash
cd dywa/exp/train

# Docker 内路径映射后一般为 /home/user/DyWA；宿主机可 export DYWA_ROOT 覆盖
DYWA_ROOT="${DYWA_ROOT:-/home/user/DyWA}"
name='dywa'
root="${DYWA_ROOT}/output/test_rma"
TEACHER_CKPT_DIR="${DYWA_ROOT}/Dywa_abs_1view/ckpt"
GPU=${1:-0}

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

PYTORCH_JIT=0 python3 test_rma.py \
+platform=debug \
+env=abs_goal_1view \
+run=teacher_base \
+student=dywa/base \
++name="$name" \
++path.root="${root}/${name}" \
++env.num_env=60 \
++global_device=cuda:${GPU} \
++student.norm="ln" \
++load_ckpt="${TEACHER_CKPT_DIR}" \
+load_student="${TEACHER_CKPT_DIR}/last.ckpt" \
++plot_pc=0 \
++dagger_train_env.anneal_step=1 \
++add_teacher_state=1 \
++student.decoder.film_mlp=1 \
++env.single_object_scene.filter_file=${DYWA_ROOT}/Dywa_abs_1view/test_set.json \
++monitor.num_env_record=60 \
++env.single_object_scene.mode=valid \
++log_categorical_results=True \
# &> "$root/$name/out.out"