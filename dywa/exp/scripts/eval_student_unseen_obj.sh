#!/bin/bash
cd dywa/exp/train

# Docker 内路径映射后一般为 /home/user/DyWA；宿主机可 export DYWA_ROOT 覆盖
DYWA_ROOT="${DYWA_ROOT:-/home/user/DyWA}"
export PYTHONPATH="${DYWA_ROOT}/dywa/src:${PYTHONPATH}"
BLOCK_DGN_ROOT="${DYWA_ROOT}/block_data_DGN"
FOUNDATIONPOSE_MODE="${FOUNDATIONPOSE_MODE:-false}"
FOUNDATIONPOSE_EVAL_ENABLED="${FOUNDATIONPOSE_EVAL_ENABLED:-false}"
# 仿真评测与 abs_goal_1view 一致：goal_cloud 用 initial（initial_cloud + initial_rel_goal）。
# 真机部署若需 goal_cloud_dict.pkl / recorded，请用部署脚本与 policy 配置，勿与此仿真评测脚本混用。
GOAL_CLOUD_DICT_PATH="${GOAL_CLOUD_DICT_PATH:-${DYWA_ROOT}/output/goal_clouds/goal_cloud_dict.pkl}"
TEST_STEP="${TEST_STEP:-4000}"
# 录制仿真视频（Nvdr + OpenCV mp4）：export USE_NVDR_RECORD_EPISODE=true
USE_NVDR_RECORD_EPISODE="${USE_NVDR_RECORD_EPISODE:-false}"
NVDR_RECORD_DIR="${NVDR_RECORD_DIR:-${DYWA_ROOT}/output/record/nvdr_episode}"
NVDR_VIDEO_FPS="${NVDR_VIDEO_FPS:-30}"
name='dywa'
root="${DYWA_ROOT}/output/test_rma"
TEACHER_CKPT_DIR="${DYWA_ROOT}/Dywa_abs_1view/ckpt"
GPU=${1:-0}

RECORD_EP_ARGS=()
_NVDR="$(echo "${USE_NVDR_RECORD_EPISODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_NVDR}" == "true" || "${_NVDR}" == "1" || "${_NVDR}" == "yes" ]]; then
  mkdir -p "${NVDR_RECORD_DIR}"
  RECORD_EP_ARGS=(
    ++use_nvdr_record_episode=true
    ++nvdr_record_episode.record_dir="${NVDR_RECORD_DIR}"
    ++nvdr_record_episode.video_fps="${NVDR_VIDEO_FPS}"
  )
fi

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
++env.single_object_scene.dgn.data_path="${BLOCK_DGN_ROOT}/meta-v8" \
++env.single_object_scene.dgn.pose_path="${BLOCK_DGN_ROOT}/meta-v8/unique_dgn_poses" \
++env.single_object_scene.filter_file="${BLOCK_DGN_ROOT}/test_set.json" \
+camera.use_color=True \
+camera.use_col=False \
+log_foundationpose_accuracy="${FOUNDATIONPOSE_EVAL_ENABLED}" \
+foundationpose_eval_url="${FOUNDATIONPOSE_EVAL_URL:-http://127.0.0.1:7780/eval_pose}" \
+foundationpose_replace_object_state="${FOUNDATIONPOSE_MODE}" \
+foundationpose_pose_url="${FOUNDATIONPOSE_POSE_URL:-http://127.0.0.1:7780/infer_pose}" \
+foundationpose_trans_thresh_m=0.05 \
+foundationpose_rot_thresh_rad=0.2 \
+foundationpose_eval_use_icp=True \
+foundationpose_goal_cloud_dict_path="${GOAL_CLOUD_DICT_PATH}" \
++monitor.num_env_record=60 \
+test_step="${TEST_STEP}" \
++env.single_object_scene.mode=valid \
++log_categorical_results=True \
++export_cfg_dir="${DYWA_ROOT}/exported_abs_goal_1view" \
"${RECORD_EP_ARGS[@]}"
# &> "$root/$name/out.out"
