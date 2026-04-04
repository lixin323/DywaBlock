#!/usr/bin/env bash
# 真机启动前公共步骤：场景 json、goal_cloud pkl、交互录制与开始确认。
# 用法：在仓库根目录由其它脚本 source：
#   source "$(dirname "$0")/start_franka_dywa_inference_common.sh"
# 依赖：调用方已 set -euo pipefail（推荐）。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-7775}"
SCENE_ID="${SCENE_ID:-4}"
SCENE_ROOT="${SCENE_ROOT:-${ROOT_DIR}/block_data/SCENEs_400_Goal_Jsons}"
GOAL_CLOUD_DIR="${GOAL_CLOUD_DIR:-${ROOT_DIR}/output/goal_clouds}"
FOUNDATIONPOSE_EXTRACT_URL="${FOUNDATIONPOSE_EXTRACT_URL:-http://127.0.0.1:7780/extract_goal_clouds}"

T_BASE_CAM_CSV="${T_BASE_CAM_CSV:-0.059375312,-0.997573331,0.036359602,0.327485000,0.740835635,0.068448557,0.668189611,-0.526087000,-0.669056899,-0.012737478,0.743102027,0.455808000,0,0,0,1}"

CAM_SIDE_SERIAL="${CAM_SIDE_SERIAL:-405622072640}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
INFERENCE_FREQUENCY="${INFERENCE_FREQUENCY:-1.0}"
MAX_ACTIONS_TO_PUBLISH="${MAX_ACTIONS_TO_PUBLISH:-20}"
ACTION_PUBLISH_INTERVAL="${ACTION_PUBLISH_INTERVAL:-0.01}"
MAX_STATE_AGE="${MAX_STATE_AGE:-0.2}"

EE_STATES_TOPIC="${EE_STATES_TOPIC:-/franka/ee_states}"
ACTION_TOPIC="${ACTION_TOPIC:-/franka/action_command}"
QUEUE_STATUS_TOPIC="${QUEUE_STATUS_TOPIC:-/franka/queue_status}"
ALLOW_INFERENCE_TOPIC="${ALLOW_INFERENCE_TOPIC:-/franka/allow_inference}"

SCENE_BASENAME="$(printf "%03d" "${SCENE_ID}")"
SCENE_JSON="${SCENE_ROOT}/${SCENE_BASENAME}.json"
GOAL_CLOUD_PKL="${GOAL_CLOUD_DIR}/${SCENE_BASENAME}.pkl"

if [ ! -f "${SCENE_JSON}" ]; then
  echo "[ERROR] scene json not found: ${SCENE_JSON}"
  exit 1
fi

mkdir -p "${GOAL_CLOUD_DIR}"
if [ ! -f "${GOAL_CLOUD_PKL}" ]; then
  echo "[INFO] goal cloud file not found: ${GOAL_CLOUD_PKL}"
  read -r -p "是否录制目标点云并保存为 ${SCENE_BASENAME}.pkl ? 输入 yes 继续: " ans_record
  if [ "${ans_record}" != "yes" ]; then
    echo "[ABORT] 未录制目标点云，退出。"
    exit 1
  fi
  python3 "${ROOT_DIR}/record_goal_cloud_dict.py" \
    --scene-json "${SCENE_JSON}" \
    --save-path "${GOAL_CLOUD_PKL}" \
    --service-url "${FOUNDATIONPOSE_EXTRACT_URL}" \
    --cam-serial "${CAM_SIDE_SERIAL}" \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --camera-fps "${CAMERA_FPS}" \
    --cloud-size 512
fi

read -r -p "是否开始操作场景 ${SCENE_BASENAME} ? 输入 yes 开始: " ans_start
if [ "${ans_start}" != "yes" ]; then
  echo "[ABORT] 用户取消操作。"
  exit 0
fi
