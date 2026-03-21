#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# -------- Policy server (A机) --------
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-33060}"
SCENE_ID="${SCENE_ID:-0}"

# -------- Hand-eye (T_base_cam, row-major 4x4) --------
T_BASE_CAM_CSV="${T_BASE_CAM_CSV:-0.106877856,-0.987151927,-0.118777928,0.335567000,0.737191404,-0.001485801,0.675682341,-0.329068000,-0.667177606,-0.159777548,0.727561116,0.710526000,0,0,0,1}"

# -------- RealSense side camera --------
CAM_SIDE_SERIAL="${CAM_SIDE_SERIAL:-405622072640}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
INFERENCE_FREQUENCY="${INFERENCE_FREQUENCY:-1.0}"
MAX_ACTIONS_TO_PUBLISH="${MAX_ACTIONS_TO_PUBLISH:-20}"
ACTION_PUBLISH_INTERVAL="${ACTION_PUBLISH_INTERVAL:-0.01}"
MAX_STATE_AGE="${MAX_STATE_AGE:-0.2}"

# -------- ROS topics --------
EE_STATES_TOPIC="${EE_STATES_TOPIC:-/franka/ee_states}"
ACTION_TOPIC="${ACTION_TOPIC:-/franka/action_command}"
QUEUE_STATUS_TOPIC="${QUEUE_STATUS_TOPIC:-/franka/queue_status}"
ALLOW_INFERENCE_TOPIC="${ALLOW_INFERENCE_TOPIC:-/franka/allow_inference}"

python3 "${ROOT_DIR}/franka_dywa_inference_node.py" \
  --policy-server-host "${POLICY_SERVER_HOST}" \
  --policy-server-port "${POLICY_SERVER_PORT}" \
  --scene-id "${SCENE_ID}" \
  --t-base-cam-csv "${T_BASE_CAM_CSV}" \
  --cam-side-serial "${CAM_SIDE_SERIAL}" \
  --image-width "${IMAGE_WIDTH}" \
  --image-height "${IMAGE_HEIGHT}" \
  --camera-fps "${CAMERA_FPS}" \
  --inference-frequency "${INFERENCE_FREQUENCY}" \
  --max-actions-to-publish "${MAX_ACTIONS_TO_PUBLISH}" \
  --action-publish-interval "${ACTION_PUBLISH_INTERVAL}" \
  --max-state-age "${MAX_STATE_AGE}" \
  --ee-states-topic "${EE_STATES_TOPIC}" \
  --action-topic "${ACTION_TOPIC}" \
  --queue-status-topic "${QUEUE_STATUS_TOPIC}" \
  --allow-inference-topic "${ALLOW_INFERENCE_TOPIC}"
