#!/usr/bin/env bash
# 真机 ROS 节点：公共前置（场景/pkl/确认）后启动 franka_dywa_inference_node.py。
# ADJUST 完成门控（ICP / FoundationPose）仅在 A 机 policy server 上由 COMPLETION_METHOD 决定，真机侧无分支。
set -euo pipefail
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SDIR}/start_franka_dywa_inference_common.sh"

exec python3 "${ROOT_DIR}/franka_dywa_inference_node.py" \
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
