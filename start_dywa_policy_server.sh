#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7775}"
DEVICE="${DEVICE:-cuda:0}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
DYWA_GOAL_OFFSET_CM_X="${DYWA_GOAL_OFFSET_CM_X:-30.0}"
DYWA_GOAL_OFFSET_CM_Y="${DYWA_GOAL_OFFSET_CM_Y:-0.0}"
DYWA_GOAL_OFFSET_CM_Z="${DYWA_GOAL_OFFSET_CM_Z:-9.5}"
GRASP_GOAL_OFFSET_CM_X="${GRASP_GOAL_OFFSET_CM_X:-30.0}"
GRASP_GOAL_OFFSET_CM_Y="${GRASP_GOAL_OFFSET_CM_Y:-15.0}"
GRASP_GOAL_OFFSET_CM_Z="${GRASP_GOAL_OFFSET_CM_Z:-9.5}"
ATTITUDE_TOL_DEG="${ATTITUDE_TOL_DEG:-8.0}"
ADJUST_REGION_X_M="${ADJUST_REGION_X_M:-0.05}"
ADJUST_REGION_Y_M="${ADJUST_REGION_Y_M:-0.05}"
ADJUST_REGION_Z_M="${ADJUST_REGION_Z_M:-0.05}"
PLACE_TOL_M="${PLACE_TOL_M:-0.02}"
GRASP_STEPS_PER_SEGMENT="${GRASP_STEPS_PER_SEGMENT:-6}"
MAX_ACTION_STEP_POS_M="${MAX_ACTION_STEP_POS_M:-0.03}"
MAX_ACTION_STEP_ROT_RAD="${MAX_ACTION_STEP_ROT_RAD:-0.35}"
MAX_ACTION_STEP_GRIPPER="${MAX_ACTION_STEP_GRIPPER:-0.6}"
LINEMOD_MATCH_THRESHOLD="${LINEMOD_MATCH_THRESHOLD:-60.0}"
SCENE_ROOT="${SCENE_ROOT:-block_data/SCENEs_400_Goal_Jsons}"
TEMPLATE_DB="${TEMPLATE_DB:-block_data/linemod_templates-real}"
BLOCK_ASSETS_DIR="${BLOCK_ASSETS_DIR:-block_data/block_assets}"
DYWA_EXPORT_DIR="${DYWA_EXPORT_DIR:-exported_abs_goal_1view}"

python3 -m dywa.src.control.dywa_policy_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --scene-root "${SCENE_ROOT}" \
  --template-db "${TEMPLATE_DB}" \
  --block-assets-dir "${BLOCK_ASSETS_DIR}" \
  --dywa-export-dir "${DYWA_EXPORT_DIR}" \
  --dywa-device "${DEVICE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --dywa-goal-offset-cm-x "${DYWA_GOAL_OFFSET_CM_X}" \
  --dywa-goal-offset-cm-y "${DYWA_GOAL_OFFSET_CM_Y}" \
  --dywa-goal-offset-cm-z "${DYWA_GOAL_OFFSET_CM_Z}" \
  --grasp-goal-offset-cm-x "${GRASP_GOAL_OFFSET_CM_X}" \
  --grasp-goal-offset-cm-y "${GRASP_GOAL_OFFSET_CM_Y}" \
  --grasp-goal-offset-cm-z "${GRASP_GOAL_OFFSET_CM_Z}" \
  --attitude-tol-deg "${ATTITUDE_TOL_DEG}" \
  --adjust-region-x-m "${ADJUST_REGION_X_M}" \
  --adjust-region-y-m "${ADJUST_REGION_Y_M}" \
  --adjust-region-z-m "${ADJUST_REGION_Z_M}" \
  --place-tol-m "${PLACE_TOL_M}" \
  --grasp-steps-per-segment "${GRASP_STEPS_PER_SEGMENT}" \
  --max-action-step-pos-m "${MAX_ACTION_STEP_POS_M}" \
  --max-action-step-rot-rad "${MAX_ACTION_STEP_ROT_RAD}" \
  --max-action-step-gripper "${MAX_ACTION_STEP_GRIPPER}" \
  --linemod-match-threshold "${LINEMOD_MATCH_THRESHOLD}"
