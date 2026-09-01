#!/usr/bin/env bash
# Replay the five ABot action-only workloads on four physical GPUs.
#
# The default prefix8 mode is a short stability preflight. Full mode replays
# the original 24-session action_world_v2 suite and raises per-worker retained
# session capacity from four to six.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/public/fanyk1/lwb/envs/telefuser_sage291/bin/python"
LIVEKIT_BIN="/tmp/livekit-bin/livekit-server"
MODEL_ZOO_PATH="/public/fanyk1/lwb/model_zoo"
PROFILE_PATH="/public/fanyk1/lwb/abot_world_eval_bench/runs/abot_full_profile/profile_evaluated.csv"
SCENARIO_PATH="tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_wave.json"
MODE="prefix8"
GPU_IDS="0,1,2,3"
RUN_ROOT=""
TRACE_ROOT=""
MAX_SESSIONS_PER_WORKER=""
DRY_RUN=0
REQUESTED_WORKLOADS=()

readonly -a ALL_WORKLOADS=(
  steady_control
  burst_join
  rapid_action_change
  interaction_pause
  mixed_nonstationary
)

usage() {
  cat <<'EOF'
Usage: tools/validation/run_abot_4gpu_action_suite.sh [options]

Replay the five ABot action-only workloads with independent LiveKit and
TeleFuser processes and independent artifacts for each workload.

Options:
  --mode prefix8|full             Trace suite to replay (default: prefix8).
  --workload NAME                 Run one workload; may be repeated.
  --gpu-ids A,B,C,D               Four physical GPU IDs (default: 0,1,2,3).
  --run-root PATH                 Artifact root; must not already exist.
  --trace-root PATH               Override the selected mode's trace root.
  --max-sessions-per-worker N     Override 4 (prefix8) or 6 (full).
  --python-bin PATH               Python interpreter.
  --livekit-bin PATH              LiveKit server binary.
  --profile PATH                  Evaluated motivation profile CSV.
  --scenario PATH                 Base LiveKit replay scenario.
  --dry-run                       Validate all selected traces without starting services.
  -h, --help                      Show this help.

Examples:
  tools/validation/run_abot_4gpu_action_suite.sh --mode prefix8 --dry-run
  tools/validation/run_abot_4gpu_action_suite.sh --mode prefix8
  tools/validation/run_abot_4gpu_action_suite.sh --mode full --workload steady_control
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      need_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --workload)
      need_value "$1" "${2:-}"
      REQUESTED_WORKLOADS+=("$2")
      shift 2
      ;;
    --gpu-ids)
      need_value "$1" "${2:-}"
      GPU_IDS="$2"
      shift 2
      ;;
    --run-root)
      need_value "$1" "${2:-}"
      RUN_ROOT="$2"
      shift 2
      ;;
    --trace-root)
      need_value "$1" "${2:-}"
      TRACE_ROOT="$2"
      shift 2
      ;;
    --max-sessions-per-worker)
      need_value "$1" "${2:-}"
      MAX_SESSIONS_PER_WORKER="$2"
      shift 2
      ;;
    --python-bin)
      need_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --livekit-bin)
      need_value "$1" "${2:-}"
      LIVEKIT_BIN="$2"
      shift 2
      ;;
    --profile)
      need_value "$1" "${2:-}"
      PROFILE_PATH="$2"
      shift 2
      ;;
    --scenario)
      need_value "$1" "${2:-}"
      SCENARIO_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "${MODE}" in
  prefix8)
    DEFAULT_TRACE_ROOT="results/experiments/motivation_diag_prefix8_20260901T000000Z/traces"
    DEFAULT_MAX_SESSIONS=4
    ;;
  full)
    DEFAULT_TRACE_ROOT="/public/fanyk1/lwb/abot_world_data_harness/outputs/action_world_v2"
    DEFAULT_MAX_SESSIONS=6
    ;;
  *)
    die "--mode must be prefix8 or full; got: ${MODE}"
    ;;
esac

TRACE_ROOT="${TRACE_ROOT:-${DEFAULT_TRACE_ROOT}}"
MAX_SESSIONS_PER_WORKER="${MAX_SESSIONS_PER_WORKER:-${DEFAULT_MAX_SESSIONS}}"
RUN_ROOT="${RUN_ROOT:-results/experiments/abot_action_suite_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"

repo_absolute_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

TRACE_ROOT="$(repo_absolute_path "${TRACE_ROOT}")"
PROFILE_PATH="$(repo_absolute_path "${PROFILE_PATH}")"
SCENARIO_PATH="$(repo_absolute_path "${SCENARIO_PATH}")"
RUN_ROOT="$(repo_absolute_path "${RUN_ROOT}")"

[[ "${MAX_SESSIONS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]] \
  || die "--max-sessions-per-worker must be a positive integer"
[[ -x "${PYTHON_BIN}" ]] || die "Python interpreter is not executable: ${PYTHON_BIN}"
[[ -f "${PROFILE_PATH}" ]] || die "Motivation profile not found: ${PROFILE_PATH}"
[[ -f "${SCENARIO_PATH}" ]] || die "Replay scenario not found: ${SCENARIO_PATH}"
[[ -d "${TRACE_ROOT}" ]] || die "Trace root not found: ${TRACE_ROOT}"

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 4 ]] || die "--gpu-ids must contain exactly four comma-separated IDs"
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${gpu}"
done

declare -A ALLOWED_WORKLOADS=()
for workload in "${ALL_WORKLOADS[@]}"; do
  ALLOWED_WORKLOADS["${workload}"]=1
done

if [[ "${#REQUESTED_WORKLOADS[@]}" -eq 0 ]]; then
  WORKLOADS=("${ALL_WORKLOADS[@]}")
else
  WORKLOADS=("${REQUESTED_WORKLOADS[@]}")
fi

declare -A SEEN_WORKLOADS=()
for workload in "${WORKLOADS[@]}"; do
  [[ -n "${ALLOWED_WORKLOADS[${workload}]:-}" ]] || die "unknown workload: ${workload}"
  [[ -z "${SEEN_WORKLOADS[${workload}]:-}" ]] || die "duplicate workload: ${workload}"
  SEEN_WORKLOADS["${workload}"]=1
  [[ -f "${TRACE_ROOT}/${workload}/events.jsonl" ]] \
    || die "trace not found: ${TRACE_ROOT}/${workload}/events.jsonl"
done

echo "ABot action suite"
echo "  mode: ${MODE}"
echo "  trace root: ${TRACE_ROOT}"
echo "  workloads: ${WORKLOADS[*]}"
echo "  GPU IDs: ${GPU_IDS}"
echo "  max sessions/worker: ${MAX_SESSIONS_PER_WORKER}"
echo "  run root: ${RUN_ROOT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  for workload in "${WORKLOADS[@]}"; do
    echo "Validating ${workload}..."
    PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" \
      tools/validation/replay_abot_livekit_action_trace.py \
      --trace "${TRACE_ROOT}/${workload}/events.jsonl" \
      --scenario "${SCENARIO_PATH}" \
      --dry-run
  done
  echo "Dry run completed; no service or GPU workload was started."
  exit 0
fi

[[ -x "${LIVEKIT_BIN}" ]] || die "LiveKit binary is not executable: ${LIVEKIT_BIN}"
[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite existing run root: ${RUN_ROOT}"

for gpu in "${GPUS[@]}"; do
  owners="$(
    nvidia-smi -i "${gpu}" \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits | sed '/^$/d'
  )"
  if [[ -n "${owners}" ]]; then
    echo "GPU ${gpu} has active compute process(es):" >&2
    echo "${owners}" >&2
    exit 1
  fi
done

for port in 7880 8088; do
  if ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .; then
    die "TCP port ${port} is already listening"
  fi
done

LOCK_PATH="/tmp/telefuser-abot-action-suite-${GPU_IDS//,/_}.lock"
exec 9>"${LOCK_PATH}"
flock -n 9 || die "another ABot action suite holds ${LOCK_PATH}"

mkdir -p "${RUN_ROOT}"
cp "${PROFILE_PATH}" "${RUN_ROOT}/profile_evaluated.csv"
cp "${SCENARIO_PATH}" "${RUN_ROOT}/scenario.json"
RUN_PROFILE_PATH="${RUN_ROOT}/profile_evaluated.csv"
RUN_SCENARIO_PATH="${RUN_ROOT}/scenario.json"
printf '%s\n' \
  "MODE=${MODE}" \
  "TRACE_ROOT=${TRACE_ROOT}" \
  "GPU_IDS=${GPU_IDS}" \
  "MAX_SESSIONS_PER_WORKER=${MAX_SESSIONS_PER_WORKER}" \
  "PROFILE_PATH=${PROFILE_PATH}" \
  "SCENARIO_PATH=${SCENARIO_PATH}" \
  "WORKLOADS=${WORKLOADS[*]}" \
  >"${RUN_ROOT}/suite-config.env"

LIVEKIT_PID=""
SERVER_PID=""
SERVING_METRICS_PID=""
GPU_METRICS_PID=""

stop_pid() {
  local pid="$1"
  local label="$2"
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  echo "${label} PID ${pid} did not stop after 30 seconds; sending SIGKILL" >&2
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup_current() {
  stop_pid "${SERVING_METRICS_PID}" "serving metrics"
  SERVING_METRICS_PID=""
  stop_pid "${GPU_METRICS_PID}" "GPU metrics"
  GPU_METRICS_PID=""
  stop_pid "${SERVER_PID}" "TeleFuser server"
  SERVER_PID=""
  stop_pid "${LIVEKIT_PID}" "LiveKit server"
  LIVEKIT_PID=""
}

cleanup_all() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_current
  exit "${status}"
}
trap cleanup_all EXIT INT TERM

trace_duration_seconds() {
  local trace_path="$1"
  "${PYTHON_BIN}" - "${trace_path}" <<'PY'
import json
import sys

duration = 0.0
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            duration = max(duration, float(json.loads(line)["time"]))
print(duration)
PY
}

validate_result() {
  local result_path="$1"
  "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)

sessions = result.get("sessions", [])
errors = [
    (
        session.get("source_trace_session_id")
        or session.get("logical_session_id")
        or session.get("server_session_id"),
        session.get("error"),
    )
    for session in sessions
    if session.get("error")
]
admission_violations = [
    (
        session.get("source_trace_session_id")
        or session.get("logical_session_id")
        or session.get("server_session_id"),
        session.get("admission_contract_violation"),
    )
    for session in sessions
    if session.get("admission_contract_violation")
]
dropped = sum(int(session.get("trace_actions_dropped", 0)) for session in sessions)
seen = sum(int(session.get("trace_actions_seen", 0)) for session in sessions)
published = sum(int(session.get("trace_actions_published", 0)) for session in sessions)

print(
    json.dumps(
        {
            "session_count": len(sessions),
            "session_errors": errors,
            "admission_contract_violations": admission_violations,
            "trace_actions_seen": seen,
            "trace_actions_published": published,
            "trace_actions_dropped": dropped,
        },
        indent=2,
        sort_keys=True,
    )
)
if errors or admission_violations or dropped:
    raise SystemExit(1)
PY
}

wait_for_port() {
  local port="$1"
  local pid="$2"
  local label="$3"
  for _ in $(seq 1 60); do
    if ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .; then
      return 0
    fi
    kill -0 "${pid}" 2>/dev/null || die "${label} exited before listening on port ${port}"
    sleep 1
  done
  die "${label} did not listen on port ${port} within 60 seconds"
}

wait_for_ready() {
  local run_dir="$1"
  for _ in $(seq 1 300); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:8088/v1/service/ready" >"${run_dir}/ready.json"; then
      [[ -s "${run_dir}/ready.json" ]] && return 0
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null \
      || die "TeleFuser exited before becoming ready; see ${run_dir}/server.log"
    sleep 1
  done
  die "TeleFuser did not become ready within 300 seconds"
}

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

for workload in "${WORKLOADS[@]}"; do
  echo "Starting workload: ${workload}"
  RUN_DIR="${RUN_ROOT}/${workload}"
  TRACE_PATH="${TRACE_ROOT}/${workload}/events.jsonl"
  mkdir -p "${RUN_DIR}"
  sha256sum "${TRACE_PATH}" >"${RUN_DIR}/trace.sha256"
  for metadata_name in manifest.json summary.json summary.md trace_metadata.json; do
    if [[ -f "${TRACE_ROOT}/${workload}/${metadata_name}" ]]; then
      cp "${TRACE_ROOT}/${workload}/${metadata_name}" "${RUN_DIR}/${metadata_name}"
    fi
  done

  TRACE_DURATION="$(trace_duration_seconds "${TRACE_PATH}")"
  METRICS_DURATION="$(awk -v duration="${TRACE_DURATION}" 'BEGIN { print int(duration + 25.999) }')"
  printf '%s\n' \
    "WORKLOAD=${workload}" \
    "TRACE_PATH=${TRACE_PATH}" \
    "TRACE_DURATION_SECONDS=${TRACE_DURATION}" \
    "METRICS_DURATION_SECONDS=${METRICS_DURATION}" \
    >"${RUN_DIR}/run-config.env"

  "${LIVEKIT_BIN}" --dev >"${RUN_DIR}/livekit.log" 2>&1 &
  LIVEKIT_PID=$!
  printf '%s\n' "${LIVEKIT_PID}" >"${RUN_DIR}/livekit.pid"
  wait_for_port 7880 "${LIVEKIT_PID}" "LiveKit"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PYTHONPATH="${REPO_ROOT}" \
  TF_MODEL_ZOO_PATH="${MODEL_ZOO_PATH}" \
  TELEFUSER_ABOT_CUDA_GRAPH_ENABLED=0 \
  TELEFUSER_ABOT_SCHEDULER_MODE=batched \
  TELEFUSER_ABOT_MAX_BATCH_SIZE=4 \
  TELEFUSER_ABOT_BATCHING_WINDOW_MS=2 \
  TELEFUSER_ABOT_MAX_DEADLINE_BATCH_WAIT_MS=1000 \
  TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_ENABLED=1 \
  TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS=3.0 \
  TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES=36 \
  TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES=4 \
  TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_GUARD_MS=50 \
  TELEFUSER_ABOT_BATCH_COMPUTE_PROFILE=h100_lf3_eager_full_pipeline_v1 \
  TELEFUSER_ABOT_BATCH_COMPUTE_SAFETY_FACTOR=1.05 \
  TELEFUSER_LIVEKIT_DISPATCH_TRACE_PATH="${RUN_DIR}/dispatch-trace.jsonl" \
  TELEFUSER_LIVEKIT_DISPATCH_TRACE_MAX_EVENTS=100000 \
  "${PYTHON_BIN}" -m telefuser.entrypoints.cli.main stream-serve \
    examples/abot_world/abot_world_livekit_service.py \
    --host 127.0.0.1 \
    --port 8088 \
    --livekit-url ws://127.0.0.1:7880 \
    --livekit-api-key devkey \
    --livekit-api-secret secret \
    --num-workers 4 \
    --worker-gpu-map '0;1;2;3' \
    --worker-mode process-nccl \
    --max-sessions-per-worker "${MAX_SESSIONS_PER_WORKER}" \
    --queue-size 0 \
    --motivation-profile "${RUN_PROFILE_PATH}" \
    --motivation-max-batch-size 4 \
    --skip-validation \
    >"${RUN_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${RUN_DIR}/server.pid"

  wait_for_port 8088 "${SERVER_PID}" "TeleFuser"
  wait_for_ready "${RUN_DIR}"
  curl --noproxy '*' -fsS "http://127.0.0.1:8088/v1/service/metadata" \
    >"${RUN_DIR}/metadata-before.json"

  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/capture_abot_serving_metrics.py \
    --server-url http://127.0.0.1:8088 \
    --duration "${METRICS_DURATION}" \
    --interval 1 \
    --output-dir "${RUN_DIR}/serving_metrics" \
    >"${RUN_DIR}/serving-metrics.log" 2>&1 &
  SERVING_METRICS_PID=$!

  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/capture_gpu_nvml_metrics.py \
    --gpu-indices "${GPU_IDS}" \
    --duration "${METRICS_DURATION}" \
    --interval 1 \
    --output-dir "${RUN_DIR}/gpu_metrics" \
    >"${RUN_DIR}/gpu-metrics.log" 2>&1 &
  GPU_METRICS_PID=$!

  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" \
    tools/validation/replay_abot_livekit_action_trace.py \
    --trace "${TRACE_PATH}" \
    --scenario "${RUN_SCENARIO_PATH}" \
    --output "${RUN_DIR}/result.json" \
    --drain-seconds 10 \
    2>&1 | tee "${RUN_DIR}/replay.log"

  curl --noproxy '*' -fsS "http://127.0.0.1:8088/v1/service/metadata" \
    >"${RUN_DIR}/metadata-after.json"
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/extract_motivation_diagnostics.py \
    --metadata "${RUN_DIR}/metadata-after.json" \
    --output "${RUN_DIR}/motivation-diagnostics.json"

  wait "${SERVING_METRICS_PID}"
  SERVING_METRICS_PID=""
  wait "${GPU_METRICS_PID}"
  GPU_METRICS_PID=""

  [[ -s "${RUN_DIR}/dispatch-trace.jsonl" ]] || die "empty dispatch trace for ${workload}"
  [[ -s "${RUN_DIR}/result.json" ]] || die "missing result for ${workload}"
  validate_result "${RUN_DIR}/result.json" | tee "${RUN_DIR}/validation-summary.json"

  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/render_abot_dispatch_timeline.py \
    --dispatch-trace "${RUN_DIR}/dispatch-trace.jsonl" \
    --result "${RUN_DIR}/result.json" \
    --output-dir "${RUN_DIR}/dispatch_analysis"
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" tools/validation/analyze_abot_serving_trace.py \
    --serving-metrics-dir "${RUN_DIR}/serving_metrics" \
    --gpu-metrics "${RUN_DIR}/gpu_metrics/gpu-metrics.jsonl" \
    --output-dir "${RUN_DIR}/serving_analysis"

  cleanup_current
  echo "Completed workload: ${workload}"
done

trap - EXIT INT TERM
cleanup_current
echo "Completed action suite: ${RUN_ROOT}"
