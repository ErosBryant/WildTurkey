#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/datasets/data}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/build/figure10_read_breakdown}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
DATA_MODE="${DATA_MODE:-auto}"
NUM="${NUM:-20000000}"
DATASET_SIZE="${DATASET_SIZE:-200M}"
VALUE_SIZE="${VALUE_SIZE:-100}"
RUNS="${RUNS:-1}"
DROP_CACHES="${DROP_CACHES:-0}"
KEEP_DB="${KEEP_DB:-0}"
KEEP_WORK="${KEEP_WORK:-1}"
CLEAR_LOGS="${CLEAR_LOGS:-1}"
WAITBG="${WAITBG:-1}"
EXIT_GRACE_SECONDS="${EXIT_GRACE_SECONDS:-30}"
POLL_SECONDS="${POLL_SECONDS:-5}"

make_abs() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${BASE_DIR}/${path}"
  fi
}

DB_BENCH="$(make_abs "${DB_BENCH}")"
DATA_DIR="$(make_abs "${DATA_DIR}")"
OUT_DIR="$(make_abs "${OUT_DIR}")"
LOG_DIR="$(make_abs "${LOG_DIR}")"

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "db_bench not found or not executable at ${DB_BENCH}" >&2
  echo "Build first: mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}/db" "${OUT_DIR}/work" "${OUT_DIR}/mplconfig"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/mplconfig}"
CASES="${OUT_DIR}/cases.csv"
SUMMARY="${OUT_DIR}/summary.csv"
echo "engine,dataset,run,num,fill_log,read_log,fill_status,read_status" > "${CASES}"

dataset_prefix() {
  case "$1" in
    osm|osm_cellids) printf "osm_cellids" ;;
    book|books) printf "books" ;;
    wiki|wiki_ts) printf "wiki_ts" ;;
    fb) printf "fb" ;;
    *) printf "%s" "$1" ;;
  esac
}

real_dataset_file() {
  printf "%s/%s_%s_uint64" "${DATA_DIR}" "$(dataset_prefix "$1")" "${DATASET_SIZE}"
}

real_datasets_available() {
  local dataset
  for dataset in "$@"; do
    [[ -f "$(real_dataset_file "${dataset}")" ]] || return 1
  done
  return 0
}

read -r -a engines <<< "${ENGINES:-wisckey:--mod=8 bourbon:--mod=7 wildturkey:--mod=10}"
read -r -a requested_datasets <<< "${DATASETS:-osm_cellids fb}"

ACTIVE_DATA_MODE="${DATA_MODE}"
if [[ "${ACTIVE_DATA_MODE}" == "auto" ]]; then
  if real_datasets_available "${requested_datasets[@]}"; then
    ACTIVE_DATA_MODE="real"
  else
    ACTIVE_DATA_MODE="builtin"
    echo "real datasets not found under ${DATA_DIR}; using db_bench built-in random data" >&2
  fi
fi

if [[ "${ACTIVE_DATA_MODE}" == "real" ]]; then
  datasets=("${requested_datasets[@]}")
elif [[ "${ACTIVE_DATA_MODE}" == "builtin" ]]; then
  read -r -a datasets <<< "${DATASETS:-builtin_random}"
else
  echo "invalid DATA_MODE='${DATA_MODE}'; expected auto, real, or builtin" >&2
  exit 1
fi

if [[ "${CLEAR_LOGS}" == "1" ]]; then
  find "${LOG_DIR}" -maxdepth 1 -type f -name "*.log" -delete
fi

CURRENT_DB_PATH=""
CURRENT_WORK_PATH=""

drop_caches_if_requested() {
  if [[ "${DROP_CACHES}" != "1" ]]; then
    return
  fi

  if [[ "${EUID}" -eq 0 ]]; then
    echo "dropping page cache..." >&2
    sync
    echo 3 > /proc/sys/vm/drop_caches
  elif sudo -n true 2>/dev/null; then
    echo "dropping page cache with sudo..." >&2
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  else
    echo "DROP_CACHES=1 requested, but passwordless sudo is unavailable; skipping cache drop" >&2
  fi
}

cleanup_case_outputs() {
  local db_path="$1"
  local work_path="$2"
  if [[ "${KEEP_DB}" != "1" ]]; then
    rm -rf "${db_path}"
  fi
  if [[ "${KEEP_WORK}" != "1" ]]; then
    rm -rf "${work_path}"
  fi
}

log_has_pattern() {
  local log_path="$1"
  local pattern="$2"
  [[ -f "${log_path}" ]] && grep -Eq "${pattern}" "${log_path}"
}

read_breakdown_benchmarks() {
  local benchmarks
  if [[ "${ACTIVE_DATA_MODE}" == "real" ]]; then
    benchmarks="fillrandom_r,flushmem,compactl0"
  else
    benchmarks="fillrandom,flushmem,compactl0"
  fi
  if [[ "${WAITBG}" == "1" ]]; then
    benchmarks="${benchmarks},waitbg"
  fi
  if [[ "${ACTIVE_DATA_MODE}" == "real" ]]; then
    benchmarks="${benchmarks},readrandom_r"
  else
    benchmarks="${benchmarks},readrandom"
  fi
  printf '%s,stats\n' "${benchmarks}"
}

run_db_bench_case() {
  local work_path="$1"
  local log_path="$2"
  local ready_pattern="$3"
  shift 3
  local pid ready_since now

  (
    cd "${work_path}" && \
    export WT_LOG_DIR="${LOG_DIR}" && \
    exec "${DB_BENCH}" "$@"
  ) > "${log_path}" 2>&1 &

  pid=$!
  ready_since=0

  while kill -0 "${pid}" 2>/dev/null; do
    if log_has_pattern "${log_path}" "${ready_pattern}"; then
      now="$(date +%s)"
      if [[ "${ready_since}" -eq 0 ]]; then
        ready_since="${now}"
      elif (( now - ready_since >= EXIT_GRACE_SECONDS )); then
        echo "warning: db_bench produced required output but did not exit after ${EXIT_GRACE_SECONDS}s; terminating pid ${pid}: ${log_path}" >&2
        kill "${pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null
        return 0
      fi
    fi
    sleep "${POLL_SECONDS}"
  done

  wait "${pid}"
}

cleanup_current_case_on_exit() {
  local status=$?
  trap - INT TERM EXIT
  if [[ -n "${CURRENT_DB_PATH}" || -n "${CURRENT_WORK_PATH}" ]]; then
    cleanup_case_outputs "${CURRENT_DB_PATH}" "${CURRENT_WORK_PATH}"
  fi
  exit "${status}"
}

trap cleanup_current_case_on_exit INT TERM EXIT

for engine_entry in "${engines[@]}"; do
  engine="${engine_entry%%:*}"
  engine_args="${engine_entry#*:}"

  for dataset in "${datasets[@]}"; do
    for run in $(seq 1 "${RUNS}"); do
      db_path="${OUT_DIR}/db/${engine}_${dataset}_run${run}"
      work_path="${OUT_DIR}/work/${engine}_${dataset}_run${run}"
      run_log="${LOG_DIR}/${engine}_${dataset}_run${run}.log"

      CURRENT_DB_PATH="${db_path}"
      CURRENT_WORK_PATH="${work_path}"
      rm -rf "${db_path}" "${work_path}"
      mkdir -p "${work_path}"

      echo "=== figure10 ${engine} dataset=${dataset} fill+read run=${run} ==="
      set +e
      if [[ "${ACTIVE_DATA_MODE}" == "real" ]]; then
        run_db_bench_case "${work_path}" "${run_log}" "^Timer 19:" \
          --benchmarks="$(read_breakdown_benchmarks)" \
          --dataset="${dataset}" \
          --dataset_size="${DATASET_SIZE}" \
          --num="${NUM}" \
          --value_size="${VALUE_SIZE}" \
          --path_real_data="${DATA_DIR}" \
          --db="${db_path}" \
          ${engine_args}
      else
        run_db_bench_case "${work_path}" "${run_log}" "^Timer 19:" \
          --benchmarks="$(read_breakdown_benchmarks)" \
          --num="${NUM}" \
          --value_size="${VALUE_SIZE}" \
          --db="${db_path}" \
          ${engine_args}
      fi
      run_status=$?
      set -e

      echo "${engine},${dataset},${run},${NUM},${run_log},${run_log},${run_status},${run_status}" >> "${CASES}"
      cleanup_case_outputs "${db_path}" "${work_path}"
      CURRENT_DB_PATH=""
      CURRENT_WORK_PATH=""
      drop_caches_if_requested
    done
  done
done

python3 "${SCRIPT_DIR}/plot_figure10_read_breakdown.py" "${CASES}" "${SUMMARY}" "${OUT_DIR}/figure10_read_breakdown.png"
echo "cases: ${CASES}"
echo "summary: ${SUMMARY}"
echo "logs: ${LOG_DIR}"
