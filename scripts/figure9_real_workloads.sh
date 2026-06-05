#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/datasets/data}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/build/figure9_real_workloads}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
DATA_MODE="${DATA_MODE:-auto}"
NUM="${NUM:-20000000}"
DATASET_SIZE="${DATASET_SIZE:-200M}"
VALUE_SIZE="${VALUE_SIZE:-100}"
RUNS="${RUNS:-1}"
DROP_CACHES="${DROP_CACHES:-0}"
KEEP_DB="${KEEP_DB:-0}"
KEEP_WORK="${KEEP_WORK:-0}"
REUSE_LOGS="${REUSE_LOGS:-0}"
CLEAR_LOGS="${CLEAR_LOGS:-1}"
INCLUDE_STATS="${INCLUDE_STATS:-0}"
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
SUMMARY="${OUT_DIR}/summary.csv"
echo "engine,dataset,workload,run,benchmark,micros_per_op,ops_per_sec,status,exit_status,log" > "${SUMMARY}"

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

read -r -a engines <<< "${ENGINES:-wildturkey:--mod=10 wisckey:--mod=8 bourbon:--mod=7}"
read -r -a requested_datasets <<< "${DATASETS:-osm_cellids books fb wiki_ts}"

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
  read -r -a workloads <<< "${WORKLOADS:-write_read:fillrandom_r,flushmem,compactl0,readrandom_r:write_only=fillrandom_r,read_only=readrandom_r balanced:fillrandom_r,flushmem,compactl0,balanced_r:balanced=balanced_r read_heavy:fillrandom_r,flushmem,compactl0,readheavy_r:read_heavy=readheavy_r}"
elif [[ "${ACTIVE_DATA_MODE}" == "builtin" ]]; then
  read -r -a datasets <<< "${DATASETS:-builtin_random}"
  read -r -a workloads <<< "${WORKLOADS:-write_read:fillrandom,flushmem,compactl0,readrandom:write_only=fillrandom,read_only=readrandom balanced:fillrandom,flushmem,compactl0,balanced:balanced=balanced read_heavy:fillrandom,flushmem,compactl0,readheavy:read_heavy=readheavy}"
else
  echo "invalid DATA_MODE='${DATA_MODE}'; expected auto, real, or builtin" >&2
  exit 1
fi

CURRENT_DB_PATH=""
CURRENT_WORK_PATH=""

if [[ "${REUSE_LOGS}" != "1" && "${CLEAR_LOGS}" == "1" ]]; then
  find "${LOG_DIR}" -maxdepth 1 -type f -name "*.log" -delete
fi

extract_micros() {
  local benchmark="$1"
  local log="$2"
  awk -v b="${benchmark}" '
    {
      pattern = b "[[:space:]]*:[[:space:]]*[0-9]+([.][0-9]+)?"
      if (match($0, pattern)) {
        value = substr($0, RSTART, RLENGTH)
        sub("^" b "[[:space:]]*:[[:space:]]*", "", value)
        print value
        exit
      }
    }
  ' "${log}"
}

append_result_rows() {
  local engine="$1"
  local dataset="$2"
  local run="$3"
  local run_status="$4"
  local log_path="$5"
  local target_specs="$6"
  local default_workload="$7"
  local success_status="$8"

  local target_entries target_entry workload benchmark micros ops_per_sec result_status
  IFS=',' read -r -a target_entries <<< "${target_specs}"
  for target_entry in "${target_entries[@]}"; do
    if [[ "${target_entry}" == *=* ]]; then
      workload="${target_entry%%=*}"
      benchmark="${target_entry#*=}"
    else
      workload="${default_workload}"
      benchmark="${target_entry}"
    fi

    if [[ ! -f "${log_path}" ]]; then
      echo "${engine},${dataset},${workload},${run},${benchmark},,,missing_log,${run_status},${log_path}" >> "${SUMMARY}"
      continue
    fi

    micros="$(extract_micros "${benchmark}" "${log_path}")"
    if [[ -z "${micros}" ]]; then
      if [[ "${run_status}" != "" && "${run_status}" -ne 0 ]]; then
        result_status="failed"
        echo "failed: ${benchmark} in ${log_path}" >&2
      else
        result_status="missing_result"
        echo "missing result for ${benchmark}: ${log_path}" >&2
      fi
      echo "${engine},${dataset},${workload},${run},${benchmark},,,${result_status},${run_status},${log_path}" >> "${SUMMARY}"
      continue
    fi

    ops_per_sec="$(awk -v us="${micros}" 'BEGIN { if (us > 0) printf "%.6f", 1000000.0 / us; else print "" }')"
    result_status="${success_status}"
    if [[ "${run_status}" != "" && "${run_status}" -ne 0 ]]; then
      result_status="ok_nonzero_exit_${run_status}"
      echo "warning: ${benchmark} produced a result but exited with status ${run_status}: ${log_path}" >&2
    fi
    echo "${engine},${dataset},${workload},${run},${benchmark},${micros},${ops_per_sec},${result_status},${run_status},${log_path}" >> "${SUMMARY}"
  done
}

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

cleanup_empty_roots() {
  if [[ "${KEEP_DB}" != "1" ]]; then
    rmdir "${OUT_DIR}/db" 2>/dev/null || true
  fi
  if [[ "${KEEP_WORK}" != "1" ]]; then
    rmdir "${OUT_DIR}/work" 2>/dev/null || true
  fi
}

benchmarks_with_optional_stats() {
  local benchmarks="$1"
  local result="${benchmarks}"
  if [[ "${WAITBG}" == "1" && ",${result}," != *",waitbg,"* ]]; then
    result="${result},waitbg"
  fi
  if [[ "${INCLUDE_STATS}" == "1" ]]; then
    printf '%s,stats\n' "${result}"
  else
    printf '%s\n' "${result}"
  fi
}

all_target_results_ready() {
  local log_path="$1"
  local target_specs="$2"
  local benchmark_arg="$3"
  local target_entries target_entry benchmark

  if [[ ! -f "${log_path}" ]]; then
    return 1
  fi

  IFS=',' read -r -a target_entries <<< "${target_specs}"
  for target_entry in "${target_entries[@]}"; do
    if [[ "${target_entry}" == *=* ]]; then
      benchmark="${target_entry#*=}"
    else
      benchmark="${target_entry}"
    fi

    if [[ -z "$(extract_micros "${benchmark}" "${log_path}")" ]]; then
      return 1
    fi
  done

  if [[ ",${benchmark_arg}," == *",waitbg," ]] &&
     [[ -z "$(extract_micros "waitbg" "${log_path}")" ]]; then
    return 1
  fi

  return 0
}

run_db_bench_case() {
  local work_path="$1"
  local log_path="$2"
  local benchmark_arg="$3"
  local target_specs="$4"
  local dataset="$5"
  local db_path="$6"
  local engine_args="$7"
  local pid ready_since now

  if [[ "${ACTIVE_DATA_MODE}" == "real" ]]; then
    (
      cd "${work_path}" && \
      export WT_LOG_DIR="${LOG_DIR}" && \
      exec "${DB_BENCH}" \
        --benchmarks="${benchmark_arg}" \
        --dataset="${dataset}" \
        --dataset_size="${DATASET_SIZE}" \
        --num="${NUM}" \
        --value_size="${VALUE_SIZE}" \
        --path_real_data="${DATA_DIR}" \
        --db="${db_path}" \
        ${engine_args}
    ) > "${log_path}" 2>&1 &
  else
    (
      cd "${work_path}" && \
      export WT_LOG_DIR="${LOG_DIR}" && \
      exec "${DB_BENCH}" \
        --benchmarks="${benchmark_arg}" \
        --num="${NUM}" \
        --value_size="${VALUE_SIZE}" \
        --db="${db_path}" \
        ${engine_args}
    ) > "${log_path}" 2>&1 &
  fi

  pid=$!
  ready_since=0

  while kill -0 "${pid}" 2>/dev/null; do
    if all_target_results_ready "${log_path}" "${target_specs}" "${benchmark_arg}"; then
      now="$(date +%s)"
      if [[ "${ready_since}" -eq 0 ]]; then
        ready_since="${now}"
      elif (( now - ready_since >= EXIT_GRACE_SECONDS )); then
        echo "warning: db_bench produced required results but did not exit after ${EXIT_GRACE_SECONDS}s; terminating pid ${pid}: ${log_path}" >&2
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

if [[ "${REUSE_LOGS}" == "1" ]]; then
  for engine_entry in "${engines[@]}"; do
    engine="${engine_entry%%:*}"

    for dataset in "${datasets[@]}"; do
      for workload_entry in "${workloads[@]}"; do
        run_name="${workload_entry%%:*}"
        rest="${workload_entry#*:}"
        target_specs="${rest#*:}"

        for run in $(seq 1 "${RUNS}"); do
          log_path="${LOG_DIR}/${engine}_${dataset}_${run_name}_run${run}.log"
          if [[ ! -f "${log_path}" ]]; then
            append_result_rows "${engine}" "${dataset}" "${run}" "" "${log_path}" "${target_specs}" "${run_name}" "missing_log"
            continue
          fi

          append_result_rows "${engine}" "${dataset}" "${run}" "" "${log_path}" "${target_specs}" "${run_name}" "ok_reused"
        done
      done
    done
  done

  cleanup_empty_roots
  python3 "${SCRIPT_DIR}/plot_figure9_real_workloads.py" "${SUMMARY}" "${OUT_DIR}/figure9_real_workloads.png"
  echo "summary: ${SUMMARY}"
  echo "logs: ${LOG_DIR}"
  exit 0
fi

for engine_entry in "${engines[@]}"; do
  engine="${engine_entry%%:*}"
  engine_args="${engine_entry#*:}"

  for dataset in "${datasets[@]}"; do
    for workload_entry in "${workloads[@]}"; do
      run_name="${workload_entry%%:*}"
      rest="${workload_entry#*:}"
      benchmarks="${rest%%:*}"
      target_specs="${rest#*:}"

      for run in $(seq 1 "${RUNS}"); do
        db_path="${OUT_DIR}/db/${engine}_${dataset}_${run_name}_run${run}"
        log_path="${LOG_DIR}/${engine}_${dataset}_${run_name}_run${run}.log"
        work_path="${OUT_DIR}/work/${engine}_${dataset}_${run_name}_run${run}"
        CURRENT_DB_PATH="${db_path}"
        CURRENT_WORK_PATH="${work_path}"
        rm -rf "${db_path}"
        rm -rf "${work_path}"
        mkdir -p "${work_path}"

        echo "=== ${engine} dataset=${dataset} workload=${run_name} run=${run} ==="
        benchmark_arg="$(benchmarks_with_optional_stats "${benchmarks}")"
        set +e
        run_db_bench_case "${work_path}" "${log_path}" "${benchmark_arg}" "${target_specs}" "${dataset}" "${db_path}" "${engine_args}"
        status=$?
        set -e

        append_result_rows "${engine}" "${dataset}" "${run}" "${status}" "${log_path}" "${target_specs}" "${run_name}" "ok"
        cleanup_case_outputs "${db_path}" "${work_path}"
        CURRENT_DB_PATH=""
        CURRENT_WORK_PATH=""
        drop_caches_if_requested
      done
    done
  done
done

cleanup_empty_roots
python3 "${SCRIPT_DIR}/plot_figure9_real_workloads.py" "${SUMMARY}" "${OUT_DIR}/figure9_real_workloads.png"
echo "summary: ${SUMMARY}"
echo "logs: ${LOG_DIR}"
