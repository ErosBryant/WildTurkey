#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/build/artifact_check}"
NUM="${NUM:-100000}"
VALUE_SIZE="${VALUE_SIZE:-100}"
BENCHMARKS="${BENCHMARKS:-fillrandom,readrandom,stats}"

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "db_bench not found or not executable at ${DB_BENCH}" >&2
  echo "Build first: mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

cases=(
  "wildturkey:--mod=10"
  "wisckey:--mod=8"
  "bourbon:--mod=7"
  "leveldb:--mod=5"
)

for entry in "${cases[@]}"; do
  name="${entry%%:*}"
  args="${entry#*:}"
  db_path="${OUT_DIR}/${name}_db"
  log_path="${OUT_DIR}/${name}.log"
  rm -rf "${db_path}"
  echo "=== artifact check: ${name} (${args}) ==="
  WT_LOG_DIR="${OUT_DIR}" "${DB_BENCH}" \
    --benchmarks="${BENCHMARKS}" \
    --num="${NUM}" \
    --value_size="${VALUE_SIZE}" \
    --db="${db_path}" \
    ${args} > "${log_path}"
  grep -q "Compactions" "${log_path}"
  grep -q "Timer 0" "${log_path}"
  echo "ok: ${log_path}"
done

echo "Artifact smoke check completed. Logs are under ${OUT_DIR}"
