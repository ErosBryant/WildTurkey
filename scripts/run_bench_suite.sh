#!/usr/bin/env bash
set -euo pipefail

# Simple runner to compare multiple db_bench configs.
NUM="${NUM:-20000000}"
BENCHMARKS="${BENCHMARKS:-fillrandom,readrandom,stats}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
# Store outputs under a dedicated directory (not the db_bench binary path).
OUT_DIR="${OUT_DIR:-${BASE_DIR}/build/db_bench_runs}"

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "db_bench not found/executable at ${DB_BENCH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

cases=(
  "고정eb=--mod=7"
  "고정lac=--lac=4"
  "동적eb=--mod=7 --adeb=1"
  "동적lac=--mod=10"
  "wildturky=--mod=10 --adeb=1"
)

for entry in "${cases[@]}"; do
  name="${entry%%=*}"
  args="${entry#*=}"
  db_path="${OUT_DIR}/${name}1"
  log_path="${OUT_DIR}/${name}1.log"

  echo "=== Running ${name} -> ${args} (db: ${db_path}) ==="
  rm -rf "${db_path}"
  "${DB_BENCH}" \
    --benchmarks="${BENCHMARKS}" \
    --num="${NUM}" \
    --db="${db_path}" \
    ${args} | tee "${log_path}"
done

echo "Done. Logs in ${OUT_DIR}"
