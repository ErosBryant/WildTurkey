#!/usr/bin/env bash
set -euo pipefail

# One-touch runner for VLDB artifact evaluation.
# Executes the common benchmark mix for Wild Turkey, WiscKey, Bourbon, and vanilla LevelDB.

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/vldb_runs}"

BENCHMARKS="${BENCHMARKS:-fillrandom,readrandom,stats}"
NUM="${NUM:-20000000}"
VALUE_SIZE="${VALUE_SIZE:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "db_bench not found or not executable at ${DB_BENCH}" >&2
  echo "Build first (see README.md), or set DB_BENCH to an existing binary." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" 

cases=(
  "wildturkey:--mod=10"
  "wisckey:--mod=8"
  # "bourbon:--mod=7"
  # "leveldb:--mod=5"
)

for entry in "${cases[@]}"; do
  IFS=":" read -r name args <<<"${entry}"
  log_path="${OUT_DIR}/${name}.log"

  echo "=== Running ${name} (${args}) ==="

  "${DB_BENCH}" \
    --benchmarks="${BENCHMARKS}" \
    --num="${NUM}" \
    --value_size="${VALUE_SIZE}" \
    ${args} ${EXTRA_ARGS} | tee "${log_path}"

  echo "Log saved to ${log_path}"
done

echo "All runs completed. Logs are under ${OUT_DIR}"
