#!/bin/bash
set -euo pipefail

# Path handling mirrors scripts/WT_test.sh to avoid hardcoded locations.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/ycsb_runs}"

python3 ./test.py testing start

# Params
nums=(10000000)
mods=(10)
number_of_runs=1

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "db_bench not found or not executable at ${DB_BENCH}" >&2
  echo "Build first (see README.md), or set DB_BENCH to an existing binary." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

drop_caches() {
  # Optional: add sync for safety
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

# --- Uniform runs (explicit --uni=1) ---
for num in "${nums[@]}"; do
  for md in "${mods[@]}"; do
    for i in $(seq 1 "$number_of_runs"); do

      out="${OUT_DIR}/mod=${md}_uni=1_num=${num}_run=${i}.txt"
      {
        echo "Running db_bench (UNIFORM) --num=${num} --mod=${md} --uni=1"
        "${DB_BENCH}" \
          --benchmarks="fillseq,ycsba,ycsbb,ycsbc,ycsbd,ycsbe,ycsbf,stats" \
          --mod="${md}" \
          --num="${num}" \
          --uni=1
        echo "-------------------------------------"
      } > "${out}"

# --max_file_size="${max_file}" \
      drop_caches
    done
  done
done

# --- Zipf runs (explicit --uni=0) ---
for num in "${nums[@]}"; do
  for md in "${mods[@]}"; do
    for i in $(seq 1 "$number_of_runs"); do

      out="${OUT_DIR}/mod=${md}_zip=1_num=${num}_run=${i}.txt"
      {
        echo "Running db_bench (UNIFORM) --num=${num} --mod=${md} --zip=1"
        "${DB_BENCH}" \
          --benchmarks="fillseq,ycsba,ycsbb,ycsbc,ycsbd,ycsbe,ycsbf,stats" \
          --mod="${md}" \
          --num="${num}" \
          --uni=0
        echo "-------------------------------------"
      } > "${out}"

         # --max_file_size="${max_file}" \
      drop_caches
    done
  done
done
python3 ./test.py testing end
