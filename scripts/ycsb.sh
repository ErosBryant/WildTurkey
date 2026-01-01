#!/bin/bash
set -euo pipefail

python3 ./test.py testing start

# Params
nums=(100000000)
mods=(8)
# max_files=(8)
lac=(4)
number_of_runs=1

output_dir="/home/eros/workspace-lsm/wicdee/KCC/ycsb/"
test_dir="/home/eros/workspace-lsm/wildturkey/build"

mkdir -p "$output_dir"

drop_caches() {
  # Optional: add sync for safety
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

# --- Uniform runs (explicit --uni=1) ---
for num in "${nums[@]}"; do
  for md in "${mods[@]}"; do
   #  for max_file in "${max_files[@]}"; do
    for lac_experiment in "${lac[@]}"; do
      for i in $(seq 1 "$number_of_runs"); do

        out="${output_dir}mod=${md}_uni=1_lac_experiment=${lac_experiment}_num=${num}_run=${i}.txt"
        {
          echo "Running db_bench (UNIFORM) --num=${num} --mod=${md} --lac_experiment=${lac_experiment} --uni=1"
          "${test_dir}/db_bench" \
            --benchmarks="fillseq,ycsba,ycsbb,ycsbc,ycsbd,ycsbe,ycsbf,stats" \
            --mod="${md}" \
            --num="${num}" \
            --lac="${lac_experiment}" \
            --uni=1
          echo "-------------------------------------"
        } > "${out}"

# --max_file_size="${max_file}" \
        drop_caches
      done
    done
  done
done

# --- Zipf runs (explicit --uni=0) ---
for num in "${nums[@]}"; do
  for md in "${mods[@]}"; do
   #  for max_file in "${max_files[@]}"; do
    for lac_experiment in "${lac[@]}"; do
      for i in $(seq 1 "$number_of_runs"); do

        out="${output_dir}mod=${md}_zip=1_lac_experiment=${lac_experiment}_num=${num}_run=${i}.txt"
        {
          echo "Running db_bench (UNIFORM) --num=${num} --mod=${md} --lac_experiment=${lac_experiment} --zip=1"
          "${test_dir}/db_bench" \
            --benchmarks="fillseq,ycsba,ycsbb,ycsbc,ycsbd,ycsbe,ycsbf,stats" \
            --mod="${md}" \
            --num="${num}" \
            --lac="${lac_experiment}" \
            --uni=0
          echo "-------------------------------------"
        } > "${out}"

         # --max_file_size="${max_file}" \
        drop_caches
      done
    done
  done
done
python3 ./test.py testing end
