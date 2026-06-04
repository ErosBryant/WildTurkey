#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"

python3 "${SCRIPT_DIR}/test.py" testing start 

# Define the desired --num values in an array
# nums=(60000)
nums=(20000000)

# lac=(5)
mod=(7)
number_of_runs=3
# file_error=(22)

file_error=(2 4 8 16 32 64 128 256 512)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define output directories
# output_dir="/mnt/lac-sec/ad-wt-bour/bourbon&wt-last/bourbon/"
output_dir="${OUT_DIR:-${BASE_DIR}/build/model_size_runs/}"


# total_experiment="/mnt/1tb/lac_experiment/"



# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
fi

# if [ ! -d "$total_experiment" ]; then
#    mkdir -p "$total_experiment"
# fi


for num in "${nums[@]}"; do

   # for md in "${mod[@]}"; do
   for err in "${file_error[@]}"; do

      for i in $(seq 1 $number_of_runs); do

               # output_file="${output_dir}mod=${md}-uni-num=${num}_${i}.txt"
               output_file="${output_dir}error=${err}-uni-fix-num=${num}_${i}.txt"
               
               echo "Running db_bench with --num=$num  --error=${err} " > "$output_file"

               # Run the benchmark
               # uni40,uniread,stats
               # osm_w,real_r,stats
               # fillrandom,readrandom
               # --lac=$lacd 
               # --bwise=$bw
               # --max_file_size=$max
               # --lsize=${max/2} 
               # --file_error=$err
               # f=$((max / 2)) 
               # --lsize=$f
               "${DB_BENCH}" --benchmarks="fillrandom,readrandom,stats" --mod=7 --file_error=$err --num=$num >> "$output_file"
               echo "-------------------------------------" >> "$output_file"

               sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
      done

   done
done


# # Execute the db_bench command for each configuration and save results
# for num in "${nums[@]}"; do

#    for md in "${mod[@]}"; do

#       for i in $(seq 1 $number_of_runs); do

#          output_file="${output_dir}mod=${md}-zip-num=${num}_${i}.txt"
               
#          echo "Running db_bench with --num=$num  --mod=${md} " > "$output_file"


#            "${DB_BENCH}" --benchmarks="fb_w,zipread,stats" --mod=7 --num=$num >> "$output_file"
#          echo "-------------------------------------" >> "$output_file"

#          sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
#       done

#    done
# done



python3 "${SCRIPT_DIR}/test.py" testing end
