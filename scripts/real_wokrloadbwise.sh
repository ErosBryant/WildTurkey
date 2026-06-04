#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DB_BENCH="${DB_BENCH:-${BASE_DIR}/build/db_bench}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${BASE_DIR}/datasets/data}"

python3 "${SCRIPT_DIR}/test.py" testing start 

# Define the desired --num values in an array
nums=(64000000)
# nums=(200000000)
DATASET_SIZE="${DATASET_SIZE:-200M}"


# Define various configurations
# memtable_size=(4)
# max_file_size=(64 32 16 8 4 2)
number_of_runs=2
# bwise=(1 0)
# lacd=(1 2 3 4 5 6 7 8 9 10 15)
# file_error=(2 4 8 16 32)

datasets=(osm_cellids fb wiki_ts books)
# lac=(5)
mod=(1)
# file_error=(22)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define output directories
# output_dir="/mnt/lac-sec/ad-wt-bour/bourbon&wt-last/bourbon/"
output_dir="${OUT_DIR:-${BASE_DIR}/build/real_workload_bwise_runs/}"


# total_experiment="${BASE_DIR}/build/results/"



# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
   # mkdir -p "${output_dir}summary_results/"
fi

# if [ ! -d "$total_experiment" ]; then
#    mkdir -p "$total_experiment"
# fi

# Execute the db_bench command for each configuration and save results
for num in "${nums[@]}"; do
   # for max in "${max_file_size[@]}"; do
      for dataset in "${datasets[@]}"; do
            for md in "${mod[@]}"; do

               for i in $(seq 1 $number_of_runs); do

               output_file="${output_dir}bwise=${md}-dataset=${dataset}-num=${num}_${i}.csv"
               
               echo "Running db_bench with --num=$num --dataset=${dataset} --bwise=${md} " > "$output_file"
            #   --max_file_size=${max}
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
               "${DB_BENCH}" --benchmarks="fillrandom_r,readrandom_r,stats" --bwise=$md --num=$num --dataset="${dataset}" --dataset_size="${DATASET_SIZE}" --path_real_data="${REAL_DATA_DIR}" >> "$output_file"
               echo "-------------------------------------" >> "$output_file"

               sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
            done
         
         done
      done
   # done
done


# workload_b=(osm_blanced fb_blanced wiki_blanced book_blanced)

# for num in "${nums[@]}"; do
#    for md in "${mod[@]}"; do
#       for wkload in "${workload_b[@]}"; do
#          for i in $(seq 1 $number_of_runs); do

#                   output_file="${output_dir}mod=${md}-wkload=${wkload}-num=${num}_${i}.csv"
                  
#                   echo "Running db_bench with --num=$num -wkload=${wkload} --mod=${md} " > "$output_file"

#                   # Run the benchmark
#                   # uni40,uniread,stats
#                   # osm_w,real_r,stats
#                   # fillrandom,readrandom
#                   # --lac=$lacd 
#                   # --bwise=$bw
#                   # --max_file_size=$max
#                   # --lsize=${max/2} 
#                   # --file_error=$err
#                   # f=$((max / 2)) 
#                   # --lsize=$f
#                   "${DB_BENCH}" --benchmarks="${wkload},stats" --mod=$md --num=$num >> "$output_file"
#                   echo "-------------------------------------" >> "$output_file"


            
#                   sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
#          done
#       done  
#    done
# done



python3 "${SCRIPT_DIR}/test.py" testing end
