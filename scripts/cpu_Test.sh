#!/bin/bash

python3 ./test.py testing start 

# Define the desired --num values in an array
nums=(20000000)
mod=(7 8)
number_of_runs=1

current_time=$(date "+%Y%m%d-%H%M%S")
output_dir="/mnt/analysis_bourbon/cpu/"
test_dir="/home/eros/workspace-lsm/wildturkey/build/"

# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
   mkdir -p "${output_dir}summary_results/"
fi

cpu_log_file="${output_dir}cpu_usage_${current_time}.log"
echo "Time,Mod,CPU_Usage" > "$cpu_log_file"
declare -A start_times

for num in "${nums[@]}"; do
   for md in "${mod[@]}"; do
   start_times[$md]=$(date +%s)
      for i in $(seq 1 $number_of_runs); do
         output_file="${output_dir}mod=${md}_zip2-num=${num}_${i}.csv"
         echo "Running db_bench with --num=$num --mod=${md}" > "$output_file"

         
         # fillrandom,readrandom,stats
         ${test_dir}/db_bench --benchmarks="zipwrite,zipread,stats" --mod=$md  --num=$num > "$output_file" &
         db_bench_pid=$!
      

         while ps -p $db_bench_pid > /dev/null; do
            echo "$(date +%s),$md,$(ps -p $db_bench_pid -o %cpu --no-headers)" >> "$cpu_log_file"
            sleep 1
         done

         while ps -p $db_bench_pid > /dev/null; do
            cpu_usage=$(ps -p $db_bench_pid -o %cpu --no-headers)
            if [[ -z "$cpu_usage" ]]; then
                cpu_usage="N/A"
            fi
            elapsed_time=$(( $(date +%s) - start_times[$md] ))
            echo "$elapsed_time,$md,$cpu_usage" >> "$cpu_log_file"
            sleep 1
         done



         echo "-------------------------------------" >> "$output_file"
         sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
      done
   done
done

# Generate CPU usage graph
python3 - <<EOF
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행하도록 설정
import matplotlib.pyplot as plt

cpu_data = pd.read_csv("$cpu_log_file")

# Normalize time so that each mod starts at 0
cpu_data['Time'] = cpu_data.groupby('Mod')['Time'].transform(lambda x: x - x.min())

plt.figure(figsize=(10, 5))

# Plot each mod separately
for mod in cpu_data["Mod"].unique():
    subset = cpu_data[cpu_data["Mod"] == mod]
    plt.plot(subset["Time"], subset["CPU_Usage"], marker='o', linestyle='-', label=f'Mod {mod}')

plt.xlabel("Elapsed Time (seconds)")
plt.ylabel("CPU Usage (%)")
plt.title("LevelDB CPU Usage Over Time (Comparison of Mods)")
plt.legend()
plt.grid()
plt.savefig("${output_dir}cpu_usage_${current_time}.png")  # GUI 없이 파일로 저장
EOF

python3 ./test.py testing end
