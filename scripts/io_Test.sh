#!/bin/bash

python3 ./test.py testing start 

# Define the desired --num values in an array
nums=(10000000)
mod=(7 8)
number_of_runs=1

current_time=$(date "+%Y%m%d-%H%M%S")
output_dir="/mnt/analysis_bourbon/io/"
test_dir="/home/eros/workspace-lsm/wildturkey/build/"

# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
   mkdir -p "${output_dir}summary_results/"
fi

io_log_file="${output_dir}io_usage_${current_time}.log"
echo "Time,Mod,KB_Read/s,KB_Written/s" > "$io_log_file"
declare -A start_times

for num in "${nums[@]}"; do
   for md in "${mod[@]}"; do
      start_times[$md]=$(date +%s)
      for i in $(seq 1 $number_of_runs); do
         output_file="${output_dir}mod=${md}-num=${num}_${i}.csv"
         echo "Running db_bench with --num=$num --mod=${md}" > "$output_file"

         # fillrandom,readrandom,stats
         ${test_dir}/db_bench --benchmarks="fillrandom,readrandom,stats" --mod=$md  --num=$num > "$output_file" &
         db_bench_pid=$!
      
         while ps -p $db_bench_pid > /dev/null; do
            io_stats=$(iostat -d -k 1 1 | grep sdc | awk '{print $3 "," $4}')  # 쉼표(,)로 구분
            if [[ -z "$io_stats" ]]; then
               io_stats="0,0"
            fi
            elapsed_time=$(( $(date +%s) - start_times[$md] ))
            echo "$elapsed_time,$md,$io_stats" >> "$io_log_file"
            sleep 1
         done


         echo "-------------------------------------" >> "$output_file"
         sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
      done
   done
done

# Generate IO Bandwidth graph
python3 - <<EOF
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행하도록 설정
import matplotlib.pyplot as plt

io_data = pd.read_csv("$io_log_file")

# Normalize time so that each mod starts at 0
io_data['Time'] = io_data.groupby('Mod')['Time'].transform(lambda x: x - x.min())

plt.figure(figsize=(10, 5))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # Create second y-axis

# Plot Read Bandwidth on left y-axis
for mod in io_data["Mod"].unique():
    subset = io_data[io_data["Mod"] == mod]
    ax1.plot(subset["Time"], subset["KB_Read/s"].astype(float), marker='o', linestyle='-', label=f"Read Mod {mod}", color='blue')

# Plot Write Bandwidth on right y-axis
for mod in io_data["Mod"].unique():
    subset = io_data[io_data["Mod"] == mod]
    ax2.plot(subset["Time"], subset["KB_Written/s"].astype(float), marker='x', linestyle='--', label=f"Write Mod {mod}", color='red')

ax1.set_xlabel("Elapsed Time (seconds)")
ax1.set_ylabel("Read IO Bandwidth (KB/s)", color='blue')
ax2.set_ylabel("Write IO Bandwidth (KB/s)", color='red')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
plt.title("LevelDB IO Bandwidth Over Time (Comparison of Mods)")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid()
plt.savefig("${output_dir}io_usage_${current_time}.png")  # GUI 없이 파일로 저장
EOF

python3 ./test.py testing end
