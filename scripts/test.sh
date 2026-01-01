
python3 ./test.py testing start 
# Define the desired --num values in an array
nums=(1000000)

# error_bound=(2)
# 2 4 8 16 32

number_of_runs=1

value_size=(100)
# sst_size=(1 2 3 4 8 16 32)
# sst_size=(2)

# 1 2 3 
sst_base_size=(2)

mods=(7)

# workload=(fb_w)

# use_bwises=(0)
# Define output directories

current_time=$(date "+%Y%m%d-%H%M%S")

output_dir="/home/eros/workspace-lsm/wildturkey/test_result/$current_time/"
test_dir="/home/eros/workspace-lsm/wildturkey/build/"
db_data_dir="/home/eros/workspace-lsm/wildturkey/db_dir"

if [ ! -d "$db_data_dir" ]; then
   mkdir -p "$db_data_dir"
fi

# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
fi


# Execute the db_bench command for each configuration and save results
for num in "${nums[@]}"; do
   for mod in "${mods[@]}"; do
      for value in "${value_size[@]}"; do
         for sst_base in "${sst_base_size[@]}"; do
            for i in $(seq 1 $number_of_runs); do

               # db_data_dir="/home/eros/workspace-lsm/wildturkey/db_dir_${i}"

               # if [ ! -d "$db_data_dir" ]; then
               #    mkdir -p "$db_data_dir"
               # fi

               # # Clean up any leftover lock file
               # if [ -f "$db_data_dir/LOCK" ]; then
               #    rm -f "$db_data_dir/LOCK"
               # fi


             # Create a new cgroup for each db_bench run
               cg_name="my_cgroup_${num}_${mod}_${i}"
               # sudo mkdir /sys/fs/cgroup/unified/${cg_name}
              if [ ! -d "/sys/fs/cgroup/${cg_name}" ]; then
                  sudo mkdir /sys/fs/cgroup/${cg_name}
               # else
               #    echo "Cgroup directory ${cg_name} already exists." >> "$output_file"
               fi

               
               # Use systemd-run to create a new transient cgroup for db_bench run
               output_file="${output_dir}_${sst_base}SST_${num}_${mod}_${i}.txt"
               echo "Running db_bench with --num=$num  --mod=$mod --times=$i --sst_base=${sst_base} --value_size=${value}" >> "$output_file"
               
# --db=$db_data_dir
               # Run db_bench with resource limitations via systemd
               # sudo systemd-run --unit=db_bench_run_${i} --scope -p CPUQuota=50% -p MemoryMax=512M \
               ${test_dir}/db_bench --benchmarks="fillrandom,readrandom,stats" \
               --max_file_size=${sst_base} --value_size=$value --mod=$mod --num=$num >> "$output_file" &
               db_bench_pid=$!
                # ${test_dir}/db_bench --benchmarks="${rwork},real_r,stats"  --max_file_size=${sst_base} --mod=$mod --num=$num --db=$db_data_dir >> "$output_file"
               

               # Check if cgroup was created
               if [ -d "/sys/fs/cgroup/${cg_name}" ]; then
                  echo "Cgroup ${cg_name} created successfully" >> "$output_file"
               else
                  echo "Failed to create Cgroup ${cg_name}" >> "$output_file"
               fi

               # Add db_bench process to the cgroup
               echo $db_bench_pid | sudo tee /sys/fs/cgroup/${cg_name}/cgroup.procs

               # Wait for db_bench to finish
               wait $db_bench_pid

               # Monitor CPU and memory usage once after the db_bench execution
               if [ -d "/sys/fs/cgroup/${cg_name}" ]; then
                  echo "CPU usage:" >> "$output_file"
                  cat /sys/fs/cgroup/${cg_name}/cpu.stat >> "$output_file"

                  echo "Memory usage:" >> "$output_file"
                  cat /sys/fs/cgroup/${cg_name}/memory.current >> "$output_file"
               else
                  echo "Cgroup directory does not exist." >> "$output_file"
               fi

               echo "-------------------------------------" >> "$output_file"

               # Clean up cgroup
               sudo rmdir /sys/fs/cgroup/${cg_name}

               sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

            done
         done
      done
   done
done

python3 ./test.py testing end



# 누적이 아닌 증가량을 측정하는 방법
# prev_cpu_usage=0
# prev_memory_usage=0

# while ps -p $db_bench_pid > /dev/null; do
#    if [ -d "/sys/fs/cgroup/${cg_name}" ]; then
#       # 获取当前的 CPU 和内存使用情况
#       current_cpu_usage=$(cat /sys/fs/cgroup/${cg_name}/cpu.stat | grep usage_usec | awk '{print $2}')
#       current_memory_usage=$(cat /sys/fs/cgroup/${cg_name}/memory.current)

#       # 计算增量
#       cpu_usage_diff=$((current_cpu_usage - prev_cpu_usage))
#       memory_usage_diff=$((current_memory_usage - prev_memory_usage))

#       # 输出增量
#       echo "CPU usage increment: $cpu_usage_diff" >> "$output_file"
#       echo "Memory usage increment: $memory_usage_diff" >> "$output_file"

#       # 更新上次的使用值
#       prev_cpu_usage=$current_cpu_usage
#       prev_memory_usage=$current_memory_usage
#    fi

#    echo "-------------------------------------" >> "$output_file"

#    # Sleep for a short interval before checking again
#    sleep 1
# done



# -------------------------------------
# 몇 초마다 측정할지 결정하는 방법
               # Monitor CPU and memory usage during the db_bench execution
               # while ps -p $db_bench_pid > /dev/null; do
               #    if [ -d "/sys/fs/cgroup/${cg_name}" ]; then
               #       echo "CPU usage:" >> "$output_file"
               #       cat /sys/fs/cgroup/${cg_name}/cpu.stat >> "$output_file"

               #       echo "Memory usage:" >> "$output_file"
               #       cat /sys/fs/cgroup/${cg_name}/memory.current >> "$output_file"
               #    else
               #       echo "Cgroup directory does not exist." >> "$output_file"
               #    fi

               #    echo "-------------------------------------" >> "$output_file"

               #    # Sleep for a short interval before checking again
               #    sleep 1
               # done