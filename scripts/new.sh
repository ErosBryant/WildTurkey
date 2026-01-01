
#!/bin/bash

python3 ./test.py testing start 

# Define the desired --num values in an array
nums=(200000)
number_of_runs=1
mod=(10 8 7 5)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define output directories
# output_dir="/mnt/lac-sec/ad-wt-bour/bourbon&wt-last/bourbon/"
output_dir="./wild_turkey_Test/"

test_dir="../build"



# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
mkdir -p "$output_dir"
fi

# if [ ! -d "$total_experiment" ]; then
#    mkdir -p "$total_experiment"
# fi

# Execute the db_bench command for each configuration and save results
for num in "${nums[@]}"; do
         for md in "${mod[@]}"; do
         # Initialize summary output file
      
            write_micros_list=()
            read_micros_list=()
            write_mb_list=()
            read_mb_list=()
            for i in $(seq 1 $number_of_runs); do
               output_file="${output_dir}mod=${md}_num=${num}_${i}.csv"
               
               echo "Running db_bench with --num=$num --mod=${md} " > "$output_file"

               dev=sdc         # 측정하려는 디바이스 이름
               sector_size=512 # 대부분 512B; 필요에 따라 확인

               # 실행 전 stat 읽기 (3열: 읽은 섹터 수, 7열: 쓴 섹터 수)
               reads_before=$(awk '{print $1}' /sys/block/$dev/stat)
               writes_before=$(awk '{print $5}' /sys/block/$dev/stat)

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
               ${test_dir}/db_bench --benchmarks="fillrandom,readrandom,stats" --mod=$md --num=$num >> "$output_file"
               echo "-------------------------------------" >> "$output_file"
               # fi

            
               

                  reads_after=$(awk '{print $1}' /sys/block/$dev/stat)
                  writes_after=$(awk '{print $5}' /sys/block/$dev/stat)

                  # 차분 → I/O 요청 횟수
                  delta_reads=$(( reads_after  - reads_before  ))
                  delta_writes=$(( writes_after - writes_before ))

                  echo "I/O 요청 횟수 → 읽기: $delta_reads, 쓰기: $delta_writes" >> "$output_file"

               # Extract performance data
               write_micros_per_op=$(grep "fillrandom" "$output_file" | awk '{for(i=1;i<=NF;i++) if($i=="micros/op;") print $(i-1)}')
               read_micros_per_op=$(grep "reads" "$output_file" | awk '{for(i=1;i<=NF;i++) if($i=="micros/op;") print $(i-1)}')
               write_mb_per_s=$(grep "fillrandom" "$output_file" | awk '{for(i=1;i<=NF;i++) if($i=="MB/s;") print $(i-1)}')
               read_mb_per_s=$(grep "reads" "$output_file" | awk '{for(i=1;i<=NF;i++) if($i=="MB/s") print $(i-1)}')
               # 99p=$(grep "99p" "$output_file" | awk '{for(i=1;i<=NF;i++) if($i=="MB/s") print $(i-1)}')
               waf=$(grep 'waf:' "$output_file" | awk -F':' '{print $2}')
               memtable_stall=$(grep 'memtable stall time' "$output_file" | awk '{print $(NF-1)}')
               l0_stall=$(grep 'L0 stall time' "$output_file" | awk '{print $(NF-1)}')
               l0_slow_stall=$(grep 'L0 slow stall time' "$output_file" | awk '{print $(NF-1)}')
               avg_segment_size=$(grep 'Average Segement Size' "$output_file" | awk '{print $NF}')
               
                        # 리스트에 성능 데이터 추가
               write_micros_list+=($write_micros_per_op)
               read_micros_list+=($read_micros_per_op)
               write_mb_list+=($write_mb_per_s)
               read_mb_list+=($read_mb_per_s)
               # Append data to summary output file
    

               # Clear system cache
               sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
            done

         
            avg_write_micros=$(echo "${write_micros_list[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            avg_read_micros=$(echo "${read_micros_list[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            avg_write_mb=$(echo "${write_mb_list[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            avg_read_mb=$(echo "${read_mb_list[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')


         done

done


python3 ./test.py testing end
