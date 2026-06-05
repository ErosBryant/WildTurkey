# Wild Turkey: Artifact for VLDB 2026 Submission

This repository provides the code, scripts, and configurations necessary to reproduce the experimental results of our VLDB 2026 submission titled *Wild Turkey*. 


This repository includes:
- Source code implementing **Wild Turkey**, including Level-Aware Compaction (LAC) and Wild-Learning.
- Benchmark scripts for comparison with **WiscKey**, and **Bourbon**.
- Configuration and usage instructions to reproduce all key results.

---

## 🔧 Build Instructions

To compile the benchmark binary:


```bash
git clone https://github.com/ErosBryant/WildTurkey.git
cd WildTurkey
```

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

---

## ▶️ Running Benchmarks

Run the following commands from the repository root after building `build/db_bench`.

Notice: For faster experiments, the number of benchmark entries is set lower than in the paper.

### Option 1: Quick Test

```bash
bash scripts/WT_test.sh
```

This runs Wild Turkey, WiscKey, Bourbon, and vanilla LevelDB with the same benchmark mix (`fillrandom,readrandom,stats`). Logs are written under `vldb_runs/` by default.

Useful overrides:

```bash
NUM=<n> OUT_DIR=<path> BENCHMARKS=fillrandom,readrandom,stats bash scripts/WT_test.sh
```

### Option 2: Availability Check

```bash
bash scripts/run_figure_reproduction.sh
```

This runs the Figure 9, Figure 10, Figure 11, and Figure 12 reproduction scripts in order:

- `figure9_real_workloads.sh`: workload matrix throughput for Wild Turkey, WiscKey, and Bourbon.
- `figure10_read_breakdown.sh`: read-path breakdown after L0 compaction.
- `figure11_write_metrics.sh`: compaction count/time, write stall time, and write amplification.
- `figure12_model_training.sh`: model size and training-time comparison.

Outputs are written under `scripts/result_figure/` by default:

- `scripts/result_figure/figure9_real_workloads/`
- `scripts/result_figure/figure10_read_breakdown/`
- `scripts/result_figure/figure11_write_metrics/`
- `scripts/result_figure/figure12_model_training/`

Common overrides:

```bash
NUM=<n> RUNS=<n> DATASET_SIZE=<size> DATASETS="osm_cellids fb" \
ENGINES="wildturkey:--mod=10 wisckey:--mod=8 bourbon:--mod=7" \
OUT_DIR=<output-root> DATA_MODE=auto DROP_CACHES=1 \
bash scripts/run_figure_reproduction.sh
```

`DATA_MODE=auto` is the default: scripts use real datasets when all requested files exist under `datasets/data/`; otherwise they fall back to db_bench built-in random keys and still generate CSV files and PNG plots. Use `DATA_MODE=real` to require real datasets or `DATA_MODE=builtin` to force built-in random-key mode.

`OUT_DIR` is treated as the output root by `run_figure_reproduction.sh`; each figure script writes into its own subdirectory under that root. `DROP_CACHES=1` runs `sync` and drops the Linux page cache after each case when running as root or when passwordless `sudo` is available; otherwise the script prints a warning and continues. `CLEAR_LOGS=1` is enabled by default and removes old `.log` files in the selected log directory before a fresh run. `KEEP_DB=0` is the default, so per-case DB directories are deleted after each benchmark to avoid leaving large `.db` directories on disk; use `KEEP_DB=1` only when debugging.

### Option 3: Manual Benchmark Commands

```bash
cd build

# Wild Turkey
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=10 > ../WildTurkey_test.log

# WiscKey
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=8 > ../WiscKey_test.log

# Bourbon
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=7 > ../Bourbon_test.log

# Vanilla LevelDB
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=0 > ../LevelDB_test.log

```
<!-- 
### Run YCSB via Script 

```bash
cd scripts
sudo bash ycsb.sh
``` -->



> 🔸 The `--mod` flag selects the storage engine mode:
> - `10`: Wild Turkey
> - `8`: WiscKey
> - `7`: Bourbon
> - `0`: vanilla LevelDB

For real-world SOSD-style workloads, run `bash datasets/download.sh` and keep the generated binary key files under `datasets/data/`, or pass `--path_real_data=/path/to/data`. Use `fillrandom_r,readrandom_r` with explicit `--dataset`, `--dataset_size`, and `--num`, for example:

```bash
./build/db_bench --benchmarks=fillrandom_r,readrandom_r,stats \
  --dataset=osm_cellids --dataset_size=200M --num=64000000 \
  --path_real_data=datasets/data --mod=10
```

Expected real dataset names are `osm_cellids`, `books`, `fb`, and `wiki_ts`. The real-data benchmark commands also support `balanced_r` and `readheavy_r` for 50% read / 50% write and 90% read / 10% write workloads. For the figure reproduction wrapper above, `DATA_MODE=auto` is the default: scripts use real datasets when all requested files exist, otherwise they fall back to db_bench built-in random keys and still generate CSV files and PNG plots. Use `DATA_MODE=real` to require real datasets or `DATA_MODE=builtin` to force the built-in random-key mode.

---

## ⏱️ Profiling Timers (Custom Stats)

The benchmark output includes internal timers for detailed profiling. Below are key timer IDs:

| Timer ID | Description (as implemented)                                           |
|----------|-------------------------------------------------------------------------|
| 0        | File candidate search per level (Version::Get find-file step)          |
| 1        | Open SSTable (TableCache::FindTable, file open + Table::Open)          |
| 2        | Seek index block                                                       |
| 3        | Seek inside data block                                                 |
| 5        | Load data block                                                        |
| 6        | Per-file key search total (wrapper around table cache Get)            |
| 7        | Compaction time                                                        |
| 8        | Learned index work: file model load during read + level-learn time     |
| 10       | Put path (fresh write) total                                           |
| 11       | File-model training time                                               |
| 12       | Value log read (WiscKey/Bourbon paths)                                 |
| 13       | Read path total (table access after mem/imm)                           |
| 14       | MemTable/ImmTable lookup                                               |
| 15       | FilteredLookup time (Bloom)                                            |
| 16       | Memtable flush/CompactMemTable                                         |
| 17       | Model prediction / range calculation                                   |
| 18       | Load predicted chunk + binary search to locate key                     |


---

## 📎 Notes

- Benchmarks match configurations in Section V of the paper.
- Keys are 16 bytes and values are 100 bytes unless otherwise noted.
- All experiments are intended to be run on SSD-backed storage; default configurations assume SATA, NVMe, or Optane drives.
