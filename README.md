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

### Option 1: Run All Tests via Script

```bash
bash scripts/WT_test.sh
```

The script runs Wild Turkey, WiscKey, Bourbon, and vanilla LevelDB with the same benchmark mix (`fillrandom,readrandom,stats`), drops any previous test DBs, and saves logs under `build/vldb_runs` by default.

### Availability Check

For a quick artifact smoke check:

```bash
bash scripts/check_artifact.sh
```

This runs a small workload for Wild Turkey, WiscKey, Bourbon, and vanilla LevelDB and verifies that benchmark statistics are emitted. Use `NUM=<n>` and `OUT_DIR=<path>` to override the default workload size and output directory.

### Option 2: Manual Benchmark Commands

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
### Option 3: Run YCSB via Script 

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

Expected real dataset names are `osm_cellids`, `books`, `fb`, and `wiki_ts`. The real-data benchmark commands also support `balanced_r` and `readheavy_r` for 50% read / 50% write and 90% read / 10% write workloads. For scripts below, `DATA_MODE=auto` is the default: scripts use real datasets when all requested files exist, otherwise they fall back to db_bench built-in random keys and still generate CSV files and PNG plots. Use `DATA_MODE=real` to require real datasets or `DATA_MODE=builtin` to force the built-in random-key mode.

### Figure Reproduction Scripts

To reproduce the Figure 9 style workload matrix for Wild Turkey, WiscKey, and Bourbon across the four datasets and workload mixes, run:

```bash
bash scripts/figure9_real_workloads.sh
```

The script reports write-only, read-only, balanced, and read-heavy throughput. It writes raw logs under `build/figure9_real_workloads/logs/`, per-case working directories under `build/figure9_real_workloads/work/`, and `summary.csv` plus `figure9_real_workloads.png` under `build/figure9_real_workloads/`. Per-case DB directories are deleted immediately after each benchmark by default; use `KEEP_DB=1` only when debugging.

To reproduce the Figure 10 style read breakdown, run:

```bash
bash scripts/figure10_read_breakdown.sh
```

The default compares WiscKey, Bourbon, and Wild Turkey on `osm_cellids` and `fb` with `NUM=20000000`. Each case runs fill, `flushmem`, `compactl0`, `waitbg`, read, and `stats` in one db_bench process, so the read breakdown is collected after L0 SSTables are compacted without closing and reopening the DB. Outputs are written under `build/figure10_read_breakdown/`, including `cases.csv`, `summary.csv`, `figure10_read_breakdown.png`, and raw logs.

To reproduce the Figure 11 style write-side metrics, run:

```bash
bash scripts/figure11_write_metrics.sh
```

The default uses the OSM dataset (`osm_cellids`) or built-in random data if OSM is unavailable, and compares Wild Turkey, WiscKey, and Bourbon with `NUM=60000000`. It produces two plots under `build/figure11_write_metrics/`: `figure11_compaction.png` for compaction count/time and `figure11_stall_waf.png` for write stall time and write amplification. It also writes `cases.csv`, `summary.csv`, and logs.

To reproduce the model-size and training-time comparison for Bourbon and Wild Turkey, run:

```bash
bash scripts/figure12_model_training.sh
```

The default uses `osm_cellids` or built-in random data if unavailable, compares Bourbon with `--always_learn=1` against Wild Turkey, and uses `NUM=20000000`. It plots live model size and index training time under `build/model_training/` as `model_size.png` and `training_time.png`. Wild Turkey's plotted training time includes the model training time reported by `ModelTrainingStats` plus the measured RL component.

Common overrides for the figure scripts are:

```bash
NUM=<n> RUNS=<n> DATASET_SIZE=<size> DATASETS="osm_cellids fb" \
ENGINES="wildturkey:--mod=10 wisckey:--mod=8 bourbon:--mod=7" \
OUT_DIR=<path> DATA_MODE=auto DROP_CACHES=1 bash scripts/figure9_real_workloads.sh
```

`DROP_CACHES=1` runs `sync` and drops the Linux page cache after each case when running as root or when passwordless `sudo` is available; otherwise the script prints a warning and continues. `CLEAR_LOGS=1` is enabled by default and removes old `.log` files in the selected log directory before a fresh run. `KEEP_DB=0` is the default, so per-case DB directories are deleted after each benchmark to avoid leaving large `.db` directories on disk.

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
