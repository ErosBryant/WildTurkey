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
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

---

## ▶️ Running Benchmarks

### Option 1: Run All Tests via Script (recommended)

```bash
cd scripts
sudo bash WT_test.sh
```

The script runs Wild Turkey, WiscKey, Bourbon, and vanilla LevelDB with the same benchmark mix (`fillrandom,readrandom,stats`), drops any previous test DBs, and saves logs under `build/vldb_runs` by default.

### Option 2: Manual Benchmark Commands

```bash
cd build

# Wild Turkey
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=10 > WildTurkey_test.log

# WiscKey
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=8 > WiscKey_test.log

# Bourbon
./db_bench --benchmarks="fillrandom,readrandom,stats" --num=20000000 --mod=7 > Bourbon_test.log

```

> 🔸 The `--mod` flag selects the storage engine mode:
> - `10`: Wild Turkey
> - `8`: WiscKey
> - `7`: Bourbon

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
- This artifact includes no author-identifiable metadata and is fully anonymized.

---

This artifact is provided solely for the purpose of anonymous review. A public release will follow upon paper acceptance.
