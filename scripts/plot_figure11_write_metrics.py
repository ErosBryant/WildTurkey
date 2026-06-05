#!/usr/bin/env python3
import csv
import os
import re
import sys
from collections import defaultdict


STALL_RE = re.compile(r"^(memtable stall time|L0 stall time|L0 slow stall time):\s*([0-9.]+)")
RAW_SIZE_RE = re.compile(r"^RawSize:\s*([0-9.]+)\s+MB")
COMPACTION_WA_RE = re.compile(r"^Total compaction WA:\s*([0-9.]+)")
DISK_WRITE_RE = re.compile(r"^Write data in disk:\s*([0-9.]+)")
COMPACTION_RE = re.compile(
    r"^\s*(\d+)\s+"
    r"(\d+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9]+)\s+"
    r"([0-9.]+)"
)


STALL_NAMES = {
    "memtable stall time": "memtable",
    "L0 stall time": "l0_stop",
    "L0 slow stall time": "l0_slow",
}

ENGINE_ORDER = ["wisckey", "bourbon", "wildturkey"]
ENGINE_LABELS = {
    "wisckey": "WiscKey",
    "bourbon": "Bourbon",
    "wildturkey": "WildTurkey",
}


def parse_log(path):
    result = {
        "stalls": defaultdict(float),
        "compaction_wa": 0.0,
        "disk_write_mb": 0.0,
        "raw_write_mb": 0.0,
        "waf": 0.0,
        "levels": defaultdict(lambda: {"time_sec": 0.0, "count": 0}),
    }
    if not os.path.exists(path):
        return result

    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()
            raw_size = RAW_SIZE_RE.match(line)
            if raw_size:
                result["raw_write_mb"] = float(raw_size.group(1))
                continue

            stall = STALL_RE.match(line)
            if stall:
                result["stalls"][STALL_NAMES[stall.group(1)]] += float(stall.group(2))
                continue

            compaction_wa = COMPACTION_WA_RE.match(line)
            if compaction_wa:
                result["compaction_wa"] = float(compaction_wa.group(1))
                continue

            disk_write = DISK_WRITE_RE.match(line)
            if disk_write:
                result["disk_write_mb"] = float(disk_write.group(1))
                continue

            compaction = COMPACTION_RE.match(raw)
            if compaction:
                level = int(compaction.group(1))
                result["levels"][level]["time_sec"] += float(compaction.group(4))
                result["levels"][level]["count"] += int(compaction.group(10))

    if result["raw_write_mb"] > 0 and result["disk_write_mb"] > 0:
        result["waf"] = result["disk_write_mb"] / result["raw_write_mb"]
    else:
        result["waf"] = result["compaction_wa"]

    return result


def status_is_ok(status):
    text = str(status).strip()
    return text in ("", "0") or text.startswith("ok")


def avg(samples):
    return sum(samples) / len(samples) if samples else 0.0


def sort_engines(engines):
    order = {engine: index for index, engine in enumerate(ENGINE_ORDER)}
    return sorted(engines, key=lambda engine: (order.get(engine, len(order)), engine))


def main():
    if len(sys.argv) != 5:
        print(
            "usage: plot_figure11_write_metrics.py <cases.csv> <summary.csv> "
            "<compaction.png> <stall_waf.png>",
            file=sys.stderr,
        )
        return 2

    cases_path, summary_path, compaction_png, stall_waf_png = sys.argv[1:]
    parsed_rows = []

    with open(cases_path, newline="") as f:
        reader = csv.DictReader(f)
        for case in reader:
            parsed = parse_log(case["log"])
            levels = parsed["levels"]
            total_stall = sum(parsed["stalls"].values())
            total_time = sum(level["time_sec"] for level in levels.values())
            total_count = sum(level["count"] for level in levels.values())
            ok = status_is_ok(case.get("status", ""))

            for level, values in sorted(levels.items()):
                parsed_rows.append(
                    {
                        "engine": case["engine"],
                        "dataset": case["dataset"],
                        "run": case["run"],
                        "num": case["num"],
                        "level": str(level),
                        "compaction_count": str(values["count"]),
                        "compaction_time_sec": f"{values['time_sec']:.9f}",
                        "total_compaction_count": str(total_count),
                        "total_compaction_time_sec": f"{total_time:.9f}",
                        "write_stall_sec": f"{total_stall:.9f}",
                        "memtable_stall_sec": f"{parsed['stalls']['memtable']:.9f}",
                        "l0_stop_stall_sec": f"{parsed['stalls']['l0_stop']:.9f}",
                        "l0_slow_stall_sec": f"{parsed['stalls']['l0_slow']:.9f}",
                        "waf": f"{parsed['waf']:.9f}",
                        "compaction_wa": f"{parsed['compaction_wa']:.9f}",
                        "raw_write_mb": f"{parsed['raw_write_mb']:.9f}",
                        "disk_write_mb": f"{parsed['disk_write_mb']:.9f}",
                        "log": case["log"],
                        "status": "ok" if ok else "failed",
                    }
                )

            if not levels:
                parsed_rows.append(
                    {
                        "engine": case["engine"],
                        "dataset": case["dataset"],
                        "run": case["run"],
                        "num": case["num"],
                        "level": "",
                        "compaction_count": "0",
                        "compaction_time_sec": "0.000000000",
                        "total_compaction_count": "0",
                        "total_compaction_time_sec": "0.000000000",
                        "write_stall_sec": f"{total_stall:.9f}",
                        "memtable_stall_sec": f"{parsed['stalls']['memtable']:.9f}",
                        "l0_stop_stall_sec": f"{parsed['stalls']['l0_stop']:.9f}",
                        "l0_slow_stall_sec": f"{parsed['stalls']['l0_slow']:.9f}",
                        "waf": f"{parsed['waf']:.9f}",
                        "compaction_wa": f"{parsed['compaction_wa']:.9f}",
                        "raw_write_mb": f"{parsed['raw_write_mb']:.9f}",
                        "disk_write_mb": f"{parsed['disk_write_mb']:.9f}",
                        "log": case["log"],
                        "status": "ok" if ok else "failed",
                    }
                )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    fields = [
        "engine",
        "dataset",
        "run",
        "num",
        "level",
        "compaction_count",
        "compaction_time_sec",
        "total_compaction_count",
        "total_compaction_time_sec",
        "write_stall_sec",
        "memtable_stall_sec",
        "l0_stop_stall_sec",
        "l0_slow_stall_sec",
        "waf",
        "compaction_wa",
        "raw_write_mb",
        "disk_write_mb",
        "log",
        "status",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parsed_rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; summary CSV was still generated", file=sys.stderr)
        return 0

    ok_rows = [row for row in parsed_rows if row["status"] == "ok"]
    if not ok_rows:
        print("no successful rows to plot", file=sys.stderr)
        return 1

    engines = []
    datasets = []
    levels = []
    for row in ok_rows:
        if row["engine"] not in engines:
            engines.append(row["engine"])
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
        if row["level"] != "" and int(row["level"]) not in levels:
            levels.append(int(row["level"]))
    levels.sort()

    count_values = defaultdict(list)
    time_values = defaultdict(list)
    stall_values = defaultdict(list)
    waf_values = defaultdict(list)
    for row in ok_rows:
        key = (row["engine"], row["dataset"], row["run"])
        if row["level"] != "":
            count_values[(row["engine"], row["dataset"], int(row["level"]))].append(
                float(row["compaction_count"])
            )
        time_values[key].append(float(row["total_compaction_time_sec"]))
        stall_values[key].append(float(row["write_stall_sec"]))
        waf_values[key].append(float(row["waf"]))

    engines = sort_engines(engines)
    x_labels = [ENGINE_LABELS.get(engine, engine) for _dataset in datasets for engine in engines]
    x_keys = [(engine, dataset) for dataset in datasets for engine in engines]
    xs = list(range(len(x_keys)))

    level_colors = {
        0: "#4c78a8",
        1: "#f58518",
        2: "#54a24b",
        3: "#e45756",
        4: "#72b7b2",
        5: "#b279a2",
        6: "#ff9da6",
    }

    fig, ax_count = plt.subplots(figsize=(max(8, 1.1 * len(xs)), 5))
    bottoms = [0.0 for _ in xs]
    for level in levels:
        ys = [
            avg(count_values.get((engine, dataset, level), []))
            for engine, dataset in x_keys
        ]
        ax_count.bar(xs, ys, bottom=bottoms, label=f"L{level} count",
                     color=level_colors.get(level))
        bottoms = [bottom + y for bottom, y in zip(bottoms, ys)]

    ax_time = ax_count.twinx()
    times = []
    for engine, dataset in x_keys:
        samples = []
        seen_runs = {
            row["run"]
            for row in ok_rows
            if row["engine"] == engine and row["dataset"] == dataset
        }
        for run in seen_runs:
            samples.extend(time_values.get((engine, dataset, run), []))
        times.append(avg(samples))
    ax_time.plot(xs, times, color="#222222", marker="o", linewidth=2.0,
                 label="compaction time")

    ax_count.set_xticks(xs)
    ax_count.set_xticklabels(x_labels)
    ax_count.set_ylabel("compaction count")
    ax_time.set_ylabel("compaction time (sec)")
    ax_count.grid(axis="y", linestyle=":", linewidth=0.7)
    h1, l1 = ax_count.get_legend_handles_labels()
    h2, l2 = ax_time.get_legend_handles_labels()
    ax_count.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(compaction_png), exist_ok=True)
    fig.savefig(compaction_png, dpi=180)
    plt.close(fig)

    fig, ax_stall = plt.subplots(figsize=(max(8, 1.1 * len(xs)), 5))
    stalls = []
    wafs = []
    for engine, dataset in x_keys:
        stall_samples = []
        waf_samples = []
        seen_runs = {
            row["run"]
            for row in ok_rows
            if row["engine"] == engine and row["dataset"] == dataset
        }
        for run in seen_runs:
            stall_samples.extend(stall_values.get((engine, dataset, run), []))
            waf_samples.extend(waf_values.get((engine, dataset, run), []))
        stalls.append(avg(stall_samples))
        wafs.append(avg(waf_samples))

    ax_stall.bar(xs, stalls, color="#4c78a8", label="write stall time")
    ax_waf = ax_stall.twinx()
    ax_waf.plot(xs, wafs, color="#e45756", marker="o", linewidth=2.0,
                label="WAF")
    ax_waf.set_ylim(bottom=0)
    ax_stall.set_xticks(xs)
    ax_stall.set_xticklabels(x_labels)
    ax_stall.set_ylabel("write stall time (sec)")
    ax_waf.set_ylabel("write amplification")
    ax_stall.grid(axis="y", linestyle=":", linewidth=0.7)
    h1, l1 = ax_stall.get_legend_handles_labels()
    h2, l2 = ax_waf.get_legend_handles_labels()
    ax_stall.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(stall_waf_png), exist_ok=True)
    fig.savefig(stall_waf_png, dpi=180)
    plt.close(fig)

    print(f"wrote plot to {compaction_png}")
    print(f"wrote plot to {stall_waf_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
