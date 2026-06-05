#!/usr/bin/env python3
import csv
import os
import re
import sys
from collections import defaultdict


STAGES = [
    "FindFiles",
    "LoadIB+FB/Model",
    "SearchIB+FB/Pred",
    "LoadDB+Correction",
    "ReadValue",
]

STAGE_TIMERS = {
    "wisckey": {
        "FindFiles": [0],
        "LoadIB+FB/Model": [1],
        "SearchIB+FB/Pred": [2, 15],
        "LoadDB+Correction": [5, 3],
        "ReadValue": [12],
    },
    "default": {
        "FindFiles": [0],
        "LoadIB+FB/Model": [1, 8],
        "SearchIB+FB/Pred": [17, 2, 15],
        "LoadDB+Correction": [18, 5, 3],
        "ReadValue": [12],
    },
}

TIMER_RE = re.compile(r"^Timer\s+(\d+):\s+(\d+)\s+:\s+([0-9.]+)(?:\s+\(([^)]*)\))?")
READ_RE = re.compile(r"^readrandom(?:_r)?\s*:\s*([0-9.]+)\s+micros/op")


def parse_log(path):
    timers = {}
    timer_names = {}
    read_micros = ""
    if not os.path.exists(path):
        return timers, timer_names, read_micros

    with open(path, errors="replace") as f:
        for line in f:
            timer_match = TIMER_RE.match(line.strip())
            if timer_match:
                timer_id = int(timer_match.group(1))
                timers[timer_id] = int(timer_match.group(2))
                timer_names[timer_id] = timer_match.group(4) or ""
                continue

            read_match = READ_RE.match(line.strip())
            if read_match:
                read_micros = read_match.group(1)

    return timers, timer_names, read_micros


def timer_map_for(engine):
    return STAGE_TIMERS.get(engine, STAGE_TIMERS["default"])


def status_is_ok(status):
    return str(status).strip() in ("", "0")


def main():
    if len(sys.argv) != 4:
        print(
            "usage: plot_figure10_read_breakdown.py <cases.csv> <summary.csv> <output.png>",
            file=sys.stderr,
        )
        return 2

    cases_path, summary_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = []

    with open(cases_path, newline="") as f:
        reader = csv.DictReader(f)
        for case in reader:
            engine = case["engine"]
            dataset = case["dataset"]
            run = case["run"]
            num = int(case["num"])
            timers, timer_names, read_micros = parse_log(case["read_log"])
            stage_timers = timer_map_for(engine)
            fill_status = case.get("fill_status", "")
            read_status = case.get("read_status", "")
            read_ok = status_is_ok(read_status)

            for stage in STAGES:
                ids = stage_timers[stage]
                total_ns = sum(timers.get(timer_id, 0) for timer_id in ids)
                per_op_us = total_ns / num / 1000.0 if num > 0 else 0.0
                if not read_ok:
                    status = "failed"
                elif total_ns > 0:
                    status = "ok"
                else:
                    status = "missing_or_zero"

                rows.append(
                    {
                        "engine": engine,
                        "dataset": dataset,
                        "run": run,
                        "fill_status": fill_status,
                        "read_status": read_status,
                        "stage": stage,
                        "timer_ids": "+".join(str(timer_id) for timer_id in ids),
                        "timer_names": "+".join(timer_names.get(timer_id, "") for timer_id in ids),
                        "total_ns": str(total_ns),
                        "per_op_us": f"{per_op_us:.9f}",
                        "readrandom_us_per_op": read_micros,
                        "read_log": case["read_log"],
                        "status": status,
                    }
                )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "engine",
            "dataset",
            "run",
            "fill_status",
            "read_status",
            "stage",
            "timer_ids",
            "timer_names",
            "total_ns",
            "per_op_us",
            "readrandom_us_per_op",
            "read_log",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; summary CSV was still generated", file=sys.stderr)
        return 0

    values = defaultdict(list)
    engines = []
    datasets = []
    for row in rows:
        if row["engine"] not in engines:
            engines.append(row["engine"])
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
        if row["status"] != "ok":
            continue
        values[(row["engine"], row["dataset"], row["stage"])].append(float(row["per_op_us"]))

    averages = {
        key: sum(samples) / len(samples)
        for key, samples in values.items()
        if samples
    }

    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(6 * max(1, len(datasets)), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {
        "FindFiles": "#4c78a8",
        "LoadIB+FB/Model": "#f58518",
        "SearchIB+FB/Pred": "#54a24b",
        "LoadDB+Correction": "#e45756",
        "ReadValue": "#72b7b2",
    }

    for ax, dataset in zip(axes, datasets):
        xs = list(range(len(engines)))
        bottoms = [0.0 for _ in engines]
        for stage in STAGES:
            ys = [averages.get((engine, dataset, stage), 0.0) for engine in engines]
            ax.bar(xs, ys, bottom=bottoms, label=stage, color=colors.get(stage))
            bottoms = [bottom + y for bottom, y in zip(bottoms, ys)]

        ax.set_title(dataset)
        ax.set_xticks(xs)
        ax.set_xticklabels(engines)
        ax.set_ylabel("microseconds / read")
        ax.grid(axis="y", linestyle=":", linewidth=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"wrote plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
