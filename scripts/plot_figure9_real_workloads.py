#!/usr/bin/env python3
import csv
import os
import sys
from collections import defaultdict


WORKLOADS = ["write_only", "read_only", "balanced", "read_heavy"]


def main():
    if len(sys.argv) != 3:
        print("usage: plot_figure9_real_workloads.py <summary.csv> <output.png>", file=sys.stderr)
        return 2

    csv_path, output_path = sys.argv[1], sys.argv[2]
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("status", "").startswith("ok"):
                continue
            rows.append(row)

    if not rows:
        print("no successful rows to plot", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; summary CSV was still generated", file=sys.stderr)
        return 0

    values = defaultdict(list)
    engines = []
    datasets = []
    for row in rows:
        key = (row["engine"], row["dataset"], row["workload"])
        values[key].append(float(row["ops_per_sec"]))
        if row["engine"] not in engines:
            engines.append(row["engine"])
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    averages = {
        key: sum(samples) / len(samples)
        for key, samples in values.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    bar_width = 0.8 / max(1, len(engines))
    x_positions = list(range(len(datasets)))

    for ax, workload in zip(axes, WORKLOADS):
        for engine_index, engine in enumerate(engines):
            xs = [x + engine_index * bar_width for x in x_positions]
            ys = [
                averages.get((engine, dataset, workload), 0.0)
                for dataset in datasets
            ]
            ax.bar(xs, ys, width=bar_width, label=engine)

        ax.set_title(workload.replace("_", "-"))
        ax.set_xticks([x + bar_width * (len(engines) - 1) / 2 for x in x_positions])
        ax.set_xticklabels(datasets)
        ax.set_ylabel("ops/sec")
        ax.grid(axis="y", linestyle=":", linewidth=0.7)

    axes[0].legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"wrote plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
