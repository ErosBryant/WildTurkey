#!/usr/bin/env python3
import csv
import os
import re
import sys
from collections import defaultdict


STAT_RE = re.compile(
    r"^ModelTrainingStats\s+model_bytes=(\d+)\s+training_ns=(\d+)\s+rl_ns=(\d+)\s+model_count=(\d+)"
    r"(?:\s+live_training_ns=(\d+)\s+live_rl_ns=(\d+))?"
    r"(?:\s+trained_model_bytes=(\d+)\s+trained_model_count=(\d+))?"
)

ENGINE_ORDER = ["bourbon", "wildturkey"]
ENGINE_LABELS = {
    "bourbon": "Bourbon",
    "wildturkey": "WildTurkey",
}


def sort_engines(engines):
    order = {engine: index for index, engine in enumerate(ENGINE_ORDER)}
    return sorted(engines, key=lambda engine: (order.get(engine, len(order)), engine))


def parse_log(path):
    result = {
        "model_bytes": 0,
        "training_ns": 0,
        "rl_ns": 0,
        "model_count": 0,
        "live_training_ns": 0,
        "live_rl_ns": 0,
        "trained_model_bytes": 0,
        "trained_model_count": 0,
    }
    if not os.path.exists(path):
        return result
    with open(path, errors="replace") as f:
        for line in f:
            match = STAT_RE.match(line.strip())
            if match:
                result["model_bytes"] = int(match.group(1))
                result["training_ns"] = int(match.group(2))
                result["rl_ns"] = int(match.group(3))
                result["model_count"] = int(match.group(4))
                result["live_training_ns"] = (
                    int(match.group(5)) if match.group(5) is not None
                    else result["training_ns"]
                )
                result["live_rl_ns"] = (
                    int(match.group(6)) if match.group(6) is not None
                    else result["rl_ns"]
                )
                result["trained_model_bytes"] = (
                    int(match.group(7)) if match.group(7) is not None
                    else result["model_bytes"]
                )
                result["trained_model_count"] = (
                    int(match.group(8)) if match.group(8) is not None
                    else result["model_count"]
                )
    return result


def avg(values):
    return sum(values) / len(values) if values else 0.0


def status_is_ok(status, parsed):
    text = str(status).strip()
    return (text in ("", "0") or text.startswith("ok")) and parsed["model_count"] > 0


def main():
    if len(sys.argv) != 5:
        print(
            "usage: plot_model_training.py <cases.csv> <summary.csv> "
            "<model_size.png> <training_time.png>",
            file=sys.stderr,
        )
        return 2

    cases_path, summary_path, model_size_png, training_time_png = sys.argv[1:]
    rows = []

    with open(cases_path, newline="") as f:
        for case in csv.DictReader(f):
            parsed = parse_log(case["log"])
            ok = status_is_ok(case.get("status", ""), parsed)
            rows.append(
                {
                    "engine": case["engine"],
                    "dataset": case["dataset"],
                    "run": case["run"],
                    "num": case["num"],
                    "model_bytes": str(parsed["model_bytes"]),
                    "model_mb": f"{parsed['model_bytes'] / 1048576.0:.9f}",
                    "training_ns": str(parsed["training_ns"]),
                    "training_sec": f"{parsed['training_ns'] / 1e9:.9f}",
                    "rl_ns": str(parsed["rl_ns"]),
                    "rl_sec": f"{parsed['rl_ns'] / 1e9:.9f}",
                    "live_training_ns": str(parsed["live_training_ns"]),
                    "live_training_sec": f"{parsed['live_training_ns'] / 1e9:.9f}",
                    "live_rl_ns": str(parsed["live_rl_ns"]),
                    "live_rl_sec": f"{parsed['live_rl_ns'] / 1e9:.9f}",
                    "index_training_sec": f"{parsed['live_training_ns'] / 1e9:.9f}",
                    "model_count": str(parsed["model_count"]),
                    "trained_model_bytes": str(parsed["trained_model_bytes"]),
                    "trained_model_mb": f"{parsed['trained_model_bytes'] / 1048576.0:.9f}",
                    "trained_model_count": str(parsed["trained_model_count"]),
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
        "model_bytes",
        "model_mb",
        "training_ns",
        "training_sec",
        "rl_ns",
        "rl_sec",
        "live_training_ns",
        "live_training_sec",
        "live_rl_ns",
        "live_rl_sec",
        "index_training_sec",
        "model_count",
        "trained_model_bytes",
        "trained_model_mb",
        "trained_model_count",
        "log",
        "status",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; summary CSV was still generated", file=sys.stderr)
        return 0

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        print("no successful rows to plot", file=sys.stderr)
        return 1

    engines = []
    datasets = []
    for row in ok_rows:
        if row["engine"] not in engines:
            engines.append(row["engine"])
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
    engines = sort_engines(engines)

    x_keys = [(engine, dataset) for dataset in datasets for engine in engines]
    x_labels = [ENGINE_LABELS.get(engine, engine) for engine, _dataset in x_keys]
    xs = list(range(len(x_keys)))

    model_values = []
    index_training_values = []
    rl_values = []
    for engine, dataset in x_keys:
        matching = [
            row for row in ok_rows
            if row["engine"] == engine and row["dataset"] == dataset
        ]
        model_values.append(avg([float(row["model_mb"]) for row in matching]))
        index_training_values.append(avg([float(row["index_training_sec"]) for row in matching]))
        rl_values.append(avg([float(row["live_rl_sec"]) for row in matching]))

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(xs)), 4.8))
    ax.bar(xs, model_values, color="#4c78a8")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("model size (MB)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(model_size_png), exist_ok=True)
    fig.savefig(model_size_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(xs)), 4.8))
    ax.bar(xs, index_training_values, color="#4c78a8", label="index training")
    ax.bar(xs, rl_values, bottom=index_training_values, color="#f58518", label="RL part")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("index training time (sec)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(training_time_png), exist_ok=True)
    fig.savefig(training_time_png, dpi=180)
    plt.close(fig)

    print(f"wrote plot to {model_size_png}")
    print(f"wrote plot to {training_time_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
