#!/usr/bin/env python3
"""
Combine one or more benchmark CSVs (or run directories) into a single QPS vs latency plot.

Inputs can be:
  - Paths to CSV files with columns: backend,qps,median_ms,avg_ms,p95_ms (from qps_latency_benchmark.py)
  - Paths to run directories that contain qps_latency_results.csv

Usage examples:
  uv run benchmark/combine_qps_latency_plots.py \
    --inputs benchmark/results/qps_latency_20250813_134449 \
             benchmark/results/qps_latency_20250813_160052 \
    --labels "FW on SM" "SM on SM" \
    --out benchmark/results/combined_qps_vs_median_latency.png

  uv run benchmark/combine_qps_latency_plots.py \
    --inputs benchmark/results/qps_latency_20250813_134449/qps_latency_results.csv \
             benchmark/results/qps_latency_20250813_160052/qps_latency_results.csv

Legend labels:
  - By default, the legend label is the subdirectory name of each input (run directory name).
    If a CSV path is provided directly, the label is the parent directory name.

Legend labels:
  - By default, the legend label is the subdirectory name of each input (run directory name).
    If a CSV path is provided directly, the label is the parent directory name.

Options:
  --y-metric: one of median_ms, avg_ms, p95_ms (default: median_ms)
  --save-merged-csv: optional path to write merged dataframe with a series label column
  --show-fail-percent: show failure rate (%) for each series
  --fail-style: how to show fail% when enabled: labels | line (default: labels)
    - labels: draw small numeric labels (e.g., "3%") near each point on the main axis
    - line:   dashed line on a secondary y-axis (right) with fail% vs QPS
"""

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _resolve_csv_path(path: str) -> Optional[str]:
    """Return a CSV file path from input which can be a CSV or a directory.
    For directories, look for qps_latency_results.csv inside.
    """
    if os.path.isdir(path):
        candidate = os.path.join(path, "qps_latency_results.csv")
        return candidate if os.path.isfile(candidate) else None
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return path
    return None


def _derive_default_label(path: str) -> str:
    base = os.path.basename(path.rstrip("/"))
    if base.lower().endswith(".csv"):
        base = os.path.basename(os.path.dirname(path))
    return base or path


def _load_labeled_frames(paths: List[str], labels: Optional[List[str]]) -> List[Tuple[pd.DataFrame, str]]:
    labeled: List[Tuple[pd.DataFrame, str]] = []
    for idx, p in enumerate(paths):
        csv_path = _resolve_csv_path(p)
        if not csv_path:
            raise FileNotFoundError(f"Could not find CSV for input: {p}")
        df = pd.read_csv(csv_path)
        label = labels[idx] if labels and idx < len(labels) else _derive_default_label(p)
        labeled.append((df, label))
    return labeled


def combine_qps_latency(
    inputs: List[str],
    labels: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    title: str = "QPS vs Latency (combined)",
    y_metric: str = "median_ms",
    save_merged_csv: Optional[str] = None,
    show_fail_percent: bool = False,
    fail_style: str = "labels",
    model_name: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> str:
    """Combine multiple benchmark runs into a single plot.

    Returns the output plot path.
    """
    assert y_metric in {"median_ms", "avg_ms", "p95_ms"}, "y_metric must be median_ms|avg_ms|p95_ms"

    labeled_frames = _load_labeled_frames(inputs, labels)

    # Build a merged dataframe with a 'series' column for plotting.
    # Series label is the run label (subdir name), regardless of backend.
    merged_rows: List[pd.DataFrame] = []
    series_order: List[str] = []
    for df, run_label in labeled_frames:
        sub = df.copy()
        sub["series"] = run_label
        merged_rows.append(sub)
        series_order.append(run_label)

    if not merged_rows:
        raise ValueError("No data rows found in provided inputs")

    merged = pd.concat(merged_rows, ignore_index=True)
    # Compute fail percent per row if available
    if {"ok", "fail"}.issubset(merged.columns):
        denom = (merged["ok"].fillna(0) + merged["fail"].fillna(0)).replace(0, pd.NA)
        merged["fail_percent"] = (merged["fail"].fillna(0) * 100.0 / denom).astype(float)

    # Save merged CSV if requested
    if save_merged_csv:
        os.makedirs(os.path.dirname(save_merged_csv) or ".", exist_ok=True)
        merged.to_csv(save_merged_csv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax2 = None
    if show_fail_percent and fail_style == "line":
        ax2 = ax.twinx()
        ax2.set_ylabel("Fail rate (%)")
        # Keep 0..100%, but place 0 slightly above the x-axis for visibility
        ax2.set_ylim(-1, 100)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        # Draw a subtle reference line at 0%
        ax2.axhline(0, color="#cccccc", linestyle=":", linewidth=0.8)

    for series_label in dict.fromkeys(series_order):  # preserve order, drop dups
        sub = merged[merged["series"] == series_label].sort_values("qps")
        if sub.empty:
            continue
        line = ax.plot(sub["qps"], sub[y_metric], marker="o", label=series_label)[0]
        if show_fail_percent and "fail_percent" in sub.columns:
            color = line.get_color()
            if fail_style == "line" and ax2 is not None:
                ax2.plot(sub["qps"], sub["fail_percent"].fillna(0), linestyle="--", marker=None, color=color, alpha=0.8)
            elif fail_style == "labels":
                # Draw small text labels near each point
                for x, y, fp in zip(sub["qps"], sub[y_metric], sub["fail_percent"].fillna(0)):
                    ax.annotate(f"{fp:.0f}%", xy=(x, y), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=color)

    ax.set_xlabel("QPS (requests/sec)")
    ylabel = {
        "median_ms": "Median total latency (ms)",
        "avg_ms": "Average total latency (ms)",
        "p95_ms": "P95 total latency (ms)",
    }[y_metric]
    ax.set_ylabel(ylabel)
    parts = [title]
    if model_name:
        parts.append(str(model_name))
    if input_tokens is not None:
        parts.append(f"Input Tokens = {int(input_tokens)}")
    if output_tokens is not None:
        parts.append(f"Output Tokens = {int(output_tokens)}")
    final_title = " | ".join(parts)
    ax.set_title(final_title)
    # Build legend with optional fail% key
    handles, labels = ax.get_legend_handles_labels()
    if show_fail_percent:
        if fail_style == "line" and ax2 is not None:
            handles.append(Line2D([0], [0], linestyle="--", color="gray"))
            labels.append("Fail rate (%) [right axis]")
        else:
            # Indicate that fail% is rendered as labels near points
            handles.append(Line2D([0], [0], linestyle=":", color="gray"))
            labels.append("Fail rate (%) shown as labels")
    ax.legend(
        handles,
        labels,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, len(labels)),
        borderaxespad=0,
        frameon=False,
    )
    plt.tight_layout()

    # Resolve output path
    if not out_path:
        # Default under <results>/combined_plots relative to the first input
        def _sanitize(name: str) -> str:
            name = name.strip().replace("/", "_").replace(" ", "-")
            return name

        labels_for_name = [
            _sanitize(lbl) for lbl in dict.fromkeys(series_order)  # preserve order, unique
        ]
        fname = "_vs_".join(labels_for_name) + ".png"

        first_input = inputs[0]
        candidate_dir = first_input if os.path.isdir(first_input) else os.path.dirname(first_input)
        candidate_dir = os.path.abspath(candidate_dir)

        # Walk up to find a parent named 'results'
        results_dir = candidate_dir
        found_results = False
        while True:
            if os.path.basename(results_dir) == "results":
                found_results = True
                break
            parent = os.path.dirname(results_dir)
            if parent == results_dir or not parent:
                break
            results_dir = parent

        if not found_results:
            maybe_results = os.path.join(candidate_dir, "results")
            if os.path.isdir(maybe_results):
                results_dir = os.path.abspath(maybe_results)
                found_results = True

        if not found_results:
            results_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))

        base_dir = os.path.join(results_dir, "combined_plots")
        out_path = os.path.join(base_dir, fname)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine benchmark CSVs into a single QPS vs latency plot")
    p.add_argument("--inputs", nargs="+", required=True, help="CSV files or run directories to include")
    p.add_argument("--labels", help="Comma-separated labels matching inputs (optional)")
    p.add_argument("--out", help="Output PNG path (optional)")
    p.add_argument("--title", default="QPS vs Latency (combined)")
    p.add_argument("--model-name", help="Optional model name to append to plot title, e.g., 'Qwen3 8B'")
    p.add_argument("--input-tokens", type=int, help="Optional input token count to append to title")
    p.add_argument("--output-tokens", type=int, help="Optional output token count to append to title")
    p.add_argument("--y-metric", default="median_ms", choices=["median_ms", "avg_ms", "p95_ms"])
    p.add_argument("--save-merged-csv", help="Optional path to save merged CSV with 'series' column")
    p.add_argument("--show-fail-percent", action="store_true", help="Show fail rate (%) for each series")
    p.add_argument("--fail-style", default="labels", choices=["labels", "line"], help="How to show fail% when enabled")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    labels = args.labels.split(",") if args.labels else None
    out_path = combine_qps_latency(
        inputs=args.inputs,
        labels=[l.strip() for l in labels] if labels else None,
        out_path=args.out,
        title=args.title,
        y_metric=args.y_metric,
        save_merged_csv=args.save_merged_csv,
        show_fail_percent=bool(args.show_fail_percent),
        fail_style=str(args.fail_style),
        model_name=args.model_name,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )
    print(f"Saved combined plot to {out_path}")
    if args.save_merged_csv:
        print(f"Saved merged CSV to {args.save_merged_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


