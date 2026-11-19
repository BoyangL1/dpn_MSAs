#!/usr/bin/env python3
"""Plot node counts, edge counts, and mean out-degree across metropolitan areas."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator
from matplotlib.font_manager import FontProperties, findfont

ROOT = Path(__file__).resolve().parents[1]


def main():
    results, figures = ROOT / "results", ROOT / "figures"
    work_dir = ROOT / ".work"
    if not (work_dir / "ready.json").exists():
        raise RuntimeError("Prepare the fresh network results before plotting.")
    source = work_dir / "msa_summary.csv"
    df = pd.read_csv(source, dtype={"GEOID": str})
    df = df.loc[df.LSAD.eq("M1")].sort_values("GEOID").copy()
    if df.empty or df.GEOID.duplicated().any():
        raise ValueError("Expected one row per metropolitan area.")
    counts = df[["n_nodes", "n_edges"]].to_numpy(float)
    if (
        not np.isfinite(counts).all()
        or (counts <= 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Expected positive integer node and edge counts.")
    df["mean_out_degree"] = df.n_edges / df.n_nodes
    figures.mkdir(exist_ok=True)
    metrics = ["n_nodes", "n_edges", "mean_out_degree"]
    summary = df[metrics].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T
    summary.to_csv(results / "msa_basic_statistics_summary.csv", index_label="metric")

    font_path = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 44,
            "axes.labelsize": 44,
            "axes.titlesize": 44,
            "axes.linewidth": 2.3,
            "xtick.labelsize": 44,
            "ytick.labelsize": 44,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "xtick.major.width": 2.3,
            "ytick.major.width": 2.3,
            "xtick.minor.size": 5.5,
            "xtick.minor.width": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#263238",
            "axes.labelcolor": "#263238",
            "axes.edgecolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "savefig.dpi": 300,
        }
    )
    labels = [r"Nodes, $N$", r"Directed edges, $E$", r"Mean out-degree, $E/N$"]
    suffixes = ["node_count", "directed_edge_count", "mean_out_degree"]
    histograms, output_names, y_limits = [], [], {}
    for i, (metric, label, suffix) in enumerate(zip(metrics, labels, suffixes)):
        fig, ax = plt.subplots(figsize=(10, 12))
        fig.subplots_adjust(left=0.21, right=0.97, bottom=0.17, top=0.97)
        values = df[metric].to_numpy()
        if i < 2:
            lower = np.floor(np.log10(values.min()) * 5) / 5
            upper = np.ceil(np.log10(values.max()) * 5) / 5
            bins = 10 ** np.linspace(lower, upper, 19)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            bins = np.arange(
                np.floor(values.min() * 2) / 2,
                np.ceil(values.max() * 2) / 2 + 0.25,
                0.5,
            )
            ax.xaxis.set_major_locator(MultipleLocator(2))
        freq, edges = np.histogram(values, bins=bins)
        if freq.sum() != len(df):
            raise ValueError(f"Histogram excludes observations: {metric}")

        y_max = 76 if freq.max() <= 76 else 20 * np.ceil((freq.max() + 6) / 20)
        ax.bar(
            edges[:-1],
            freq,
            width=np.diff(edges),
            align="edge",
            color="#35677D",
            edgecolor="white",
            linewidth=1.6,
            zorder=2,
        )
        median = float(np.median(values))
        ax.axvline(
            median, color="#BA7752", linewidth=3.3, linestyle=(0, (4, 3)), zorder=3
        )
        median_text = (
            f"{median:,.1f}"
            if i == 0
            else (f"{median:,.0f}" if i == 1 else f"{median:.2f}")
        )
        ax.text(
            0.97,
            0.96,
            f"Median = {median_text}\n$n$ = {len(df)} MSAs",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=44,
            linespacing=1.35,
        )
        ax.set_xlabel(label, labelpad=22)
        ax.set_ylabel("Number of MSAs", labelpad=22)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, y_max)
        y_limits[metric] = [0, float(y_max)]
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.tick_params(axis="both", which="major", pad=15)
        output_name = f"msa_{suffix}_distribution.png"
        fig.savefig(figures / output_name, facecolor="white", bbox_inches=None)
        plt.close(fig)
        output_names.append(output_name)
        histograms.extend(
            {
                "metric": metric,
                "bin_left": left,
                "bin_right": right,
                "msa_count": int(count),
                "msa_fraction": count / len(df),
            }
            for left, right, count in zip(edges[:-1], edges[1:], freq)
        )
    pd.DataFrame(histograms).to_csv(
        work_dir / "msa_basic_statistics_histogram_bins.csv", index=False
    )
    notes = {
        "input": str(source.relative_to(ROOT)),
        "n_msa": len(df),
        "node_definition": "All active nodes in the cleaned national network located within each MSA, including nodes with only external links. A node is a retained polygon identifier, potentially containing multiple visible POI names.",
        "edge_definition": "Within-MSA directed records in the cleaned NIJ > 50 network.",
        "mean_out_degree": "Within-MSA E/N; nodes with no internal outgoing links contribute zero. MSA observations are equally weighted.",
        "scope": "Metropolitan LSAD M1 rows; cleaned 2021 network and 2023 CBSA boundaries.",
        "histograms": "N and E: 18 equal log-width bins. E/N: linear bins of width 0.5. Bar heights are MSA counts. Bins are left-inclusive/right-exclusive except for the final bin, which includes its right endpoint.",
        "median_marker": "Muted copper dashed line",
        "figure_size_inches": [10, 12],
        "fixed_canvas_pixels": [3000, 3600],
        "y_axis_limits": y_limits,
        "font_family": "Arial",
        "font_file": font_path,
        "font_size_pt": 44,
        "dpi": 300,
        "ppt_width_cm": 6,
        "outputs": output_names,
    }
    (work_dir / "msa_basic_statistics_notes.json").write_text(
        json.dumps(notes, indent=2) + "\n"
    )
    print(summary.to_string())
    print("Saved PNG figures: " + ", ".join(output_names))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
