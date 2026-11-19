#!/usr/bin/env python3
"""Plot metropolitan in-strength and out-strength distributions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator
from distribution_panel_layout import (
    FIGURE_SIZE,
    WIDTH_CM,
    HEIGHT_CM,
    FONT_SIZES,
    make_panel,
)


ROOT = Path(__file__).resolve().parents[1]
BIN_EDGES = np.logspace(-4, np.log10(600), 50)
BIN_CENTERS = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
MIN_NODES = 50
MIN_SUPPORT_FRACTION = 0.5
MAX_ERRORBAR_CV = 0.5
MAX_ERRORBARS = 10
STRENGTH_COLUMNS = ("in_str", "out_str")


def configure_style(font_path: str | None = None) -> str:
    """Configure figure typography and colors."""
    if font_path:
        font_manager.fontManager.addfont(font_path)
    resolved = font_manager.findfont(
        font_manager.FontProperties(family="Arial"), fallback_to_default=False
    )
    if font_manager.FontProperties(fname=resolved).get_name() != "Arial":
        raise ValueError(f"Actual Arial is required; resolved {resolved}")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZES["base"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "axes.linewidth": 0.65,
            "axes.labelsize": FONT_SIZES["axis_label"],
            "axes.titlesize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "text.color": "#252a2e",
            "axes.labelcolor": "#252a2e",
            "axes.edgecolor": "#42494e",
            "axes.labelpad": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )
    return resolved


def positive_strength_histogram(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Probability mass per log bin, normalized by positive-strength nodes."""
    positive = values[values > 0]
    if len(positive) == 0:
        raise ValueError("An MSA has no positive strengths; review the denominator")
    counts, _ = np.histogram(positive, bins=BIN_EDGES)
    if counts.sum() != len(positive):
        raise ValueError(
            "Strength outside the fixed grid [0.0001, 600]: "
            f"min={positive.min()}, max={positive.max()}. Review before plotting."
        )
    return counts / len(positive), len(positive)


def display_statistics(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the equal-MSA mean, population SD, and display thresholds."""
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=0)
    support = (curves > 0).sum(axis=0)
    show_mean = (
        np.isfinite(mean) & (mean > 0) & (support >= MIN_SUPPORT_FRACTION * len(curves))
    )
    show_errorbar = show_mean & (sd <= MAX_ERRORBAR_CV * mean)
    indices = np.flatnonzero(show_errorbar)
    if indices.size > MAX_ERRORBARS:
        indices = indices[
            np.linspace(0, indices.size - 1, MAX_ERRORBARS).round().astype(int)
        ]
    return {
        "mean": mean,
        "sd": sd,
        "support": support,
        "show_mean": show_mean,
        "show_errorbar_eligible": show_errorbar,
        "errorbar_indices": indices,
    }


def run(root: Path = ROOT) -> dict:
    work_dir = root / ".work"
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    node_file = work_dir / "network_nodes.parquet"
    msa_file = work_dir / "msa_summary.csv"
    config_file = work_dir / "plot_config.json"
    with config_file.open() as handle:
        config = json.load(handle)
    font_path = configure_style(config.get("font_path"))
    node = pd.read_parquet(node_file, columns=["node", "GEOID", *STRENGTH_COLUMNS])
    if node["GEOID"].isna().any():
        raise ValueError("Missing MSA assignments")
    node["GEOID"] = node["GEOID"].astype(str).str.zfill(5)
    if node["node"].duplicated().any():
        raise ValueError("Duplicate full polygon identifiers in network_nodes.parquet")
    values = node[list(STRENGTH_COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Node strengths must be finite and nonnegative")

    msa = pd.read_csv(msa_file, dtype={"GEOID": str}).set_index("GEOID")
    msa.index = msa.index.str.zfill(5)
    sizes = node.groupby("GEOID").size()
    expected_sizes = msa.loc[sizes.index, "n_nodes"].astype(int)
    if not np.array_equal(sizes.to_numpy(), expected_sizes.to_numpy()):
        raise ValueError("Node counts disagree with the common MSA summary")
    metro = set(msa.index[msa["LSAD"] == "M1"])
    included = sizes.index[(sizes >= MIN_NODES) & sizes.index.isin(metro)]
    if len(included) == 0:
        raise ValueError("No metropolitan areas meet the 50-node threshold")
    node = node[node["GEOID"].isin(included)]
    groups = list(node.groupby("GEOID", sort=True))
    geoids = [geoid for geoid, _ in groups]
    node_counts = np.array([len(frame) for _, frame in groups], dtype=np.int64)

    vmin = float(config.get("norm_vmin", msa["n_nodes"].min()))
    vmax = float(config.get("norm_vmax", msa["n_nodes"].max()))
    if vmin != msa["n_nodes"].min() or vmax != msa["n_nodes"].max():
        raise ValueError("Shared node-count normalization must span all current MSAs")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "viridis_readable", plt.get_cmap("viridis")(np.linspace(0.03, 0.88, 256))
    )

    distributions = {}
    positive_counts = {}
    stats = {}
    for col in STRENGTH_COLUMNS:
        per_msa = [
            positive_strength_histogram(sub[col].to_numpy(float)) for _, sub in groups
        ]
        distributions[col] = np.stack([item[0] for item in per_msa])
        positive_counts[col] = np.array([item[1] for item in per_msa], dtype=np.int64)
        if not np.allclose(distributions[col].sum(axis=1), 1.0):
            raise ValueError("Per-MSA fractions do not sum to one")
        stats[col] = display_statistics(distributions[col])

    all_positive = np.concatenate(
        [curves[curves > 0] for curves in distributions.values()]
    )
    lower_endpoints = [all_positive.min()]
    upper_endpoints = [all_positive.max()]
    for s in stats.values():
        eligible = s["show_errorbar_eligible"]
        if eligible.any():
            lower_endpoints.append((s["mean"] - s["sd"])[eligible].min())
            upper_endpoints.append((s["mean"] + s["sd"])[eligible].max())
    common_ylim = (min(lower_endpoints) * 0.65, max(upper_endpoints) * 1.65)

    bin_rows, pooled_rows, msa_rows, image_paths = [], [], [], []
    labels = {
        "in_str": r"In-strength, $s^{\mathrm{in}}$",
        "out_str": r"Out-strength, $s^{\mathrm{out}}$",
    }
    for col in STRENGTH_COLUMNS:
        curves, s = distributions[col], stats[col]
        fig, ax = make_panel()
        for index in np.argsort(node_counts):
            ax.plot(
                BIN_CENTERS,
                np.where(curves[index] > 0, curves[index], np.nan),
                color=cmap(norm(node_counts[index])),
                lw=0.35,
                alpha=0.16,
                zorder=2,
            )
        ax.plot(
            BIN_CENTERS,
            np.where(s["show_mean"], s["mean"], np.nan),
            color="#202629",
            lw=1.4,
            zorder=6,
        )
        indices = s["errorbar_indices"]
        ax.errorbar(
            BIN_CENTERS[indices],
            s["mean"][indices],
            yerr=s["sd"][indices],
            fmt="none",
            ecolor="#202629",
            elinewidth=0.55,
            capsize=1.3,
            capthick=0.55,
            zorder=7,
            alpha=0.9,
            label="MSA mean ± SD",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*common_ylim)
        if col == "in_str":
            ax.xaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-4, 3, 2)))
        ax.set_xlabel(labels[col])
        ax.set_ylabel("Node fraction per log bin")
        ax.legend(
            frameon=False,
            loc="lower left",
            handlelength=1.1,
            handletextpad=0.5,
            borderaxespad=0.35,
            labelspacing=0.4,
        )
        if len(fig.axes) != 1:
            raise ValueError("Main strength panel must not contain a colorbar")
        image_path = (
            figures_dir / f"national_{col.split('_')[0]}_strength_distribution.png"
        )

        fig.savefig(image_path, facecolor="white", bbox_inches=None, pad_inches=0)
        plt.close(fig)
        image_paths.append(str(image_path.relative_to(root)))

        positive = node.loc[node[col] > 0, col].to_numpy(float)
        pooled_rows.append(
            dict(
                strength=col,
                n_msa=len(groups),
                n_nodes=len(node),
                n_positive=len(positive),
                n_zero=int((node[col] == 0).sum()),
                pooled_median=float(np.median(positive)),
                pooled_mean=float(positive.mean()),
            )
        )
        for j, center in enumerate(BIN_CENTERS):
            bin_rows.append(
                dict(
                    strength=col,
                    bin_left=BIN_EDGES[j],
                    bin_right=BIN_EDGES[j + 1],
                    bin_center=center,
                    n_msa=len(groups),
                    n_nonzero_msa=int(s["support"][j]),
                    mean=s["mean"][j],
                    sd=s["sd"][j],
                    mean_displayed=bool(s["show_mean"][j]),
                    errorbar_displayed=bool(j in indices),
                )
            )
            for k, geoid in enumerate(geoids):
                msa_rows.append(
                    dict(
                        strength=col,
                        GEOID=geoid,
                        n_nodes=int(node_counts[k]),
                        n_positive=int(positive_counts[col][k]),
                        n_zero=int(node_counts[k] - positive_counts[col][k]),
                        bin_left=BIN_EDGES[j],
                        bin_right=BIN_EDGES[j + 1],
                        bin_center=center,
                        fraction=curves[k, j],
                    )
                )

    pd.DataFrame(bin_rows).to_csv(work_dir / "strength_bin_summary.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(
        work_dir / "strength_pooled_summary.csv", index=False
    )
    pd.DataFrame(msa_rows).to_csv(work_dir / "strength_msa_histograms.csv", index=False)
    metadata = {
        "node_input": str(node_file.relative_to(root)),
        "msa_input": str(msa_file.relative_to(root)),
        "n_msa": len(groups),
        "n_nodes": len(node),
        "minimum_nodes_per_msa": MIN_NODES,
        "strength_definition": "National network: in_str(j)=sum_i WIJ_ij; out_str(i)=sum_j WIJ_ij. Cross-MSA edges are included.",
        "histogram_definition": "Fraction of positive-strength nodes per fixed log bin; not density per unit strength. Zero-strength nodes are excluded from each direction's denominator.",
        "aggregation": "Equal-MSA mean and population SD (ddof=0); empty bins contribute zero.",
        "mean_display_min_nonzero_msa_fraction": MIN_SUPPORT_FRACTION,
        "errorbar_display_max_sd_over_mean": MAX_ERRORBAR_CV,
        "maximum_errorbars_per_panel": MAX_ERRORBARS,
        "all_bins_retained_in_csv": True,
        "bin_edges": BIN_EDGES.tolist(),
        "common_y_limits": list(common_ylim),
        "font_family": "Arial",
        "font_path": font_path,
        "font_sizes_pt": FONT_SIZES,
        "figure_size_inches": list(FIGURE_SIZE),
        "figure_size_cm": [WIDTH_CM, HEIGHT_CM],
        "tight_crop": False,
        "dpi": 600,
        "colormap": "viridis sampled from 0.03 to 0.88",
        "normalization": {
            "type": "LogNorm",
            "vmin": vmin,
            "vmax": vmax,
            "variable": "active national-network nodes per MSA",
        },
        "embedded_colorbar": False,
        "figures": image_paths,
        "pooled_statistics": pooled_rows,
    }
    with (work_dir / "strength_figure_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {"n_msa": len(groups), "n_nodes": len(node), "figures": image_paths},
            indent=2,
        )
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Network statistics output directory"
    )
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
