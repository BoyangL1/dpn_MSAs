#!/usr/bin/env python3
"""Plot within-metropolitan dependency-weight distributions."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont
from distribution_panel_layout import (
    FIGURE_SIZE,
    WIDTH_CM,
    HEIGHT_CM,
    FONT_SIZES,
    make_panel,
)

ROOT = Path(__file__).resolve().parents[1]


def load_internal_weights(meta):
    """Read only within-MSA rows and verify their weights and group counts."""
    edge_path = ROOT / ".work/network_edges.parquet"
    table = pq.read_table(
        edge_path,
        columns=["WIJ", "NIJ", "NI_SOURCE", "source_msa", "target_msa"],
        filters=[("is_intra_msa", "=", True)],
    )
    edges = table.to_pandas()
    if not np.array_equal(edges.source_msa.to_numpy(), edges.target_msa.to_numpy()):
        raise ValueError("An internal edge has different endpoint MSAs.")
    geoids = edges.source_msa.astype(str).to_numpy()
    values = edges.WIJ.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or (values > 1).any():
        raise ValueError("Expected finite dependency weights in (0, 1].")
    if not np.allclose(
        values,
        edges.NIJ.to_numpy() / edges.NI_SOURCE.to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("WIJ differs from the original NIJ / NI_SOURCE ratio.")
    codes = pd.Categorical(geoids, categories=meta.GEOID).codes
    if (codes < 0).any():
        raise ValueError("Internal edges contain an MSA absent from the summary.")
    counts = np.bincount(codes, minlength=len(meta))
    if not np.array_equal(counts, meta.n_edges.to_numpy()):
        raise ValueError("Internal-edge counts differ from the fresh MSA summary.")
    order = np.argsort(codes, kind="stable")
    values = values[order]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return values, starts, counts


def draw_weight_panel(empirical, grid, median, meta):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZES["base"],
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
            "xtick.color": "black",
            "ytick.color": "black",
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
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )
    fig, ax = make_panel()
    cmap = LinearSegmentedColormap.from_list(
        "viridis_readable", plt.get_cmap("viridis")(np.linspace(0.03, 0.88, 256))
    )
    norm = LogNorm(meta.n_nodes.min(), meta.n_nodes.max())
    for k in np.argsort(meta.n_nodes.to_numpy()):
        x, y = empirical[k]
        ax.step(
            x,
            y,
            where="pre",
            color=cmap(norm(meta.n_nodes.iloc[k])),
            lw=0.35,
            alpha=0.16,
            zorder=2,
        )
    positive = median > 0
    ax.plot(
        grid[positive],
        median[positive],
        color="#202629",
        lw=1.4,
        zorder=6,
        label="MSA median",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(grid[0], 1.10 * grid[-1]),
        ylim=(1e-5, 1.12),
        xlabel=r"Dependency weight, $w_{ij}$",
        ylabel=r"$P(W\geq w)$",
    )
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.legend(
        frameon=False,
        loc="lower left",
        handlelength=1.1,
        handletextpad=0.5,
        borderaxespad=0.35,
        labelspacing=0.4,
    )
    if len(fig.axes) != 1:
        raise ValueError(
            "The plot must contain exactly one axes and no embedded colorbar."
        )
    return fig


def main():
    work_dir, figures = ROOT / ".work", ROOT / "figures"
    if not (work_dir / "ready.json").exists():
        raise RuntimeError("Prepare the fresh network results before plotting.")
    meta = pd.read_csv(work_dir / "msa_summary.csv", dtype={"GEOID": str})
    meta = meta.loc[meta.LSAD.eq("M1")].sort_values("GEOID").reset_index(drop=True)
    if meta.empty or meta.GEOID.duplicated().any() or (meta.n_edges <= 0).any():
        raise ValueError("Expected unique MSAs with positive edge counts.")
    config = json.loads((work_dir / "plot_config.json").read_text())
    if config["node_count_vmin"] != int(meta.n_nodes.min()) or config[
        "node_count_vmax"
    ] != int(meta.n_nodes.max()):
        raise ValueError(
            "MSA node counts disagree with the shared colorbar normalization."
        )
    font_path = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    values, starts, sizes = load_internal_weights(meta)
    bounds = np.floor(np.log10(values.min())), np.ceil(np.log10(values.max()))
    grid = np.logspace(*bounds, 1000)
    curves, empirical = [], []
    for start, n in zip(starts, sizes):
        s = np.sort(values[start : start + n])
        curve = (n - np.searchsorted(s, grid, side="left")) / n
        x = np.unique(s)
        y = (n - np.searchsorted(s, x, side="left")) / n
        if y[0] != 1 or (np.diff(y) > 0).any():
            raise ValueError("Invalid empirical CCDF normalization or monotonicity.")
        curves.append(curve)
        empirical.append((x, y))
    curves = np.asarray(curves)
    if np.any(np.diff(curves, axis=1) > 0) or (curves < 0).any() or (curves > 1).any():
        raise ValueError("CCDF must be nonincreasing and bounded by [0,1].")
    median = np.median(curves, axis=0)
    q25, q75 = np.percentile(curves, [25, 75], axis=0)
    fit_mask = (median >= 0.001) & (median <= 0.1)
    if fit_mask.sum() < 3:
        raise ValueError("Insufficient support for the descriptive tail trend.")
    lx, ly = np.log10(grid[fit_mask]), np.log10(median[fit_mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    prediction = slope * lx + intercept
    r2 = 1 - np.sum((ly - prediction) ** 2) / np.sum((ly - ly.mean()) ** 2)
    pd.DataFrame(
        dict(
            value=grid,
            ccdf_median=median,
            ccdf_q25=q25,
            ccdf_q75=q75,
            n_msa_positive=(curves > 0).sum(axis=0),
            used_for_tail_fit=fit_mask,
        )
    ).to_csv(work_dir / "msa_dependency_weight_ccdf_summary.csv", index=False)
    np.savez_compressed(
        work_dir / "msa_dependency_weight_ccdf_curves.npz",
        geoids=meta.GEOID.to_numpy(dtype=str),
        value=grid,
        ccdf=curves,
    )
    notes = {
        "n_msa": len(meta),
        "n_observations": int(sizes.sum()),
        "input": ".work/network_edges.parquet",
        "n_zero": int((values == 0).sum()),
        "minimum_weight": float(values.min()),
        "maximum_weight": float(values.max()),
        "metric": "wij",
        "metric_definition": "Within-MSA directed-edge weight WIJ = NIJ/NI_SOURCE in the cleaned NIJ > 50 network. The cleaned national source denominator is retained when restricting to internal edges.",
        "ccdf_definition": "P(W >= w); each MSA normalized separately by its own number of internal directed edges.",
        "median_definition": "Pointwise median, each MSA equally weighted; zero CCDF values retained in summaries.",
        "grid": {"n_points": len(grid), "log10_bounds": [float(x) for x in bounds]},
        "fit_method": "Descriptive equal-log-grid OLS on log10 of the pointwise median CCDF; not displayed in the figure.",
        "fit_probability_window": [0.001, 0.1],
        "fit_s_min": float(grid[fit_mask].min()),
        "fit_s_max": float(grid[fit_mask].max()),
        "n_fit_grid_points": int(fit_mask.sum()),
        "ccdf_exponent": float(-slope),
        "log10_amplitude": float(intercept),
        "descriptive_r2": float(r2),
        "fit_displayed": False,
        "fit_interpretation": "Not a power-law goodness-of-fit test. Grid points are dependent; no regression p-value or CI is reported. A straight segment does not establish a common power-law distribution across cities.",
        "figure_size_inches": list(FIGURE_SIZE),
        "figure_size_cm": [WIDTH_CM, HEIGHT_CM],
        "dpi": 600,
        "tight_crop": False,
        "font_family": "Arial",
        "font_file": font_path,
        "font_size_pt": FONT_SIZES["base"],
        "axis_label_size_pt": FONT_SIZES["axis_label"],
        "tick_and_legend_size_pt": FONT_SIZES["tick_label"],
        "color": {
            "quantity": "national-network nodes in each MSA",
            "normalization": "log",
            "vmin": int(meta.n_nodes.min()),
            "vmax": int(meta.n_nodes.max()),
            "colormap": "viridis",
            "sample_interval": [0.03, 0.88],
            "samples": 256,
            "embedded_colorbar": False,
        },
        "output": "figures/msa_dependency_weight_ccdf.png",
    }
    figures.mkdir(exist_ok=True)
    fig = draw_weight_panel(empirical, grid, median, meta)
    fig.savefig(
        figures / "msa_dependency_weight_ccdf.png",
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    (work_dir / "msa_dependency_weight_ccdf_notes.json").write_text(
        json.dumps(notes, indent=2) + "\n"
    )
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
