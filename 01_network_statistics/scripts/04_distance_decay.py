#!/usr/bin/env python3
"""Estimate distance decay over 0.3-10 km and its association with network size."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr
from sklearn.neighbors import BallTree


ROOT = Path(__file__).resolve().parents[1]
BIN_EDGES_KM = np.logspace(-1.0, 3.0, 17)
BIN_EDGES_M = BIN_EDGES_KM * 1000.0
CENTERS_KM = np.sqrt(BIN_EDGES_KM[:-1] * BIN_EDGES_KM[1:])
N_BINS = len(CENTERS_KM)
EARTH_RADIUS_M = 6_371_000.0
NEAR_RADIUS_M = 10**1.5 * 1000.0
SAMPLED_SOURCES = 512
SAMPLED_PAIRS = 500_000
SEED = 12345
BOOTSTRAPS = 2000
MIN_BIN_EDGES = 20
MIN_FIT_CENTER_KM = 0.3
MIN_FIT_BINS = 3


STATE: dict = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def haversine_m(la1, lo1, la2, lo2):
    """Great-circle distances, coordinates supplied in radians."""
    dphi, dlambda = la2 - la1, lo2 - lo1
    a = np.sin(dphi / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_total = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - predicted) ** 2) / ss_total if ss_total > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def estimate_opportunities(coords, rng, query_batch=32):
    """Estimate directed node-pair opportunities using batched spatial queries."""
    n = len(coords)
    tree = BallTree(coords, metric="haversine")
    sources = rng.choice(n, min(SAMPLED_SOURCES, n), replace=False)
    near_counts = np.zeros(N_BINS, dtype=np.float64)
    for start in range(0, len(sources), query_batch):
        sampled = sources[start : start + query_batch]
        neighbors, distances = tree.query_radius(
            coords[sampled], r=NEAR_RADIUS_M / EARTH_RADIUS_M, return_distance=True
        )
        for source, nearby, distance in zip(sampled, neighbors, distances):
            distance = distance[nearby != source] * EARTH_RADIUS_M
            bins = np.digitize(distance, BIN_EDGES_M) - 1
            good = (bins >= 0) & (bins < N_BINS)

            near_counts += np.bincount(bins[good], minlength=N_BINS)
    near_counts *= n / len(sources)

    source = rng.integers(0, n, SAMPLED_PAIRS)
    target = rng.integers(0, n, SAMPLED_PAIRS)
    different = source != target
    source, target = source[different], target[different]
    distance = haversine_m(
        coords[source, 0], coords[source, 1], coords[target, 0], coords[target, 1]
    )
    bins = np.digitize(distance, BIN_EDGES_M) - 1
    good = (bins >= 0) & (bins < N_BINS)
    far_counts = np.bincount(bins[good], minlength=N_BINS).astype(np.float64)
    far_estimate = far_counts / max(int(different.sum()), 1) * (float(n) * (n - 1))
    near = BIN_EDGES_M[1:] <= NEAR_RADIUS_M
    return np.where(near, near_counts, far_estimate)


def compute_msa(k):
    info = STATE["metadata"].iloc[k]
    ids = STATE["node_groups"][k]
    coords = STATE["coords"][ids]
    estimate = estimate_opportunities(coords, np.random.default_rng(SEED + k))
    start, end = STATE["edge_starts"][k], STATE["edge_ends"][k]
    edge_bins = STATE["edge_bins"][start:end]
    weights = STATE["weights"][start:end]
    count = np.zeros(N_BINS, dtype=np.int64)
    weight_sum = np.zeros(N_BINS, dtype=np.float64)
    positive_mean = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        wb = weights[edge_bins == b]
        count[b] = len(wb)
        if len(wb):
            weight_sum[b] = wb.sum()
            positive_mean[b] = wb.mean()
    opportunities = np.maximum(estimate, count.astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = np.where(opportunities > 0, weight_sum / opportunities, np.nan)
        p_edge = np.where(opportunities > 0, count / opportunities, np.nan)
    usable = (
        (count >= MIN_BIN_EDGES)
        & (opportunities > 0)
        & np.isfinite(mean_w)
        & (mean_w > 0)
        & (CENTERS_KM >= MIN_FIT_CENTER_KM)
    )
    max_fit_center_km = STATE.get("max_fit_center_km")
    if max_fit_center_km is not None:
        usable &= CENTERS_KM <= max_fit_center_km
    fit_centers = CENTERS_KM[usable]
    if usable.sum() >= MIN_FIT_BINS:
        beta, intercept, r2 = fit_line(np.log10(fit_centers), np.log10(mean_w[usable]))
        status = "fitted"
    else:
        beta, intercept, r2 = np.nan, np.nan, np.nan
        status = "insufficient_fit_bins"
    row = {
        "GEOID": str(info.GEOID),
        "NAME": info.NAME,
        "LSAD": info.LSAD,
        "n_nodes": len(ids),
        "n_edges": end - start,
        "beta": beta,
        "log10_intercept": intercept,
        "r2": r2,
        "n_fit_bins": int(usable.sum()),
        "fit_dmin_km": float(fit_centers.min()) if len(fit_centers) else np.nan,
        "fit_dmax_km": float(fit_centers.max()) if len(fit_centers) else np.nan,
        "fit_centers_km": ";".join(f"{v:.12g}" for v in fit_centers),
        "sample_seed": SEED + k,
        "status": status,
    }
    bins = [
        {
            "GEOID": str(info.GEOID),
            "bin_left_km": BIN_EDGES_KM[b],
            "bin_right_km": BIN_EDGES_KM[b + 1],
            "bin_center_km": CENTERS_KM[b],
            "n_pos_edges": int(count[b]),
            "sum_w": weight_sum[b],
            "n_possible_pairs_raw": estimate[b],
            "n_possible_pairs": opportunities[b],
            "p_edge": p_edge[b],
            "mean_w": mean_w[b],
            "mean_w_given_edge": positive_mean[b],
            "used_in_beta_fit": bool(usable[b]),
        }
        for b in range(N_BINS)
    ]
    return k, row, bins


def configure_fonts():
    arial = Path(font_manager.findfont("Arial", fallback_to_default=False)).resolve()
    if (
        not arial.is_file()
        or font_manager.FontProperties(fname=str(arial)).get_name() != "Arial"
    ):
        raise RuntimeError("Install Arial to reproduce the figure typography.")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "axes.labelpad": 5,
            "text.color": "#252a2e",
            "axes.labelcolor": "#252a2e",
            "axes.edgecolor": "#42494e",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial:italic",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.fallback": None,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )
    return str(arial)


def render_scatter(df, figures_dir, results_dir, *, write_results=True):
    font_path = configure_fonts()
    fitted = df.loc[df.status == "fitted"].copy()
    x, y = np.log10(fitted.n_nodes.to_numpy(float)), fitted.beta.to_numpy(float)
    slope, intercept, r2 = fit_line(x, y)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    pearson_p = float(pearsonr(x, y, alternative="two-sided").pvalue)
    if pearson_p == 0:
        p_label = r"p < 10^{-308}"
    elif pearson_p < 0.001:
        mantissa, exponent = f"{pearson_p:.2e}".split("e")
        p_label = rf"p = {mantissa} \times 10^{{{int(exponent)}}}"
    else:
        p_label = f"p = {pearson_p:.3f}"
    xgrid = np.linspace(x.min(), x.max(), 150)
    rng = np.random.default_rng(SEED)
    bands = []
    for _ in range(BOOTSTRAPS):
        ix = rng.integers(0, len(x), len(x))
        m, c, _ = fit_line(x[ix], y[ix])
        bands.append(m * xgrid + c)
    low, high = np.percentile(np.asarray(bands), [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(4.5, 3.6), layout="constrained")
    ax.fill_between(10**xgrid, low, high, color="#3B6EA5", alpha=0.13, lw=0)
    ax.scatter(
        10**x,
        y,
        s=9.6,
        color="#3B6EA5",
        alpha=0.60,
        edgecolors="white",
        linewidths=0.12,
        zorder=3,
    )
    ax.plot(10**xgrid, slope * xgrid + intercept, color="#253e50", lw=1.05, zorder=4)
    ax.text(
        0.97,
        0.97,
        f"$n={len(fitted)}$ MSAs\nPearson $r={pearson_r:.3f}$\n${p_label}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        linespacing=1.4,
    )
    ax.set(
        xscale="log",
        xlabel="Nodes per MSA, $N$",
        ylabel=r"Distance-decay exponent, $\beta$",
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(
        handles=[Line2D([], [], color="#253e50", lw=1.05, label="Linear fit")],
        frameon=False,
        loc="lower left",
        handlelength=1.8,
    )
    fig.savefig(
        figures_dir / "msa_nodes_vs_distance_decay.png",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    summary = {
        "n_msa": len(fitted),
        "n_msa_input": len(df),
        "median_beta": float(fitted.beta.median()),
        "median_beta_fit_r2": float(fitted.r2.median()),
        "regression_slope": slope,
        "regression_intercept": intercept,
        "pearson_r": pearson_r,
        "pearson_p_two_sided": pearson_p,
        "regression_r2": r2,
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "font_family": "Arial",
    }
    if write_results:
        work_dir = results_dir.parent / ".work"
        work_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(
            results_dir / "nodes_vs_distance_decay_regression.csv", index=False
        )
        pd.DataFrame(
            {
                "n_nodes": 10**xgrid,
                "beta_fit": slope * xgrid + intercept,
                "beta_bootstrap_lower_95": low,
                "beta_bootstrap_upper_95": high,
            }
        ).to_csv(work_dir / "nodes_vs_distance_decay_bootstrap_band.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Write results/, figures/, and .work/ here; read prepared tables from --root.",
    )
    fit_range = parser.add_mutually_exclusive_group()
    fit_range.add_argument(
        "--max-fit-center-km",
        type=float,
        default=10.0,
        help="Upper cutoff on fitted bin centers in km; default 10.",
    )
    fit_range.add_argument(
        "--no-fit-upper-limit",
        dest="max_fit_center_km",
        action="store_const",
        const=None,
        help="Reproduce the earlier fit with no shared upper cutoff.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_fit_center_km is not None and (
        not np.isfinite(args.max_fit_center_km)
        or args.max_fit_center_km < MIN_FIT_CENTER_KM
    ):
        parser.error("--max-fit-center-km must be finite and at least 0.3")
    input_dir = args.root / ".work"
    output_root = args.output_root if args.output_root is not None else args.root
    results_dir, figures_dir = output_root / "results", output_root / "figures"
    work_dir = output_root / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ready_path = input_dir / "ready.json"
    if not ready_path.is_file():
        raise RuntimeError(
            "Prepared tables are not ready: run the preparation script first."
        )
    input_paths = [
        input_dir / "network_nodes.parquet",
        input_dir / "network_edges.parquet",
        input_dir / "msa_summary.csv",
        ready_path,
        input_dir / "preparation_manifest.json",
    ]
    log("Reading newly prepared nodes and intra-MSA edges")
    nodes = pq.read_table(
        input_paths[0], columns=["node_id", "node", "GEOID", "LATITUDE", "LONGITUDE"]
    ).to_pandas()
    nodes["GEOID"] = nodes.GEOID.astype(str)
    nodes = nodes.sort_values("node_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(nodes.node_id.to_numpy(), np.arange(len(nodes))):
        raise ValueError("Prepared node_id must be unique and contiguous from zero.")
    metadata = (
        pd.read_csv(input_paths[2], dtype={"GEOID": str})
        .sort_values("GEOID")
        .reset_index(drop=True)
    )
    if not (metadata.LSAD == "M1").all():
        raise ValueError("This figure requires only metropolitan statistical areas.")
    gid_to_k = dict(zip(metadata.GEOID, metadata.index))
    group = nodes.GEOID.map(gid_to_k)
    if group.isna().any():
        raise ValueError("Every active node must belong to the prepared MSA universe.")
    group = group.to_numpy(dtype=np.int64)
    coords = np.radians(nodes[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
    if not np.isfinite(coords).all():
        raise ValueError("Active nodes have missing or nonfinite coordinates.")

    ordered = nodes.sort_values(["GEOID", "node"], kind="stable")
    node_groups = [
        ordered.loc[ordered.GEOID == g, "node_id"].to_numpy(np.int64)
        for g in metadata.GEOID
    ]
    edge = pq.read_table(
        input_paths[1],
        columns=["src_idx", "tgt_idx", "WIJ", "NI_SOURCE", "is_intra_msa"],
    ).to_pandas()
    edge = edge.loc[
        edge.is_intra_msa & (edge.NI_SOURCE > 0) & np.isfinite(edge.WIJ)
    ].copy()
    sid, tid, weight = (
        edge.src_idx.to_numpy(np.int64),
        edge.tgt_idx.to_numpy(np.int64),
        edge.WIJ.to_numpy(float),
    )
    if not np.all(group[sid] == group[tid]) or not np.all(weight > 0):
        raise ValueError("Expected positive intra-MSA edges.")
    edge_group = group[sid]
    counts = np.bincount(edge_group, minlength=len(metadata))
    node_counts = np.bincount(group, minlength=len(metadata))
    if not np.array_equal(counts, metadata.n_edges) or not np.array_equal(
        node_counts, metadata.n_nodes
    ):
        raise ValueError(
            "Prepared summary does not match the active nodes / intra-MSA edges."
        )
    if np.any(counts < 200) or np.any(node_counts < 50):
        raise ValueError("An MSA does not meet the minimum of 50 nodes and 200 edges.")
    distance = haversine_m(
        coords[sid, 0], coords[sid, 1], coords[tid, 0], coords[tid, 1]
    )
    edge_bins = np.digitize(distance, BIN_EDGES_M) - 1
    edge_order = np.argsort(edge_group, kind="stable")
    sorted_group = edge_group[edge_order]
    STATE.update(
        metadata=metadata,
        node_groups=node_groups,
        coords=coords,
        max_fit_center_km=args.max_fit_center_km,
        edge_starts=np.searchsorted(
            sorted_group, np.arange(len(metadata)), side="left"
        ),
        edge_ends=np.searchsorted(sorted_group, np.arange(len(metadata)), side="right"),
        edge_bins=edge_bins[edge_order],
        weights=weight[edge_order],
    )
    distance_audit = {
        "intra_msa_edges": len(edge),
        "below_0_1_km": int((edge_bins < 0).sum()),
        "at_or_above_1000_km": int((edge_bins >= N_BINS).sum()),
        "in_distance_bins": int(((edge_bins >= 0) & (edge_bins < N_BINS)).sum()),
    }
    del (
        edge,
        nodes,
        ordered,
        sid,
        tid,
        edge_order,
        weight,
        edge_group,
        sorted_group,
        distance,
    )
    log(
        f"Recomputing opportunity estimates for {len(metadata)} MSAs with {args.workers} workers"
    )
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for result in pool.imap_unordered(compute_msa, range(len(metadata))):
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(metadata):
                log(f"Computed {len(results)} / {len(metadata)} MSAs")
    results.sort(key=lambda item: item[0])
    df = pd.DataFrame([item[1] for item in results])
    bins = pd.DataFrame([row for item in results for row in item[2]])
    df.to_csv(results_dir / "msa_distance_decay_beta.csv", index=False)
    bins.to_csv(work_dir / "msa_distance_decay_bins.csv", index=False)
    if not (df.status == "fitted").all():
        raise RuntimeError(
            "Some MSAs have too few supported bins; inspect saved status before plotting."
        )
    regression = render_scatter(df, figures_dir, results_dir)
    provenance = {
        "node_universe": "All cleaned national-network active polygon IDs assigned to each MSA, including cross-MSA-only nodes.",
        "node_coordinate": "Stored LATITUDE/LONGITUDE of exact full PLACEKEY polygon ID.",
        "numerator": "Exact sum of newly cleaned intra-MSA WIJ per distance bin; missing edges have zero weight.",
        "bin_edges_km": BIN_EDGES_KM.tolist(),
        "earth_radius_m": EARTH_RADIUS_M,
        "sampled_sources_without_replacement": SAMPLED_SOURCES,
        "near_radius_m": NEAR_RADIUS_M,
        "near_estimator": "All neighbors of sampled sources, excluding self; counts scaled by N / sampled sources.",
        "far_pair_draws": SAMPLED_PAIRS,
        "far_estimator": "Independent uniform source/target draws; discard self pairs; bin fractions times N(N-1).",
        "opportunity_lower_bound": "max(raw opportunity estimate, exact observed edge count) per bin",
        "seed": SEED,
        "per_msa_seed_rule": "12345 + zero-based rank among current 376 GEOIDs sorted lexicographically",
        "seed_scope_note": "Original upstream ranks all spatially matched CBSAs. The current cleaned universe contains 376 MSAs only.",
        "within_msa_node_order": "lexicographic full polygon node/PLACEKEY string",
        "min_observed_edges_per_fit_bin": MIN_BIN_EDGES,
        "min_fit_center_km": MIN_FIT_CENTER_KM,
        "min_fit_bins": MIN_FIT_BINS,
        "max_fit_center_km": args.max_fit_center_km,
        "max_fit_distance": (
            "Each MSA's largest supported distance bin; no shared upper cutoff."
            if args.max_fit_center_km is None
            else f"Supported bin centers <= {args.max_fit_center_km:g} km."
        ),
        "fit": "Equal-bin-weight OLS: log10(mean WIJ) = beta*log10(distance in km) + intercept.",
        "regression": regression,
        "confidence_band": "Pointwise 95% percentile band over 2,000 resamples of MSAs; regression mean only, not prediction or propagated beta-estimation uncertainty.",
        "distance_audit": distance_audit,
        "inputs": {
            str(p.resolve()): {
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in input_paths
        },
        "preparation_provenance": json.loads(
            (input_dir / "preparation_manifest.json").read_text()
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure": {
            "filename": "msa_nodes_vs_distance_decay.png",
            "canvas_inches": [4.5, 3.6],
            "dpi": 600,
            "bbox_inches": "tight",
            "font_family": "Arial",
            "colorbar": False,
        },
    }
    (work_dir / "distance_decay_metadata.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    log(
        f"Saved plot; {len(df)} MSA fits, median beta={df.beta.median():.6f}, Pearson r={regression['pearson_r']:.6f}"
    )


if __name__ == "__main__":
    main()
