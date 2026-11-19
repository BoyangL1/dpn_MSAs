"""Render primary GNN transferability and the eight-cluster partition."""

from __future__ import annotations
import argparse
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_gnn_matplotlib")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, MaxNLocator
from .cluster_analysis import analyze_matrix, case_at_k
from .select_analysis_cohort import retained_ids

ROOT = Path(__file__).resolve().parents[2]
VIEW = "gnn_all_pairs"
COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#354B5E",
    "#800000",
]
MAP_LAND, MAP_STATE_EDGE = ("#F5F5F3", "#8B908E")
MAP_UNASSIGNED, MAP_UNASSIGNED_EDGE = ("#E8E9E7", "#D8DAD7")
INK, WARM, ACCENT = ("#28617f", "#91816d", "#B75B32")
from .data_paths import CBSA as CBSA_SHP, STATES as STATE_SHP


def style(size=10):
    font = findfont(FontProperties(family="Arial"), fallback_to_default=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size - 1,
            "ytick.labelsize": size - 1,
            "axes.linewidth": 0.7,
            "text.color": "#263640",
            "axes.labelcolor": "#263640",
            "axes.edgecolor": "#34434c",
            "xtick.color": "#34434c",
            "ytick.color": "#34434c",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.fallback": None,
        }
    )
    return font


def save(fig, path, dpi=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for subfig in getattr(fig, "subfigs", []):
        texts.extend(subfig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
    for text in texts:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert (
                box.x0 >= fig.bbox.x0 - 1
                and box.y0 >= fig.bbox.y0 - 1
                and (box.x1 <= fig.bbox.x1 + 1)
                and (box.y1 <= fig.bbox.y1 + 1)
            ), (path.name, text.get_text(), box)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def short_name(name, include_state=True):
    name = str(name)
    city, _, state = name.rpartition(",")
    if not city:
        city, state = (name, "")
    for principal in ("Winston-Salem", "Wilkes-Barre", "Bristol", "Dover"):
        if city.startswith(principal):
            city = principal
            break
    else:
        city = city.split("-")[0]
    if city == "Louisville/Jefferson County":
        city = "Louisville"
    if city == "New York":
        city = "New York City"
    state = state.strip().split("-")[0]
    return f"{city}, {state}" if include_state and state else city


def read_matrix(root, view=VIEW):
    if view != VIEW:
        raise ValueError("Only the primary GNN transfer matrix is supported")
    paths = [
        root / "results/gnn_all_pairs/transfer_pearson_analysis.csv",
        root / ".work/gnn_all_pairs/transfer_pearson.csv",
    ]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Run 02_train_transfer.py and 03_select_cohort.py first"
        )
    frame = pd.read_csv(path, index_col=0)
    frame.index, frame.columns = (frame.index.astype(str), frame.columns.astype(str))
    assert frame.index.is_unique and frame.columns.is_unique
    assert set(frame.index) == set(frame.columns)
    frame = frame.loc[frame.index, frame.index]
    assert np.isfinite(frame.to_numpy()).all(), "Undefined transfer coefficients"
    assert np.abs(frame.to_numpy()).max() <= 1 + 1e-06
    return frame


def load_meta(root, matrix):
    candidates = [
        root / ".work/msa_metadata.csv",
        root / "results/gnn_all_pairs/k_8/cluster_membership.csv",
        root / ".work/build/msa_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            meta = pd.read_csv(candidate, dtype=str)
            meta = meta.rename(
                columns={
                    "msa_geoid": "geoid",
                    "GEOID": "geoid",
                    "NAME": "name",
                    "msa_name": "name",
                }
            )
            if {"geoid", "name", "n_nodes"} <= set(meta.columns):
                meta["n_nodes"] = pd.to_numeric(meta.n_nodes)
                meta = meta.set_index("geoid")
                if set(matrix.index) <= set(meta.index):
                    return meta.loc[matrix.index].copy()
    raise FileNotFoundError("Need MSA metadata with geoid, name, n_nodes")


def heatmap(case, norm, out):
    """Plot observed source-to-target scores using the existing city clusters.

    Rows are training/source MSAs and columns are test/target MSAs. The
    symmetric similarity is used only to construct the supplied hierarchy.
    """
    style(15)
    order, z = (case["order"], case["z"])
    order = order[np.argsort(case["labels"][order], kind="stable")]
    labels = case["labels"][order]
    values = case["values"][np.ix_(order, order)]
    inverse = np.argsort(order)
    np.testing.assert_array_equal(values[np.ix_(inverse, inverse)], case["values"])
    width, height, size, left, bottom = (180.0, 163.5, 128.333, 20.2, 19.1)
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=150)

    def panel(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    top = panel(left, bottom + size + 5.2, size, 8.9)
    side = panel(10.2, bottom, 6.8, size)
    n = len(order)
    centers = np.zeros(2 * n - 1)
    centers[order] = np.arange(n)
    heights = np.r_[np.zeros(n), z[:, 2]]
    groups = np.r_[case["labels"], np.full(n - 1, -1)]
    included = []
    for i, (left_node, right_node, merge_height, _) in enumerate(z):
        left_node, right_node = (int(left_node), int(right_node))
        centers[n + i] = (centers[left_node] + centers[right_node]) / 2
        if groups[left_node] == groups[right_node] and groups[left_node] > 0:
            groups[n + i] = groups[left_node]
            x = [
                centers[left_node],
                centers[left_node],
                centers[right_node],
                centers[right_node],
            ]
            y = [heights[left_node], merge_height, merge_height, heights[right_node]]
            top.plot(x, y, color="#94999D", lw=0.55)
            side.plot(y, x, color="#94999D", lw=0.55)
            included.append(i)
    assert len(included) == n - case["k"]
    high = float(z[included, 2].max()) * 1.05 if included else 1.0
    top.set(xlim=(-0.5, n - 0.5), ylim=(0, high))
    side.set(xlim=(high, 0), ylim=(n - 0.5, -0.5))
    top.axis("off")
    side.axis("off")
    rgba = np.array([to_rgba(COLORS[c - 1]) for c in labels])
    for bounds, array in (
        ((left, bottom + size + 2.5, size, 1.7), rgba[None]),
        ((17.5, bottom, 1.7, size), rgba[:, None]),
    ):
        ax = panel(*bounds)
        ax.imshow(array, aspect="auto", interpolation="nearest")
        ax.axis("off")
    ax = panel(left, bottom, size, size)
    image = ax.imshow(
        values, cmap="magma_r", norm=norm, interpolation="nearest", aspect="equal"
    )
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    np.testing.assert_array_equal(labels[starts], np.arange(1, case["k"] + 1))
    mids = (starts + ends - 1) / 2
    positions = mids.copy()
    min_gap = len(labels) * 0.052
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + min_gap)
    if positions[-1] > len(labels) - 1:
        positions -= positions[-1] - (len(labels) - 1)
    for i in range(len(positions) - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - min_gap)
    ax.set_xticks(
        positions, [f"C{labels[s]}" for s in starts], rotation=90, fontsize=16.5
    )
    ax.set_yticks([])
    ax.tick_params(length=0, pad=5)
    for original, moved in zip(mids, positions):
        if not np.isclose(original, moved):
            ax.plot(
                [original, moved],
                [0, -0.023],
                color="#94999D",
                lw=0.5,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in starts[1:]:
        ax.axhline(boundary - 0.5, color="white", alpha=0.5, lw=0.8)
        ax.axvline(boundary - 0.5, color="white", alpha=0.5, lw=0.8)
    for start, end in zip(starts, ends):
        ax.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                end - start,
                end - start,
                fill=False,
                edgecolor="#FFEB00",
                lw=2,
                clip_on=False,
            )
        )
    fig.text(
        (left + size / 2) / width,
        4.3 / height,
        "Target MSA",
        ha="center",
        va="center",
        fontsize=18,
    )
    fig.text(
        4.3 / width,
        (bottom + size / 2) / height,
        "Source MSA",
        rotation=90,
        ha="center",
        va="center",
        fontsize=18,
    )
    cax = panel(152.0, 27.9, 3.0, 111.0)
    step = 0.1 if norm.vmax - norm.vmin < 0.75 else 0.2
    ticks = np.arange(np.ceil(norm.vmin / step), np.floor(norm.vmax / step) + 1) * step
    colorbar = fig.colorbar(image, cax=cax, ticks=ticks)
    colorbar.set_label("Directed Pearson r", fontsize=18, labelpad=9)
    colorbar.ax.tick_params(labelsize=15, pad=5, length=2.5, width=0.6)
    colorbar.outline.set_linewidth(0.5)
    assert colorbar.mappable is image and colorbar.norm is norm
    np.testing.assert_array_equal(image.get_array(), values)
    fig.canvas.draw()
    boxes = [
        label.get_window_extent(fig.canvas.get_renderer())
        for label in ax.get_xticklabels()
    ]
    assert all(
        (a.x1 + 1 <= b.x0 for a, b in zip(boxes, boxes[1:]))
    ), "Heatmap cluster labels overlap"
    for label in colorbar.ax.get_yticklabels():
        box = label.get_window_extent(fig.canvas.get_renderer())
        if norm.vmin <= label.get_position()[1] <= norm.vmax:
            assert (
                box.x0 >= 0 and box.x1 <= fig.bbox.width
            ), "Colorbar tick label clipped"
    save(fig, out / "transfer_heatmap.png", dpi=900)


def geometry():
    states = gpd.read_file(STATE_SHP)
    states = states.loc[
        ~states.STUSPS.isin({"AK", "HI", "PR", "GU", "VI", "MP", "AS"})
    ].to_crs(5070)
    msa = gpd.read_file(CBSA_SHP)
    msa = msa.loc[msa.LSAD.eq("M1")].to_crs(5070)
    msa = msa.loc[msa.intersects(states.union_all())].copy()
    return (msa, states)


def cluster_map(case, meta, excluded, out, geo, legend_examples=None):
    style(44)
    plt.rcParams["hatch.linewidth"] = 1.9
    msa, states = geo
    msa = msa.copy()
    msa["cluster"] = msa.GEOID.map(dict(zip(case["ids"], case["labels"])))
    missing = (set(case["ids"]) | set(excluded)) - set(msa.GEOID)
    assert not missing, f"MSA boundaries missing: {missing}"
    fig = plt.figure(figsize=(30, 30 * 8.5 / 18 * max(1, case["k"] / 8)))
    ax = fig.add_axes([0.005, 0.025, 0.635, 0.95])
    states.plot(ax=ax, facecolor=MAP_LAND, edgecolor="none", zorder=0)
    msa.plot(
        ax=ax,
        facecolor=MAP_UNASSIGNED,
        edgecolor=MAP_UNASSIGNED_EDGE,
        linewidth=0.5,
        zorder=1,
    )
    for c in range(1, case["k"] + 1):
        msa.loc[msa.cluster.eq(c)].plot(
            ax=ax, color=COLORS[c - 1], edgecolor="white", linewidth=0.93, zorder=3
        )
    dropped = msa.loc[msa.GEOID.isin(excluded)]
    if len(dropped):
        dropped.plot(
            ax=ax,
            facecolor="#D3D3D3",
            edgecolor="#888888",
            linewidth=0.93,
            hatch="///",
            zorder=3,
        )
    states.boundary.plot(ax=ax, color=MAP_STATE_EDGE, linewidth=1.35, zorder=4)
    ax.margins(0.01)
    ax.axis("off")
    entries, examples = ([], [])
    renderer = fig.canvas.get_renderer()
    legend_font = FontProperties(family="Arial", size=40)
    max_text_width = (0.995 - 0.675) * fig.bbox.width

    def label_width(label):
        return max(
            (
                renderer.get_text_width_height_descent(line, legend_font, ismath=False)[
                    0
                ]
                for line in label.split("\n")
            )
        )

    for c in range(1, case["k"] + 1):
        ids = case["ids"][case["labels"] == c]
        if legend_examples is None:
            ranked = (
                meta.loc[ids]
                .rename_axis("geoid")
                .reset_index()
                .sort_values(
                    ["n_nodes", "geoid"], ascending=[False, True], kind="stable"
                )
                .set_index("geoid")
                .head(3)
            )
        else:
            selected = legend_examples.loc[legend_examples.cluster.eq(c)].sort_values(
                "rank"
            )
            assert len(selected) == min(3, len(ids)) and selected.geoid.is_unique
            assert set(selected.geoid) <= set(ids)
            ranked = meta.loc[selected.geoid]
        names = [short_name(name) for name in ranked.name]
        for rank, (gid, row) in enumerate(ranked.iterrows(), 1):
            examples.append(
                dict(
                    cluster=c,
                    geoid=gid,
                    name=row["name"],
                    n_nodes=row.n_nodes,
                    rank=rank,
                )
            )
        label = f"C{c} ({', '.join(names)})"
        if label_width(label) > max_text_width and len(names) > 1:
            alternatives = [
                f"C{c} ({', '.join(names[:split])},\n     {', '.join(names[split:])})"
                for split in range(1, len(names))
            ]
            label = min(alternatives, key=label_width)
            if label_width(label) > max_text_width:
                label = f"C{c} (" + ",\n     ".join(names) + ")"
        assert (
            label_width(label) <= max_text_width
        ), f"Map legend needs a shorter city label: {label}"
        entries.append((label, COLORS[c - 1], False))
    if len(excluded):
        entries.append((f"Excluded ({len(excluded)})", "#D3D3D3", True))
    heights = []
    for label, _, _ in entries:
        temporary = fig.text(0.675, 0.5, label, fontsize=40, linespacing=1.15)
        heights.append(temporary.get_window_extent(renderer).height / fig.bbox.height)
        temporary.remove()
    gap = (0.86 - sum(heights)) / max(len(entries) - 1, 1)
    assert gap >= 0.006, "Map city labels require more vertical room"
    gap = min(gap, 0.08)
    cursor = 0.5 + (sum(heights) + gap * max(len(entries) - 1, 0)) / 2
    for (label, color, is_excluded), height in zip(entries, heights):
        y = cursor - height / 2
        cursor -= height + gap
        swatch_height = 0.014 * fig.get_figwidth() / fig.get_figheight()
        fig.add_artist(
            Rectangle(
                (0.654, y - swatch_height / 2),
                0.014,
                swatch_height,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="#888888" if is_excluded else "none",
                linewidth=1.9,
                hatch="///" if is_excluded else None,
            )
        )
        fig.text(0.675, y, label, fontsize=40, ha="left", va="center", linespacing=1.15)
    save(fig, out / "transfer_cluster_map.png", dpi=300)
    result = pd.DataFrame(examples)
    if legend_examples is not None:
        result = result.merge(
            legend_examples.drop(columns=["name", "n_nodes"]),
            on=["cluster", "geoid", "rank"],
            validate="one_to_one",
        )
    return result


def cluster_sizes(case, out):
    style(10)
    k = int(case["k"])
    counts = np.bincount(case["labels"], minlength=k + 1)[1:]
    fig = plt.figure(figsize=(18 / 2.54, 6.5 / 2.54))
    ax = fig.add_axes([0.1, 0.19, 0.86, 0.64])
    fig.text(
        0.53,
        0.98,
        f"Descriptive cluster sizes · k = {k}",
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.bar(range(1, k + 1), counts, color=COLORS[:k], width=0.65)
    for cluster, count in enumerate(counts, 1):
        ax.text(
            cluster,
            float(count) + max(counts) * 0.025,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        xticks=np.arange(1, k + 1),
        xticklabels=[f"C{i}" for i in range(1, k + 1)],
        xlabel="Cluster",
        ylabel="MSAs",
        ylim=(0, max(counts) * 1.2),
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, pad=2)
    save(fig, out / "cluster_sizes.png")


def overall_distribution(case, out):
    style(10)
    n = len(case["ids"])
    values = case["values"][~np.eye(n, dtype=bool)]
    assert len(values) == n * (n - 1) and np.isfinite(values).all()
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    fig = plt.figure(figsize=(18 / 2.54, 5.5 / 2.54))
    ax = fig.add_axes([0.1, 0.24, 0.85, 0.62])
    ax.hist(
        values,
        bins=np.linspace(-1, 1, 81),
        weights=np.full(len(values), 100 / len(values)),
        color=INK,
        edgecolor="white",
        linewidth=0.45,
    )
    for value in quantiles:
        ax.axvline(value, color=WARM, ls="--", lw=0.9)
    ax.set(
        xlim=(min(-0.025, np.floor(values.min() * 10) / 10 - 0.025), 1),
        xlabel="Cross-city $T_{A\\rightarrow B}$ (Pearson $r$)",
        ylabel="Directed pairs (%)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.text(
        0.02,
        0.94,
        f"{n} MSAs\n{len(values):,} directed pairs",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, pad=2)
    qaxes = (quantiles - ax.get_xlim()[0]) / np.ptp(ax.get_xlim())
    labels = [
        ax.text(
            x,
            1.035,
            f"P{q} = {value:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        for x, q, value in zip(qaxes, (10, 50, 90), quantiles)
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = np.array(
        [label.get_window_extent(renderer).width / ax.bbox.width for label in labels]
    )
    positions = qaxes.copy()
    positions[0] = min(positions[0], positions[1] - (widths[0] + widths[1]) / 2 - 0.018)
    positions[2] = max(positions[2], positions[1] + (widths[1] + widths[2]) / 2 + 0.018)
    for label, actual, position in zip(labels, qaxes, positions):
        label.set_x(position)
        ax.plot(
            [actual, position],
            [1, 1.025],
            color=WARM,
            lw=0.5,
            transform=ax.transAxes,
            clip_on=False,
        )
    fig.canvas.draw()
    boxes = [label.get_window_extent(fig.canvas.get_renderer()) for label in labels]
    assert all(
        (left.x1 < right.x0 for left, right in zip(boxes[:-1], boxes[1:]))
    ), "Overlapping percentile labels"
    save(fig, out / "cross_msa_transferability_distribution.png")
    return dict(
        n_msas=n,
        n_directed_pairs=len(values),
        mean_r=np.mean(values),
        p10_r=quantiles[0],
        median_r=quantiles[1],
        p90_r=quantiles[2],
        min_r=values.min(),
        max_r=values.max(),
        sd_r=values.std(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip representative network profiles",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Skip the map requiring geographic inputs",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    matrix = read_matrix(root)
    training_path = root / ".work/within_msa_performance.csv"
    if training_path.is_file():
        training = pd.read_csv(training_path, dtype={"geoid": str}).set_index("geoid")
        ids = retained_ids(training)
        excluded = training.index.difference(ids).to_numpy()
        if not set(ids) <= set(matrix.index):
            raise ValueError("The transfer matrix does not match the selected cohort")
        matrix = matrix.loc[ids, ids]
    else:
        if not args.skip_map:
            raise FileNotFoundError(
                "Run 03_select_cohort.py to identify excluded MSAs for the map"
            )
        ids = matrix.index.to_numpy()
        excluded = np.array([], dtype=str)
    meta = load_meta(root, matrix)
    case = case_at_k(analyze_matrix(matrix, ids), 8)
    output = root / "figures/gnn_all_pairs"
    cut_output = output / "k_8"
    results = root / "results/gnn_all_pairs"
    (results / "k_8").mkdir(parents=True, exist_ok=True)
    matrix.to_csv(results / "transfer_pearson_analysis.csv")
    membership = pd.DataFrame(
        {
            "geoid": case["ids"],
            "name": meta.loc[case["ids"], "name"].to_numpy(),
            "cluster": case["labels"],
            "n_nodes": meta.loc[case["ids"], "n_nodes"].to_numpy(),
            "leaf_position": np.argsort(case["order"]),
        }
    )
    membership.sort_values("leaf_position").to_csv(
        results / "k_8/cluster_membership.csv", index=False
    )
    off_diagonal = case["values"][~np.eye(len(case["values"]), dtype=bool)]
    norm = Normalize(
        vmin=float(np.percentile(off_diagonal, 0.5)),
        vmax=float(off_diagonal.max()),
        clip=True,
    )
    heatmap(case, norm, cut_output)
    cluster_sizes(case, cut_output)
    overall_distribution(case, output)
    if not args.skip_map:
        from .map_legend_representatives import curated_map_examples

        examples = curated_map_examples(root, VIEW, case, meta)
        cluster_map(
            case, meta, excluded, cut_output, geometry(), legend_examples=examples
        )
    if not args.skip_profiles:
        from .representative_profiles import render_profiles

        render_profiles(root, {"gnn_all_pairs/k_8": case}, meta)
    print(f"Completed primary GNN figures for {len(ids)} MSAs")


if __name__ == "__main__":
    main()
