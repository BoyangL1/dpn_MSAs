"""Representative networks and profiles for fixed descriptive cluster cuts."""

from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_gnn_matplotlib")
import re
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from .plot_transfer_clusters import COLORS, short_name, save, style
from .data_paths import ACS

SOC = [
    ("median_household_income", "Income"),
    ("bachelor_plus_share", "Education"),
    ("young_adult_share_18_34", "Young adults"),
    ("senior_share_65plus", "Seniors 65+"),
]
POI = [
    ("cat_retail_trade", "Retail"),
    ("cat_accommodation_and_food_s", "Food/lodging"),
    ("cat_health_care_and_social_a", "Health care"),
    ("cat_real_estate_and_rental_a", "Real estate"),
    ("cat_other_services_except_pu", "Other svc."),
    ("cat_arts_entertainment_and_r", "Arts/rec."),
    ("cat_educational_services", "Education"),
    ("cat_manufacturing", "Manufact."),
]


def community_cmap():
    return LinearSegmentedColormap.from_list(
        "interaction_intensity",
        [
            (0.0, "#FCFCFD"),
            (0.06, "#E9D6DE"),
            (0.18, "#CA8AA8"),
            (0.40, "#9A4E7F"),
            (0.70, "#523157"),
            (1.0, "#232D45"),
        ],
    )


def build_profiles(root, meta, cohort):
    """Reuse final raw profiles for a cohort-only update, then rescale the cohort.

    A prior imputed value cannot safely be treated as raw data. In that case the
    full graph/static-data build is required, just as for an initial run.
    """
    if (root / ".work/build/nodes.parquet").exists():
        return _build_profiles_from_sources(root, meta, cohort)
    path = root / ".work/msa_context_profiles.csv"
    profiles = pd.read_csv(path, index_col=0)
    profiles.index = profiles.index.astype(str)
    assert profiles.index.is_unique and set(meta.index) <= set(profiles.index)
    profiles = profiles.loc[meta.index].copy()
    profiles.index.name = "geoid"
    assert (
        profiles.acs_missing_columns.fillna("").eq("").all()
    ), "Previously imputed ACS values require rebuilding raw ACS profiles"
    assert np.array_equal(profiles.n_nodes.to_numpy(), meta.n_nodes.to_numpy())
    for col, _ in SOC:
        assert np.isfinite(profiles[col]).all()
        minimum, maximum = (
            profiles.loc[cohort, col].min(),
            profiles.loc[cohort, col].max(),
        )
        profiles[col + "_minmax"] = (profiles[col] - minimum) / (
            maximum - minimum if maximum > minimum else 1
        )
    profiles.to_csv(path)
    return profiles


def _build_profiles_from_sources(root, meta, cohort):
    path = root / ".work/build/nodes.parquet"
    nodes = pd.read_parquet(
        path, columns=["msa_geoid", "county_fips", "TOP_CATEGORY_2DIGIT"]
    )
    nodes["msa_geoid"] = nodes.msa_geoid.astype(str)
    nodes["county_fips"] = nodes.county_fips.astype("string")
    valid_county = nodes.county_fips.str.fullmatch(r"\d{5}", na=False)
    crosswalk = (
        nodes.loc[valid_county]
        .groupby("county_fips")
        .msa_geoid.agg(lambda s: s.value_counts().index[0])
    )
    source = ACS
    acs = pd.read_csv(
        source,
        usecols=["GEOID", "population", *[col for col, _ in SOC]],
        dtype={"GEOID": str},
    )
    acs["msa_geoid"] = acs.GEOID.str[:5].map(crosswalk)
    acs = acs.dropna(subset=["msa_geoid"])
    weight = pd.to_numeric(acs.population, errors="coerce").clip(lower=0).fillna(0)
    profiles = meta[["name", "n_nodes"]].copy()
    profiles.index.name = "geoid"
    for col, _ in SOC:
        values = pd.to_numeric(acs[col], errors="coerce")
        valid = values.notna() & weight.gt(0)
        profiles[col] = (values[valid] * weight[valid]).groupby(
            acs.msa_geoid[valid]
        ).sum() / weight[valid].groupby(acs.msa_geoid[valid]).sum()
    profiles["acs_missing_columns"] = profiles.apply(
        lambda row: "|".join(col for col, _ in SOC if pd.isna(row[col])), axis=1
    )

    for col, _ in SOC:
        median = profiles.loc[cohort, col].median()
        if not np.isfinite(median):
            raise ValueError(f"No usable ACS observations for {col}")
        profiles[col] = profiles[col].fillna(median)
        minimum, maximum = (
            profiles.loc[cohort, col].min(),
            profiles.loc[cohort, col].max(),
        )
        profiles[col + "_minmax"] = (profiles[col] - minimum) / (
            maximum - minimum if maximum > minimum else 1
        )
    nodes["sector"] = (
        nodes.TOP_CATEGORY_2DIGIT.fillna("Unknown").str.split("|").str[0].str.strip()
    )
    counts = nodes.groupby(["msa_geoid", "sector"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [
        "cat_" + re.sub("[^a-z0-9]+", "_", str(c).lower()).strip("_")[:24]
        for c in shares.columns
    ]
    assert shares.columns.is_unique
    for col, _ in POI:
        profiles[col] = (
            shares[col].reindex(profiles.index).fillna(0) if col in shares else 0.0
        )
    profiles["n_nodes_with_county"] = (
        nodes.loc[valid_county]
        .groupby("msa_geoid")
        .size()
        .reindex(profiles.index)
        .fillna(0)
        .astype(int)
    )
    profiles["n_nodes_unknown_sector"] = (
        nodes.loc[nodes.sector.eq("Unknown")]
        .groupby("msa_geoid")
        .size()
        .reindex(profiles.index)
        .fillna(0)
        .astype(int)
    )
    (root / ".work").mkdir(parents=True, exist_ok=True)
    profiles.to_csv(root / ".work/msa_context_profiles.csv")
    return profiles


def load_positive_edges(root, meta, ids):
    path = root / ".work/build/edges.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            "Run 01_build_network.py before rendering network profiles."
        )
    edges = pd.read_parquet(
        path,
        columns=["src_idx", "tgt_idx", "msa_geoid", "nij"],
        filters=[("nij", ">", 0), ("msa_geoid", "in", ids)],
    )
    edges["msa_geoid"] = edges.msa_geoid.astype(str)
    return edges


def community_matrix(payload):
    geoid, source, target, weights = payload
    ew = defaultdict(float)
    for a, b, w in zip(source, target, weights):
        if a != b:
            ew[(int(a), int(b)) if a < b else (int(b), int(a))] += float(w)
    graph = nx.Graph()
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in ew.items())
    if graph.number_of_nodes() == 0:
        raise ValueError(f"No positive edges for representative MSA {geoid}")
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    communities = sorted(
        nx.community.louvain_communities(
            giant, weight="weight", seed=1, resolution=0.25
        ),
        key=len,
        reverse=True,
    )
    order = [node for community in communities for node in sorted(community)]
    n, bins = len(order), 95
    bin_of = {node: int(i * bins // n) for i, node in enumerate(order)}
    values = np.zeros((bins, bins))
    for (a, b), weight in ew.items():
        if a in bin_of and b in bin_of:
            x, y = bin_of[a], bin_of[b]
            values[x, y] += weight
            values[y, x] += weight
    sizes = np.bincount(list(bin_of.values()), minlength=bins)
    denominator = np.outer(sizes, sizes).astype(float)
    denominator[np.diag_indices(bins)] -= sizes
    values = np.divide(
        values, denominator, out=np.zeros_like(values), where=denominator > 0
    )
    info = dict(
        geoid=geoid,
        n_communities=len(communities),
        giant_nodes=n,
        total_nodes=graph.number_of_nodes(),
        giant_fraction=n / graph.number_of_nodes(),
        largest_community_fraction=len(communities[0]) / n,
        modularity=nx.community.modularity(
            giant, communities, weight="weight", resolution=0.25
        ),
    )
    return geoid, values, info


def choose_representatives(case, profiles):
    selected = []
    k = int(case["k"])
    assert np.array_equal(np.unique(case["labels"]), np.arange(1, k + 1))
    for cluster in range(1, k + 1):
        mask = case["labels"] == cluster
        ids = case["ids"][mask]

        within = case["distance"][np.ix_(mask, mask)]
        scores = (
            within.sum(axis=1) / (len(ids) - 1) if len(ids) > 1 else np.full(1, np.nan)
        )
        ranked = pd.DataFrame({"geoid": ids, "mean_transfer_dissimilarity": scores})
        ranked = ranked.sort_values(
            ["mean_transfer_dissimilarity", "geoid"],
            ascending=[True, True],
            na_position="last",
        ).head(3)
        for rank, row in enumerate(ranked.itertuples(), 1):
            selected.append(
                dict(
                    cluster=cluster,
                    geoid=row.geoid,
                    name=profiles.loc[row.geoid, "name"],
                    mean_transfer_dissimilarity=row.mean_transfer_dissimilarity,
                    rank=rank,
                    n_cluster_msas=int(mask.sum()),
                    selection_method="Smallest mean direct 1-S transfer dissimilarity to other cluster members; GEOID breaks ties",
                )
            )
    return pd.DataFrame(selected)


def draw_panel(
    fig, cluster, chosen, case, profiles, matrices, share_limit, grouped=True
):
    style(40)
    plt.rcParams.update(
        {
            "axes.linewidth": 2,
            "text.color": "#28343E",
            "axes.labelcolor": "#28343E",
            "axes.edgecolor": "#ABB4BC",
            "xtick.color": "#52616C",
            "ytick.color": "#28343E",
        }
    )
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.05, 1.05, 1.95],
        left=0.30,
        right=0.98,
        top=0.775,
        bottom=0.065,
        hspace=0.30,
        wspace=0.20,
    )
    color = COLORS[cluster - 1]
    city_labels = chosen.name.map(lambda name: short_name(name, include_state=False))
    repeated_cities = set(city_labels[city_labels.duplicated(keep=False)])
    for col in range(len(chosen), 3):
        ax = fig.add_subplot(gs[0, col])
        ax.text(
            0.5,
            0.5,
            "No additional\nMSA",
            ha="center",
            va="center",
            fontsize=32,
            color="#6f7a82",
        )
        ax.axis("off")
    for col, row in enumerate(chosen.itertuples()):
        g, name = row.geoid, row.name
        ax = fig.add_subplot(gs[0, col])
        matrix = matrices[g]
        ax.imshow(
            matrix / (matrix.max() or 1),
            cmap=community_cmap(),
            norm=PowerNorm(0.4, vmin=0, vmax=1),
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#D5DBDF")
            spine.set_linewidth(2)
        label = short_name(
            name, include_state=short_name(name, include_state=False) in repeated_cities
        )
        ax.set_title(
            textwrap.fill(label, 11, break_long_words=False, break_on_hyphens=False),
            fontsize=44,
            fontweight="medium",
            pad=18,
        )
        for row_index, features in ((1, SOC), (2, POI)):
            ax = fig.add_subplot(gs[row_index, col])
            values = [
                profiles.loc[g, field + "_minmax" if row_index == 1 else field]
                for field, _ in features
            ]
            ax.barh(
                range(len(features)),
                values,
                color=color,
                height=0.52,
                zorder=3,
                alpha=0.9,
                linewidth=0,
            )
            limit = 1 if row_index == 1 else share_limit
            ax.set_xlim(0, limit)
            ax.set_ylim(-0.6, len(features) - 0.4)
            ax.invert_yaxis()
            ax.set_xticks([0, limit])
            if row_index == 2:
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(length=0, labelsize=42, pad=12, axis="x")
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
            ax.tick_params(axis="x", labelbottom=col == 0)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(
                [label for _, label in features] if col == 0 else [], fontsize=46
            )
            ax.tick_params(axis="y", length=0, pad=12)
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color="#E8ECEF", linewidth=2.117)
            ax.spines["left"].set_color("#C7CFD5")
            ax.spines[["top", "right", "bottom"]].set_visible(False)
    for row_index, label in enumerate(["Network", "ACS · 0–1", "POI · %"]):
        box = gs[row_index, 0].get_position(fig)
        fig.text(
            0.025,
            box.y1 + 0.018,
            label,
            fontsize=48,
            color="#52616C",
            weight="bold",
            ha="left",
            va="bottom",
        )
    x0, x1 = gs[0, 0].get_position(fig).x0, gs[0, 2].get_position(fig).x1
    n = int(np.sum(case["labels"] == cluster))
    fig.text(
        (x0 + x1) / 2,
        0.974,
        f"C{cluster}",
        color=color,
        fontsize=56,
        weight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        (x0 + x1) / 2,
        0.916,
        f"{n} " + ("MSA" if n == 1 else "MSAs"),
        color=color,
        fontsize=42,
        ha="center",
        va="center",
    )
    fig.add_artist(
        Line2D(
            [x0, x1],
            [0.864, 0.864],
            transform=fig.transSubfigure if grouped else fig.transFigure,
            color=color,
            linewidth=4.657,
            alpha=0.65,
        )
    )


def profile_figure(rows, columns):
    style(40)
    height = 17.5 * rows + 1.8
    fig = plt.figure(figsize=(18 * columns, height))
    layout = fig.add_gridspec(
        rows + 1,
        columns,
        height_ratios=[17.5] * rows + [1.8],
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )
    panels = [
        fig.add_subfigure(layout[row, col])
        for row in range(rows)
        for col in range(columns)
    ]
    ax = fig.add_axes([0.30, 1.12 / height, 0.40, 0.22 / height])
    bar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=PowerNorm(0.4, vmin=0, vmax=1), cmap=community_cmap()
        ),
        cax=ax,
        orientation="horizontal",
        ticks=[0, 0.1, 0.5, 1],
    )
    bar.set_label(
        "Within-MSA normalized interaction intensity", fontsize=32, labelpad=8
    )
    bar.ax.tick_params(labelsize=30, length=5, width=1.2, pad=6)
    bar.outline.set_linewidth(1.2)
    return fig, panels


def render_profiles(root, cases, meta):
    cohort = next(iter(cases.values()))["ids"]
    profiles = build_profiles(root, meta, cohort)
    selections = {
        view: choose_representatives(case, profiles) for view, case in cases.items()
    }
    ids = sorted(set(g for selected in selections.values() for g in selected.geoid))
    share_limit = max(
        0.05, np.ceil(profiles.loc[ids, [c for c, _ in POI]].to_numpy().max() * 20) / 20
    )
    print(
        f"Computing fresh community matrices for {len(ids)} distinct representative MSAs",
        flush=True,
    )
    edges = load_positive_edges(root, meta, ids)
    payloads = [
        (g, part.src_idx.to_numpy(), part.tgt_idx.to_numpy(), part.nij.to_numpy())
        for g, part in edges.groupby("msa_geoid", sort=True)
    ]
    assert len(payloads) == len(ids)
    matrices, info = {}, []
    workers = min(4, len(payloads))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for g, matrix, record in pool.map(community_matrix, payloads):
            matrices[g] = matrix
            info.append(record)
            print(f"  Community matrix {len(matrices)}/{len(ids)}: {g}", flush=True)
    info_frame = pd.DataFrame(info).set_index("geoid")
    for view, case in cases.items():
        (root / ".work" / view).mkdir(parents=True, exist_ok=True)
        selected = selections[view]
        selected.join(info_frame, on="geoid").join(
            profiles.drop(columns=["name"]), on="geoid"
        ).to_csv(root / ".work" / view / "representative_msa_profiles.csv", index=False)
        output = root / "figures" / view
        for first in range(1, int(case["k"]) + 1, 4):
            last = min(first + 3, int(case["k"]))
            count = last - first + 1

            rows, columns = (2, 2) if count == 4 else (1, count)
            combined, panels = profile_figure(rows, columns)
            assert len(panels) == count
            for panel, cluster in zip(panels, range(first, last + 1)):
                chosen = selected.loc[selected.cluster.eq(cluster)]
                assert len(chosen) > 0
                draw_panel(
                    panel, cluster, chosen, case, profiles, matrices, share_limit
                )
            path = output / f"representative_cluster_profiles_C{first}-C{last}.png"
            save(combined, path, dpi=300)
