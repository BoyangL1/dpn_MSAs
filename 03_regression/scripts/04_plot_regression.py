"""Render regression contributors and cluster feature heatmaps."""

from regression_lib.analysis_paths import MAIN, INPUTS, RESULTS, FIGURES, CONFIG
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties

I = INPUTS
SHARED = RESULTS
from lib.cluster_analysis import transfer_similarity, display_labels

DOMAINS = ["Sociodemographic", "POI composition", "Network attributes"]
SOC = [
    ("median_household_income", "Household\nincome", False),
    ("median_home_value", "Home value", True),
    ("median_gross_rent", "Gross rent", False),
    ("hispanic_share", "Hispanic share", False),
    ("black_share", "Black share", False),
    ("bachelor_plus_share", "Bachelor’s\ndegree+", False),
    ("young_adult_share_18_34", "Young adults\n18–34", False),
    ("senior_share_65plus", "Seniors 65+", False),
    ("public_transit_share", "Public-transit\ncommute", True),
    ("no_vehicle_share", "No-vehicle\nhouseholds", False),
    ("renter_share", "Renter share", False),
    ("pop_density", "Population\ndensity", True),
]
LABELS = {
    "scn_Grocery___convenience_stores": "Grocery &\nconvenience",
    "scn_Gas_stations": "Gas stations",
    "scn_Transit__bus___taxi_services": "Transit, bus\n& taxi",
    "scn_Shopping_malls___centers": "Shopping\ncenters",
    "scn_Schools___colleges": "Schools &\ncolleges",
    "scn_Parks___nature_areas": "Parks &\nnature areas",
    "scn_Gyms___recreation_centers": "Gyms &\nrecreation",
    "scn_Hotels___motels": "Hotels &\nmotels",
    "scn_Restaurants___bars": "Restaurants\n& bars",
    "log_n_nodes": "POI count (log)",
    "net_modularity": "Louvain\nmodularity",
    "net_transitivity": "Triangle\nclosure",
    "net_giant_component_share": "Giant-comp.\nshare",
    "net_pagerank_gini": "PageRank Gini",
    "top1_share_visits": "Top 1%\nvisit share",
    "incoming_strength_gini": "Dependency\nstrength Gini",
    "edge_weight_gini": "Edge-weight\nGini",
    "w_row_sum_mean": "Mean dep.\nstrength",
    "log_top1_pair_distance": "Top-1% spatial\nseparation",
}
LABELS.update({c: l for c, l, _ in SOC})


def read(p):
    return pd.read_csv(p, dtype={"msa_geoid": str}).set_index("msa_geoid")


def wrap_feature_label(fig, label, width_cm=1.90, fontsize=9.5):
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family="Arial", size=fontsize)
    limit = width_cm / 2.54 * fig.dpi
    lines = []
    for original in label.split("\n"):
        line = ""
        for word in original.split():
            candidate = (line + " " + word).strip()
            if (
                line
                and renderer.get_text_width_height_descent(candidate, font, False)[0]
                > limit
            ):
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def prepare():
    T = pd.read_csv(
        MAIN / "results/gnn_all_pairs/transfer_pearson_analysis.csv", index_col=0
    )
    T.index = T.index.astype(str)
    T.columns = T.columns.astype(str)
    ids = T.index.tolist()
    T = T.loc[ids, ids]
    S = transfer_similarity(T.to_numpy())
    D = np.maximum(1 - S, 0)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=True), method="ward")
    order = leaves_list(Z)
    np.save(I / "ward_linkage.npy", Z)
    metadata_path = MAIN / ".work/msa_metadata.csv"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path, dtype={"GEOID": str})
    else:
        meta = pd.read_csv(
            MAIN / "results/gnn_all_pairs/k_8/cluster_membership.csv",
            dtype={"geoid": str},
        ).rename(columns={"geoid": "GEOID", "name": "NAME"})
    meta = meta.set_index("GEOID").loc[ids]
    nd = read(I / "node_based_msa_features.csv").loc[ids]
    network = read(I / "network_msa_features.csv").loc[ids]
    social = read(I / "demographic_raw_msa_features.csv").loc[ids]
    network["incoming_strength_gini"] = read(
        I / "raw_w_strength_features.csv"
    ).incoming_strength_gini
    network["log_n_nodes"] = np.log1p(nd.n_nodes)
    dict_ = pd.read_csv(CONFIG / "original_variable_dictionary.csv")
    poi = (
        dict_[dict_.component.eq("POI composition")]
        .term.str.removeprefix("diff_")
        .tolist()
    )
    net = ["log_n_nodes"] + dict_[
        dict_.component.eq("Network structure")
    ].term.str.removeprefix("diff_").tolist()
    pools = {DOMAINS[0]: [c for c, _, _ in SOC], DOMAINS[1]: poi, DOMAINS[2]: net}
    vals = pd.concat([social[pools[DOMAINS[0]]], nd[poi], network[net]], axis=1)
    raw = vals.copy()
    assert vals.columns.is_unique
    for c, _, log in SOC:
        if log:
            vals[c] = np.log1p(vals[c].clip(lower=0))
    assert np.isfinite(vals).all().all()
    mu, sd = vals.mean(), vals.std(ddof=0)
    assert (sd > 0).all()
    z = (vals - mu) / sd
    raw.to_csv(I / "heatmap_msa_features_raw.csv")
    z.to_csv(I / "heatmap_msa_features_z.csv")
    pd.DataFrame({"mean_transformed": mu, "sd_ddof0": sd}).to_csv(
        I / "heatmap_feature_scaling.csv"
    )
    checks = []
    outputs = {}
    crosscuts = pd.DataFrame(index=ids)
    for k in range(4, 9):
        folder = RESULTS / f"k{k}"
        folder.mkdir(parents=True, exist_ok=True)
        rawlabels = fcluster(Z, k, criterion="maxclust")
        labels, _ = display_labels(rawlabels, order, ids)
        assert len(np.unique(labels)) == k
        membership = pd.DataFrame(
            {
                "geoid": ids,
                "name": meta.NAME.to_numpy(),
                "cluster": labels,
                "raw_cluster": rawlabels,
                "n_nodes": nd.n_nodes.to_numpy(),
                "within_msa_r": np.diag(T),
                "leaf_position": np.argsort(order),
            }
        )
        oldpath = MAIN / f"results/gnn_all_pairs/k_{k}/cluster_membership.csv"
        if oldpath.exists():
            old = pd.read_csv(oldpath, dtype={"geoid": str}).set_index("geoid").loc[ids]
            assert np.array_equal(old.cluster, labels)
            assert np.array_equal(old.n_nodes, nd.n_nodes)
        membership.sort_values("leaf_position").to_csv(
            folder / "cluster_membership.csv", index=False
        )
        crosscuts[f"k{k}"] = labels
        summaries = membership.groupby("cluster").agg(
            n_msa=("geoid", "size"),
            total_network_nodes=("n_nodes", "sum"),
            median_network_nodes=("n_nodes", "median"),
            mean_within_msa_r=("within_msa_r", "mean"),
            median_within_msa_r=("within_msa_r", "median"),
        )
        for c in range(1, k + 1):
            mask = labels == c
            inside = T.to_numpy()[np.ix_(mask, mask)]
            summaries.loc[c, "mean_within_cluster_transfer_r"] = (
                inside[~np.eye(mask.sum(), dtype=bool)].mean()
                if mask.sum() > 1
                else np.nan
            )
            summaries.loc[c, "mean_transfer_to_other_clusters"] = T.to_numpy()[
                np.ix_(mask, ~mask)
            ].mean()
        summaries.to_csv(folder / "cluster_summary.csv")
        long = []
        ranks = []
        for domain, features in pools.items():
            items = []
            for f in features:
                v = z[f].to_numpy()
                ss = ((v - v.mean()) ** 2).sum()
                between = sum(
                    (labels == c).sum() * (v[labels == c].mean() - v.mean()) ** 2
                    for c in range(1, k + 1)
                )
                eta = float(between / ss)
                items.append(
                    dict(
                        domain=domain,
                        feature=f,
                        label=LABELS[f].replace("\n", " "),
                        eta_squared=eta,
                    )
                )
                for c in range(1, k + 1):
                    vv = v[labels == c]
                    rv = raw[f].to_numpy()[labels == c]
                    qq = np.quantile(vv, [0.1, 0.25, 0.5, 0.75, 0.9])
                    long.append(
                        dict(
                            domain=domain,
                            feature=f,
                            label=LABELS[f].replace("\n", " "),
                            cluster=c,
                            n=len(vv),
                            eta_squared=eta,
                            mean_z=float(vv.mean()),
                            raw_mean=float(rv.mean()),
                            raw_median=float(np.median(rv)),
                            **dict(zip(["q10", "q25", "median", "q75", "q90"], qq)),
                        )
                    )
            ranks_ = pd.DataFrame(items).sort_values(
                ["eta_squared", "feature"], ascending=[False, True]
            )
            ranks_["rank"] = np.arange(1, len(ranks_) + 1)
            ranks_["selected"] = ranks_["rank"] <= 4
            ranks.append(ranks_)
        ranking = pd.concat(ranks, ignore_index=True)
        allstats = pd.DataFrame(long)
        selected = allstats[
            allstats.feature.isin(ranking[ranking.selected].feature)
        ].copy()
        assert len(selected) == 12 * k
        ranking.to_csv(folder / f"feature_ranking_k{k}.csv", index=False)
        allstats.to_csv(folder / f"all_feature_group_statistics_k{k}.csv", index=False)
        selected.to_csv(
            folder / f"cluster_grouped_feature_intervals_numeric_k{k}.csv", index=False
        )
        outputs[k] = (selected, ranking, summaries)
        checks.append(
            dict(
                k=k,
                n_msa=len(membership),
                n_clusters=k,
                sizes=summaries.n_msa.to_dict(),
                matches_existing_labels=True if oldpath.exists() else None,
            )
        )
    crosscuts.index.name = "geoid"
    crosscuts.to_csv(RESULTS / "cluster_membership_all_k.csv")
    return outputs


def plot_heatmaps(outputs, only_k=None):
    limit = max(
        0.5,
        float(
            np.ceil(max(t[0]["median"].abs().max() for t in outputs.values()) * 2) / 2
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "text.color": "#28333A",
            "savefig.bbox": None,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "blue_neutral_red", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"]
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    validation = []
    for k, (stats, rank, summary) in outputs.items():
        if only_k is not None and k != only_k:
            continue

        footer = 0.50
        W, H = 18.0, 6.6 + footer
        fig = plt.figure(figsize=(W / 2.54, H / 2.54), dpi=300)
        texts = []
        cells = []
        for block, domain in enumerate(DOMAINS):
            chosen = rank[(rank.domain == domain) & rank.selected].sort_values("rank")
            features = chosen.feature.tolist()
            pivot = (
                stats[stats.domain.eq(domain)]
                .pivot(index="feature", columns="cluster", values="median")
                .loc[features, range(1, k + 1)]
            )
            x = 2.10 + block * 6.0
            bottom = 1.18 + footer
            height = 4.80
            width = 3.83
            ax = fig.add_axes([x / W, bottom / H, width / W, height / H])
            im = ax.imshow(
                pivot.to_numpy(),
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_xticks(range(k), [f"C{c}" for c in range(1, k + 1)], fontsize=9.5)
            ax.set_yticks([])
            ax.tick_params(length=0, pad=3)
            ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            texts.extend(ax.get_xticklabels())
            texts.append(
                fig.text(
                    (x + width / 2) / W,
                    (bottom + height + 0.36) / H,
                    domain,
                    ha="center",
                    va="center",
                    fontsize=10.4,
                    fontweight="bold",
                )
            )
            for i, r in enumerate(chosen.itertuples()):
                cy = bottom + height * (1 - (i + 0.5) / 4)
                name = wrap_feature_label(fig, LABELS[r.feature])
                shift = 0.13 if "\n" in name else 0
                texts.append(
                    fig.text(
                        (x - 0.12) / W,
                        (cy + 0.02 - shift) / H,
                        name,
                        ha="right",
                        va="bottom",
                        fontsize=9.5,
                        linespacing=1.05,
                    )
                )
                texts.append(
                    fig.text(
                        (x - 0.12) / W,
                        (cy - 0.055 - shift) / H,
                        f"η² = {r.eta_squared:.2f}",
                        ha="right",
                        va="top",
                        fontsize=9.5,
                        color="#4B555C",
                    )
                )
                for c in range(1, k + 1):
                    cells.append(
                        dict(
                            domain=domain,
                            feature=r.feature,
                            label=r.label,
                            eta_squared=r.eta_squared,
                            cluster=c,
                            n_msa=int(summary.loc[c, "n_msa"]),
                            median_z=pivot.loc[r.feature, c],
                        )
                    )
        cax = fig.add_axes([(W - 5.5) / 2 / W, (0.52 + footer) / H, 5.5 / W, 0.18 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks(
            np.arange(np.ceil(-limit), np.floor(limit) + 1)
            if limit >= 1
            else [-limit, 0, limit]
        )
        cb.ax.tick_params(labelsize=9.5, length=2, pad=2)
        cb.outline.set_visible(False)
        texts.extend(cb.ax.get_xticklabels())
        cb.set_label("Median z-score", fontsize=9.5, labelpad=3, color="#28333A")
        texts.append(cb.ax.xaxis.label)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for t in texts:
            box = t.get_window_extent(renderer)
            assert (
                box.x0 >= 0
                and box.y0 >= 0
                and box.x1 <= fig.bbox.width
                and box.y1 <= fig.bbox.height
            ), t.get_text()
            boxes.append((t.get_text(), box))
        for j, (name, box) in enumerate(boxes):
            for name2, box2 in boxes[j + 1 :]:
                assert not box.overlaps(box2), (name, name2)
        FIGURES.mkdir(parents=True, exist_ok=True)
        stem = FIGURES / f"cluster_grouped_feature_heatmap_k{k}"
        fig.savefig(
            stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=None
        )
        plt.close(fig)
        if only_k is None:
            pd.DataFrame(cells).to_csv(
                RESULTS / f"k{k}/cluster_grouped_feature_heatmap_numeric_k{k}.csv",
                index=False,
            )
        validation.append(
            dict(
                k=k,
                cells=len(cells),
                color_limit=limit,
                colorbar_label="Median z-score",
                colorbar_label_position="centered_below_ticks",
                added_bottom_margin_cm=footer,
                panel_dimensions_cm=[3.83, 4.80],
                no_clipping=True,
                no_text_overlap=True,
                no_text_outside_canvas=True,
                top_note=False,
                bottom_note=False,
                cluster_sample_counts_shown=False,
                canvas_cm=[W, H],
                font="Arial",
                minimum_font_pt=9.5,
                title_font_pt=10.4,
                dpi=300,
            )
        )
        print(f"Saved k{k} heatmap; sizes={summary.n_msa.tolist()}", flush=True)


def contributor():
    from regression_lib.plot_contributors import main as plot_contributors

    plot_contributors()


def main():
    for path in [INPUTS, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    contributor()
    outputs = prepare()
    plot_heatmaps(outputs, only_k=8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
