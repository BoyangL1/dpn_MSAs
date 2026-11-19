"""Current 43-predictor results in the original Arial style, with compact margins."""

from .analysis_paths import RESULTS, PUBLISHED, FIGURES
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter

HERE = RESULTS
STEM = "regression_contributors"
URBAN = "Urban structure & function"
URBAN_COLOR = "#287F78"
GROUPS = [
    ("Distance & Size", "Distance & size", 2, "#946B84", "#946B84"),
    ("Socioeconomic", "Socioeconomic", 8, "#34688A", "#34688A"),
    ("POI composition", "POI composition", 3, URBAN_COLOR, "#000000"),
    ("Network structure", "Network structure", 5, URBAN_COLOR, "#000000"),
    ("Urban form", "Urban form", 4, URBAN_COLOR, "#000000"),
]
SHORT = {
    "log_n_nodes": "POI count (log)",
    "log_geo_dist": "Geographic distance",
    "acs_population": "Host-CBG population",
    "acs_households": "Host-CBG households",
    "acs_median_household_income": "Household income",
    "acs_per_capita_income": "Per-capita income",
    "acs_median_home_value": "Home value",
    "acs_unemployment_rate": "Unemployment rate",
    "acs_car_commute_share": "Car commute share",
    "acs_public_transit_share": "Transit commute share",
    "acs_no_vehicle_share": "No-vehicle households",
    "acs_renter_share": "Renter share",
    "acs_bachelor_plus_share": "Bachelor’s degree+",
    "acs_young_adult_share_18_34": "Young adults 18–34",
    "acs_senior_share_65plus": "Seniors 65+",
    "acs_white_share": "White share",
    "acs_black_share": "Black share",
    "acs_asian_share": "Asian share",
    "acs_hispanic_share": "Hispanic share",
    "net_modularity": "Louvain modularity",
    "net_transitivity": "Transitivity",
    "net_giant_component_share": "Giant-component share",
    "net_pagerank_gini": "PageRank Gini",
    "scn_Grocery___convenience_stores": "Grocery & convenience",
    "scn_Gas_stations": "Gas stations",
    "scn_Transit__bus___taxi_services": "Transit, bus & taxi",
    "scn_Shopping_malls___centers": "Shopping centers",
    "scn_Schools___colleges": "Schools & colleges",
    "scn_Parks___nature_areas": "Parks & nature areas",
    "scn_Gyms___recreation_centers": "Gyms & recreation",
    "scn_Hotels___motels": "Hotels & motels",
    "scn_Restaurants___bars": "Restaurants & bars",
    "uf_poi_density": "POI density (log)",
    "uf_poi_entropy": "POI functional diversity",
    "uf_poly_K": "Spatial centers (log)",
    "uf_poly_D": "Largest-center dominance",
    "rd_intersection_density": "Intersection density (log)",
    "rd_orientation_entropy": "Street-orientation entropy",
}
NETWORK_LABELS = {
    "diff_top1_share_visits": "Top-1% co-visitation-volume share",
    "diff_incoming_strength_gini": "Dependency-strength Gini",
    "diff_edge_weight_gini": "Co-visitation edge-count Gini",
    "diff_w_row_sum_mean": "Mean within-MSA co-visitation fraction",
    "diff_log_top1_pair_distance": "Top-1% dependency-hub separation",
}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PUBLISHED / "regression_coefficients.csv")
    data = data[data.model.eq("M6") & data.term.ne("const")].copy()
    assert (
        len(data) == 43
        and data.term.is_unique
        and data.covariance.eq("source_MSA_clustered").all()
    )
    assert set(data.loc[data.component.eq("Distance & Size"), "term"]) == {
        "log_N_diff",
        "log_geo_dist",
    }
    assert (
        data.loc[~data.component.eq("Distance & Size"), "term"]
        .str.startswith("diff_")
        .all()
    )

    network_labels = NETWORK_LABELS
    labels = SHORT | {
        term.removeprefix("diff_"): label for term, label in network_labels.items()
    }
    for term, label in network_labels.items():
        data.loc[data.term.eq(term), "label"] = label
    data["abs_coef"] = data.coef.abs()
    data["display_label"] = [
        (
            ("|Δ| " + labels[t.removeprefix("diff_")])
            if t.startswith("diff_")
            else {
                "log_N_diff": "Signed log-size difference",
                "log_geo_dist": "Geographic distance",
            }[t]
        )
        for t in data.term
    ]
    shares = pd.read_csv(HERE / "group_contribution_shapley.csv").set_index("group")
    comparison = pd.read_csv(HERE / "model_comparison.csv").set_index("model").loc["M6"]
    assert np.allclose(data.r2, comparison.r2)
    chosen = []
    for component, _, top, _, _ in GROUPS:
        q = data[data.component.eq(component)].nlargest(top, "abs_coef").copy()
        q["rank_within_component"] = np.arange(1, len(q) + 1)
        chosen.append(q)
    shown = pd.concat(chosen, ignore_index=True)
    shown["ci_excludes_zero"] = (shown.ci_low > 0) | (shown.ci_high < 0)
    selected = pd.read_csv(HERE / "top_contributors_full.csv")
    assert shown.term.tolist() == selected.term.tolist()
    assert np.allclose(
        shown[["coef", "ci_low", "ci_high"]], selected[["coef", "ci_low", "ci_high"]]
    )
    assert np.isclose(shares.share_of_total_explained_variance.sum(), 1)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 24,
            "text.color": "#29332F",
            "axes.labelcolor": "#29332F",
            "axes.unicode_minus": True,
        }
    )
    fig = plt.figure(figsize=(18, 9))
    axes = [
        fig.add_axes([0.325, 0.145, 0.165, 0.775]),
        fig.add_axes([0.825, 0.145, 0.155, 0.775]),
    ]
    texts = []
    share_annotations = []
    for ax, groups in zip(axes, (GROUPS[:2], GROUPS[2:])):
        ax.axvline(0, color="#A3ACA7", lw=1.7, zorder=0)
        ticks = []
        labels = []
        y = 0.0
        for component, title, _, color, title_color in groups:
            is_block = component in shares.index
            texts.append(
                ax.text(
                    -0.05,
                    y,
                    title,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=26 if is_block else 23,
                    weight="normal" if is_block else "bold",
                    color=title_color,
                )
            )
            if is_block:
                share = shares.loc[component, "share_of_total_explained_variance"]
                st = ax.text(
                    1,
                    y,
                    f"{share:.1%}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=24,
                    color=color,
                )
                texts.append(st)
                share_annotations.append((st, share, color))
            y += 1
            for row in shown[shown.component.eq(component)].itertuples():
                ax.plot(
                    [row.ci_low, row.ci_high],
                    [y, y],
                    color=color,
                    lw=2.5,
                    solid_capstyle="round",
                )
                ax.plot(
                    row.coef,
                    y,
                    "o",
                    ms=9.5,
                    mfc=color if row.ci_excludes_zero else "white",
                    mec=color,
                    mew=2,
                )
                ticks.append(y)
                labels.append(row.display_label)
                y += 1
            y += 0.45
        ax.set_yticks(ticks, labels, fontsize=24)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_ylim(y - 0.85, -0.65)
        subset = shown[shown.component.isin([g[0] for g in groups])]
        lo, hi = min(0, subset.ci_low.min()), max(0, subset.ci_high.max())
        pad = (hi - lo) * 0.13
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(
            MultipleLocator(0.02)
            if ax is axes[0]
            else MaxNLocator(nbins=3, steps=[1, 2, 5, 10])
        )
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda v, _: "0" if abs(v) < 1e-12 else f"{v:.2f}".replace("-", "−")
            )
        )
        ax.tick_params(axis="x", labelsize=22, width=1.5, length=5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("#69756F")
        texts += ax.get_yticklabels()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_spans = []
    for ax in axes:
        labels = list(ax.get_yticklabels()) + [
            t for t in ax.texts if t.get_position()[0] < 0
        ]
        label_spans.append(
            (ax.bbox.x0 - min(t.get_window_extent(renderer).x0 for t in labels))
            / fig.dpi
        )
    margin, gap, right_margin = 0.14, 0.48, 0.18
    plot_widths = [3.30, 18 * 0.155]
    left_x = margin + label_spans[0]
    right_label_x = left_x + plot_widths[0] + gap
    right_x = right_label_x + label_spans[1]
    width = right_x + plot_widths[1] + right_margin
    fig.set_size_inches(width, 9)
    for ax, x, w in zip(axes, [left_x, right_x], plot_widths):
        ax.set_position([x / width, 0.145, w / width, 0.775])
    texts.append(
        fig.text(
            right_label_x / width,
            0.965,
            URBAN,
            fontsize=26,
            color=URBAN_COLOR,
            va="center",
        )
    )
    share = shares.loc[URBAN, "share_of_total_explained_variance"]
    st = fig.text(
        (width - right_margin) / width,
        0.965,
        f"{share:.1%}",
        fontsize=24,
        ha="right",
        va="center",
        color=URBAN_COLOR,
    )
    texts.append(st)
    share_annotations.append((st, share, URBAN_COLOR))
    texts.append(
        fig.text(
            margin / width,
            0.07,
            "Change in Pearson r per 1 SD",
            fontsize=23,
            va="center",
        )
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            ms=9.5,
            mfc=fill,
            mec="#52616C",
            mew=2,
            label=label,
        )
        for fill, label in [
            ("#52616C", "95% CI excludes 0"),
            ("white", "95% CI includes 0"),
        ]
    ]
    leg = fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.59, 0.07),
        ncol=2,
        frameon=False,
        fontsize=23,
        handlelength=1,
        handletextpad=0.4,
        columnspacing=1,
        borderaxespad=0,
    )
    texts += list(leg.get_texts())
    texts.append(
        fig.text(
            (width - right_margin) / width,
            0.07,
            f"R² = {comparison.r2:.3f}",
            fontsize=23,
            ha="right",
            va="center",
            color="#52616C",
        )
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        texts.extend(
            t.label1
            for t in ax.xaxis.get_major_ticks()
            if ax.get_xlim()[0] <= t.get_loc() <= ax.get_xlim()[1]
        )
    boxes = []
    for item in texts:
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert (
                box.x0 >= 0
                and box.y0 >= 0
                and box.x1 <= fig.bbox.width
                and box.y1 <= fig.bbox.height
            ), ("Text outside canvas: " + item.get_text())
            boxes.append((item.get_text(), box))
    for j, (name, box) in enumerate(boxes):
        for other, ob in boxes[j + 1 :]:
            assert not box.overlaps(ob), "Text overlap: " + name + " / " + other
    for item, share, color in share_annotations:
        box = item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        bar_width, bar_height = 1.62 / width, 0.072 / 9
        left, bottom = box.x1 - bar_width, box.y0 - 0.005 - bar_height
        for w, alpha in [(bar_width, 0.13), (bar_width * share, 1)]:
            fig.add_artist(
                Rectangle(
                    (left, bottom),
                    w,
                    bar_height,
                    transform=fig.transFigure,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                )
            )
    fig.savefig(
        FIGURES / f"{STEM}.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    shown.to_csv(HERE / f"{STEM}_plotted_coefficients.csv", index=False)
    print(f"Saved {STEM}.png: 22 coefficients, R²={comparison.r2:.6f}", flush=True)


if __name__ == "__main__":
    main()
