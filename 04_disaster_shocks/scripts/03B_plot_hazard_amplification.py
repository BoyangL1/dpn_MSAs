"""Show each observed AR's percentage difference from its random-shock mean.

Use the same 323 retained study MSAs as the hazard-coverage map. Within that
cohort, flooding shows its top 30 observed ARs; every eligible city is shown
for the other hazards. Random controls are means of the saved 100 draws,
with their sample SDs retained in source data. A clearly marked broken axis
retains exceptionally large wildfire effects without compressing other cities.
"""

from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "shock_ar_by_msa_mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from shock_lib.shock_paths import ROOT, DATA_DIR

STEM = "fig_ar_by_msa_nature_random"
AR = "amplification_ratio_unw"
RAND, SD = (AR + "_rand", AR + "_rand_sd")
ORDER = ["IFLD", "WFIR", "TRND", "ERQK", "HRCN"]
COLOURS = {
    "IFLD": "#3679AA",
    "WFIR": "#B88624",
    "TRND": "#6C5B9B",
    "ERQK": "#C36161",
    "HRCN": "#CD7246",
}
INK, GREY, GREY_EDGE = ("#262626", "#C9C6C0", "#86817A")
WIDTH, HEIGHT, DPI = (180, 216, 600)
EFFECT = "ar_difference_from_random_pct"


def short_name(name):
    city, states = name.rsplit(", ", 1)
    return (
        f"{city.split('-')[0].replace('/Jefferson County', '')}, {states.split('-')[0]}"
    )


def prepare(root, config):
    cohort = pd.read_csv(
        DATA_DIR / config["hazard_coverage"]["cohort_file"], dtype={"msa_geoid": str}
    )
    keep = cohort.included_in_coverage.astype(str).str.lower().eq("true")
    assert keep.equals(cohort.best_epoch.gt(1))
    assert int(keep.sum()) == config["hazard_coverage"]["study_msa_count"]
    study_ids = set(cohort.loc[keep, "msa_geoid"])
    raw = pd.read_csv(root / "results/shock_results.csv", dtype={"msa_geoid": str})
    raw = raw.loc[raw.msa_geoid.isin(study_ids)].copy()
    assert set(raw.hazard) <= set(ORDER)
    assert not raw.duplicated(["msa_geoid", "hazard"]).any()
    valid = (
        raw.status.eq("ok")
        & raw.k_pois_removed.gt(0)
        & np.isfinite(raw[[AR, RAND, SD]]).all(axis=1)
        & raw[RAND].gt(0)
        & raw[SD].ge(0)
    )
    excluded = raw.loc[
        ~valid,
        ["msa_geoid", "msa_name", "hazard", "status", "criticality_status", "beta_rho"],
    ].copy()
    excluded["exclusion_reason"] = np.where(
        excluded.status.eq("ok"),
        "invalid_or_nonfinite_ar",
        excluded.criticality_status.where(
            excluded.status.eq("non_subcritical"), excluded.status
        ),
    )
    raw = raw.loc[valid].copy()
    assert np.isfinite(raw[[AR, RAND, SD]]).all().all()
    assert raw[RAND].gt(0).all() and raw[SD].ge(0).all()
    assert raw.n_rand.eq(config["random_draws"]).all()
    assert np.allclose(raw[AR] / raw[RAND], raw.ar_premium_vs_rand, atol=1e-12, rtol=0)
    columns = [
        "msa_geoid",
        "msa_name",
        "hazard",
        "hazard_name",
        "n_pois",
        "k_pois_removed",
        "n_rand",
        AR,
        RAND,
        SD,
        "status",
        "criticality_status",
        "beta_rho",
    ]
    prepared = []
    for hazard in ORDER:
        d = (
            raw.loc[raw.hazard.eq(hazard), columns]
            .sort_values(AR, ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        if not len(d):
            continue
        shown = np.arange(len(d)) < (30 if hazard == "IFLD" else len(d))
        y, r = (d[AR].to_numpy(), d[RAND].to_numpy())
        labels = d.msa_name.map(short_name)
        assert labels.is_unique, f"Ambiguous shortened city labels: {hazard}"
        d = d.assign(
            display_label=labels,
            rank_within_hazard=np.arange(len(d)) + 1,
            above_random_mean=y > r,
            ar_relative_to_random=y / r,
            shown_in_figure=shown,
            off_scale=False,
        )
        d[EFFECT] = 100 * (y / r - 1)
        d["display_observed_y"] = np.where(shown, y, np.nan)
        d["display_effect_y"] = np.where(shown, d[EFFECT], np.nan)
        prepared.append(d)
    data = pd.concat(prepared, ignore_index=True)
    data["display_axis"] = "main"
    wildfire = (
        data.loc[data.hazard.eq("WFIR") & data.shown_in_figure, EFFECT]
        .sort_values()
        .to_numpy()
    )
    if len(wildfire) > 1 and np.diff(wildfire).max() >= 40:
        gap = int(np.argmax(np.diff(wildfire)))
        threshold = (wildfire[gap] + wildfire[gap + 1]) / 2
        data.loc[
            data.hazard.eq("WFIR") & data[EFFECT].gt(threshold), "display_axis"
        ] = "upper"
    for group_name in ["main", "upper"]:
        values = data.loc[
            data.shown_in_figure & data.display_axis.eq(group_name), EFFECT
        ]
        if values.empty:
            continue
        step = 2.0 if group_name == "main" else 5.0
        pad = max(
            1.25 if group_name == "main" else 2.0, 0.05 * (values.max() - values.min())
        )
        low = (
            step * np.floor((min(values.min(), 0.0) - pad) / step)
            if group_name == "main"
            else step * np.floor((values.min() - pad) / step)
        )
        high = (
            step * np.ceil((max(values.max(), 0.0) + pad) / step)
            if group_name == "main"
            else step * np.ceil((values.max() + pad) / step)
        )
        mask = data.display_axis.eq(group_name)
        data.loc[mask, "display_ymin"] = low
        data.loc[mask, "display_ymax"] = high
        assert values.between(low, high).all()
    if data.display_axis.eq("upper").any():
        assert (
            data.loc[data.display_axis.eq("main"), "display_ymax"].max()
            < data.loc[data.display_axis.eq("upper"), "display_ymin"].min()
        )
    return (data, excluded)


def draw(data, hazards):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK,
            "text.color": INK,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.5,
            "xtick.major.size": 1.5,
            "ytick.major.size": 2,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=DPI, facecolor="white")
    panels = {"IFLD": (14, 171, 163, 24), "WFIR": (14, 99, 163, 32)}
    bottom = ["TRND", "ERQK", "HRCN"]
    counts = (
        data.loc[data.shown_in_figure]
        .groupby("hazard")
        .size()
        .reindex(ORDER, fill_value=0)
    )
    available, left, gap = (154.0, 10.5, 7.0)
    for hazard in bottom:
        panel_counts = counts.loc[bottom].clip(lower=1)
        width = available * panel_counts[hazard] / panel_counts.sum()
        panels[hazard] = (left, 36.5, width, 24)
        left += width + gap
    for hazard in ORDER:
        d = data.loc[data.hazard.eq(hazard) & data.shown_in_figure]
        xm, ym, wm, hm = panels[hazard]
        if not len(d):
            ax = fig.add_axes([xm / WIDTH, ym / HEIGHT, wm / WIDTH, hm / HEIGHT])
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, "No convergent cases", ha="center", va="center", fontsize=9
            )
            fig.text(
                xm / WIDTH,
                (ym + hm + 2.2) / HEIGHT,
                f"{hazards[hazard]}  (n = 0)",
                fontsize=9.5,
                weight="bold",
            )
            continue
        x = np.arange(len(d))
        y = d[EFFECT].to_numpy()
        above = d.above_random_mean.to_numpy()
        high_points = d.display_axis.eq("upper").to_numpy()
        broken = bool(high_points.any())
        main_height = hm - 8 if broken else hm
        ax = fig.add_axes([xm / WIDTH, ym / HEIGHT, wm / WIDTH, main_height / HEIGHT])
        axes = [("main", ax)]
        if broken:
            upper = fig.add_axes(
                [xm / WIDTH, (ym + main_height + 2) / HEIGHT, wm / WIDTH, 6 / HEIGHT]
            )
            axes.append(("upper", upper))
        for section, target in axes:
            in_section = high_points if section == "upper" else ~high_points
            source = d.loc[d.display_axis.eq(section)].iloc[0]
            ymin, ymax = source[["display_ymin", "display_ymax"]]
            for flag in (False, True):
                mask = above == flag
                colour = COLOURS[hazard] if flag else GREY
                target.vlines(
                    x[mask], 0.0, y[mask], color=colour, linewidth=0.65, zorder=2
                )
                points = mask & in_section
                target.scatter(
                    x[points],
                    y[points],
                    s=3.8**2,
                    color=colour,
                    edgecolors=GREY_EDGE,
                    linewidths=0.35,
                    zorder=4,
                )
            target.set(xlim=(-0.5, len(d) - 0.5), ylim=(ymin, ymax))
            target.spines[["top", "right"]].set_visible(False)
            target.tick_params(axis="y", labelsize=9, pad=2)
            target.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:+g}" if value else "0")
            )
            if section == "main":
                target.axhline(0.0, color="#656565", lw=0.7, ls=(0, (3, 2)), zorder=1)
                ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(
                    ymin, ymax
                )
                target.set_yticks(ticks[(ticks >= ymin) & (ticks <= ymax)])
            else:
                target.set_yticks([ymin, ymax])
                target.spines["bottom"].set_visible(False)
                target.tick_params(axis="x", bottom=False, labelbottom=False)
                for j in np.flatnonzero(in_section):
                    target.annotate(
                        f"{y[j]:+.1f}%",
                        (x[j], y[j]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                        va="center",
                    )
        if broken:
            for target, edge in [(ax, 1.0), (upper, 0.0)]:
                dx = 0.6 / wm
                dy = 0.35 / (main_height if target is ax else 6.0)
                for side in [0.0, 1.0]:
                    target.plot(
                        [side - dx, side + dx],
                        [edge - dy, edge + dy],
                        transform=target.transAxes,
                        color=INK,
                        lw=0.7,
                        clip_on=False,
                    )
        ax.set_xticks(x, d.display_label, rotation=90, fontsize=9)
        ax.tick_params(axis="x", pad=1.8)
        total = data.hazard.eq(hazard).sum()
        title = f"{hazards[hazard]}  (n = {total})"
        if hazard == "IFLD":
            title = f"Inland flooding  ·  Top {len(d)} of {total} MSAs by observed AR"
        fig.text(
            xm / WIDTH, (ym + hm + 2.2) / HEIGHT, title, fontsize=9.5, weight="bold"
        )
    fig.text(
        2.5 / WIDTH,
        0.51,
        "AR difference from random mean (%)",
        rotation=90,
        fontsize=9.5,
        ha="center",
        va="center",
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=3.8,
            color=COLOURS["IFLD"],
            label="Above random mean",
        ),
        Line2D(
            [], [], marker="o", ls="", ms=3.8, color=GREY, label="At/below random mean"
        ),
        Line2D([], [], ls="--", lw=0.7, color="#656565", label="Random mean (0%)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        frameon=False,
        ncol=3,
        fontsize=9,
        handlelength=1.3,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="Redraw the saved figure summary without input data.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config.json").read_text())
    if args.from_results:
        data = pd.read_csv(
            root / "results/ar_by_msa_plot_data.csv", dtype={"msa_geoid": str}
        )
    else:
        data, excluded = prepare(root, config)
    fig = draw(data, config["hazards"])
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "figures" / f"{STEM}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    data.to_csv(root / "results/ar_by_msa_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
