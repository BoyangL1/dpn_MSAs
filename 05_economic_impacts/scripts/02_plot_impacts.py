#!/usr/bin/env python3
"""Map initial losses, propagated impacts, and hazard-to-random impact ratios."""
from pathlib import Path
import argparse
import os
import tempfile

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
DATA = (
    Path(os.environ.get("DPN_DATA_DIR", ANALYSIS.parent / "data")).expanduser()
    / "disaster_shocks"
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpn_economic_matplotlib")
)

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_hex

from economic_lib import map_style as reference

maps, plt = reference.maps, reference.plt
WIDTH, HEIGHT = reference.WIDTH, reference.HEIGHT
OUT = ANALYSIS / "figures"
INPUT = ANALYSIS / "results/msa_hazard_summary.csv"
RAMPS = {
    "IFLD": ["#FAFAF9", "#D5E5F2", "#287DB8", "#002B55"],
    "HRCN": ["#FAFAF9", "#D5EADB", "#209860", "#003D26"],
    "ERQK": ["#FAFAF9", "#F4E6CA", "#CF911B", "#573000"],
    "WFIR": ["#FAFAF9", "#F2DAD4", "#DC4C3C", "#650D19"],
    "TRND": ["#FAFAF9", "#E6DCEE", "#9253B9", "#381557"],
}
maps.LAND = "#F3F3F1"
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "impact_ratio", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4B879", "#B35806"]
)
SPECS = [
    dict(
        stem="initial_loss",
        column="initial_building_value_loss_dollars",
        known="known_coverage_hazard_initial_loss_dollars",
        title="Initial loss",
        ratio=False,
    ),
    dict(
        stem="propagated_loss",
        column="propagated_dollar_impact",
        known="known_coverage_hazard_propagated_impact_dollars",
        title="Propagated impact",
        ratio=False,
    ),
    dict(
        stem="hazard_vs_random",
        column="hazard_random_propagated_loss_ratio",
        title="Propagated impact",
        ratio=True,
    ),
]


def load(data_dir):
    raw = pd.read_csv(INPUT, dtype={"msa_geoid": str})
    if raw.duplicated(["msa_geoid", "hazard"]).any():
        raise ValueError("Hazard cases must be unique within each MSA.")
    boundaries = (
        gpd.read_parquet(data_dir / "msa_boundaries.parquet")
        .set_index("GEOID")
        .to_crs("EPSG:5070")
    )
    data = gpd.GeoDataFrame(
        raw,
        geometry=boundaries.loc[raw.msa_geoid].geometry.to_numpy(),
        crs=boundaries.crs,
    )
    states = gpd.read_parquet(data_dir / "states.parquet").to_crs(boundaries.crs)
    return data, states


def select_values(data, spec):
    """Select the largest untransformed value, with the same rule for every city."""
    data = data.copy()
    column = spec["column"]
    if spec["ratio"]:
        known = (
            data.known_coverage_hazard_propagated_impact_dollars
            / data.known_coverage_mean_random_propagated_impact_dollars
        )
    else:
        known = data[spec["known"]]
    complete = data[column].notna()
    data["partial_coverage"] = ~complete & known.notna()
    data[column] = data[column].fillna(known)
    data["_order"] = data.hazard.map({h: i for i, h in enumerate(maps.ORDER)})
    shown = (
        data.sort_values(
            ["msa_geoid", column, "_order"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop_duplicates("msa_geoid")
        .copy()
    )
    incomplete = set(data.loc[data[column].isna(), "msa_geoid"])
    shown.loc[shown.msa_geoid.isin(incomplete), column] = np.nan
    assert shown[column].dropna().gt(0).all()
    return shown


def absolute_scale(selections):
    values = np.concatenate(
        [
            selections[s["stem"]][s["column"]].dropna().to_numpy() / 1e9
            for s in SPECS
            if not s["ratio"]
        ]
    )
    nice = np.array([1.0, 2.0, 5.0, 10.0])
    lo_power = 10 ** np.floor(np.log10(values.min()))
    hi_power = 10 ** np.floor(np.log10(values.max()))
    low = nice[nice <= values.min() / lo_power][-1] * lo_power
    high = nice[nice >= values.max() / hi_power][0] * hi_power
    assert low < high and low <= values.min() <= values.max() <= high
    middle = 10 ** round(np.log10(np.sqrt(low * high)))
    return LogNorm(low, high), [float(low), float(middle), float(high)]


def legend(fig, spec, norm, ticks, partial, missing):
    center = 16.25
    fig.text(
        center / WIDTH,
        9.04 / HEIGHT,
        spec["title"],
        fontsize=9.5,
        ha="center",
        va="center",
    )
    if spec["ratio"]:
        fig.text(
            center / WIDTH,
            8.55 / HEIGHT,
            "Hazard / mean random",
            fontsize=9,
            ha="center",
            va="center",
        )
        fig.text(
            center / WIDTH,
            8.12 / HEIGHT,
            "Log scale",
            fontsize=9,
            ha="center",
            va="center",
        )
        ax = fig.add_axes([15.05 / WIDTH, 4.45 / HEIGHT, 0.35 / WIDTH, 3.2 / HEIGHT])
        ax.imshow(
            np.linspace(0, 1, 256)[:, None],
            extent=[0, 1, norm.vmin, norm.vmax],
            origin="lower",
            cmap=RATIO_CMAP,
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        ax.set(xticks=[], yticks=np.log2(ticks))
        ax.set_yticklabels([f"{value:g}×" for value in ticks], fontsize=9)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", length=2, width=0.5, pad=4)
        ax.axhline(0, color="#555555", lw=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, hazard in enumerate(maps.ORDER):
            y = 3.66 - i * 0.59
            fig.add_artist(
                plt.Line2D(
                    [15.06 / WIDTH],
                    [y / HEIGHT],
                    transform=fig.transFigure,
                    linestyle="none",
                    marker=reference.MARKERS[hazard],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgecolor="#292929",
                    markeredgewidth=0.65,
                )
            )
            fig.text(
                15.48 / WIDTH,
                y / HEIGHT,
                maps.base.HAZARDS[hazard],
                fontsize=9,
                ha="left",
                va="center",
            )
    else:
        fig.text(
            center / WIDTH,
            8.57 / HEIGHT,
            "US$ billions · log scale",
            fontsize=9,
            ha="center",
            va="center",
        )
        for i, hazard in enumerate(maps.ORDER):
            y = 7.79 - i * 1.37
            fig.text(
                center / WIDTH,
                y / HEIGHT,
                maps.base.HAZARDS[hazard],
                fontsize=9,
                ha="center",
                va="center",
                color=maps.COLOURS[hazard],
            )
            fig.add_artist(
                plt.Line2D(
                    [14.82 / WIDTH],
                    [y / HEIGHT],
                    transform=fig.transFigure,
                    linestyle="none",
                    marker=reference.MARKERS[hazard],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgecolor="#292929",
                    markeredgewidth=0.65,
                )
            )
            ax = fig.add_axes(
                [15 / WIDTH, (y - 0.47) / HEIGHT, 2.5 / WIDTH, 0.20 / HEIGHT]
            )
            ax.imshow(
                np.linspace(0, 1, 256)[None, :],
                extent=[0, 1, 0, 1],
                cmap=maps.CMAPS[hazard],
                aspect="auto",
                vmin=0,
                vmax=1,
            )
            ax.set(yticks=[], xticks=norm(ticks))
            ax.set_xticklabels([f"{value:g}" for value in ticks], fontsize=9)
            ax.tick_params(axis="x", length=1.8, width=0.5, pad=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if partial:
        fig.text(
            center / WIDTH,
            0.30 / HEIGHT,
            "* Partial coverage",
            fontsize=9,
            ha="center",
            va="center",
        )
    if missing:
        fig.add_artist(
            plt.Rectangle(
                (15.05 / WIDTH, 0.66 / HEIGHT),
                0.25 / WIDTH,
                0.23 / HEIGHT,
                transform=fig.transFigure,
                fc="white",
                ec="#666666",
                lw=0.5,
                hatch="////",
            )
        )
        fig.text(15.48 / WIDTH, 0.775 / HEIGHT, "Missing", fontsize=9, va="center")


def render(shown, states, spec, amount_norm, amount_ticks):
    column, ratio = spec["column"], spec["ratio"]
    shown = shown.copy()
    missing = shown[column].isna()
    numeric = shown.loc[~missing].copy()
    partial = numeric.loc[numeric.partial_coverage]
    if ratio:
        transformed = np.log2(numeric[column].to_numpy())
        largest_fold = max(2.0, float(2.0 ** np.max(np.abs(transformed))))
        power = 10.0 ** np.floor(np.log10(largest_fold))
        nice = np.array([1.0, 2.0, 5.0, 10.0])
        upper = float(nice[nice >= largest_fold / power][0] * power)
        extent = np.log2(upper)
        norm = Normalize(-extent, extent)
        above = sorted(
            {
                v * 10.0**p
                for p in range(int(np.ceil(np.log10(upper))) + 1)
                for v in [1.0, 2.0, 5.0]
                if 1.0 < v * 10.0**p <= upper
            }
        )
        ticks = [1 / v for v in above[::-1]] + [1.0] + above
        colours = RATIO_CMAP(norm(transformed))
        numeric["display_log2_ratio"] = transformed
        numeric["difference_from_random_pct"] = 100 * (numeric[column] - 1)
    else:
        norm, ticks = amount_norm, amount_ticks
        transformed = numeric[column].to_numpy() / 1e9
        colours = [maps.CMAPS[h](norm(v)) for h, v in zip(numeric.hazard, transformed)]
    assert np.all((norm(transformed) >= 0) & (norm(transformed) <= 1))
    numeric["map_colour"] = [to_hex(c) for c in colours]
    shown.loc[numeric.index, "map_colour"] = numeric.map_colour
    if ratio:
        shown.loc[
            numeric.index, ["display_log2_ratio", "difference_from_random_pct"]
        ] = numeric[["display_log2_ratio", "difference_from_random_pct"]]
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54), dpi=150, facecolor="white")
    ax = fig.add_axes([0.05 / WIDTH, 0.10 / HEIGHT, 14.4 / WIDTH, 9.22 / HEIGHT])
    states.plot(ax=ax, color=maps.LAND, edgecolor="white", linewidth=0.45, zorder=0)
    numeric.plot(
        ax=ax, color=numeric.map_colour, edgecolor="white", linewidth=0.3, zorder=2
    )
    numeric.boundary.plot(
        ax=ax,
        color="#777777" if ratio else numeric.hazard.map(maps.COLOURS),
        linewidth=0.5 if ratio else 0.65,
        zorder=3,
    )
    if missing.any():
        shown.loc[missing].plot(
            ax=ax,
            facecolor="white",
            edgecolor="#666666",
            linewidth=0.45,
            hatch="////",
            zorder=3,
        )
    for hazard in maps.ORDER:
        points = numeric.loc[numeric.hazard.eq(hazard)].geometry.representative_point()
        ax.plot(
            points.x,
            points.y,
            linestyle="none",
            marker=reference.MARKERS[hazard],
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor="#292929",
            markeredgewidth=0.6,
            zorder=4,
        )
    if len(partial):
        points = partial.geometry.representative_point()
        for x, y in zip(points.x, points.y):
            ax.annotate(
                "*",
                (x, y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="bottom",
                color="#292929",
                zorder=5,
            )
    x0, x1, y0, y1 = maps.BOUNDS
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    leaders = (
        numeric.sort_values([column, "msa_geoid"], ascending=[False, True])
        .groupby("hazard", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    points = leaders.geometry.representative_point()
    leaders["x"], leaders["y"] = points.x.to_numpy(), points.y.to_numpy()
    leaders["rank"] = [
        f"{maps.base.short_name(r.msa_name)}\n{maps.base.HAZARDS[r.hazard]} · "
        + (
            f"{getattr(r, column):.2f}×"
            if ratio
            else f"${getattr(r, column)/1e9:,.1f}B"
        )
        + ("*" if r.partial_coverage else "")
        for r in leaders.itertuples()
    ]
    callouts = reference.city_callouts(
        fig,
        ax,
        leaders,
        shown,
        fontsize=9,
        radii=(14, 20, 28, 36, 45, 60, 75, 90, 110, 130, 155),
    )
    if ratio:
        for annotation in callouts:
            annotation.arrow_patch.set_color("#555555")
    legend(fig, spec, norm, ticks, len(partial) > 0, missing.any())
    stem = spec["stem"]
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"{stem}: {len(numeric)} MSAs; {missing.sum()} missing; saved to {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", choices=[s["stem"] for s in SPECS] + ["all"], default="all"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Prepared disaster-shock boundary inputs.",
    )
    args = parser.parse_args()
    reference.configure()
    maps.CMAPS = {
        h: LinearSegmentedColormap.from_list(h, list(zip([0, 0.35, 0.7, 1], colors)))
        for h, colors in RAMPS.items()
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data, states = load(args.data_dir.expanduser())
    selections = {s["stem"]: select_values(data, s) for s in SPECS}
    norm, ticks = absolute_scale(selections)
    for spec in SPECS:
        if args.metric in ("all", spec["stem"]):
            render(selections[spec["stem"]], states, spec, norm, ticks)


if __name__ == "__main__":
    main()
