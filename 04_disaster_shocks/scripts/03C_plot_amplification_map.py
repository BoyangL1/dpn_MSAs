"""Plot the hazard/random AR map from this experiment's local results and polygons."""

from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "shock_summary_mpl")
)
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_hex, to_rgb
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from shapely.geometry import box as geometry_box
from shock_lib.shock_paths import ROOT, DATA_DIR

STEM = "fig_ar_over_random_ar_all_msa_slide_contrast_markers"
WIDTH, HEIGHT, DPI = (18.0, 9.4, 600)
BOUNDS = (-2480000.0, 2350000.0, 170000.0, 3260000.0)
PALETTE = {
    "IFLD": "#1965B0",
    "HRCN": "#008568",
    "ERQK": "#B47A08",
    "WFIR": "#C12E42",
    "TRND": "#7546A2",
}
MARKERS = {"IFLD": "o", "HRCN": "D", "ERQK": "s", "WFIR": "^", "TRND": "P"}
LAND, LOW, INK, VMAX = ("#F3F3F1", "#DCDDDC", "#1f1f1f", 1.35)
COLOUR_GAMMA = 0.5
NON_SUBCRITICAL = "#74556D"


def short_name(name):
    city, sep, state = str(name).rpartition(", ")
    if not sep:
        return str(name)
    return (
        f"{city.split('-')[0].replace('/Jefferson County', '')}, {state.split('-')[0]}"
    )


def colourmaps():
    maps = {}
    for hazard, colour in PALETTE.items():
        rgb = np.asarray(to_rgb(colour))
        maps[hazard] = LinearSegmentedColormap.from_list(
            hazard, [(0, 0.9 + 0.1 * rgb), (0.65, rgb), (1, 0.55 * rgb)], N=1024
        )
    return maps


def colour_normalization():
    """Use one shared mapping to reveal small excesses above the random mean."""
    return PowerNorm(gamma=COLOUR_GAMMA, vmin=1.0, vmax=VMAX, clip=True)


def load(root, config, cmaps, norm=None):
    if norm is None:
        norm = colour_normalization()
    raw = pd.read_csv(root / "results/shock_results.csv", dtype={"msa_geoid": str})
    required = {
        "msa_geoid",
        "msa_name",
        "hazard",
        "amplification_ratio_unw",
        "amplification_ratio_unw_rand",
    }
    assert required <= set(raw), f"Missing result columns: {required - set(raw)}"
    assert "status" in raw and "criticality_status" in raw
    assert not raw.duplicated(["msa_geoid", "hazard"]).any()
    assert set(raw.hazard) <= set(config["hazards"]) <= set(PALETTE)
    valid = (
        raw.status.eq("ok")
        & raw.k_pois_removed.gt(0)
        & np.isfinite(
            raw[["amplification_ratio_unw", "amplification_ratio_unw_rand"]]
        ).all(axis=1)
        & raw.amplification_ratio_unw_rand.gt(0)
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
    assert (
        np.isfinite(raw[["amplification_ratio_unw", "amplification_ratio_unw_rand"]])
        .all()
        .all()
    )
    assert raw.amplification_ratio_unw_rand.gt(0).all()
    if "n_rand" in raw:
        assert raw.n_rand.eq(config["random_draws"]).all()
    raw["relative_ar"] = raw.amplification_ratio_unw / raw.amplification_ratio_unw_rand
    if "ar_premium_vs_rand" in raw:
        assert np.allclose(
            raw.relative_ar, raw.ar_premium_vs_rand, rtol=1e-12, atol=1e-12
        )
    raw["_hazard_order"] = raw.hazard.map(
        {h: i for i, h in enumerate(config["hazards"])}
    )
    selected = (
        raw.sort_values(
            ["msa_geoid", "relative_ar", "_hazard_order"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("msa_geoid")
        .drop(columns="_hazard_order")
    )
    selected["hazard_name"] = selected.hazard.map(config["hazards"])
    selected["map_colour"] = [
        LOW if ratio <= 1 else to_hex(cmaps[h](norm(ratio)))
        for h, ratio in zip(selected.hazard, selected.relative_ar)
    ]
    selected["above_random_mean"] = selected.relative_ar.gt(1)
    selected["colour_capped"] = selected.relative_ar.gt(VMAX)
    msa = gpd.read_parquet(DATA_DIR / "msa_boundaries.parquet")
    msa["GEOID"] = msa.GEOID.astype(str)
    assert msa.GEOID.is_unique and set(selected.msa_geoid) <= set(msa.GEOID)
    msa = msa.set_index("GEOID").to_crs(5070)
    shown = gpd.GeoDataFrame(
        selected, geometry=msa.loc[selected.msa_geoid].geometry.to_numpy(), crs=msa.crs
    )
    unavailable = excluded.loc[excluded.status.eq("non_subcritical")].drop_duplicates(
        "msa_geoid"
    )
    unavailable = gpd.GeoDataFrame(
        unavailable,
        geometry=msa.loc[unavailable.msa_geoid].geometry.to_numpy(),
        crs=msa.crs,
    )
    states = gpd.read_parquet(DATA_DIR / "states.parquet").to_crs(msa.crs)
    assert shown.geometry.is_valid.all() and states.geometry.is_valid.all()
    assert len(shown) == shown.msa_geoid.nunique() == raw.msa_geoid.nunique()
    return (raw, shown, states, excluded, unavailable)


def map_panel(fig, shown, states, unavailable):
    ax = fig.add_axes([0.05 / WIDTH, 0.1 / HEIGHT, 14.4 / WIDTH, 9.22 / HEIGHT])
    states.plot(ax=ax, color=LAND, edgecolor="white", linewidth=0.45, zorder=0)
    if len(unavailable):
        unavailable.plot(
            ax=ax,
            facecolor="white",
            edgecolor=NON_SUBCRITICAL,
            linewidth=0.6,
            hatch="////",
            zorder=2,
        )
    low, high = (
        shown.loc[~shown.above_random_mean],
        shown.loc[shown.above_random_mean],
    )
    if len(low):
        low.plot(ax=ax, color=LOW, edgecolor="white", linewidth=0.25, zorder=1)
    if len(high):
        high.plot(
            ax=ax, color=high.map_colour, edgecolor="white", linewidth=0.3, zorder=2
        )
        high.boundary.plot(ax=ax, color=high.map_colour, linewidth=0.7, zorder=3)
        for hazard in MARKERS:
            points = high.loc[high.hazard.eq(hazard)].geometry.representative_point()
            ax.plot(
                points.x,
                points.y,
                linestyle="none",
                marker=MARKERS[hazard],
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor="#292929",
                markeredgewidth=0.6,
                zorder=4,
            )
    ax.set(xlim=BOUNDS[:2], ylim=BOUNDS[2:], aspect="equal")
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    return ax


def callouts(fig, ax, leaders, shown):
    """Move labels only; preserve polygons and avoid text over screened cities."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    region = ax.get_window_extent(renderer).padded(-2)
    occupied, paths = ([], [])
    for i in sorted(
        leaders.index, key=lambda j: len(leaders.loc[j, "label"]), reverse=True
    ):
        row = leaders.loc[i]
        point = ax.transData.transform((row.x, row.y))
        ann = ax.annotate(
            row.label,
            (row.x, row.y),
            xytext=(0.5, 0.5),
            textcoords="axes fraction",
            fontsize=8.5,
            ha="center",
            va="center",
            color=INK,
            zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.95, pad=0.6),
            arrowprops=dict(
                arrowstyle="-", color=PALETTE[row.hazard], lw=0.65, shrinkA=2, shrinkB=0
            ),
        )
        ann.update_positions(renderer)
        bounds = Text.get_window_extent(ann, renderer)
        candidates = []
        for radius in (14, 20, 28, 36, 45, 60, 75, 90, 110, 130, 155, 180, 210, 240):
            for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                center = point + radius * fig.dpi / 72 * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                rect = Bbox.from_bounds(
                    center[0] - bounds.width / 2,
                    center[1] - bounds.height / 2,
                    bounds.width,
                    bounds.height,
                )
                padded = rect.padded(2)
                if not (
                    region.contains(padded.x0, padded.y0)
                    and region.contains(padded.x1, padded.y1)
                ):
                    continue
                if any((padded.overlaps(b) for b in occupied)):
                    continue
                corners = ax.transData.inverted().transform(
                    [[padded.x0, padded.y0], [padded.x1, padded.y1]]
                )
                if shown.intersects(geometry_box(*corners.ravel())).any():
                    continue
                end = [
                    np.clip(point[0], rect.x0, rect.x1),
                    np.clip(point[1], rect.y0, rect.y1),
                ]
                path = MplPath([point, end])
                if any((path.intersects_bbox(b, filled=False) for b in occupied)):
                    continue
                if any((p.intersects_bbox(padded, filled=False) for p in paths)):
                    continue
                candidates.append((center, padded, path))
            if candidates:
                break
        assert candidates, f"No unobstructed callout position for {row.msa_name}"
        center, padded, path = candidates[0]
        ann.set_position(ax.transAxes.inverted().transform(center))
        occupied.append(padded)
        paths.append(path)
        leaders.loc[i, ["label_x", "label_y"]] = ax.transData.inverted().transform(
            center
        )


def legend(fig, hazards, cmaps, has_unavailable=False, norm=None):
    if norm is None:
        norm = colour_normalization()
    center = 16.2
    fig.text(
        center / WIDTH,
        9.04 / HEIGHT,
        "Hazard AR",
        fontsize=10,
        ha="center",
        va="center",
        color=INK,
    )
    fig.add_artist(
        plt.Line2D(
            [14.91 / WIDTH, 17.49 / WIDTH],
            [8.81 / HEIGHT] * 2,
            transform=fig.transFigure,
            color=INK,
            lw=0.6,
        )
    )
    fig.text(
        center / WIDTH,
        8.55 / HEIGHT,
        "Mean random AR",
        fontsize=10,
        ha="center",
        va="center",
        color=INK,
    )
    fig.text(
        center / WIDTH,
        8.16 / HEIGHT,
        "Square-root colour scale",
        fontsize=7.5,
        ha="center",
        va="center",
        color="#555555",
    )
    for i, (hazard, name) in enumerate(hazards.items()):
        y = 7.79 - i * 1.37
        fig.text(
            center / WIDTH,
            y / HEIGHT,
            name,
            fontsize=9,
            ha="center",
            va="center",
            color=INK,
        )
        fig.add_artist(
            plt.Line2D(
                [14.82 / WIDTH],
                [y / HEIGHT],
                transform=fig.transFigure,
                linestyle="none",
                marker=MARKERS[hazard],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="#292929",
                markeredgewidth=0.65,
            )
        )
        ax = fig.add_axes([15 / WIDTH, (y - 0.47) / HEIGHT, 2.4 / WIDTH, 0.2 / HEIGHT])
        ax.imshow(
            np.linspace(1, VMAX, 256)[None, :],
            extent=[1, VMAX, 0, 1],
            cmap=cmaps[hazard],
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set(yticks=[], xticks=[1, VMAX])
        ax.set_xticklabels(["1", "≥1.35"], fontsize=8)
        ax.tick_params(axis="x", length=1.8, width=0.5, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.add_artist(
        plt.Rectangle(
            (15.5 / WIDTH, 0.59 / HEIGHT),
            0.4 / WIDTH,
            0.23 / HEIGHT,
            transform=fig.transFigure,
            fc=LOW,
            ec="none",
        )
    )
    fig.text(16.08 / WIDTH, 0.705 / HEIGHT, "≤1", fontsize=9, va="center", ha="left")
    if has_unavailable:
        fig.add_artist(
            plt.Rectangle(
                (14.85 / WIDTH, 0.14 / HEIGHT),
                0.4 / WIDTH,
                0.23 / HEIGHT,
                transform=fig.transFigure,
                fc="white",
                ec=NON_SUBCRITICAL,
                linewidth=0.5,
                hatch="////",
            )
        )
        fig.text(
            15.45 / WIDTH,
            0.255 / HEIGHT,
            "Non-subcritical",
            fontsize=8.5,
            va="center",
            ha="left",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    root = parser.parse_args().root.resolve()
    config = json.loads((root / "config.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "text.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
            "axes.linewidth": 0.65,
        }
    )
    cmaps = colourmaps()
    norm = colour_normalization()
    _, shown, states, _, unavailable = load(root, config, cmaps, norm)
    leaders = (
        shown.loc[shown.above_random_mean]
        .sort_values(
            ["relative_ar", "msa_geoid"], ascending=[False, True], kind="stable"
        )
        .groupby("hazard", sort=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )
    points = leaders.geometry.representative_point()
    leaders["x"], leaders["y"] = (points.x.to_numpy(), points.y.to_numpy())
    leaders["label"] = [
        f"{short_name(r.msa_name)}\n{config['hazards'][r.hazard]} · {r.relative_ar:.2f}×"
        for r in leaders.itertuples()
    ]
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54), dpi=150, facecolor="white")
    ax = map_panel(fig, shown, states, unavailable)
    callouts(fig, ax, leaders, pd.concat([shown, unavailable], ignore_index=True))
    legend(fig, config["hazards"], cmaps, bool(len(unavailable)), norm)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "figures" / f"{STEM}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(
        f"Summary map: {len(shown)} MSAs; {shown.above_random_mean.sum()} above random mean; {len(leaders)} hazard leaders.",
        flush=True,
    )


if __name__ == "__main__":
    main()
