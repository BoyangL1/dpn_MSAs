"""Plot metropolitan coverage for the five hazard scenarios."""

from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "shock_coverage_mpl")
)
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from shock_lib.shock_paths import ROOT, DATA_DIR

STEM = "fig_si_hazard_coverage_18cm"
BACKGROUND, HIGHLIGHT = ("#D9DDE0", "#216B8A")
WIDTH, HEIGHT, PREVIEW_DPI = (18.0, 8.9, 150)


def load(root, config):
    msas = gpd.read_parquet(DATA_DIR / "msa_boundaries.parquet").to_crs(5070)
    msas["GEOID"] = msas.GEOID.astype(str).str.zfill(5)
    assert len(msas) == msas.GEOID.nunique() == config["base_msa_count"]
    cohort_config = config["hazard_coverage"]
    cohort = pd.read_csv(
        DATA_DIR / cohort_config["cohort_file"], dtype={"msa_geoid": str}
    )
    assert len(cohort) == cohort.msa_geoid.nunique() == config["base_msa_count"]
    assert set(cohort.msa_geoid) == set(msas.GEOID)
    keep = cohort.included_in_coverage.astype(str).str.lower().eq("true")
    assert keep.equals(cohort.best_epoch.gt(1))
    assert int(keep.sum()) == cohort_config["study_msa_count"]
    assert int((~keep).sum()) == cohort_config["excluded_msa_count"]
    study_ids = set(cohort.loc[keep, "msa_geoid"])
    msas = msas.loc[msas.GEOID.isin(study_ids)].copy()
    states = gpd.read_parquet(DATA_DIR / "states.parquet").to_crs(5070)
    assert not states.STUSPS.isin({"AK", "HI", "PR", "GU", "VI", "MP", "AS"}).any()
    assert msas.geometry.is_valid.all() and states.geometry.is_valid.all()
    assert msas.intersects(states.union_all()).all()
    raw = pd.read_csv(root / "results/shock_results.csv", dtype={"msa_geoid": str})
    raw = raw.loc[raw.msa_geoid.isin(study_ids)].copy()
    selected = {
        hazard: set(raw.loc[raw.hazard.eq(hazard), "msa_geoid"])
        for hazard in config["hazards"]
    }
    return (msas, states, raw, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use at least 300 dpi for the manuscript export")
    root = args.root.resolve()
    config = json.loads((root / "config.json").read_text())
    assert len(config["hazards"]) == 5
    (root / "figures").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "text.color": "#252A2E",
            "savefig.bbox": None,
        }
    )
    msas, states, _, selected = load(root, config)
    counts = {hazard: len(ids) for hazard, ids in selected.items()}
    msas.geometry = msas.geometry.simplify(250, preserve_topology=True)
    states.geometry = states.geometry.simplify(500, preserve_topology=True)
    x0, y0, x1, y1 = states.total_bounds
    dx, dy = (0.015 * (x1 - x0), 0.015 * (y1 - y0))
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54), facecolor="white")
    panel_w, panel_h = (5.7, 3.45)
    positions = [(0.15, 4.85), (6.15, 4.85), (12.15, 4.85), (3.15, 0.85), (9.15, 0.85)]
    for (hazard, title), (left, bottom) in zip(config["hazards"].items(), positions):
        ax = fig.add_axes(
            [left / WIDTH, bottom / HEIGHT, panel_w / WIDTH, panel_h / HEIGHT]
        )
        states.boundary.plot(ax=ax, color="#CDD2D5", linewidth=0.2, zorder=0)
        msas.plot(
            ax=ax, facecolor=BACKGROUND, edgecolor="white", linewidth=0.06, zorder=1
        )
        msas.loc[msas.GEOID.isin(selected[hazard])].plot(
            ax=ax, facecolor=HIGHLIGHT, edgecolor=HIGHLIGHT, linewidth=0.1, zorder=2
        )
        ax.set_xlim(x0 - dx, x1 + dx)
        ax.set_ylim(y0 - dy, y1 + dy)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.text(
            (left + panel_w / 2) / WIDTH,
            (bottom + panel_h + 0.1) / HEIGHT,
            f"{title} (n = {counts[hazard]})",
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="center",
        )
    fig.legend(
        handles=[
            Patch(facecolor=BACKGROUND, label=f"Study MSAs ({len(msas)})"),
            Patch(facecolor=HIGHLIGHT, label="Included MSAs"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
        handleheight=0.85,
        columnspacing=2.2,
        borderaxespad=0,
        labelcolor="#51585E",
    )
    fig.savefig(
        root / "figures" / f"{STEM}.png",
        dpi=args.dpi,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
