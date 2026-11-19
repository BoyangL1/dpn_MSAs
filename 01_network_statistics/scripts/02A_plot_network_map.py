#!/usr/bin/env python3
"""Plot metropolitan dependency networks."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    Path(os.environ.get("DPN_DATA_DIR", ROOT.parent / "data")).expanduser().resolve()
)
STATE_SHP = Path(
    os.environ.get(
        "DPN_STATES_PATH",
        DATA_DIR / "geography/cb_2018_us_state_500k/cb_2018_us_state_500k.shp",
    )
).expanduser()
ALBERS = "EPSG:5070"


def main():
    work_dir, figures = ROOT / ".work", ROOT / "figures"
    assert (work_dir / "ready.json").exists(), "Run 01_prepare_network.py first"
    config = json.loads((work_dir / "plot_config.json").read_text())
    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "text.color": "#334653"})
    nodes = pd.read_parquet(
        work_dir / "network_nodes.parquet", columns=["node_id", "LATITUDE", "LONGITUDE"]
    ).sort_values("node_id")
    edges = pd.read_parquet(
        work_dir / "network_edges.parquet",
        columns=["src_idx", "tgt_idx", "source_msa"],
        filters=[("is_intra_msa", "=", True)],
    )
    active = set(edges.source_msa)
    sampled = np.arange(len(edges))
    if len(sampled) > 400_000:
        sampled = np.random.default_rng(7).choice(sampled, size=400_000, replace=False)
    selected = edges.iloc[sampled]
    lon, lat = nodes.LONGITUDE.to_numpy(), nodes.LATITUDE.to_numpy()
    si, ti = selected.src_idx.to_numpy(), selected.tgt_idx.to_numpy()
    tf = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    sx, sy = tf.transform(lon[si], lat[si])
    tx, ty = tf.transform(lon[ti], lat[ti])
    segments = np.stack([np.column_stack([sx, sy]), np.column_stack([tx, ty])], axis=1)
    assert np.isfinite(segments).all()
    states = gpd.read_file(STATE_SHP)
    states = states.loc[
        ~states.STUSPS.isin({"AK", "HI", "PR", "GU", "VI", "MP", "AS"})
    ].to_crs(ALBERS)
    cbsa = gpd.read_file(config["cbsa_shapefile"])
    cbsa = cbsa.loc[cbsa.LSAD.eq("M1")].to_crs(ALBERS)
    active_cbsa, idle_cbsa = (
        cbsa.loc[cbsa.GEOID.isin(active)],
        cbsa.loc[~cbsa.GEOID.isin(active)],
    )
    width = 13.34 / 0.6
    fig = plt.figure(figsize=(width, width * 12.5 / 20), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    states.plot(
        ax=ax, facecolor="#f6f7f7", edgecolor="#ffffff", linewidth=0.65, zorder=0
    )
    idle_cbsa.boundary.plot(ax=ax, color="#e0e5e8", linewidth=0.18, zorder=1)
    active_cbsa.plot(
        ax=ax, facecolor="#e6eef3", edgecolor="#a7bdcb", linewidth=0.32, zorder=2
    )
    states.dissolve().boundary.plot(ax=ax, color="#bcc5cb", linewidth=0.55, zorder=3)
    ax.add_collection(
        LineCollection(
            segments,
            colors=[(*to_rgb("#145579"), 0.24)],
            linewidths=0.28,
            capstyle="round",
            zorder=4,
            rasterized=True,
        )
    )
    minx, miny, maxx, maxy = states.total_bounds
    w, h = maxx - minx, maxy - miny
    sy = max(1.09 * h, 1.06 * w / (20 / 12.5))
    sx = (20 / 12.5) * sy
    cx, top = 0.5 * (minx + maxx), maxy + 0.045 * h
    ax.set_xlim(cx - sx / 2, cx + sx / 2)
    ax.set_ylim(top - sy, top)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color="#145579", lw=3, label="Within-MSA\ndependency links"
            ),
            Patch(
                facecolor="#e6eef3",
                edgecolor="#a7bdcb",
                linewidth=0.6,
                label="Metropolitan area",
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=44,
        bbox_to_anchor=(0.025, 0.015),
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.65,
        borderaxespad=0,
        alignment="left",
    )
    output = figures / "metro_dependency_network_map.png"
    fig.savefig(output, dpi=300, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    metadata = {
        "n_active_msa": len(active),
        "n_within_msa_edges": len(edges),
        "n_drawn_edges": len(selected),
        "sampling": "At most 400,000 edges uniformly without replacement; seed 7; statistics use all edges",
        "crs": ALBERS,
        "font_family": "Arial",
        "font_path": font,
        "legend_font_size": 44,
        "figure_inches": [width, width * 12.5 / 20],
        "dpi": 300,
        "state_shapefile": str(STATE_SHP),
        "cbsa_shapefile": config["cbsa_shapefile"],
        "embedded_colorbar": False,
        "output": str(output.relative_to(ROOT)),
    }
    (work_dir / "metro_map_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"Saved {output}; {len(active)} MSAs, {len(selected):,} rendered links",
        flush=True,
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
