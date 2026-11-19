"""Map styling and collision-aware callout placement."""

from types import SimpleNamespace
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.path import Path as MplPath
from shapely.geometry import box as geometry_box

WIDTH, HEIGHT = 18.0, 9.4
HAZARDS = {
    "IFLD": "Inland flooding",
    "HRCN": "Hurricane",
    "ERQK": "Earthquake",
    "WFIR": "Wildfire",
    "TRND": "Tornado",
}
PALETTE = {
    "IFLD": "#1965B0",
    "HRCN": "#008568",
    "ERQK": "#B47A08",
    "WFIR": "#C12E42",
    "TRND": "#7546A2",
}
MARKERS = {"IFLD": "o", "HRCN": "D", "ERQK": "s", "WFIR": "^", "TRND": "P"}


def short_name(name):
    city, states = str(name).rsplit(", ", 1)
    return (
        city.split("-")[0].replace("/Jefferson County", "")
        + ", "
        + states.split("-")[0]
    )


base = SimpleNamespace(INK="#1f1f1f", HAZARDS=HAZARDS, short_name=short_name)
maps = SimpleNamespace(
    base=base,
    CM=1 / 2.54,
    ORDER=list(HAZARDS),
    BOUNDS=(-2.48e6, 2.35e6, 0.17e6, 3.26e6),
    LAND="#F3F3F1",
    LOW="#DCDDDC",
)


def configure():
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "text.color": base.INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
            "axes.linewidth": 0.65,
            "hatch.linewidth": 0.28,
        }
    )
    maps.COLOURS = PALETTE.copy()
    maps.CMAPS = {}
    for hazard, colour in PALETTE.items():
        rgb = np.asarray(to_rgb(colour))
        maps.CMAPS[hazard] = LinearSegmentedColormap.from_list(
            hazard, [(0, 0.40 + 0.60 * rgb), (0.43, 0.15 + 0.85 * rgb), (1, rgb)]
        )


def city_callouts(
    fig, ax, leaders, shown, fontsize=9, radii=(14, 20, 28, 36, 45, 60, 75, 90)
):
    """Keep every label outside all screened MSA polygons, including low AR."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    region = ax.get_window_extent(renderer).padded(-2)
    occupied, paths, artists = [], [], []
    # Longest labels first, since coastal placement is more constrained.
    for i in sorted(
        leaders.index, key=lambda j: len(leaders.loc[j, "rank"]), reverse=True
    ):
        row = leaders.loc[i]
        point = ax.transData.transform((row.x, row.y))
        ann = ax.annotate(
            row["rank"],
            (row.x, row.y),
            xytext=(0.5, 0.5),
            textcoords="axes fraction",
            fontsize=fontsize,
            ha="center",
            va="center",
            color=maps.base.INK,
            zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.95, pad=0.6),
            arrowprops=dict(
                arrowstyle="-",
                color=maps.COLOURS[row.hazard],
                lw=0.65,
                shrinkA=2,
                shrinkB=0,
            ),
        )
        ann.update_positions(renderer)
        bounds = Text.get_window_extent(ann, renderer)
        candidates = []
        for radius in radii:
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
                if any(padded.overlaps(b) for b in occupied):
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
                if any(path.intersects_bbox(b, filled=False) for b in occupied):
                    continue
                if any(p.intersects_bbox(padded, filled=False) for p in paths):
                    continue
                candidates.append((center, padded, path))
            if candidates:
                break
        assert candidates, f"No unobstructed callout position for {row.msa_name}"
        center, padded, path = candidates[0]
        ann.set_position(ax.transAxes.inverted().transform(center))
        occupied.append(padded)
        paths.append(path)
        artists.append(ann)
        leaders.loc[i, ["label_x", "label_y"]] = ax.transData.inverted().transform(
            center
        )
    return artists
