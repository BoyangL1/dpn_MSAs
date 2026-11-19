#!/usr/bin/env python3
"""18-cm presentation layout for the existing k=3..10 city trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from lib.plot_gnn_cluster_tracking import KS, prepare
from lib.plot_transfer_clusters import COLORS, style


PALETTE = list(COLORS)


def curve(y, reverse=False):
    from matplotlib.path import Path as PlotPath

    stages = list(reversed(KS)) if reverse else KS
    direction = -1 if reverse else 1
    vertices = [(stages[0], y[stages[0]])]
    codes = [PlotPath.MOVETO]
    for left, right in zip(stages[:-1], stages[1:]):
        vertices.extend(
            [
                (left + direction * 0.42, y[left]),
                (right - direction * 0.42, y[right]),
                (right, y[right]),
            ]
        )
        codes.extend([PlotPath.CURVE4] * 3)
    return vertices, codes


def render(root, data):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import Patch, PathPatch, Rectangle

    style(10)
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": "#252525",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4))
    ax = fig.add_axes([0.038, 0.10, 0.932, 0.84])
    half_step = 0.4 / len(data["ids"])
    reference = data["saved8"].cluster.to_numpy(dtype=int)

    for group in dict.fromkeys(data["labels"][10][data["order"]]):
        members = data["order"][data["labels"][10][data["order"]] == group]
        cluster8 = np.unique(reference[members])
        assert len(cluster8) == 1
        upper = {k: float(data["positions"][k][members[0]] - half_step) for k in KS}
        lower = {k: float(data["positions"][k][members[-1]] + half_step) for k in KS}
        assert all(
            np.isclose(lower[k] - upper[k], 0.8 * len(members) / len(data["ids"]))
            for k in KS
        )
        top_vertices, top_codes = curve(upper)
        bottom_vertices, bottom_codes = curve(lower, reverse=True)
        vertices = top_vertices + bottom_vertices + [top_vertices[0]]
        codes = top_codes + [PlotPath.LINETO] + bottom_codes[1:] + [PlotPath.CLOSEPOLY]
        ax.add_patch(
            PathPatch(
                PlotPath(vertices, codes),
                facecolor=PALETTE[cluster8[0] - 1],
                alpha=0.16,
                edgecolor="none",
                zorder=0,
            )
        )

    for index in data["order"]:
        vertices, codes = curve({k: float(data["positions"][k][index]) for k in KS})
        ax.add_patch(
            PathPatch(
                PlotPath(vertices, codes),
                facecolor="none",
                edgecolor=PALETTE[reference[index] - 1],
                linewidth=0.19,
                alpha=0.44,
                zorder=1,
            )
        )
    n_nodes = 0
    for k in KS:
        for node in data["nodes"][k]:
            left, width = k - 0.024, 0.048
            ax.add_patch(
                Rectangle(
                    (left, node["top"]),
                    width,
                    node["height"],
                    facecolor="#FEFEFE",
                    edgecolor="#9EA3A8",
                    linewidth=0.20,
                    zorder=2,
                )
            )
            members = data["order"][data["labels"][k][data["order"]] == node["cluster"]]
            for cluster8 in dict.fromkeys(reference[members]):
                part = members[reference[members] == cluster8]
                top = float(data["positions"][k][part[0]] - half_step)
                bottom = float(data["positions"][k][part[-1]] + half_step)
                ax.add_patch(
                    Rectangle(
                        (left, top),
                        width,
                        bottom - top,
                        facecolor=PALETTE[cluster8 - 1],
                        edgecolor="none",
                        zorder=3,
                    )
                )
            n_nodes += 1
    assert n_nodes == sum(KS)
    ax.set(
        xlim=(2.74, 10.26),
        ylim=(1.012, -0.012),
        xticks=KS,
        xticklabels=[f"$k={k}$" for k in KS],
        yticks=[],
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=7, labelsize=11, colors="#252525")
    ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
    handles = [
        Patch(facecolor=color, edgecolor="none", label=f"C{i + 1}")
        for i, color in enumerate(PALETTE)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.018),
        ncol=8,
        frameon=False,
        fontsize=10,
        title="Cluster at k = 8",
        title_fontsize=10,
        handlelength=1.1,
        handleheight=0.65,
        columnspacing=1.6,
        handletextpad=0.55,
        labelspacing=0.7,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=plt.Text):
        if item.get_visible() and item.get_text():
            assert item.get_fontsize() >= 10
            box = item.get_window_extent(renderer)
            assert (
                box.x0 >= -1
                and box.y0 >= -1
                and box.x1 <= fig.bbox.width + 1
                and box.y1 <= fig.bbox.height + 1
            ), item.get_text()
    output = root / "figures/gnn_all_pairs/cluster_tracking_k3_k10_nature"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    settings = dict(
        width_mm=180,
        height_mm=180,
        font="Arial",
        font_sizes_pt=[10, 11],
        minimum_font_size_pt_at_18cm_width=10,
        upper_left_title=False,
        per_k_node_labels=False,
        node_width_in_k_units=0.048,
        palette_source="plot_transfer_clusters.COLORS; same as existing k=8 figures",
        palette=PALETTE,
        city_line_width_pt=0.19,
        city_line_alpha=0.44,
        ribbon_alpha=0.16,
        n_city_paths=len(data["ids"]),
        n_terminal_ribbons=10,
        n_nodes=n_nodes,
        color_reference_k=8,
        pdf_fonttype=42,
        svg_fonttype="none",
        png_dpi=600,
        layout="Original city order, cluster memberships and vertical positions retained",
    )
    (data["output"] / "nature_style_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    print(f"Saved cluster tracking: {output}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    root = parser.parse_args().root.resolve()
    data = prepare(root)
    render(root, data)
