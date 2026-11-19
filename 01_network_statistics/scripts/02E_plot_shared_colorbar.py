#!/usr/bin/env python3
"""Plot the shared metropolitan node-count colorbar."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / ".work/plot_config.json").read_text())
    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "text.color": "#252a2e",
            "axes.labelcolor": "#252a2e",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "viridis_readable", plt.get_cmap("viridis")(np.linspace(0.03, 0.88, 256))
    )
    norm = LogNorm(config["norm_vmin"], config["norm_vmax"])
    fig = plt.figure(figsize=(1.4, 3.6))
    ax = fig.add_axes([0.08, 0.08, 0.20, 0.84])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    bar.set_label(r"Nodes per MSA, $N$", fontsize=12, labelpad=6)
    bar.locator = LogLocator(base=10, subs=[1], numticks=5)
    bar.formatter = LogFormatterMathtext(base=10)
    bar.update_ticks()
    bar.ax.yaxis.set_minor_formatter(NullFormatter())
    bar.ax.tick_params(which="major", labelsize=11.5, length=3, width=0.65)
    bar.ax.tick_params(which="minor", length=1.6, width=0.5)
    bar.outline.set_linewidth(0.65)
    output = ROOT / "figures/msa_node_count_colorbar.png"
    fig.savefig(
        output, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03
    )
    plt.close(fig)
    meta = {
        "font_path": font,
        "norm_vmin": config["norm_vmin"],
        "norm_vmax": config["norm_vmax"],
        "colormap": "viridis truncated to 0.03..0.88",
        "normalization": "LogNorm",
        "shared_by": [
            "national_in_strength_distribution.png",
            "national_out_strength_distribution.png",
            "msa_dependency_weight_ccdf.png",
        ],
        "dpi": 600,
        "embedded_in_main_figures": False,
    }
    (ROOT / ".work/shared_colorbar_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
