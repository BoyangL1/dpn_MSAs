"""Shared layout for distribution figures."""

import matplotlib.pyplot as plt

WIDTH_CM, HEIGHT_CM = 5.41, 6.6
FIGURE_SIZE = (WIDTH_CM / 2.54, HEIGHT_CM / 2.54)
FONT_SIZES = {"base": 9.5, "axis_label": 10, "tick_label": 9.5, "legend": 9.5}


def make_panel():
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    fig.subplots_adjust(
        left=1.46 / WIDTH_CM,
        right=1 - 0.30 / WIDTH_CM,
        bottom=1.55 / HEIGHT_CM,
        top=1 - 0.26 / HEIGHT_CM,
    )
    ax.tick_params(axis="x", which="major", pad=6)
    return fig, ax
