import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# IEEE-like styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Raw data
yaw_targets = [30, 90, 180]
yaw_measured = [
    [32, 30, 30, 28, 29, 31, 31, 29, 30, 30, 30, 30, 29, 28, 30],
    [92, 89, 90, 90, 90, 91, 91, 90, 90, 91, 90, 92, 90, 90, 89],
    [181, 181, 182, 180, 180, 180, 180, 182, 180, 180, 179, 181, 182, 180, 180]
]

dist_target = 100
dist_measured = [101.5, 100.1, 100.2, 99.8, 100.4, 100.0, 99.5, 100.5, 101.2,
                 100.2, 100.5, 99.5, 101.3, 99.6, 100.3]

x_offset_target = 0
y_offset_target = 0
#x_offset_measured = [0.3, 0.6, 0.3, 0.5, 0.6, 1.1, 1.1, 2.2, 1, 0.8, 1.8, 0.3, 1.1, 1.3, 0.4]
x_offset_measured = [-0.3, 0.6, -0.3, 0.5, 0.6, 1.1, -1.1, -2.2, 1, -0.8, 1.8, 0.3, -1.1, 1.3, 0.4]
#y_offset_measured = [0.3, 1, 1.6, 2, 2.3, 2.1, 2.1, 1.1, 2.7, 2.6, 1.6, 1, 2.9, 1.2, 2.6]
y_offset_measured = [-0.3, 1, 1.6, 2, -2.3, -2.1, -2.1, 1.1, -2.7, 2.6, 1.6, 1, -2.9, 1.2, 2.6]


# Convert to error relative to target
yaw_error = [np.array(m) - t for m, t in zip(yaw_measured, yaw_targets)]
dist_error = [np.array(dist_measured) - dist_target]
x_offset_error = np.array(x_offset_measured) - x_offset_target
y_offset_error = np.array(y_offset_measured) - y_offset_target

def ieee_boxplot(data, labels, title, ylabel, filename, colors, hatches):
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)  # single-column IEEE style

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanline=False,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        flierprops=dict(marker='o', markerfacecolor='white',
                        markeredgecolor='black', markersize=3, linestyle='none'),
        meanprops=dict(marker='D', markerfacecolor='black',
                       markeredgecolor='black', markersize=3)
    )

    for patch, color, hatch in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_hatch(hatch)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Axis")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", hatch=hatches[i], label=labels[i])
        for i in range(len(labels))
    ]
    legend_handles += [
        Line2D([0], [0], color="black", lw=1.5, label="Median"),
        Line2D([0], [0], marker="D", color="black", linestyle="None", markersize=4, label="Mean"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),  # (x, y) in axes coordinates
        frameon=False
    )

    plt.tight_layout()
    # plt.savefig(filename, bbox_inches="tight")
    plt.show()

# Yaw plot
ieee_boxplot(
    data=yaw_error,
    labels=["30°", "90°", "180°"],
    title="",
    ylabel="Error [°]",
    filename="yaw_error_boxplot_ieee.pdf",
    colors=["0.88", "0.72", "0.56"],
    hatches=["///", "...", "\\\\\\\\"]
)

# Distance plot
ieee_boxplot(
    data=dist_error,
    labels=["100 cm"],
    title="",
    ylabel="Error [cm]",
    filename="distance_error_boxplot_ieee.pdf",
    colors=["0.80"],
    hatches=["///"]
)

# X,Y offset plot
ieee_boxplot(
    data=[x_offset_error, y_offset_error],
    labels=["X", "Y"],
    title="",
    ylabel="Error [cm]",
    filename="square_error_boxplot_ieee.pdf",
    colors=["0.88", "0.56"],
    hatches=["///", "\\\\\\\\"]
)