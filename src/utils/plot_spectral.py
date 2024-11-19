import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def plot_spectral(ax, data_list, y_max=None, label=False):
    """
    Usage:
    plot_loss_history(data_list, save_path=None, y_max=None)
    """
    plt.style.use("ggplot")

    for entry in data_list:
        data = entry["data"]
        color = entry["color"]
        name = entry["name"]
        alpha = entry["alpha"]

        # sns.scatterplot(
        #     x=np.arange(len(data)),
        #     y=data,
        #     ax=ax,
        #     label=f"{name}",
        #     color=color,
        #     alpha=0.5,
        #     s=20,  # Decrease the size of the dots
        # )
        sns.lineplot(
            x=np.arange(len(data)),
            y=data,
            ax=ax,
            color=color,
            alpha=alpha,
            linestyle="-",
            linewidth=2,
            label=f"{name}",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Epochs (x$10^3$) →", fontsize=15, color="grey")
    ax.set_ylabel(r"$\lambda$ (log) →", fontsize=15, color="grey")
    ax.tick_params(axis="both", labelsize=14, colors="grey")
    ax.spines["top"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.spines["left"].set_color("grey")
    ax.spines["right"].set_color("grey")

    # Set the background color to white and grid lines to very light grey
    ax.set_facecolor("white")
    ax.grid(True, color="lightgrey")

    if y_max is not None:
        ax.set_ylim(top=y_max)

    # Stop the legend from being displayed
    ax.legend().set_visible(False)

    return ax
