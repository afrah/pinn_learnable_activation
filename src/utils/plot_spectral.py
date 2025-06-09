import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.plot_loss import smooth_loss

fontsize = 15


def plot_spectral(
    ax, data_list, index=1, y_max=None, label=False, smoothed_data=True, fontsize=15
):
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
        window = entry["window"] if "window" in entry else 2
        polyorder = entry["polyorder"] if "polyorder" in entry else 1
        show_avg = entry["show_avg"] if "show_avg" in entry else False
        show_lower = entry["show_lower"] if "show_lower" in entry else False
        # Apply smoothing and plot the smoothed data
        smooth_alpha = 0.1
        polyorder = 1

        if smoothed_data:
            data = smooth_loss(data, smooth_alpha, window, polyorder)
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
    if index == 0:
        ax.set_xlabel(r"Epochs (x$10^3$) →", fontsize=fontsize, color="grey")
        ax.set_ylabel(r"log($\lambda_{max}$) →", fontsize=fontsize, color="grey")
    ax.tick_params(axis="both", labelsize=fontsize, colors="grey")
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
