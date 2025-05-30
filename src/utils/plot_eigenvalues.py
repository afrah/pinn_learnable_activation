import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.plot_loss import smooth_loss


def plot_ntk_eigenvalues(
    ax,
    data_list,
    row=0,
    col=None,
    y_max=None,
    label=False,
    smoothed_data=True,
    eigenvalue_threshold=1e-6,
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
        max_meaningful_idx = 0

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
        meaningful_indices = np.where(np.array(data) > eigenvalue_threshold)[0]
        if len(meaningful_indices) > 0:
            max_meaningful_idx = max(max_meaningful_idx, meaningful_indices[-1])

    ax.set_yscale("log")
    if row == 2 and col == 0:
        # ax.set_xlabel(r"# Eigenvalues →", fontsize=15, color="grey")
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
    cutoff_idx = int(max_meaningful_idx * 1.2)
    ax.set_xlim(right=cutoff_idx)
    # Stop the legend from being displayed
    ax.legend().set_visible(False)

    return ax
