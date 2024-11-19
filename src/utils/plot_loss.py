import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def exponential_moving_average(data, alpha=0.1):
    """Applies Exponential Moving Average (EMA) smoothing to the data."""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def smooth_loss(data, alpha=0.1, window_length=51, polyorder=3):
    """Applies a combination of EMA and Savitzky-Golay smoothing to the data."""
    ema_data = exponential_moving_average(data, alpha=alpha)
    if len(ema_data) < window_length:  # Avoid issues with short data
        window_length = len(ema_data) - 1 if len(ema_data) % 2 == 0 else len(ema_data)
    smoothed_data = savgol_filter(
        ema_data, window_length=window_length, polyorder=polyorder
    )
    return smoothed_data


def spline_smoothing(data, s=0.5):
    """Applies spline smoothing to the data."""
    x = np.arange(len(data))
    spline = UnivariateSpline(x, data, s=s)
    return spline(x)


def smoothed_min(data):
    # Calculate rolling minimum
    window_size = 1000
    min_values = np.minimum.accumulate(data)
    for i in range(window_size, len(data)):
        min_values[i] = min(data[i - window_size + 1 : i + 1])

    # Smooth the minimum values
    window_length = 51  # must be odd and less than data size
    poly_order = 3  # must be less than window_length
    smoothed_min = savgol_filter(min_values, window_length, poly_order)
    return smoothed_min


def plot_loss_history(data_list, save_path=None, y_max=None):
    """
    Usage:
    plot_loss_history(data_list, save_path=None, y_max=None)
    """

    # Set ggplot style
    plt.style.use("ggplot")

    fig, ax = plt.subplots()
    fig.set_size_inches([15, 6])

    for entry in data_list:
        data = entry["data"]
        color = entry["color"]
        name = entry["name"]
        alpha = entry["alpha"]
        window = entry["window"]
        show_avg = entry["show_avg"]
        show_lower = entry["show_lower"]

        # Apply smoothing and plot the smoothed data
        smooth_alpha = 0.1
        polyorder = 1
        smoothed_data = smooth_loss(data, smooth_alpha, window, polyorder)

        # Plot the original data
        sns.lineplot(
            x=np.arange(len(data)),
            y=smoothed_data,
            ax=ax,
            label=f"{name}",
            color=color,
            linestyle="-",
            alpha=alpha,
        )

        if show_avg:
            sns.lineplot(
                x=np.arange(len(smoothed_data)),
                y=smoothed_data,
                ax=ax,
                color=color,
                linewidth=0.5,
            )
        if show_lower:
            smoothed_lower = smoothed_min(data)
            sns.lineplot(
                x=np.arange(len(smoothed_lower)),
                y=smoothed_lower,
                ax=ax,
                color=color,
                linewidth=1.5,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Epochs →", fontsize=15, color="grey")
    ax.set_ylabel("Loss (log) →", fontsize=15, color="grey")
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

    # Draw the legend at the top, horizontally aligned
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_facecolor("white")  # Set legend box background to white

    plt.tight_layout()
    plt.savefig(save_path)
