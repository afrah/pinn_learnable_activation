import os
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt


def lp_error(pred, exact, text, logger, p):
    num = np.sum(np.abs(pred - exact) ** p)
    denum = np.sum(np.abs(exact) ** p)
    if denum == 0.0:
        denum = 1.0
        text = text + " (Absolute (denominator is zero))"
    result = ((num / denum) ** (1 / p)) * 100
    logger.print("%s  : %5.3e " % (text, result))
    return result


def plot_time_profile(dirname, x, time, u, uPINN, ylabel, time_step):
    """_summary_

    Args:
        dirname (_type_): _description_
        x (_type_): _description_
        time (_type_): _description_
        u (_type_): _description_
        uPINN (_type_): _description_
        ylabel (_type_): _description_
        time_step (_type_): _description_
    """

    N = 50

    idx_x = np.random.choice(x.shape[0], N, replace=False)
    x0 = x[idx_x]

    # Row 1: h(t,x) slices
    gs1 = gridspec.GridSpec(1, len(time_step))
    gs1.update(top=0.85, bottom=0.25, left=0.05, right=0.95, wspace=0.75, hspace=0.5)

    text = ["Data", "Exact", "Precition"]
    color = ["rx", "r--", "b-"]

    for j, _ in enumerate(time_step):
        ax = plt.subplot(gs1[0, j])
        ax.plot(x, u[time_step[j], :], "g.", markersize=4, label="Exact", alpha=0.2)

        u0 = u[time_step[j] : time_step[j] + 1, idx_x].T
        # ax.plot(x0, u0, "r.", markersize=4, label="Data" , alpha=0.2)
        ax.plot(
            x, uPINN[time_step[j], :], "b.", markersize=4, label="Precition", alpha=0.2
        )

        ax.set_ylabel(ylabel)
        ax.set_title("$t = %.2f$ " % (time[time_step[j]]), fontsize=10)
        ax.set_xlim([x.min(), x.max()])
        ax.set_xlabel("$x$")
        if j == 1:
            plt.legend(
                loc="upper center", bbox_to_anchor=(0.3, -0.1), ncol=3, frameon=False
            )

    plt.tight_layout()
    text = "time_profile" + str(ylabel) + ".png"
    plt.savefig(os.path.join(dirname, text), dpi=300, bbox_inches="tight")
    plt.close(
        "all",
    )
