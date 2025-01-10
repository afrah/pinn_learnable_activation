import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import ImageGrid


def plot_time_profile_regular_2D_cavity(dirname, steps, txy, exact, pred, part, N=30):

    [tstep, xstep, ystep] = steps
    [x, y, t] = txy

    minVal = min(exact.min(), pred.min())
    maxVal = max(exact.max(), pred.max())

    # print(minVal , maxVal)

    x = x.reshape(tstep, xstep, ystep)[0, 0, :]
    y = y.reshape(tstep, xstep, ystep)[0, :, 0]
    t = t.reshape(tstep, xstep, ystep)[:, 0, 0]

    # print(x , y , t)

    exact = exact.reshape(tstep, xstep, ystep)
    pred = pred.reshape(tstep, xstep, ystep)

    timeStp = [0, 10,20]
    yStep = [0, 10, 20]
    idx_x = np.random.choice(x.shape[0], N, replace=False)
    x0 = x[idx_x]
    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(top=0.85, bottom=0.25, left=0.05, right=0.95, wspace=0.45, hspace=0.9)

    text = ["Data", "Exact", "Precition"]
    color = ["rx", "r--", "b-"]
    for i in range(len(yStep)):
        for j in range(len(timeStp)):
            ax = plt.subplot(gs1[i, j])
            ax.plot(
                x,
                exact[:, yStep[i], :][timeStp[j], :],
                color="#A9A9A9",
                linewidth=3,
                alpha=0.5,
                label="Exact",
            )
            ax.plot(
                x,
                pred[:, yStep[i], :][timeStp[j], :],
                color="#007FFF",
                linestyle="--",
                dashes=(3, 3),  # Equal length of dashes and gaps
                dash_capstyle="butt",  # Optional: change the cap style of dashes
                linewidth=1,
                label="Prediction",
            )

            ax.set_ylabel(part)
            ax.set_title(
                "$t = %.2f$ , y =  %.2f" % (t[timeStp[j]], y[yStep[i]]), fontsize=10
            )
            ax.set_xlim([x.min() - 0.1, x.max() + 0.1])
            ax.set_ylim([minVal, maxVal])
            if i == 2:
                ax.set_xlabel("$x$")
                if j == 1:
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.7),
                        ncol=2,
                        frameon=False,
                    )

    # plt.tight_layout()
    plt.savefig(
        os.path.join(dirname, "time_profile_" + part + ".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        "all",
    )


def draw_contourf_regular_2D(
    tf,
    xf,
    yf,
    data,
    titles,
    nrows_ncols,
    values,
    xref,
    yref,
    model_dirname,
    img_width,
    img_height,
    ticks,
    fontsize,
    labelsize,
    axes_pad,
):

    yf = yref * xf
    xf = xref * yf
    # minmax = [xf.min(), xf.max(), yf.min(), yf.max()]

    for timeStp in values:
        file = os.path.join(model_dirname, "tricontourf_" + str(timeStp) + ".png")
        fig, grid, pcfsets, kwargs = grid_contour_plots_regular(
            data,
            nrows_ncols,
            titles,
            xf,
            yf,
            timeStp,
            file,
            img_width,
            img_height,
            ticks,
            fontsize=fontsize,
            labelsize=labelsize,
            axes_pad=axes_pad,
        )

###############################################################################################
def grid_contour_plots_regular(
    data,
    nrows_ncols,
    titles,
    x,
    y,
    time_step,
    dirname,
    img_width,
    img_height,
    ticks,
    fontsize,
    labelsize,
    axes_pad,
):

    # CREATE FIGURE AND AXIS
    fig = plt.figure()

    grid = ImageGrid(
        fig,
        111,
        direction="row",
        nrows_ncols=nrows_ncols,
        label_mode="1",
        axes_pad=axes_pad,
        share_all=False,
        cbar_mode="each",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=0.0,
    )

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []

    for d in data:
        min_ = np.min(d[time_step, :])
        max_ = np.max(d[time_step, :])
        if min_ == max_ == 0:
            min_ += -1e-16
            max_ += 1e-6
        minmax_list.append([min_, max_])
        kwargs_list.append(
            dict(
                levels=np.linspace(minmax_list[-1][0], minmax_list[-1][1], 60),
                cmap="coolwarm",
                vmin=minmax_list[-1][0],
                vmax=minmax_list[-1][1],
            )
        )

    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(
        grid, data, kwargs_list, minmax_list, titles
    ):
        pcf = [ax.contourf(x, y, z[time_step, :, :], **kwargs)]
        pcfsets.append(pcf)
        cb = ax.cax.colorbar(
            pcf[0], ticks=np.linspace(minmax[0], minmax[1], ticks), format="%.3e"
        )
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.set_ylabel("y", labelpad=labelsize, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize)
        # ax.set_xlim(x.min(), x.max())
        # ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")

    fig.set_size_inches(img_width, img_height, True)
    fig.subplots_adjust(
        left=1.3, bottom=0.1, right=1.5, top=0.12, wspace=None, hspace=None
    )
    plt.tight_layout()
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close(
        "all",
    )
    return fig, grid, pcfsets, kwargs_list



###################################################
def update_contourf_regular(i, xs, ys, data, axis, pcfsets, kwargs):

    list_of_collections = []

    for x, y, z, ax, pcfset, kw in zip(xs, ys, data, axis, pcfsets, kwargs):

        for tp in pcfset[0].collections:
            tp.remove()

        pcfset[0] = ax.contourf(x, y, z[i, :, :], **kw)
        list_of_collections += pcfset[0].collections

    return list_of_collections
