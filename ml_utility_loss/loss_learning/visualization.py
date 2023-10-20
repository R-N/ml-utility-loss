import numpy as np
import matplotlib.pyplot as plt


def plot_grad(loss, grad, fig=None, ax=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    series = np.array(list(zip(loss, grad)))
    series = series[series[:, 0].argsort()]
    ax.plot(series[:, 0], series[:, 1], **kwargs)
    return fig

def plot_grad_2(y, models, loss="loss", grad="grad", **kwargs):
    fig, ax = plt.subplots()
    axes = []
    for m in models:
        yi = y[m]
        plot_grad(yi[loss], yi[grad], fig=fig, ax=ax, **kwargs)
        axes.append(m)
    ax.legend(axes)
    return fig