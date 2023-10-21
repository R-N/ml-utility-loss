import numpy as np
import matplotlib.pyplot as plt
import os
from ..util import sorted_nicely
import pandas as pd


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

def plot_synth_real_density(info_path, synth="synth", fig=None, ax=None, real=True, real_linestyle="solid", col="synth_value", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    df = pd.read_csv(info_path)
    df[col].plot.kde(alpha=0.5, ax=ax, linestyle="dashed", **kwargs)
    if real:
        df["real_value"].plot.kde(alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs)
    axes = [synth]
    if real:
        axes.append("real")
    
    leg = ax.legend(axes)
    return fig

def plot_synths_density(data_dir, sizes=None, fig=None, ax=None, real=False, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    sizes = sizes or list(sorted_nicely(os.listdir(data_dir)))
    for size in sizes:
        info_dir_1 = os.path.join(data_dir, size)
        info_path = os.path.join(info_dir_1, "info.csv")
        plot_synth_real_density(info_path, fig=fig, ax=ax, real=real, real_linestyle="dashed", synth=size, **kwargs)
    ax.legend(sizes)
    return fig

def plot_synth_real_box(info_path, synth="synth", fig=None, ax=None, real=True, col="synth_value", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    df = pd.read_csv(info_path)
    axes = [synth]
    cols = [col]
    if real:
        axes.append("real")
        cols.append("real_value")

    df.boxplot(column=cols, **kwargs)
    ax.set_xticklabels(axes)
    return fig

def plot_synths_box(info_path, sizes=None, fig=None, ax=None, col="synth_value", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
  
    sizes = sizes or list(sorted_nicely(os.listdir(info_dir)))
    df = pd.DataFrame()
    for size in sizes:
        info_dir_1 = os.path.join(aug_dir, dataset, size)
        info_path = os.path.join(info_dir_1, "info.csv")
        
        dfi = pd.read_csv(info_path)
        s = pd.Series(dfi[col], name=size)
        df[size] = s
    df.boxplot(column=list(df.columns), **kwargs)
    ax.set_xticklabels(sizes)
    return fig
