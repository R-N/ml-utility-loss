import numpy as np
import matplotlib.pyplot as plt
import os
from ..util import sorted_nicely
import pandas as pd


def plot_grad(loss, grad, fig=None, ax=None, name=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    series = np.array(list(zip(loss, grad)))
    series = series[series[:, 0].argsort()]
    ax.plot(series[:, 0], series[:, 1], **kwargs)
    if name:
        ax.legend([name])
    return fig

def plot_grad_2(y, models, loss="loss", grad="grad", g="g", **kwargs):
    fig, ax = plt.subplots()
    axes = []
    for m in models:
        yi = y[m]
        plot_grad(yi[loss], yi[grad], fig=fig, ax=ax, **kwargs)
        plot_grad(yi[loss], yi[g], fig=fig, ax=ax, **kwargs)
        axes.append(m)
    ax.legend(axes)
    return fig

def plot_synth_real_density(info_path, synth="synth", fig=None, ax=None, real=True, real_linestyle="solid", col="synth_value", real_col="real_value", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    df = pd.read_csv(info_path)
    df[col].plot.kde(alpha=0.5, ax=ax, linestyle="dashed", **kwargs)
    if real:
        df[real_col].plot.kde(alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs)
    axes = [synth]
    if real:
        axes.append("real")
    
    leg = ax.legend(axes)
    return fig

def plot_pred_density(pred, y, fig=None, ax=None, real_linestyle="solid", title=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    pd.Series(pred).plot.kde(alpha=0.5, ax=ax, linestyle="dashed", **kwargs)
    pd.Series(y).plot.kde(alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs)
    axes = ["pred", "y"]
    
    leg = ax.legend(axes)

    if title:
        ax.set_title(title)

    return fig

def plot_pred_density_2(results, **kwargs):
    for model, result in results.items():
        plot_pred_density(result["pred"], y=result["y"], title=model, **kwargs)

def plot_synths_density(info_dir, sizes=None, fig=None, ax=None, real=False, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    sizes = sizes or list(sorted_nicely(os.listdir(info_dir)))
    for size in sizes:
        info_dir_1 = os.path.join(info_dir, size)
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


def plot_pred_box(pred, y, fig=None, ax=None, title=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["pred"] = pred
    df["y"] = y

    cols = list(df.columns)
    df.boxplot(column=cols, **kwargs)
    ax.set_xticklabels(cols)

    if title:
        ax.set_title(title)

    return fig

def plot_pred_box_2(results, **kwargs):
    for model, result in results.items():
        plot_pred_box(result["pred"], y=result["y"], title=model, **kwargs)

def plot_synths_box(info_dir, sizes=None, fig=None, ax=None, col="synth_value", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
  
    sizes = sizes or list(sorted_nicely(os.listdir(info_dir)))
    df = pd.DataFrame()
    for size in sizes:
        info_dir_1 = os.path.join(info_dir, size)
        info_path = os.path.join(info_dir_1, "info.csv")
        
        dfi = pd.read_csv(info_path)
        s = pd.Series(dfi[col], name=size)
        df[size] = s
    df.boxplot(column=list(df.columns), **kwargs)
    ax.set_xticklabels(sizes)
    return fig
