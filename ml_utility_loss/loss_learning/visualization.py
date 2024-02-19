import numpy as np
import matplotlib.pyplot as plt
import os
from ..util import sorted_nicely
import pandas as pd
from scipy.linalg import LinAlgError
from sklearn.linear_model import LinearRegression


def plot_grad(error, grad, fig=None, ax=None, name=None, sqrt=False, abs=False, xlabel="Error", ylabel="Gradient norm", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    if sqrt:
        error = np.sqrt(error)
    if abs:
        error = np.abs(error)
    series = np.array(list(zip(error, grad)))
    series = series[series[:, 0].argsort()]
    ax.plot(series[:, 0], series[:, 1], **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if name:
        ax.legend([name])
    return fig

def plot_grad_2(y, models, error="error", grad="grad", g="g", xlabel="Error", ylabel="Gradient norm", **kwargs):
    fig, ax = plt.subplots()
    axes = []
    for m in models:
        yi = y[m]
        plot_grad(yi[error], yi[grad], fig=fig, ax=ax, **kwargs)
        axes.append(f"{m}_{grad}")
        if g in yi:
            plot_grad(yi[error], yi[g], fig=fig, ax=ax, **kwargs)
            axes.append(f"{m}_{g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(axes)
    return fig

def plot_grad_3(error, grad, fig=None, ax=None, name=None, g_name="g_linear", **kwargs):
    if not ax:
        fig, ax = plt.subplots()

    plot_grad(error, grad, fig=fig, ax=ax, name=name, **kwargs)
    
    sign = np.sign(error)
    X = error[..., np.newaxis]
    y = sign * grad
    y1 = LinearRegression(positive=True, fit_intercept=False).fit(X, y).predict(X)
    y1 = sign * y1

    plot_grad(
        np.concatenate([error, [0]]), 
        np.concatenate([y1, [0]]), 
        fig=fig, ax=ax, name=g_name, **kwargs
    )

    if name:
        ax.legend([name, g_name])
    return fig

def plot_density(series, *args, xlabel="ML utility", ylabel="Density", **kwargs):
    try:
        ax = series.plot.kde(*args, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    except LinAlgError:
        pass

def plot_synth_real_density(info_path, synth="synth", fig=None, ax=None, real=True, real_linestyle="solid", col="synth_value", real_col="real_value", label="", limit=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(info_path, pd.DataFrame):
        df = info_path
    else:
        df = pd.read_csv(info_path)
    if limit:
        df = df[:limit]

    axes = []
    if plot_density(df[col], alpha=0.5, ax=ax, linestyle="dashed", **kwargs):
        axes.append(synth if not label else (f"{label}_{synth}" if real else label))
    if real:
        if plot_density(df[real_col], alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
            axes.append(f"{label}_real" if label else "real")
    
    leg = ax.legend(axes)

    return fig

def plot_pred_density(pred, y, fig=None, ax=None, real_linestyle="solid", title=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    axes = []
    if plot_density(pd.Series(pred), alpha=0.5, ax=ax, linestyle="dashed", **kwargs):
        axes.append("pred")
    if plot_density(pd.Series(y), alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
        axes.append("y")
    
    leg = ax.legend(axes)

    if title:
        ax.set_title(title)

    return fig

def plot_pred_density_2(results, **kwargs):
    for model, result in results.items():
        plot_pred_density(result["pred"], y=result["y"], title=model, **kwargs)

def plot_synths_density(info_dir, sizes=None, fig=None, ax=None, real=False, start_size=0, skip_last=True, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    sizes = sizes or list(sorted_nicely(os.listdir(info_dir)))
    sizes = sizes[start_size:]
    if skip_last:
        sizes = sizes[:-2] + sizes[-1:]
    for size in sizes:
        info_dir_1 = os.path.join(info_dir, size)
        info_path = os.path.join(info_dir_1, "info.csv")
        plot_synth_real_density(info_path, fig=fig, ax=ax, real=real, real_linestyle="dashed", label=size, **kwargs)
    ax.legend(sizes)
    return fig

def plot_box(df, column=None, ax=None, ylabel="ML utility", xlabel="Dataset", **kwargs):
    ax = df.boxplot(column=column, ax=ax, ylabel=ylabel, xlabel=xlabel, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(column or list(df.columns))
    return ax

def plot_synth_real_box(info_path, synth="synth", fig=None, ax=None, real=True, col="synth_value", real_col="real_value", label="", limit=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(info_path, pd.DataFrame):
        df = info_path
    else:
        df = pd.read_csv(info_path)
    if limit:
        df = df[:limit]

    axes = [synth if not label else (f"{label}_{synth}" if real else label)]
    cols = [col]
    if real:
        axes.append(f"{label}_real" if label else "real")
        cols.append(real_col)

    ax = plot_box(df, ax=ax, column=cols, **kwargs)
    ax.set_xticklabels(axes)
    return fig


def plot_pred_box(pred, y, fig=None, ax=None, title=None,  **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["pred"] = pred
    df["y"] = y

    cols = list(df.columns)
    ax = plot_box(df, ax=ax, column=cols, **kwargs)

    if title:
        ax.set_title(title)

    return fig

def plot_pred_box_2(results, **kwargs):
    for model, result in results.items():
        plot_pred_box(result["pred"], y=result["y"], title=model, **kwargs)

def plot_synths_box(info_dir, sizes=None, fig=None, ax=None, col="synth_value", limit=None, start_size=0, skip_last=True, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
  
    sizes = sizes or list(sorted_nicely(os.listdir(info_dir)))
    sizes = sizes[start_size:]
    if skip_last:
        sizes = sizes[:-2] + sizes[-1:]

    df = pd.DataFrame()
    for size in sizes:
        info_dir_1 = os.path.join(info_dir, size)
        info_path = os.path.join(info_dir_1, "info.csv")
        
        dfi = pd.read_csv(info_path)
        s = pd.Series(dfi[col], name=size)
        df[size] = s

    if limit:
        df = df[:limit]

    plot_box(df, ax=ax, column=list(df.columns), **kwargs)
    return fig

def plot_box_3(values, y=None, y_name="target", **kwargs):
    values = dict(values)
    if y is not None:
        values[y_name] = y
    fig, ax = plt.subplots()
    df_box = pd.DataFrame()
    for k, v in values.items():
        df_box[k] = v
    plot_box(df_box, ax=ax, **kwargs)
    return fig

def plot_density_3(values, y=None, y_name="target", real_linestyle="solid", linestyle="dashed", legend_loc=None, legend_prop={}, **kwargs):
    fig, ax = plt.subplots()

    axes = []
    for k, v in values.items():
        if plot_density(pd.Series(v), alpha=0.5, ax=ax, linestyle=linestyle, **kwargs):
            axes.append(k)
        
    if y is not None and plot_density(pd.Series(y), alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
        axes.append(y_name)

    leg = ax.legend(axes, loc=legend_loc, prop=legend_prop)
    return fig
