import numpy as np
import matplotlib.pyplot as plt
import os
from ..util import sorted_nicely
import pandas as pd
from scipy.linalg import LinAlgError
from sklearn.linear_model import LinearRegression


def plot_grad(error, grad, fig=None, ax=None, name=None, sqrt=False, abs=False, xlabel="Error", ylabel="Gradient norm", group=None, scatter=False, s=1, alpha=1.0, alpha2=0.2, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    if group is not None:
        for g in sorted(list(set(group))):
            mask = group==g
            name_g = name
            if name_g:
                name_g = f"{name_g}_{g}"
            fig = plot_grad(
                error=error[mask],
                grad=grad[mask],
                fig=fig,
                ax=ax,
                name=name_g,
                sqrt=sqrt,
                abs=abs,
                xlabel=xlabel,
                ylabel=ylabel,
                **kwargs,
            )
        return fig
    if sqrt:
        error = np.sqrt(error)
    if abs:
        error = np.abs(error)
    series = np.array(list(zip(error, grad)))
    series = series[series[:, 0].argsort()]
    if scatter:
        ax.scatter(series[:, 0], series[:, 1], s=s, alpha=alpha, **kwargs)
        ax.plot(series[:, 0], series[:, 1], alpha=alpha2, **kwargs)
    else:
        ax.plot(series[:, 0], series[:, 1], alpha=alpha, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if name:
        ax.legend([name])
    return fig

def plot_grad_2(y, models, error="error", grad="grad", g="g", xlabel="Error", ylabel="Gradient norm", group=None, **kwargs):
    fig, ax = plt.subplots()
    axes = []
    for m in models:
        yi = y[m]
        yg = None
        if group is not None and group in yi:
            yg = yi[group]
        plot_grad(yi[error], yi[grad], fig=fig, ax=ax, group=yg, **kwargs)
        axes.append(f"{m}_{grad}")
        if g in yi:
            plot_grad(yi[error], yi[g], fig=fig, ax=ax, group=yg, **kwargs)
            axes.append(f"{m}_{g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(axes)
    return fig

def plot_grad_3(error, grad, fig=None, ax=None, name=None, g_name="g_linear", group=None, scatter=True, s=1, colors=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()

    main_kwargs = dict(kwargs)
    if scatter:
        main_kwargs["s"] = s

    if group is not None:
        axes_all = []
        colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = iter(colors)
        for g in sorted(list(set(group))):
            color = next(colors)
            mask = group==g
            name_g = name
            if name_g:
                name_g = f"{name_g}_{g}"
            g_name_g = g_name
            if g_name_g:
                g_name_g = f"{g_name_g}_{g}"
            fig, axes = plot_grad_3(
                error=error[mask],
                grad=grad[mask],
                fig=fig,
                ax=ax,
                name=None,
                g_name=None,
                scatter=scatter,
                color=color,
                **main_kwargs,
            )
            ax.plot([], [], '-o', color=color, label = name_g)
            #axes_all.extend(axes)
            #axes_all.append(name_g)
        if name:
            ax.legend()
        return fig

    plot_grad(error, grad, fig=fig, ax=ax, name=name, scatter=scatter, **main_kwargs)
    
    sign = np.sign(error)
    X = error[..., np.newaxis]
    y = sign * grad
    y1 = LinearRegression(positive=True, fit_intercept=False).fit(X, y).predict(X)
    y1 = sign * y1

    plot_grad(
        np.concatenate([error, [0]]), 
        np.concatenate([y1, [0]]), 
        fig=fig, ax=ax, name=g_name, 
        scatter=False,
        **kwargs
    )

    axes = []
    if name:
        if scatter:
            axes = [name, name, g_name]
        else:
            axes = [name, g_name]
        ax.legend(axes)
    return fig, axes

def plot_density(series, *args, xlabel="ML utility", ylabel="Density", group=None, ax=None, **kwargs):
    try:
        if group is not None:
            if isinstance(group, str):
                series = series.groupby(group)
                ax = series.plot.kde(*args, xlabel=xlabel, ylabel=ylabel, **kwargs)
            else:
                for g in sorted(list(set(group))):
                    mask = group==g
                    ax = series[mask].plot.kde(*args, xlabel=xlabel, ylabel=ylabel, **kwargs)
        else:
            ax = series.plot.kde(*args, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    except LinAlgError:
        pass

def plot_synth_real_density(info_path, synth="synth", fig=None, ax=None, real=True, real_linestyle="solid", col="synth_value", real_col="real_value", label="", limit=None, group=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()

    if isinstance(info_path, pd.DataFrame):
        df = info_path
    else:
        df = pd.read_csv(info_path)
    if limit:
        df = df[:limit]

    axes = []

    df_synth = df
    if group is not None:
        df_synth = df_synth.groupby(group)

    if plot_density(df_synth[col], alpha=0.5, ax=ax, linestyle="dashed", **kwargs):
        axes.append(synth if not label else (f"{label}_{synth}" if real else label))
    if real:
        if plot_density(df[real_col], alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
            axes.append(f"{label}_real" if label else "real")
    
    leg = ax.legend(axes)

    return fig

def plot_pred_density(pred, y, fig=None, ax=None, real_linestyle="solid", title=None, group=None, plot_y=True, pred_axis="pred", **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    axes = []
    if group is not None:
        for g in sorted(list(set(group))):
            mask = group==g
            title_g = title
            if title_g:
                title_g = f"{title_g}_{g}"
            if plot_density(pd.Series(pred[mask]), alpha=0.5, ax=ax, linestyle="dashed", **kwargs):
                axes.append(f"{pred_axis}_{g}")
    else:
        if plot_density(pd.Series(pred), alpha=0.5, ax=ax, linestyle="dashed", **kwargs):
            axes.append(pred_axis)
    if plot_y and plot_density(pd.Series(y), alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
        axes.append("y")
    leg = ax.legend(axes)

    if title:
        ax.set_title(title)

    return fig

def plot_pred_density_2(results, group=None, exclude=["run"], **kwargs):
    result_g = None
    if group:
        result_g = results[group]["pred"]
    for model, result in results.items():
        if model == group or model in exclude:
            continue
        plot_pred_density(result["pred"], y=result["y"], title=model, group=result_g, **kwargs)

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

def plot_box(df, columns=None, ax=None, ylabel="ML utility", xlabel="Dataset", group=None, exclude=["run"], colors=None, scatter=True, s=1, linewidth=2, meansize=8, fliersize=4, scatter_jitter=0.025, scatter_alpha=0.2, **kwargs):
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    if not columns:
        columns = [x for x in df.columns if x not in exclude]
    ax = df.boxplot(
        column=columns, 
        ax=ax, 
        ylabel=ylabel, 
        xlabel=xlabel, 
        showmeans=True,
        boxprops=dict(
            linewidth=linewidth, 
        ),
        medianprops=dict(
            linewidth=linewidth, 
            color="black",
        ),
        meanprops=dict(
            linewidth=linewidth, 
            marker='o',
            markersize=meansize,
            markeredgecolor='black',
        ),
        capprops=dict(
            linewidth=linewidth, 
        ),
        whiskerprops=dict(
            linewidth=linewidth, 
        ),
        flierprops=dict(
            linewidth=1, 
            marker='o',
            markersize=fliersize,
            markeredgecolor='black',
        ),
        **kwargs
    )

    if scatter:
        groups = [0]
        x_group = None
        mask = None
        if group is not None:
            if isinstance(group, str):
                if group in df.columns:
                    x_group = df[group]
                    groups = sorted(list(set(df[group])))
            else:
                x_group = group
            groups = sorted(list(set(x_group)))
        
        for i in range(len(columns)):
            for i_g, g in enumerate(groups):
                color = colors[i_g]
                y = df[columns[i]].dropna()
                # Add some random "jitter" to the x-axis
                if x_group is not None:
                    mask = x_group == g
                    y = y[mask]
                x = np.random.normal(i+1, scatter_jitter, size=len(y))
                ax.scatter(x, y, color=color, alpha=scatter_alpha, s=s)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(columns or list(df.columns))
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

def plot_box_3(values, y=None, y_name="target", group=None, exclude=["run"], **kwargs):
    values = dict(values)
    if y is not None:
        values[y_name] = y
    fig, ax = plt.subplots()
    df_box = pd.DataFrame()
    for k, v in values.items():
        if k in exclude:
            continue
        df_box[k] = v
    if group is not None:
        if isinstance(group, str):
            if group in values:
                df_box[group] = values[group]
        else:
            df_box["run"] = group
    plot_box(df_box, ax=ax, group=group, **kwargs)
    return fig

def plot_density_3(values, y=None, y_name="target", real_linestyle="solid", linestyle="dashed", legend_loc=None, legend_prop={}, group=None, exclude=["run"], **kwargs):
    fig, ax = plt.subplots()

    axes = []
    v_g = None
    if group is not None:
        v_g = values[group]
    for k, v in values.items():
        if k == group or k in exclude:
            continue
        if plot_density(pd.Series(v), alpha=0.5, ax=ax, linestyle=linestyle, group=v_g, **kwargs):
            axes.append(k)
        
    if y is not None and plot_density(pd.Series(y), alpha=0.5, ax=ax, linestyle=real_linestyle, **kwargs):
        axes.append(y_name)

    leg = ax.legend(axes, loc=legend_loc, prop=legend_prop)
    return fig
