# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from microcavities.utils import square


def imshow(img, ax=None, diverging=True, scaling=None, cbar=True, xlabel=None, ylabel=None, **kwargs):
    """Utility imshow, wraps commonly used plotting tools

    :param img: 2D array
    :param ax: pyplot.axes
    :param diverging: whether to use a diverging colormap, centered around 0
    :param scaling: a 2-tuple or a float. The pixel to unit conversion value
    :param cbar: whether to add a colorbar
    :param xlabel: str
    :param ylabel: str
    :param kwargs: any other named arguments are passed to plt.imshow
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    if scaling is not None:
        xaxis = np.arange(img.shape[0], dtype=np.float)
        yaxis = np.arange(img.shape[1], dtype=np.float)
        xaxis -= np.mean(xaxis)
        yaxis -= np.mean(yaxis)
        try:
            xaxis *= scaling[0]
            yaxis *= scaling[1]
        except:
            xaxis *= scaling
            yaxis *= scaling
        extent = [yaxis.min(), yaxis.max(), xaxis.min(), xaxis.max()]
    else:
        extent = None

    if diverging:
        val = np.max(np.abs([np.max(img), np.min(img)]))
        cmap = 'RdBu'
        vmin = -val
        vmax = val
    else:
        vmin = None
        vmax = None
        cmap = None

    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, **kwargs)
    if cbar: fig.colorbar(im, ax=ax)

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig, ax


def subplots(datas, plotting_func, axes=(0, ), fig_shape=None, *args, **kwargs):
    """Utility function for plotting multiple datasets

    >>>> subplots(np.random.random((4, 4, 100))-0.5, plt.plot, (0, 1))
    >>>> subplots(np.random.random((4, 100, 200))-0.5, imshow, (0, ), scaling=1)
    >>>> subplots(np.random.random((4, 3, 10, 200))-0.5, imshow, (0, 1), scaling=(100, 1), aspect='auto')

    :param datas:
    :param plotting_func:
    :param axes:
    :param fig_shape: float
    :param args:
    :param kwargs:
    :return:
    """
    if len(axes) == 1:
        n_images = datas.shape[axes[0]]
        a, b = square(n_images)
    elif len(axes) == 2:
        a, b = datas.shape[axes[0]], datas.shape[axes[1]]
    else:
        raise ValueError

    if fig_shape is None:
        if len(datas.shape) - len(axes) == 2:
            fig_shape = (b/a) * (datas.shape[-2]/datas.shape[-1])
        else:
            fig_shape = 1

    fig = plt.figure(figsize=(8, 8*fig_shape))
    gs = gridspec.GridSpec(b, a)
    axs = []
    for idx in range(a):
        for idx2 in range(b):
            if len(axes) == 1:
                indx = idx * b + idx2
                data = datas[indx]
            elif len(axes) == 2:
                data = datas[idx, idx2]
            ax = plt.subplot(gs[idx2, idx])
            try:
                plotting_func(data, ax, *args, **kwargs)
            except:
                plotting_func(data, *args, **kwargs)
            axs += [ax]
    return fig, axs
