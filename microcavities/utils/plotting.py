# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from microcavities.utils import square
from microcavities.analysis.utils import normalize


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
        kwargs['extent'] = [yaxis.min(), yaxis.max(), xaxis.min(), xaxis.max()]

    if diverging:
        val = np.max(np.abs([np.max(img), np.min(img)]))
        kwargs['cmap'] = 'RdBu'
        kwargs['vmin'] = -val
        kwargs['vmax'] = val

    im = ax.imshow(img, **kwargs)
    if cbar: fig.colorbar(im, ax=ax)

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig, ax


def combined_imshow(red=None, green=None, blue=None, ax=None, norm_args=(0, 100), from_black=True, *args, **kwargs):
    """
    :param red:
    :param green:
    :param blue:
    :param ax:
    :param norm_args:
    :param from_black:
    :param args:
    :param kwargs:
    :return:
    """
    # Prepare the arrays to be within 0...1
    if red is None:
        if green is not None:
            red = np.zeros(green.shape)
        elif blue is not None:
            red = np.zeros(blue.shape)
        else:
            raise ValueError('All channels are None')
    else:
        red = normalize(red, norm_args, True)
    if green is None:
        green = np.zeros(red.shape)
    else:
        green = normalize(green, norm_args, True)
    if blue is None:
        blue = np.zeros(red.shape)
    else:
        blue = normalize(blue, norm_args, True)

    # Have the matplotlib imshow go from black to RGB or from white to RGB
    if from_black:
        img = np.rollaxis(np.array([red, green, blue]), 0, 3)
    else:
        img = np.rollaxis(np.array([1-green-blue, 1-red-blue, 1-red-green]), 0, 3)
    kwargs['diverging'] = False
    kwargs['cbar'] = False
    return imshow(img, ax, *args, **kwargs)


def subplots(datas, plotting_func, axes=(0, ), fig_shape=None, figsize=8,sharex=False, sharey=False,
             gridspec_kwargs=None, *args, **kwargs):
    """Utility function for plotting multiple datasets

    >>>> subplots(np.random.random((4, 4, 100))-0.5, plt.plot, (0, 1))
    >>>> subplots(np.random.random((4, 100, 200))-0.5, imshow, (0, ), scaling=1)
    >>>> subplots(np.random.random((4, 3, 10, 200))-0.5, imshow, (0, 1), scaling=(100, 1), aspect='auto')

    :param datas:
    :param plotting_func:
    :param axes:
    :param fig_shape:
    :param figsize:
    :param sharex:
    :param sharey:
    :param gridspec_kwargs:
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

    try:
        fig_size = tuple(iter(figsize))
    except:
        if fig_shape is None:
            if len(datas.shape) - len(axes) == 2:
                fig_shape = (b/a) * (datas.shape[-2]/datas.shape[-1])
            else:
                fig_shape = 1
        fig_size = np.array([figsize, figsize*fig_shape])
        if any(fig_size < 4):
            fig_size *= 4 / np.min(fig_size)
        if any(fig_size > 20):
            fig_size = np.array([figsize, figsize])

    fig = plt.figure(figsize=tuple(fig_size))
    if gridspec_kwargs is None:
        gridspec_kwargs = dict()
    gs = gridspec.GridSpec(b, a, **gridspec_kwargs)
    axs = []
    for idx2 in range(b):
        for idx in range(a):
            try:
                if len(axes) == 1:
                    indx = idx2 * a + idx
                    data = datas[indx]
                elif len(axes) == 2:
                    data = datas[idx, idx2]
            except IndexError:
                continue
            _kwargs = dict()
            if len(axs) > 0:
                if sharex:
                    _kwargs['sharex'] = axs[0]
                if sharey:
                    _kwargs['sharey'] = axs[0]

            ax = plt.subplot(gs[idx2, idx], **_kwargs)
            axs += [ax]

            try:
                plotting_func(data, ax, *args, **kwargs)
            except:
                plotting_func(data, *args, **kwargs)
    return fig, axs, gs


def waterfall(lines, ax=None, x_axis=None, offset=None, *args, **kwargs):
    if offset is None:
        offset = 1.05 * np.abs(np.min(np.diff(lines, axis=0)))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    if x_axis is None:
        x_axis = np.arange(lines.shape[1])
    [ax.plot(x_axis, x + offset * idx, *args, **kwargs) for idx, x in enumerate(lines)]
    return fig, ax


def pcolormesh(img, x, y, ax=None, cbar=True, diverging=True, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    if diverging:
        val = np.max(np.abs([np.max(img), np.min(img)]))
        vmin = -val
        vmax = val
        if 'vmin' not in kwargs:
            kwargs['vmin'] = vmin
        if 'vmax' not in kwargs:
            kwargs['vmax'] = vmax
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu'
    sort_indxs_x = np.argsort(x)
    sort_indxs_y = np.argsort(y)
    _sorted_img = img[sort_indxs_x]
    sorted_img = _sorted_img[:, sort_indxs_y]
    x = x[sort_indxs_x]
    y = y[sort_indxs_y]

    middle_edges_x = np.mean([x[:-1], x[1:]], 0)
    middle_edges_y = np.mean([y[:-1], y[1:]], 0)

    edges_x = [x[0] - (middle_edges_x[0] - x[0])] + list(middle_edges_x) + [x[-1] + (x[-1] - middle_edges_x[-1])]
    edges_y = [y[0] - (middle_edges_y[0] - y[0])] + list(middle_edges_y) + [y[-1] + (y[-1] - middle_edges_y[-1])]

    im = ax.pcolormesh(edges_x, edges_y, sorted_img.transpose(), *args, **kwargs)

    if cbar:
        cb = fig.colorbar(im, ax=ax)
        ax = (ax, cb.ax)

    return fig, ax


def label_grid(figure, grid, label, position, offset=0.07):
    """Simple labelling of matplotlib.gridspec grids

    :param figure: matplotlib.figure
    :param grid: matplotlib.gridspec
    :param label: string
    :param position: string
    :param offset: float
    :return:
    """
    _pos = grid.get_grid_positions(figure)
    if position == 'bottom':
        figure.text(np.mean(_pos[2:]), _pos[0][-1]-offset, label, va='top', ha='center')
    elif position == 'top':
        figure.text(np.mean(_pos[2:]), _pos[1][0]+offset, label, va='bottom', ha='center')
    elif position == 'left':
        figure.text(_pos[2][0]-offset, np.mean(_pos[:2]), label, va='center', ha='right', rotation=90)
    elif position == 'right':
        figure.text(_pos[3][-1]+offset, np.mean(_pos[:2]), label, va='center', ha='left', rotation=-90)
