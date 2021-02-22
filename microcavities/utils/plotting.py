# -*- coding: utf-8 -*-

import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors, cm
import numpy as np
from microcavities.utils import square, get_data_path
from microcavities.analysis.utils import normalize
import os


def default_save(figure, name, base_path=None):
    if base_path is None:
        base_path = os.path.dirname(get_data_path(None, False))
        assert os.path.exists(base_path)
    if not os.path.exists(os.path.join(base_path, 'figures')):
        os.makedirs(os.path.join(base_path, 'figures'))
    figure.savefig(os.path.join(base_path, 'figures', '%s.png' % name), dpi=1200, bbox_inches='tight')


def imshow(img, ax=None, diverging=True, scaling=None, xaxis=None, yaxis=None, cbar=True, xlabel=None, ylabel=None,
           cbar_label=None, **kwargs):
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

    if xaxis is None:
        xaxis = np.arange(img.shape[1], dtype=np.float)
    if yaxis is None:
        yaxis = np.arange(img.shape[0], dtype=np.float)
    if scaling is not None:
        try:
            xaxis *= scaling[0]
            yaxis *= scaling[1]
        except:
            xaxis *= scaling
            yaxis *= scaling
    kwargs['extent'] = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

    if diverging:
        val = np.max(np.abs([np.max(img), np.min(img)]))
        kwargs['cmap'] = 'RdBu'
        kwargs['vmin'] = -val
        kwargs['vmax'] = val

    im = ax.imshow(img, **kwargs)
    if cbar: fig.colorbar(im, ax=ax, label=cbar_label)

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig, ax


def colorful_imshow(images, ax=None, norm_args=(0, 100), from_black=True, cmap='hsv', *args, **kwargs):
    normed = np.array([normalize(x, norm_args) for x in images])

    _cmap = cm.get_cmap(cmap, normed.shape[0] + 1)
    full = np.zeros(images.shape[1:] + (4,))
    for idx in range(normed.shape[0]):
        tst = np.tile(normed[idx], (4, 1, 1))
        plain_color = np.tile(_cmap(idx), images.shape[1:] + (1, ))
        full += plain_color * np.moveaxis(tst, 0, -1)

    if from_black:
        r = full[..., 0]
        g = full[..., 1]
        b = full[..., 2]
        full = np.moveaxis(np.array([r, g, b]), 0, -1)

    kwargs['diverging'] = False
    kwargs['cbar'] = False
    return imshow(full, ax, *args, **kwargs), _cmap(range(normed.shape[0]))


def rgb_imshow(red=None, green=None, blue=None, ax=None, norm_args=(0, 100), from_black=True, *args, **kwargs):
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


def imshow_transparency(img, alpha=None, percentiles=(0, 100), vmin=None, vmax=None,
                        diverging=True, cbar=False, cmap='coolwarm_r', ax=None,
                        *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    if diverging:
        if vmin is not None or vmax is not None:
            warnings.warn('Both diverging and vmin/vmax given. Defaulting to diverging.')
        val = np.max(np.abs([img.min(), img.max()]))
        norm_colour = colors.Normalize(-val, +val)
    else:
        norm_colour = colors.Normalize(vmin, vmax)

    img_array = plt.get_cmap(cmap)(norm_colour(img))

    if alpha is not None:
        norm_alpha = colors.Normalize(np.percentile(alpha, percentiles[0]), np.percentile(alpha, percentiles[1]))
        img_array[..., 3] = norm_alpha(alpha)

    kwargs['diverging'] = False
    kwargs['cbar'] = False
    kwargs['ax'] = ax
    kwargs['vmin'] = None
    kwargs['vmax'] = None
    imshow(img_array, *args, **kwargs)

    cbar_ax = None
    if cbar:
        _cbar = fig.colorbar(cm.ScalarMappable(norm=norm_colour, cmap=cmap), ax=ax)
        cbar_ax = _cbar.ax
    return fig, ax, cbar_ax


def combined_imshow(images, axes=(0, ), normalise=False, normalise_kwargs=None, *args, **kwargs):
    """    For making arrays of images, faster than making tons of subplots.
    Makes a large array with NaNs to separate different images, which can then be plotted in a single Matplotlib artist

    :param images:
    :param axes:
    :param normalise:
    :param normalise_kwargs:
    :param args:
    :param kwargs:
    :return:
    """
    shape = images.shape
    assert len(shape) - len(axes) == 2
    other_axis = set(range(len(shape))) - set(axes)
    if len(axes) == 1:
        a, b = square(shape[axes[0]])
    elif len(axes) == 2:
        a, b = [shape[idx] for idx in axes]
    else:
        raise NotImplementedError("Images shape %s and axes %s don't combine" % (shape, axes))

    stepx, stepy = [images.shape[x] for x in other_axis]
    combined_image = np.zeros((a * (stepx + 1), b * (stepy + 1))) + np.nan
    for idx in range(a):
        for idx2 in range(b):
            if len(axes) == 1:
                indx = idx * a + idx2
                img = images[indx]
            elif len(axes) == 2:
                img = images[idx, idx2]
            img = np.array(img, dtype=np.float)
            if normalise:
                if normalise_kwargs is None:
                    normalise_kwargs = dict()
                img = normalize(img, **normalise_kwargs)
            combined_image[idx * stepx + idx:(idx + 1) * stepx + idx, idx2 * stepy + idx2:(idx2 + 1) * stepy + idx2] = img
    return imshow(combined_image, *args, **kwargs)


def subplots(datas, plotting_func, axes=(0, ), fig_shape=None, figsize=8, sharex=False, sharey=False,
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


def label_grid(figure, grid, label, position, offset=0.07, **kwargs):
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
        figure.text(np.mean(_pos[2:]), _pos[0][-1]-offset, label, va='top', ha='center', **kwargs)
    elif position == 'top':
        figure.text(np.mean(_pos[2:]), _pos[1][0]+offset, label, va='bottom', ha='center', **kwargs)
    elif position == 'left':
        figure.text(_pos[2][0]-offset, np.mean(_pos[:2]), label, va='center', ha='right', rotation=90, **kwargs)
    elif position == 'right':
        figure.text(_pos[3][-1]+offset, np.mean(_pos[:2]), label, va='center', ha='left', rotation=-90, **kwargs)
