# -*- coding: utf-8 -*-

import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors, cm, collections, colorbar
import numpy as np
from microcavities.utils import square, get_data_path
from microcavities.analysis.utils import normalize
import os
from collections import OrderedDict
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from shapely.geometry import MultiLineString
plt.style.use(os.path.join(os.path.dirname(__file__), 'default_style.mplstyle'))


# Utils
def default_save(figure, name, base_path=None):
    if base_path is None:
        base_path = os.path.dirname(get_data_path(None, False))
        assert os.path.exists(base_path)
    if not os.path.exists(os.path.join(base_path, 'figures')):
        os.makedirs(os.path.join(base_path, 'figures'))
    figure.savefig(os.path.join(base_path, 'figures', '%s.png' % name), dpi=1200, bbox_inches='tight')


def _make_axes(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    return fig, ax


def subplots(datas, plotting_func, axes=(0, ), subplots_shape=None, fig_shape=None, figsize=8,
             sharex=False, sharey=False, gridspec_loc=None, gridspec_kwargs=None, *args, **kwargs):
    """Utility function for plotting multiple datasets

    >>>> subplots(np.random.random((4, 4, 100))-0.5, plt.plot, (0, 1))
    >>>> subplots(np.random.random((4, 100, 200))-0.5, imshow, (0, ), scaling=1)
    >>>> subplots(np.random.random((4, 3, 10, 200))-0.5, imshow, (0, 1), scaling=(100, 1), aspect='auto')

    :param datas:
    :param plotting_func:
    :param axes:
    :param fig_shape:
    :param figsize: float or tuple
    :param sharex:
    :param sharey:
    :param gridspec_kwargs:
    :param args:
    :param kwargs:
    :return:
    """
    if len(axes) > 2:
        raise ValueError('Can only make subplots for 1 or 2 axes')

    # Getting the shape of the subplot grid
    if subplots_shape is None:
        if len(axes) == 1:
            n_images = datas.shape[axes[0]]
            a, b = square(n_images)
        elif len(axes) == 2:
            a, b = datas.shape[axes[0]], datas.shape[axes[1]]
    else:
        a, b = subplots_shape

    # Getting the size of the figure. If figsize is an iterable, use it. If it's a
    try:
        fig_size = tuple(iter(figsize))
    except TypeError:
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

    if gridspec_kwargs is None:
        gridspec_kwargs = dict()
    if gridspec_loc is None:
        fig = plt.figure(figsize=tuple(fig_size))
        gs = gridspec.GridSpec(b, a, **gridspec_kwargs)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(b, a, gridspec_loc, **gridspec_kwargs)
        fig = gs.figure
    axs = []
    for idx2 in range(b):
        for idx in range(a):
            if len(axes) == 1:
                indx = idx2 * a + idx
                data = np.take(datas, indx, axes[0])
            elif len(axes) == 2:
                _data = np.take(datas, idx, axes[0])
                if axes[1] > axes[0]:
                    data = np.take(_data, idx2, axes[1]-1)
                else:
                    data = np.take(_data, idx2, axes[1])
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


def unique_legend(ax, *args, **kwargs):
    """Removes repeated labels in a legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), *args, **kwargs)


def colour_axes(ax, colour):
    ax.tick_params(color=colour, labelcolor=colour, which='both')
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
    ax.xaxis.label.set_color(colour)
    ax.yaxis.label.set_color(colour)


# 1D plots
def waterfall(lines, ax=None, cmap=None, xaxis=None, offsets=None,
              labels=None, label_kwargs=None,
              xlabel=None, ylabel=None,
              peak_positions=None, peak_kwargs=None,
              **kwargs):
    if xaxis is not None and peak_positions is not None:
        # assert np.diff(xaxis)[0] > 0, 'To find the correct peak heights, the xaxis needs to be increasing'
        if np.diff(xaxis)[0] < 0:
            lines = lines[:, ::-1]
            xaxis = xaxis[::-1]
    fig, ax = _make_axes(ax)
    if offsets is None:
        offsets = 1.05 * np.abs(np.nanmin(np.diff(lines, axis=0))) * np.ones(len(lines))
    elif offsets == 'auto':
        offsets = 1.05 * np.abs(np.nanmin(np.diff(lines, axis=0), 1))
    else:
        try:
            len(offsets)
        except:
            offsets = np.ones(len(lines)) * offsets

    default_label_kwargs = dict(ha='right', va='bottom')
    if label_kwargs is None:
        label_kwargs = dict()
    for key, value in default_label_kwargs.items():
        if key not in label_kwargs:
            label_kwargs[key] = value
    if xaxis is None:
        xaxis = np.arange(lines.shape[1])
    if cmap is None:
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(lines))]
    else:
        try:
            colours = cm.get_cmap(cmap, len(lines) + 1)(range(len(lines)))
        except:
            colours = cmap

    if peak_positions is not None:
        default_peak_kwargs = dict(ls='-', marker='.', color='k')
        if peak_kwargs is None:
            peak_kwargs = dict()
        for key, value in default_peak_kwargs.items():
            if key not in peak_kwargs:
                peak_kwargs[key] = value
        peak_lines = []
        if len(np.unique([len(x) for x in peak_positions])) == 1:
            join_peaks = True
        else:
            join_peaks = False
            peak_kwargs['ls'] = "None"
    for idx, line in enumerate(lines):
        offset_line = line + np.sum(offsets[:idx])
        _kwargs = dict(kwargs)
        if 'color' not in _kwargs:
            _kwargs['color'] = colours[idx]
        ax.plot(xaxis, offset_line, **_kwargs)
        if peak_positions is not None:
            _peak_lines = []
            for peak in peak_positions[idx]:
                interpolation = np.interp(peak, xaxis, offset_line)
                _peak_lines += [(peak, interpolation)]
            peak_lines += [_peak_lines]
        if labels is not None:
            _label_kwargs = dict(label_kwargs)
            if 'color' not in _label_kwargs:
                _label_kwargs['color'] = colours[idx]
            ax.text(xaxis.max(), offset_line[-1], labels[idx], **_label_kwargs)
    if peak_positions is not None:
        if join_peaks:
            peak_lines = np.array(peak_lines)
            [ax.plot(*pkline, **peak_kwargs) for pkline in np.transpose(peak_lines, (1, 2, 0))]
        else:
            [ax.plot(*np.transpose(pkline), **peak_kwargs) for pkline in peak_lines]

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

    return fig, ax


def colorline(y, ax=None, z=None, xaxis=None, cmap='copper', vmin=None, vmax=None, xlabel=None, ylabel=None,
              cbar=True, cbar_kwargs=None, *args, **kwargs):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    fig, ax = _make_axes(ax)

    if xaxis is None:
        xaxis = np.arange(len(y))

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(xaxis))

    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    norm_colour = colors.Normalize(vmin, vmax)

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)
    segments = _make_segments(xaxis, y)
    lc = collections.LineCollection(segments, array=z, cmap=cmap, norm=norm_colour, *args, **kwargs)
    ax.add_collection(lc)

    ax.set_xlim(xaxis.min(), xaxis.max())
    ax.set_ylim(y.min(), y.max())

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if cbar:
        if cbar_kwargs is None:
            cbar_kwargs = dict()

        ax_divider = make_axes_locatable(ax)
        if 'orientation' in cbar_kwargs:
            if cbar_kwargs['orientation'] == 'horizontal':
                cax = ax_divider.append_axes("top", size="7%", pad="2%")
            else:
                cax = ax_divider.append_axes("right", size="7%", pad="2%")
        else:
            cax = ax_divider.append_axes("right", size="7%", pad="2%")

        fig.colorbar(cm.ScalarMappable(norm=norm_colour, cmap=cmap), cax=cax, **cbar_kwargs)
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')

    return fig, ax


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_fill(x, y_array, ax=None):
    """Plotting following seaborn.lineplot
    Given an array of lines, plots a single central average line and a shadowed region to show the standard deviation
    :param x:
    :param y_array:
    :param ax:
    :return:
    """
    fig, ax = _make_axes(ax)
    y_mean = np.nanmean(y_array, 0)
    y_err = np.nanstd(y_array, 0)
    ax.plot(x, y_mean)
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, alpha=0.3)
    return fig, ax


# 2D plots
def imshow(img, ax=None, diverging=True, scaling=None, xaxis=None, yaxis=None, cbar=True, cbar_kwargs=None,
           xlabel=None, ylabel=None, **kwargs):
    """Utility imshow, wraps commonly used plotting tools

    :param img: 2D array
    :param ax: pyplot.axes
    :param diverging: bool. Whether to use a diverging colormap, centered around 0
    :param scaling: 2-tuple or a float. The pixel to unit conversion value
    :param xaxis: 1D array
    :param yaxis: 1D array
    :param cbar: bool. Whether to add a colorbar
    :param cbar_kwargs: dict or None
    :param xlabel: str
    :param ylabel: str
    :param kwargs: dict. any other named arguments are passed to plt.imshow
    :return:
    """

    fig, ax = _make_axes(ax)

    if xaxis is None:
        xaxis = np.arange(-0.5, img.shape[1]+0.5, dtype=np.float)
    if yaxis is None:
        yaxis = np.arange(-0.5, img.shape[0]+0.5, dtype=np.float)
    assert len(xaxis) == img.shape[1]
    assert len(yaxis) == img.shape[0]
    if scaling is not None:
        try:
            xaxis *= scaling[0]
            yaxis *= scaling[1]
        except:
            xaxis *= scaling
            yaxis *= scaling
    kwargs['extent'] = [xaxis[0], xaxis[-1], yaxis[-1], yaxis[0]]

    if diverging:
        kwargs['cmap'] = 'RdBu'
        val = np.max(np.abs([np.max(img), np.min(img)]))
        if 'vmin' in kwargs and 'vmax' not in kwargs:
                kwargs['vmax'] = -kwargs['vmin']
        elif 'vmin' not in kwargs and 'vmax' in kwargs:
                kwargs['vmin'] = -kwargs['vmax']
        else:
            kwargs['vmin'] = -val
            kwargs['vmax'] = val
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'

    im = ax.imshow(img, **kwargs)
    if cbar:
        if cbar_kwargs is None:
            cbar_kwargs = dict()

        ax_divider = make_axes_locatable(ax)
        if 'orientation' in cbar_kwargs:
            if cbar_kwargs['orientation'] == 'horizontal':
                cax = ax_divider.append_axes("top", size="7%", pad="2%")
            else:
                cax = ax_divider.append_axes("right", size="7%", pad="2%")
        else:
            cax = ax_divider.append_axes("right", size="7%", pad="2%")

        fig.colorbar(im, cax=cax, **cbar_kwargs)
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

    return fig, ax


def colorful_imshow(images, ax=None, norm_args=(0, 100), from_black=True, cmap='hsv', *args, **kwargs):
    images = np.asarray(images)
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


def imshow_transparency(img, ax=None, alpha=None, percentiles=(0, 100), vmin=None, vmax=None,
                        diverging=True, cbar=False, cmap='coolwarm_r',
                        *args, **kwargs):
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
    kwargs['vmin'] = None
    kwargs['vmax'] = None
    fig, ax = imshow(img_array, ax, *args, **kwargs)

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
            else:
                raise ValueError
            img = np.array(img, dtype=np.float)
            if normalise:
                if normalise_kwargs is None:
                    normalise_kwargs = dict()
                img = normalize(img, **normalise_kwargs)
            combined_image[idx * stepx + idx:(idx + 1) * stepx + idx, idx2 * stepy + idx2:(idx2 + 1) * stepy + idx2] = img
    return imshow(combined_image, *args, **kwargs)


def pcolormesh(img, ax=None, x=None, y=None, cbar=True, cbar_label=None, diverging=True, xlabel=None, ylabel=None, *args, **kwargs):
    fig, ax = _make_axes(ax)
    if x is None:  x = np.arange(img.shape[0])
    if y is None:  y = np.arange(img.shape[1])

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
        cb = fig.colorbar(im, ax=ax, label=cbar_label)
        ax = (ax, cb.ax)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig, ax


def contour_intersections(images, contour_levels, ax=None, xs=None, ys=None, colours=None):
    if colours is None:
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(images))]

    if xs is None:
        xs = [0.1*np.arange(image.shape[1]) for image in images]
    if ys is None:
        ys = [np.arange(image.shape[0]) for image in images]
    fig, ax = _make_axes(ax)

    lines = []
    intersections = []
    for image, x, y, contour, colour in zip(images, xs, ys, contour_levels, colours):
        X, Y = np.meshgrid(x, y)
        contour = ax.contour(X, Y, image, contour, colors=[colour])
        line = MultiLineString(
            [path.interpolated(1).vertices for linecol in contour.collections for path in linecol.get_paths()])
        for prev_line in lines:
            points = line.intersection(prev_line).geoms
            for pnt in points:
                ax.plot(*pnt.xy, 'ko')
            intersections += [pnt.xy for pnt in points]
        lines += [line]
    return fig, ax, np.squeeze(intersections)


# Tests
def test_1D():
    x = np.linspace(-2*np.pi, 2*np.pi, 201)

    fig, axs = plt.subplots(1, 5, figsize=(8, 4))
    lines = np.array([np.sin(x + ph) for ph in np.linspace(-np.pi, np.pi, 10)])
    waterfall(lines, axs[0], xaxis=x, xlabel='Phase', ylabel='amplitude')
    waterfall(lines, axs[1], color='k', alpha=0.1, offsets=0.1)
    waterfall(lines, axs[2], cmap='jet', labels=range(10))
    waterfall(lines, axs[3], xaxis=x, peak_positions=np.transpose([np.linspace(1, -1, 10), np.linspace(3, 2, 10)]))
    waterfall(lines, axs[4], xaxis=x, peak_positions=np.transpose([np.linspace(1, -1, 10), np.linspace(3, 2, 10)]),
              peak_kwargs=dict(ls='--', color='r'))
    fig.tight_layout()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    x = np.linspace(-2*np.pi, 2*np.pi, 201)
    colorline(np.sin(x), axs[0], np.cos(x), xaxis=x, xlabel='Phase', ylabel='amplitude')
    colorline(np.sin(x), axs[1], 10*np.cos(2*x), xaxis=x, xlabel='Phase', ylabel='amplitude',
              cbar_kwargs=dict(orientation='horizontal', label='Anything'))
    fig.tight_layout()

def test_2D():
    _x = np.linspace(-np.pi, np.pi, 201)
    _y = np.linspace(-4*np.pi, 4*np.pi, 101)
    x, y = np.meshgrid(_x, _y)
    imshow(np.cos(x) * np.cos(y), xaxis=_x, yaxis=_y, xlabel='$x$', ylabel='$y$', cbar_kwargs=dict(label=r'$cos(x) \cdot cos(y)$'))

    contour_intersections([x**2 - y**2, x**2+y**2], [[2, 4, 6], [3, 5]])


if __name__ == '__main__':
    test_1D()
    test_2D()
    plt.show(block=True)
