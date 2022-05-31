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
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from matplotlib import transforms
import matplotlib.patches as mpatches
import re
import imageio
plt.style.use(os.path.join(os.path.dirname(__file__), 'default_style.mplstyle'))


def default_extension(path, default):
    name, extension = os.path.splitext(path)
    if extension == '':
        extension = default
    return ''.join([name, extension])


# Utils
def default_save(figure, name, base_path=None, dpi=1200):
    if base_path is None:
        base_path = os.path.dirname(get_data_path(None, False))
        assert os.path.exists(base_path)
    if not os.path.exists(os.path.join(base_path, 'figures')):
        os.makedirs(os.path.join(base_path, 'figures'))
    name = default_extension(name, '.png')
    figure.savefig(os.path.join(base_path, 'figures', name), dpi=dpi, bbox_inches='tight')


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
        gs = gridspec.GridSpec(b, a, figure=fig, **gridspec_kwargs)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(b, a, gridspec_loc, **gridspec_kwargs)
        fig = gs.figure
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    for idx2 in range(b):
        for idx in range(a):
            if len(axes) == 1:
                indx = idx2 * a + idx
                try:
                    data = np.take(datas, indx, axes[0])
                except IndexError:
                    continue
            elif len(axes) == 2:
                try:
                    _data = np.take(datas, idx, axes[0])
                    if axes[1] > axes[0]:
                        data = np.take(_data, idx2, axes[1]-1)
                    else:
                        data = np.take(_data, idx2, axes[1])
                except IndexError:
                    continue
            _kwargs = dict()

            if a == 1 and b == 1:
                ax = axs
            elif a == 1:
                ax = axs[idx2]
            elif b == 1:
                ax = axs[idx]
            else:
                ax = axs[idx2, idx]

            try:
                plotting_func(data, ax, *args, **kwargs)
            except:
                plotting_func(data, *args, **kwargs)
    return fig, axs, gs


def label_axes(ax, xlabel=None, ylabel=None, title=None, xlabel_kw=None, ylabel_kw=None, letter=None, letter_kw=None,
               letter_position=(-0.025, 1.025)):
    if xlabel_kw is None: xlabel_kw = dict()
    if ylabel_kw is None: ylabel_kw = dict()
    if letter_kw is None: letter_kw = dict()
    letter_kw = {**dict(weight='bold', fontsize=9, ha='right', va='bottom', transform=ax.transAxes), **letter_kw}

    if xlabel is not None:
        ax.set_xlabel(xlabel, **xlabel_kw)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **ylabel_kw)
    if title is not None:
        ax.set_title(title)

    if letter is not None:
        fig = ax.figure
        fig.text(letter_position[0], letter_position[1], letter, **letter_kw)


def label_grid(figure_grid, label, position, offset=0.07, **kwargs):
    """Simple labelling of matplotlib.gridspec grids

    :param figure_grid: matplotlib.figure or matplotlib.gridspec
    :param label: string
    :param position: string
    :param offset: float
    :return:
    """
    if isinstance(figure_grid, plt.Figure):
        assert len(figure_grid._gridspecs)
        grid = figure_grid._gridspecs[0]
        figure = figure_grid
    elif isinstance(figure_grid, gridspec.GridSpecBase):
        grid = figure_grid
        figure = figure_grid.figure
    else:
        raise ValueError('Unrecognised type for figure_grid: %s' % figure_grid)
    _pos = grid.get_grid_positions(figure)
    if position == 'bottom':
        figure.text(np.mean(_pos[2:]), _pos[0][-1]-offset, label, va='top', ha='center', **kwargs)
    elif position == 'top':
        figure.text(np.mean(_pos[2:]), _pos[1][0]+offset, label, va='bottom', ha='center', **kwargs)
    elif position == 'left':
        figure.text(_pos[2][0]-offset, np.mean(_pos[:2]), label, va='center', ha='right', rotation=90, **kwargs)
    elif position == 'right':
        figure.text(_pos[3][-1]+offset, np.mean(_pos[:2]), label, va='center', ha='left', rotation=-90, **kwargs)


def unique_legend(ax, sort=False, *args, **kwargs):
    """Removes repeated labels in a legend"""
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    values = np.array(list(by_label.values()))
    keys = np.array(list(by_label.keys()))
    if sort:
        if type(sort) == bool:
            regex_str = '-?[0-9]*.[0-9]*'
        else:
            regex_str = sort
        parsed_values = [float(re.match(regex_str, key).groups()[0]) for key in keys]
        indices = np.argsort(parsed_values)
        values = values[indices]
        keys = keys[indices]
    ax.legend(values, keys, *args, **kwargs)


def colour_axes(ax, colour, axis='both', which='both'):
    ax.tick_params(axis=axis, color=colour, labelcolor=colour, which='both')
    if axis == 'both':
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
        ax.xaxis.label.set_color(colour)
        ax.yaxis.label.set_color(colour)
    elif axis in ['y', 'left', 'right']:
        ax.yaxis.label.set_color(colour)
        if which == 'both':
            ax.spines['right'].set_edgecolor(colour)
            ax.spines['left'].set_edgecolor(colour)
        else:
            ax.spines[which].set_edgecolor(colour)
    elif axis in ['x', 'bottom', 'top']:
        ax.xaxis.label.set_color(colour)
        if which == 'both':
            ax.spines['top'].set_edgecolor(colour)
            ax.spines['bottom'].set_edgecolor(colour)
        else:
            ax.spines[which].set_edgecolor(colour)


def connect_axes(ax, ax2, ax2_ypos=None, ax2_xpos=None, offsets=(0.1, 0.2), arrow_props=None):
    if arrow_props is None:
        arrow_props = dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='black', connectionstyle="arc3")
    if ax2_ypos is not None:
        for _yaxis in ax.get_ylim():
            ax.annotate("",
                        xy=(ax.get_xlim()[0], _yaxis), xycoords='data',
                        xytext=(ax.get_xlim()[0] - offsets[0], np.mean(ax.get_ylim())), textcoords='data',
                        arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, color='black',
                                        connectionstyle="arc3"))
        ax.annotate("",
                    xytext=(ax.get_xlim()[0] - offsets[0], np.mean(ax.get_ylim())), textcoords='data',
                    xy=(ax2.get_xlim()[1] - offsets[1], ax2_ypos), xycoords=ax2.transData,
                    arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, color='black',
                                    connectionstyle="arc3"))
        ax.annotate("",
                    xytext=(ax2.get_xlim()[1] - offsets[1], ax2_ypos), textcoords=ax2.transData,
                    xy=(ax2.get_xlim()[1], ax2_ypos), xycoords=ax2.transData,
                    arrowprops=arrow_props)
    elif ax2_xpos is not None:
        for _xaxis in ax.get_xlim():
            ax.annotate("",
                        xy=(_xaxis, ax.get_ylim()[1]), xycoords='data',
                        xytext=(np.mean(ax.get_xlim()), ax.get_ylim()[1] + offsets[0]), textcoords='data',
                        arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, color='black',
                                        connectionstyle="arc3"))
        ax.annotate("",
                    xytext=(np.mean(ax.get_xlim()), ax.get_ylim()[1] + offsets[0]), textcoords='data',
                    xy=(ax2_xpos, ax2.get_ylim()[0] - offsets[1]), xycoords=ax2.transData,
                    arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, color='black',
                                    connectionstyle="arc3"))
        ax.annotate("",
                    xytext=(ax2_xpos, ax2.get_ylim()[0] - offsets[1]), textcoords=ax2.transData,
                    xy=(ax2_xpos, ax2.get_ylim()[0]), xycoords=ax2.transData,
                    arrowprops=arrow_props)
    else:
        raise ValueError('Need to provide ax2_ypos or ax2_xpos')


def my_annotate(ax, text, xy, xy_end, length=None, *args, **kwargs):
    xy = np.array(xy, dtype=float)
    xy_end = np.array(xy_end, dtype=float)
    unit_vector = xy_end - xy
    if length is None:
        length = np.sqrt(np.sum(unit_vector**2))
    unit_vector /= np.sqrt(np.sum(unit_vector**2))
    # print(length, unit_vector, xy, xy+length*unit_vector)
    ax.annotate(text, xy, xy-length*unit_vector, *args, **kwargs)
    ax.annotate(text, xy_end, xy_end+length*unit_vector, *args, **kwargs)


def make_gif(figures, gif_path, **kwargs):
    filenames = []
    for i, fig in enumerate(figures):
        # create file name and append it to a list
        filename = get_data_path(f'gif{i}.png')
        filenames.append(filename)
        # save frame
        default_save(fig, filename, os.path.dirname(gif_path))
    # build gif
    gif_path = default_extension(gif_path, '.gif')
    with imageio.get_writer(gif_path, mode='I', **kwargs) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in filenames:
            os.remove(get_data_path(filename))


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = text.get_transform() + transforms.Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + transforms.Affine2D().translate(0, ex.height)


# 1D plots
def waterfall(lines, ax=None, cmap=None, xaxis=None, offsets=None,
              labels=None, label_kwargs=None,
              xlabel=None, ylabel=None,
              peak_positions=None, peak_kwargs=None, join_peaks=True,
              **kwargs):
    """

    :param lines: 2D or 3D
    :param ax:
    :param cmap:
    :param xaxis:
    :param offsets:
    :param labels:
    :param label_kwargs:
    :param xlabel:
    :param ylabel:
    :param peak_positions:
    :param peak_kwargs:
    :param join_peaks:
    :param kwargs:
    :return:
    """
    fig, ax = _make_axes(ax)

    # Handling defaults
    if label_kwargs is None: label_kwargs = dict()
    default_label_kwargs = dict(ha='right', va='bottom')
    label_kwargs = {**default_label_kwargs, **label_kwargs}
    if peak_kwargs is None: peak_kwargs = dict()
    default_peak_kwargs = dict(ls='-', marker='.', color='k')
    peak_kwargs = {**default_peak_kwargs, **peak_kwargs}

    # Handling the shape of lines. If it is 2D, add an additional axis that will be handled by plot_fill
    lines = np.asarray(lines)
    if len(lines.shape) == 2:
        lines = lines[:, np.newaxis, :]

    # To find the correct peak heights, the xaxis needs to be increasing (a feature of np.interpolate)
    if xaxis is not None and peak_positions is not None:
        if np.diff(xaxis)[0] < 0:
            lines = lines[..., ::-1]
            xaxis = xaxis[::-1]

    # Finding offsets between lines
    if offsets is None:
        offsets = 1.05 * np.abs(np.nanmin(np.diff(lines, axis=0))) * np.ones(len(lines))
    elif offsets == 'auto':
        offsets = 1.05 * np.abs(np.nanmin(np.diff(lines, axis=0), 1))
    else:
        try:
            len(offsets)
        except:
            offsets = np.ones(len(lines)) * offsets

    if xaxis is None:
        xaxis = np.arange(lines.shape[-1])
    if cmap is None:
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(lines))]
    else:
        try:
            colours = cm.get_cmap(cmap, len(lines) + 1)(range(len(lines)))
        except:
            colours = cmap

    if peak_positions is not None:
        # Handling the joining of lines
        if join_peaks:
            peak_lines = []
            if len(np.unique([len(x) for x in peak_positions])) != 1:
                peak_kwargs['ls'] = "None"
                join_peaks = False
                warnings.warn('join_peaks == True but # of peaks per line is not the same')

    for idx, line in enumerate(lines):
        offset_line = line + np.sum(offsets[:idx])
        _kwargs = {**dict(color=colours[idx]), **kwargs}
        plot_fill(offset_line, ax, xaxis, **_kwargs)
        # ax.plot(xaxis, offset_line, **_kwargs)

        # Labelling the lines
        if labels is not None:
            _label_kwargs = {**dict(color=colours[idx]), **label_kwargs}
            ax.text(xaxis.max(), np.nanmean(offset_line[:, -1]), labels[idx], **_label_kwargs)

        # Making the peak lines:
        if peak_positions is not None:
            _peak_lines = []
            for peak in peak_positions[idx]:
                interpolation = np.interp(peak, xaxis, np.nanmean(offset_line, 0))
                _peak_lines += [(peak, interpolation)]
            if join_peaks:
                peak_lines += [_peak_lines]
            else:
                ax.plot(*np.transpose(_peak_lines), **peak_kwargs)
    if peak_positions is not None and join_peaks:
        [ax.plot(*pkline, **peak_kwargs) for pkline in np.transpose(peak_lines, (1, 2, 0))]

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


def plot_fill(y_array, ax=None, x=None, *args, **kwargs):
    """Plotting following seaborn.lineplot
    Given an array of lines, plots a single central average line and a shadowed region to show the standard deviation
    :param x:
    :param y_array:
    :param ax:
    :return:
    """
    if x is None: x = np.arange(y_array.shape[1])
    fig, ax = _make_axes(ax)
    y_mean = np.nanmean(y_array, 0)
    y_err = np.nanstd(y_array, 0)
    ax.plot(x, y_mean, *args, **kwargs)
    kwargs['label'] = None
    if 'alpha' in kwargs:
        warnings.warn('You have given an alpha for both the line and the fill')
        kwargs.pop('alpha')
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, alpha=0.3, **kwargs)
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
        # xaxis = np.linspace(-0.5, img.shape[1]-0.5, img.shape[1])
        xaxis = np.linspace(0, img.shape[1]-1, img.shape[1])
    # else:
    #     xdiff = np.mean(np.diff(xaxis))
    #     xaxis = np.linspace(xaxis.min() - xdiff/2, xaxis.max() + xdiff/2, len(xaxis))
    if yaxis is None:
        # yaxis = np.linspace(-0.5, img.shape[0]-0.5, img.shape[0])
        yaxis = np.linspace(0, img.shape[0]-1, img.shape[0])
    # else:
    #     ydiff = np.mean(np.diff(yaxis))
    #     yaxis = np.linspace(yaxis.min() - ydiff/2, yaxis.max() + yaxis/2, len(yaxis))

    assert len(xaxis) == img.shape[1]
    assert len(yaxis) == img.shape[0]
    if scaling is not None:
        try:
            xaxis *= scaling[0]
            yaxis *= scaling[1]
        except:
            xaxis *= scaling
            yaxis *= scaling
        xaxis -= np.mean(xaxis)
        yaxis -= np.mean(yaxis)

    xdiff = np.mean(np.diff(xaxis))
    ydiff = np.mean(np.diff(yaxis))
    kwargs['extent'] = [xaxis[0] - xdiff/2, xaxis[-1] + xdiff/2,
                        yaxis[-1] + ydiff/2, yaxis[0] - ydiff/2]

    if diverging:
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu'
        val = np.nanmax(np.abs([np.nanmax(img), np.nanmin(img)]))
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
    cax = None
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

    label_axes(ax, xlabel, ylabel)

    return fig, ax, cax


def colorful_imshow(images, ax=None, norm_args=(0, 100), from_black=True, cmap='hsv', labels=None,
                    legend_kwargs=None, *args, **kwargs):
    """Displays a list of images, each with a different colormap going from white/black to a saturated color

    :param images: list of 2D arrays
    :param ax:
    :param norm_args:
    :param from_black:
    :param cmap:
    :param labels:
    :param legend_kwargs:
    :param args:
    :param kwargs:
    :return:
    """
    images = np.asarray(images)
    normed = np.array([normalize(x, norm_args) for x in images])

    _cmap = cm.get_cmap(cmap, normed.shape[0] + 0.01)
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
    fig, ax = imshow(full, ax, *args, **kwargs)

    clrs = _cmap(range(normed.shape[0]))
    if labels is not None:
        patches = []
        for lbl, clr in zip(labels, clrs):
            patches += [mpatches.Patch(color=clr, label=lbl)]
        if legend_kwargs is None: legend_kwargs = dict()
        ax.legend(handles=patches, **legend_kwargs)
    return (fig, ax), clrs


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


def combined_imshow(images, ax=None, axes=(0, ), normalise=False, normalise_kwargs=None, *args, **kwargs):
    """    For making arrays of images, faster than making tons of subplots.
    Makes a large array with NaNs to separate different images, which can then be plotted in a single Matplotlib artist

    :param images:
    :param ax:
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
    return imshow(combined_image, ax, *args, **kwargs)


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


def contour_intersections(images, contour_levels, ax=None, xs=None, ys=None, colours=None,
                          extrapolate=False, max_extrapolation_percent=0.1):
    if colours is None:
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(images))]

    if xs is None:
        xs = [np.arange(image.shape[1]) for image in images]
    if ys is None:
        ys = [np.arange(image.shape[0]) for image in images]
    fig, ax = _make_axes(ax)

    lines = []
    intersections = []
    for image, x, y, contour, colour in zip(images, xs, ys, contour_levels, colours):
        X, Y = np.meshgrid(x, y)
        contour = ax.contour(X, Y, image, contour, colors=[colour])
        path_points = [path.interpolated(1).vertices for linecol in contour.collections for path in linecol.get_paths()]
        if extrapolate:
            xranges = [np.max(pp[:, 0])-np.min(pp[:, 0]) for pp in path_points]
            funcs = [interp1d(pp[:, 0], pp[:, 1], fill_value="extrapolate") for pp in path_points]
            new_x = [np.linspace(np.min(pp[:, 0])-xrange*max_extrapolation_percent,
                                 np.max(pp[:, 0])+xrange*max_extrapolation_percent) for pp, xrange in zip(path_points, xranges)]
            path_points = [np.transpose([xrange, func(xrange)]) for xrange, func in zip(new_x, funcs)]
        line = MultiLineString(path_points)
        for prev_line in lines:
            points = line.intersection(prev_line)
            try:
                for pnt in points.geoms:
                    ax.plot(*pnt.xy, 'ko')
                intersections += [pnt.xy for pnt in points]
            except:
                try:
                    ax.plot(*points.xy, 'ko')
                    intersections += [points.xy]
                except:
                    pass
        lines += [line]
    return fig, ax, np.squeeze(intersections), lines


# Tests
def test_1D():
    x = np.linspace(-2*np.pi, 2*np.pi, 201)

    fig, axs = plt.subplots(1, 5, figsize=(8, 4))
    lines = np.array([np.sin(x + ph) for ph in np.linspace(-np.pi, np.pi, 10)])
    waterfall(lines, axs[0], xaxis=x, xlabel='Phase', ylabel='amplitude')
    waterfall(lines, axs[1], color='k', alpha=0.1, offsets=0.1)
    waterfall(lines, axs[2], cmap='jet', labels=range(10))
    waterfall(lines, axs[3], xaxis=x, peak_positions=[np.linspace(-6, 6, np.random.randint(15)) for _ in range(10)])
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

    colorful_imshow([np.exp(-(x+1)**2 - (y+1)**2), np.exp(-(x-1)**2 - (y-1)**2)], xaxis=_x, yaxis=_y, labels=['++', '--'])


if __name__ == '__main__':
    test_1D()
    test_2D()
    plt.show(block=True)
