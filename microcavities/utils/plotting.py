# -*- coding: utf-8 -*-
import itertools
import warnings, os, re, imageio
import numpy as np
from scipy.interpolate import interp1d
from microcavities.utils import square, get_data_path, normalize, run_once
from itertools import cycle
from skimage.segmentation import watershed

import matplotlib.pyplot as plt
from matplotlib import gridspec, colors, cm, collections, transforms
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as mpatches
from shapely.geometry import MultiLineString

plt.style.use(os.path.join(os.path.dirname(__file__), 'default_style.mplstyle'))
plt.rcParams['pdf.fonttype'] = 'truetype'
plt.rcParams['svg.fonttype'] = 'none'


# Colormaps
_cmap = LinearSegmentedColormap.from_list('BlueYellowRed', [(1, 1, 1),
                                                            'xkcd:royal blue', 'xkcd:canary yellow', 'xkcd:cherry red',
                                                            (0, 0, 0)])
run_once(lambda: plt.register_cmap(cmap=_cmap))()  # only register colormap once, useful when importing multiple times

cdict = {'red': [(0.0, 0.0, 1.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75, 1.0, 1.0), (1.0, 0.0, 0.0)],
         'green': [(0.0, 0.0, 1.0), (0.25, 0.4, 0.4), (0.5, 1.0, 1.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)],
         'blue': [(0.0, 1.0, 1.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)]}
_cmap = LinearSegmentedColormap('Michael', cdict, 256)
run_once(lambda: plt.register_cmap(cmap=_cmap))()  # only register colormap once, useful when importing multiple times

greek_alphabet = {
    'Alpha': u'\u0391', 'Beta': u'\u0392', 'Gamma': u'\u0393', 'Delta': u'\u0394', 'Epsilon': u'\u0395',
    'Zeta': u'\u0396', 'Eta': u'\u0397', 'Theta': u'\u0398', 'Iota': u'\u0399', 'Kappa': u'\u039A', 'Lamda': u'\u039B',
    'Mu': u'\u039C', 'Nu': u'\u039D', 'Xi': u'\u039E', 'Omicron': u'\u039F', 'Pi': u'\u03A0', 'Rho': u'\u03A1',
    'Sigma': u'\u03A3', 'Tau': u'\u03A4', 'Upsilon': u'\u03A5', 'Phi': u'\u03A6', 'Chi': u'\u03A7', 'Psi': u'\u03A8',
    'Omega': u'\u03A9',
    'alpha': u'\u03B1', 'beta': u'\u03B2', 'gamma': u'\u03B3', 'delta': u'\u03B4', 'epsilon': u'\u03B5',
    'zeta': u'\u03B6', 'eta': u'\u03B7', 'theta': u'\u03B8', 'iota': u'\u03B9', 'kappa': u'\u03BA', 'lamda': u'\u03BB',
    'mu': u'\u03BC', 'nu': u'\u03BD', 'xi': u'\u03BE', 'omicron': u'\u03BF', 'pi': u'\u03C0', 'rho': u'\u03C1',
    'sigma': u'\u03C3', 'tau': u'\u03C4', 'upsilon': u'\u03C5', 'phi': u'\u03C6', 'chi': u'\u03C7', 'psi': u'\u03C8',
    'omega': u'\u03C9',
}
unicodes = {'hbar': '\u0127', 'deg': '\u03B1'}


# Utils
def default_extension(path, default):
    name, extension = os.path.splitext(path)
    if extension == '':
        extension = default
    return ''.join([name, extension])


def figure(aspect_ratio=1.5, columns='double', margins=5, column_separation=5, *args, **kwargs):
    """Wrapper for plt.figure to make paper-ready figures

    :param aspect_ratio:
    :param columns:
    :param margins:
    :param column_separation:
    :param args:
    :param kwargs:
    :return:
    """

    if 'figsize' not in kwargs:
        a4_width = 210
        if columns == 'double':
            width = (a4_width - 2 * margins) * (1 / 25.4)  # in inches
        elif columns == 'single':
            width = ((a4_width - 2 * margins - column_separation) / 2) * (1 / 25.4)  # in inches
        else:
            raise ValueError('Unrecognised columns: %s' % columns)
        kwargs['figsize'] = (width, width/aspect_ratio)

    return plt.figure(*args, **kwargs)


def default_save(figure, name, base_path=None, dpi=1200, **kwargs):
    if base_path is None:
        base_path = os.path.dirname(get_data_path(None, False))
        assert os.path.exists(base_path)
    if not os.path.exists(os.path.join(base_path, 'figures')):
        os.makedirs(os.path.join(base_path, 'figures'))
    name = default_extension(name, '.png')
    figure.savefig(os.path.join(base_path, 'figures', name), dpi=dpi, bbox_inches='tight', **kwargs)


def create_axes(ax=None, subplots_shape=(1, 1), fig_kw=None):
    """Utility function to create/return a (fig,ax) tuple"""
    if ax in [False]:
        return None, None
    elif ax in [None, True]:
        if fig_kw is None: fig_kw = dict()
        defaults = dict(aspect_ratio=subplots_shape[1]/subplots_shape[0])
        fig_kw = {**defaults, **fig_kw}
        fig = figure(**fig_kw)
        ax = fig.subplots(*subplots_shape)
    elif isinstance(ax, tuple):
        fig, ax = ax
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    elif isinstance(ax, plt.Figure):
        fig = ax
        ax = fig.get_axes()
        if len(ax) == 1:
            ax = ax[0]
    else:
        raise ValueError('Cannot handle %s' % type(ax))
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
    else:
        raise ValueError()


def unique_legend(ax, sort=False, multi_artist=False, *args, **kwargs):
    """Removes repeated labels in a legend"""

    # Simple extension of maptlotlib.legend_handler.HandlerTuple to stack artists vertically instead of horizontally
    class HandlerTuple2(HandlerTuple):
        """
        Handler for Tuple.

        Additional kwargs are passed through to `HandlerBase`.

        Parameters
        ----------
        ndivide : int, default: 1
            The number of sections to divide the legend area into. If None,
            use the length of the input tuple.
        pad : float, default: :rc:`legend.borderpad`
            Padding in units of fraction of font size.
        direction : str, default: 'horizontal'
            Direction in which to stack artists
        """

        def __init__(self, ndivide=1, pad=None, direction='vertical', **kwargs):
            self._direction = direction
            super(HandlerTuple2, self).__init__(ndivide, pad, **kwargs)

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize,
                           trans):
            if self._direction == 'horizontal':
                return super(HandlerTuple2, self).create_artists(legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize,
                           trans)
            elif self._direction == 'vertical':
                # Literally just copied from the matplotlib code
                handler_map = legend.get_legend_handler_map()

                if self._ndivide is None:
                    ndivide = len(orig_handle)
                else:
                    ndivide = self._ndivide

                if self._pad is None:
                    pad = legend.borderpad * fontsize
                else:
                    pad = self._pad * fontsize

                if ndivide > 1:
                    height = (height - pad * (ndivide - 1)) / ndivide

                yds_cycle = cycle(ydescent - (height + pad) * np.arange(ndivide))

                a_list = []
                for handle1 in orig_handle:
                    handler = legend.get_legend_handler(handler_map, handle1)
                    _a_list = handler.create_artists(
                        legend, handle1,
                        xdescent, next(yds_cycle), width, height, fontsize, trans)
                    a_list.extend(_a_list)
                return a_list

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict()
    for label, handle in zip(labels, handles):
        if label not in by_label:
            by_label[label] = (handle, )
        else:
            by_label[label] += (handle, )
    for label in labels:
        if len(by_label[label]) == 1:
            by_label[label] = by_label[label][0]
    _values = np.array(list(by_label.values()))
    keys = np.array(list(by_label.keys()))
    if sort:
        if type(sort) == bool:
            regex_str = '-?[0-9]*.[0-9]*'
        else:
            regex_str = sort
        parsed_values = [float(re.match(regex_str, key).groups()[0]) for key in keys]
        indices = np.argsort(parsed_values)
        _values = _values[indices]
        keys = keys[indices]

    values = []
    for v in _values:
        try:
            values += [tuple(v)]
        except:
            values += [v]

    if multi_artist:
        kwargs = {**dict(handler_map={tuple: HandlerTuple2(ndivide=None)}), **kwargs}
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


def colorbar_wrapper(ax, mappable, cbar_kwargs=None, cax_kwargs=None):
    """Creates an axes_divider and places a colorbar in it

    :param ax:
    :param mappable:
    :param cbar_kwargs:
    :param cax_kwargs:
    :return:
    """
    fig, ax = create_axes(ax)
    if cbar_kwargs is None: cbar_kwargs = dict()
    cbar_kwargs = {**dict(orientation='vertical'), **cbar_kwargs}

    if cax_kwargs is None: cax_kwargs = dict()
    cax_kwargs = {**dict(size="7%", pad="2%"), **cax_kwargs}

    ax_divider = make_axes_locatable(ax)
    if cbar_kwargs['orientation'] == 'horizontal':
        cax = ax_divider.append_axes("top", **cax_kwargs)
    else:
        cax = ax_divider.append_axes("right", **cax_kwargs)

    fig.colorbar(mappable, cax=cax, **cbar_kwargs)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    return cax


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
    fig, ax = create_axes(ax)

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

    label_axes(ax, xlabel, ylabel)

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
    def _make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        return np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = create_axes(ax)

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

    label_axes(ax, xlabel, ylabel)

    if cbar:
        if cbar_kwargs is None: cbar_kwargs = dict()
        colorbar_wrapper(ax, cm.ScalarMappable(norm=norm_colour, cmap=cmap), cbar_kwargs=cbar_kwargs)

    return fig, ax


def plot_fill(y_array, ax=None, xaxis=None, *args, **kwargs):
    """Plotting following seaborn.lineplot
    Given an array of lines, plots a single central average line and a shadowed region to show the standard deviation
    :param y_array:
    :param ax:
    :param xaxis:
    :param args:
    :param kwargs:
    :return:
    """
    if xaxis is None: xaxis = np.arange(y_array.shape[1])
    fig, ax = create_axes(ax)
    y_mean = np.nanmean(y_array, 0)
    y_err = np.nanstd(y_array, 0)
    ax.plot(xaxis, y_mean, *args, **kwargs)
    kwargs['label'] = None  # ensures that if a legend is made, only the mean line is used
    if 'alpha' in kwargs:
        warnings.warn('You have given an alpha for both the line and the fill')
        kwargs.pop('alpha')
    ax.fill_between(xaxis, y_mean-y_err, y_mean+y_err, alpha=0.3, **kwargs)
    # todo: add a legend option that creates a square patch and a line
    return fig, ax


# 2D plots
def imshow(img, ax=None, diverging=True, scaling=None, xaxis=None, yaxis=None,
           cbar=True, cbar_kwargs=None, cax_kwargs=None,
           xlabel=None, ylabel=None, **kwargs):
    """Utility imshow, wraps commonly used plotting tools

    :param img: 2D array
    :param ax: pyplot.Axes object
    :param diverging: bool. Whether to use a diverging colormap, centered around 0
    :param scaling: 2-tuple or a float. The pixel to unit conversion value
    :param xaxis: 1D array. Corresponds to the second axis in img
    :param yaxis: 1D array. Corresponds to the first axis in img
    :param cbar: bool. Whether to add a colorbar
    :param cbar_kwargs: dict or None. To be passed to plt.colorbar
    :param cax_kwargs: dict or None. To be passed to ax_divider
    :param xlabel: str
    :param ylabel: str
    :param kwargs: dict. any other named arguments are passed to plt.imshow
    :return:
    """

    fig, ax = create_axes(ax)

    # Extent handling
    if 'extent' not in kwargs:
        # Define image axes if not provided
        if xaxis is None:
            xaxis = np.linspace(0, img.shape[1]-1, img.shape[1])
        if yaxis is None:
            yaxis = np.linspace(0, img.shape[0]-1, img.shape[0])
        assert len(xaxis) == img.shape[1]
        assert len(yaxis) == img.shape[0]

        # Scale and center image axes. Useful when we want to quickly quantify distances on the imshow
        if scaling is not None:
            try:
                xaxis *= scaling[0]
                yaxis *= scaling[1]
            except:
                xaxis *= scaling
                yaxis *= scaling
            xaxis -= np.mean(xaxis)
            yaxis -= np.mean(yaxis)

        # Create an "extent" to be passed to plt.imshow so that each value in xaxis/yaxis is at the center of each pixel
        xdiff = np.mean(np.diff(xaxis))
        ydiff = np.mean(np.diff(yaxis))
        kwargs['extent'] = [xaxis[0] - xdiff/2, xaxis[-1] + xdiff/2,
                            yaxis[-1] + ydiff/2, yaxis[0] - ydiff/2]

    # Colormap handling
    if diverging:
        kwargs = {**dict(cmap='RdBu'), **kwargs}  # Using a diverging colormap if not provided
        if 'vmin' not in kwargs and 'vmax' not in kwargs:
            val = np.nanmax(np.abs([np.nanmax(img), np.nanmin(img)]))
            kwargs = {**dict(vmin=-val, vmax=val), **kwargs}
        elif 'vmax' not in kwargs:
            kwargs['vmax'] = -kwargs['vmin']
        elif 'vmin' not in kwargs:
            kwargs['vmin'] = -kwargs['vmax']
        else:
            warnings.warn('Provided both vmin and vmax. You sure you want diverging=True?')

    # plt.imshow call
    kwargs = {**dict(aspect='auto'), **kwargs}
    im = ax.imshow(img, **kwargs)

    cax = None
    if cbar:
        cax = colorbar_wrapper(ax, im, cbar_kwargs=cbar_kwargs, cax_kwargs=cax_kwargs)

    label_axes(ax, xlabel, ylabel)

    return fig, ax, cax


def colorful_imshow(images, ax=None, norm_args=(0, 100), from_black=False, cmap='rainbow', labels=None,
                    legend_kwargs=None, **kwargs):
    """Qualitative 2D plot to show multiple spatially-separated features simultaneously

    Displays a list of images, each with a different colormap going from white/black to a saturated color

    :param images: list of 2D arrays
    :param ax: plt.Axes object
    :param norm_args: 2-tuple. Percentile values to normalise each image
    :param from_black: bool
    :param cmap: str
    :param labels: iterable of strings. If not None, creates a legend with these labels
    :param legend_kwargs: dict. to be passed to pyplot.Axes.legend
    :param kwargs: to be passed to microcavities.utils.plotting.imshow
    :return:
    """
    images = np.asarray(images)

    # Normalise each image, so the color saturation in the imshow is the same for each
    normed = np.array([normalize(x, norm_args) for x in images])

    # Creating the RGBA array
    _cmap = cm.get_cmap(cmap, images.shape[0])
    colour_list = _cmap(range(normed.shape[0]))
    rgb_array = np.zeros(images.shape[1:] + (4,))  # start with a black array
    for idx in range(normed.shape[0]):
        # Create an (N, M, 4) array that repeats the image 4 times. Acts like an intensity mask on the colour
        _image = np.moveaxis(np.tile(normed[idx], (4, 1, 1)), 0, -1)
        # Create an (N, M, 4) array that repeats the colormap RGBA value
        plain_color = np.tile(colour_list[idx], images.shape[1:] + (1, ))
        rgb_array += plain_color * _image
    if from_black:
        rgb_array = rgb_array[..., :3]  # removing the alpha channel

    # Imshow call
    kwargs['cbar'] = False  # colorbar doesn't make sens for this plot
    fig, ax, cax = imshow(rgb_array, ax, **kwargs)

    # Legend handling
    if labels is not None:
        patches = []
        for lbl, clr in zip(labels, colour_list):
            patches += [mpatches.Patch(color=clr, label=lbl)]
        if legend_kwargs is None: legend_kwargs = dict()
        ax.legend(handles=patches, **legend_kwargs)
    return (fig, ax, cax), colour_list


def imshow_transparency(img, ax=None, alpha=None, alpha_percentiles=(0, 100),
                        color_vmin=None, color_vmax=None, colour_diverging=True, cbar=False, cmap='hsv',
                        cbar_kwargs=None, cax_kwargs=None, **kwargs):
    """Qualitative 2D plot that masks an image with an alpha channel

    :param img: NxM array
    :param ax: plt.Axes object
    :param alpha: NxM array
    :param alpha_percentiles: 2-tuple
    :param color_vmin: float
    :param color_vmax: float
    :param colour_diverging: bool. If True, the colormap normalization will be centered at 0
    :param cbar: bool. Whether to add a colorbar
    :param cmap: str
    :param kwargs: to be passed to microcavities.utils.plotting.imshow
    :return:
    """
    # Creating a linear normalization of the colormap
    if colour_diverging:
        if color_vmin is not None and color_vmax is not None:
            warnings.warn('Both vmin/vmax given and diverging=True. Overriding and using diverging=False')
            norm_colour = colors.Normalize(-color_vmin, color_vmax)
        else:
            if color_vmin is not None:
                val = np.abs(color_vmin)
            elif color_vmax is not None:
                val = np.abs(color_vmax)
            else:
                val = np.max(np.abs([img.min(), img.max()]))
            norm_colour = colors.Normalize(-val, +val)
    else:
        norm_colour = colors.Normalize(color_vmin, color_vmax)

    # Create RGBA array
    img_array = plt.get_cmap(cmap)(norm_colour(img))

    # Mask the RGBA array with a normalized alpha channel
    if alpha is not None:
        img_array[..., 3] = normalize(alpha, alpha_percentiles, cut=True)

    # plt.imshow call
    kwargs['cbar'] = False  # colorbar doesn't make sens for this plot
    fig, ax, _ = imshow(img_array, ax, **kwargs)

    # Colorbar handling
    cax = None
    if cbar:
        cax = colorbar_wrapper(ax, cm.ScalarMappable(norm=norm_colour, cmap=cmap),
                               cbar_kwargs=cbar_kwargs, cax_kwargs=cax_kwargs)
    return fig, ax, cax


def combined_imshow(images, ax=None, axes=(0, ), normalise=False, normalise_kwargs=None, spacing=3, *args, **kwargs):
    """    For making arrays of images, faster than making tons of subplots.
    Stacks images into a larger array to be plotted by a single Matplotlib artist. Spaces in between images are

    :param images: 3 or 4 dimensional array
    :param ax: pyplot.Axes object
    :param axes: 1- or 2-tuple. Axes over which to iterate in stacking the images
    :param normalise: bool. Whether to normalise each image when stacking
    :param normalise_kwargs: dict or None
    :param spacing: int
    :param args: to be passed to microcavities.utils.plotting.imshow
    :param kwargs: to be passed to microcavities.utils.plotting.imshow
    :return:
    """
    images = np.array(images)
    shape = images.shape
    other_axes = set(range(len(shape))) - set(axes)  # axes that are not stacked
    assert len(other_axes) == 2, 'non-stacked axes need to be 2-dimensional'

    # Finding the square shape on which to stack images
    if len(axes) == 1:
        a, b = square(shape[axes[0]])
        images = np.moveaxis(images, axes[0], 0)
        images = np.reshape(images, (a, b) + images.shape[1:])
    elif len(axes) == 2:
        a, b = [shape[idx] for idx in axes]
        images = np.moveaxis(images, axes, (0, 1))
    else:
        raise ValueError("Can't stack over more than 2 axes")

    # Creating the stacked array
    stepx, stepy = [shape[x]+spacing for x in other_axes]  # the size of the images to be stacked
    combined_image = np.full((a * stepx - spacing, b * stepy - spacing), np.nan)  # create the full array of np.nan
    for idx in range(a):
        for idx2 in range(b):
            img = images[idx, idx2]
            img = np.array(img, dtype=np.float)
            if normalise:
                if normalise_kwargs is None: normalise_kwargs = dict()
                img = normalize(img, **normalise_kwargs)

            x_index = idx * stepx
            y_index = idx2 * stepy
            combined_image[x_index: x_index + (stepx-spacing), y_index: y_index + (stepy-spacing)] = img

    return imshow(combined_image, ax, *args, **kwargs)


def pcolormesh(img, ax=None, xaxis=None, yaxis=None, cbar=True, cbar_kwargs=None, cax_kwargs=None, diverging=True, xlabel=None, ylabel=None, **kwargs):
    """Alternative to imshow when the axes are not linear

    :param img: 2d array
    :param ax: pyplot.Axes object
    :param xaxis: 1D array. Corresponds to the second axis in img
    :param yaxis: 1D array. Corresponds to the first axis in img
    :param cbar: bool. Whether to add a colorbar
    :param cbar_kwargs: dict or None. To be passed to plt.colorbar
    :param cax_kwargs: dict or None. To be passed to ax_divider
    :param diverging: bool. Whether to use a diverging colormap, centered around 0
    :param xlabel: str
    :param ylabel: str
    :param kwargs: to be passed to pyplot.pcolormesh
    :return:
    """

    fig, ax = create_axes(ax)

    # Define image axes if not provided
    if xaxis is None:
        xaxis = np.linspace(0, img.shape[1] - 1, img.shape[1])
    if yaxis is None:
        yaxis = np.linspace(0, img.shape[0] - 1, img.shape[0])
    assert len(xaxis) == img.shape[1]
    assert len(yaxis) == img.shape[0]

    # Colormap handling
    if diverging:
        kwargs = {**dict(cmap='RdBu'), **kwargs}  # Using a diverging colormap if not provided
        if 'vmin' not in kwargs and 'vmax' not in kwargs:
            val = np.nanmax(np.abs([np.nanmax(img), np.nanmin(img)]))
            kwargs = {**dict(vmin=-val, vmax=val), **kwargs}
        elif 'vmax' not in kwargs:
            kwargs['vmax'] = -kwargs['vmin']
        elif 'vmin' not in kwargs:
            kwargs['vmin'] = -kwargs['vmax']
        else:
            warnings.warn('Provided both vmin and vmax. You sure you want diverging=True?')

    # Creating the edge of each pixel on the pcolormesh
    middle_edges_x = np.mean([xaxis[:-1], xaxis[1:]], 0)
    middle_edges_y = np.mean([yaxis[:-1], yaxis[1:]], 0)
    edges_x = [xaxis[0] - (middle_edges_x[0] - xaxis[0])] + list(middle_edges_x) + [xaxis[-1] + (xaxis[-1] - middle_edges_x[-1])]
    edges_y = [yaxis[0] - (middle_edges_y[0] - yaxis[0])] + list(middle_edges_y) + [yaxis[-1] + (yaxis[-1] - middle_edges_y[-1])]

    im = ax.pcolormesh(edges_x, edges_y, img, **kwargs)

    # Colorbar handling
    cax = None
    if cbar:
        cax = colorbar_wrapper(ax, im, cbar_kwargs=cbar_kwargs, cax_kwargs=cax_kwargs)

    label_axes(ax, xlabel, ylabel)

    return fig, ax, cax


def contour_intersections(images, contour_levels, ax=None, xs=None, ys=None, colours=None,
                          extrapolate=False, max_extrapolation_percent=0.1):
    """Plots image countours and uses them to calculate and marks the intersection between the contours

    # TODO: modify this function into an microcavities.analysis function to simply get the contour intersections

    :param images:
    :param contour_levels:
    :param ax:
    :param xs:
    :param ys:
    :param colours:
    :param extrapolate:
    :param max_extrapolation_percent:
    :return:
    """
    if colours is None:
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(images))]

    if xs is None:
        xs = [np.arange(image.shape[1]) for image in images]
    if ys is None:
        ys = [np.arange(image.shape[0]) for image in images]
    fig, ax = create_axes(ax)

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


def polygonal_image(points, z_scale=None, margins=(0.1, 0.1), min_points=(101, 101), plot_kwargs=None, *args, **kwargs):
    """Creates skimage.segmentation.watershed labelled image from a series of points

    Example:
    >>> x = [-1, 0, 1, 2]
    >>> y = [0, -1, 0, 1]
    >>> scales = np.array([0, -1, 1, 2])
    >>> points = np.array([(_x, _y) for _x, _y in zip(x, y)])
    >>> polygonal_image(points, scales)

    :param points: (Nx2) ndarray. (x, y) positions defining the position of each watershed-defined region
    :param z_scale: (N, ) ndarray. Values to evaluate the colormap of each region
    :param margins: 2-tuple of floats. Percentage of spatial range to add as margins on the edges
    :param min_points: 2-tuple of int. Minimum number of grid points to use in the boundary image
    :param plot_kwargs: dict or None. To be passed to pyplot.plot
    :param args: to be passed to imshow
    :param kwargs: to be passed to imshow
    :return:
    """
    if z_scale is None:
        z_scale = np.arange(len(points))
    normalized_z = -normalize(z_scale)  # Setting the zscale to go from -1 to 0 so that watershed segments correctly
    z_scale = np.append(z_scale, np.nan)  # Adding a np.nan for the boundary pixels

    # Edges of the spatial grid
    min_x, min_y = np.min(points, 0)
    max_x, max_y = np.max(points, 0)

    # Adding margins
    range_x = (max_x - min_x)
    range_y = (max_y - min_y)
    min_x -= range_x * margins[0]
    max_x += range_x * margins[0]
    min_y -= range_y * margins[1]
    max_y += range_y * margins[1]

    # Need enough points to separate any two points
    pairs = np.array(list(itertools.product(points, points)))  # pairs of points
    distances_x = np.abs(pairs[..., 0, 0]-pairs[..., 1, 0])  # x distance between every two points
    distances_y = np.abs(pairs[..., 0, 1]-pairs[..., 1, 1])  # y distance between every two points

    # Creating the grids, ensuring there's at least 10 pixels in between the nearest points
    dx = np.min(distances_x[np.nonzero(distances_x)]) * 1e-1
    dy = np.min(distances_y[np.nonzero(distances_y)]) * 1e-1
    nx = np.max([int((max_x - min_x) / dx), min_points[0]])
    ny = np.max([int((max_y - min_y) / dy), min_points[1]])
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)

    # Creating an image with single pixels at points set to values in zscale
    data = np.ones((len(x), len(y)))
    for z, point in zip(normalized_z, points):
        idx = np.argmin(np.abs(x - point[0]))
        idy = np.argmin(np.abs(y - point[1]))
        data[idx, idy] = z

    # Finding the boundaries using skimage.segmentation.watershed
    boundaries = watershed(data, watershed_line=True)  # nx by ny array with region indices
    scaled = z_scale[boundaries-1]  # nx by ny array with region z-values

    # Plotting
    fig, ax, cax = imshow(scaled.transpose(), xaxis=x, yaxis=y, *args, **kwargs)
    if plot_kwargs is not False:
        defaults = dict(color='k', marker='x', ls='none')
        if plot_kwargs is None:
            plot_kwargs = dict()
        plot_kwargs = {**defaults, **plot_kwargs}
        ax.plot(*points.transpose(), **plot_kwargs)
    return fig, ax, cax


# Tests
def test_1D():
    x = np.linspace(-2*np.pi, 2*np.pi, 201)

    fig = figure()
    axs = fig.subplots(1, 5)
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
    _x = np.linspace(-np.pi, np.pi, 21)
    _y = np.linspace(-4*np.pi, 4*np.pi, 11)
    x, y = np.meshgrid(_x, _y)
    imshow(np.cos(x) * np.cos(y), xaxis=_x, yaxis=_y, xlabel='$x$', ylabel='$y$', cbar_kwargs=dict(label=r'$cos(x) \cdot cos(y)$'))

    contour_intersections([x**2 - y**2, x**2+y**2], [[2, 4, 6], [3, 5]])

    colorful_imshow([np.exp(-(x+1)**2 - (y+1)**2), np.exp(-(x-1)**2 - (y-1)**2)], xaxis=_x, yaxis=_y, labels=['++', '--'])

    imshow_transparency(np.arctan2(y, x), None, np.exp(-(x/3)**2 - (y/3)**2), xaxis=_x, yaxis=_y, cbar=True)

    combined_imshow(np.moveaxis([np.random.rand()*np.exp(-(x+np.cos(theta))**2 - (y+np.sin(theta))**2) for theta in 2*np.pi*np.random.rand(9)], 0, 1), axes=(1, ), normalise=True)
    combined_imshow([[r*np.exp(-(x+np.cos(theta))**2 - (y+np.sin(theta))**2) for theta in 2*np.pi*np.random.rand(3)] for r in np.random.rand(3)], axes=(0, 1), normalise=True)

    fig, axs = plt.subplots(1, 2)
    img = np.random.random(x.shape)
    pcolormesh(img, axs[0], xaxis=_x, yaxis=_y)
    imshow(img, axs[1], xaxis=_x, yaxis=_y)

    pcolormesh(img, xaxis=np.exp(_x), yaxis=_y)


if __name__ == '__main__':
    test_1D()
    test_2D()
    plt.show(block=True)
