# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from microcavities.utils.functools import lorentzianNd, gaussianNd
import collections


def fit_function(name):
    """
    Given a name, return a lmfit Model with a guess method implemented
    :param name:
    :return:
    """


def centroids(array, axis=-1):
    """Reduce an array along an axis, giving the index centroid along that axis"""
    axes_array = [np.arange(x) for x in array.shape]
    mesh = np.meshgrid(*axes_array, indexing='ij')
    img = mesh[axis]
    return np.ma.average(img, axis, array)


def remove_spikes(array, thresh=5, smooth=30, max_iterations=10):
    # Modification of the nplab version, to use on
    cleaned = np.copy(array)  # prevent modification in place

    for i in range(max_iterations):
        noise_spectrum = cleaned / gaussian_filter(cleaned, smooth)
        # ^ should be a flat, noisy array, with a large spike where there's a cosmic ray.
        noise_level = np.sqrt(np.var(noise_spectrum))
        # average deviation of a datapoint from the mean
        mean_noise = noise_spectrum.mean()  # should be == 1
        spikes = np.nonzero(noise_spectrum > mean_noise + (thresh * noise_level))
        cleaned[spikes] = gaussian_filter(cleaned, smooth)[spikes]
        if len(spikes[0]) == 0:
            break
    return cleaned


def remove_outliers(x, axis=None, bar=1.5, side='both', method='IQR', return_mask=False):
    """Uses the IQR or zscore method
    From https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

    :param x: array
    :param axis: axis to remove along
    :param bar:
    :param side:
    :param method:
    :return:
    """
    assert side in ['gt', 'lt', 'both'], 'Side should be `gt`, `lt` or `both`.'
    assert method in ['IQR', 'zscore'], 'Method should be `IQR` or `zscore`.'
    xcopy = np.copy(x)
    if method == 'IQR':
        d_iqr = np.nanpercentile(xcopy, 75, axis=axis) - np.nanpercentile(xcopy, 25, axis=axis)
        d_q1 = np.nanpercentile(xcopy, 25, axis=axis)
        d_q3 = np.nanpercentile(xcopy, 75, axis=axis)
        iqr_distance = np.multiply(d_iqr, bar)
        upper_range = d_q3 + iqr_distance
        lower_range = d_q1 - iqr_distance
    elif method == 'zscore':
        xcopy = zscore(xcopy, axis=axis)
        upper_range = bar
        lower_range = -bar
    stat_shape = list(xcopy.shape)
    if axis is None:
        axis = list(range(len(stat_shape)))
    if isinstance(axis, collections.abc.Iterable):
        for single_axis in axis:
            stat_shape[single_axis] = 1
    else:
        stat_shape[axis] = 1

    if side in ['gt', 'both']:
        upper_outlier = np.greater(xcopy - upper_range.reshape(stat_shape), 0)
    if side in ['lt', 'both']:
        lower_outlier = np.less(xcopy - lower_range.reshape(stat_shape), 0)

    if side == 'gt':
        mask = upper_outlier
    elif side == 'lt':
        mask = lower_outlier
    elif side == 'both':
        mask = np.logical_or(upper_outlier, lower_outlier)
    x[mask] = np.nan
    if return_mask:
        return x, mask
    else:
        return x


def around_step(value, step):
    """Round a float to the nearest step
    :param value:
    :param step:
    :return:
    """
    up_down = (value % step // (step/2))
    if up_down:
        offset = -value % step
    else:
        offset = -(value % step)
    return value + offset


def stitch_datasets(x_sets, y_sets, interpolation='even'):
    """Stitching

    TODO: allow for more complicated stitching than just averaging: previous, next, or a smooth transition from one to the next

    :param x_sets: iterable of iterables
    :param y_sets: iterable of iterables
    :param interpolation: str. Currently just a way of selecting x values
        'even'  -  selects a range of x-values that is evenly spaced between the minimum and the maximum x values
        'same'  -  selects an range of x-values that follows x_sets but removes the overlaps.
    :return:
    """

    # Choosing an array of x-values
    if interpolation == 'even':
        x_step = [np.diff(x)[0] for x in x_sets]
        x_new = np.arange(np.min(np.concatenate(x_sets)),
                          np.max(np.concatenate(x_sets))+np.min(x_step)/2,
                          np.min(x_step))
    elif interpolation == 'same':
        x_new = []
        for idx, x in enumerate(x_sets):
            if idx == 0:
                next_x = x_sets[idx + 1]
                if np.max(x) > np.min(next_x):
                    max_index = np.min(np.argwhere(x > np.min(next_x))) - 1
                    next_index = np.max(np.argwhere(next_x <= np.max(x))) + 1
                    arrays = [x[max_index:], next_x[:next_index]]
                    joined = np.concatenate(arrays)
                    try:
                        min_step = np.min([np.diff(arr) for arr in arrays])
                        rounded = around_step(joined, min_step)
                    except ValueError:
                        rounded = joined
                    x_new += [np.concatenate([x[:max_index], np.unique(rounded)])]
                else:
                    x_new += [x]
            elif idx == len(x_sets) - 1:
                prev_x = x_sets[idx - 1]
                if np.min(x) < np.max(prev_x):
                    min_index = np.max(np.argwhere(x <= np.max(prev_x))) + 1
                    x_new += [x[min_index:]]
                else:
                    x_new += [x]
            else:
                next_x = x_sets[idx + 1]
                prev_x = x_sets[idx - 1]
                if np.max(x) > np.min(next_x) and np.min(x) < np.max(prev_x):
                    min_index = np.max(np.argwhere(x <= np.max(prev_x))) + 1
                    max_index = np.min(np.argwhere(x > np.min(next_x))) - 1
                    next_index = np.max(np.argwhere(next_x <= np.max(x))) + 1
                    arrays = [x[max_index:], next_x[:next_index]]
                    joined = np.concatenate(arrays)
                    try:
                        min_step = np.min([np.diff(arr) for arr in arrays])
                        rounded = around_step(joined, min_step)
                    except ValueError:
                        rounded = joined
                    x_new += [np.concatenate([x[min_index:max_index], np.unique(rounded)])]
                elif np.max(x) > np.min(next_x):
                    max_index = np.min(np.argwhere(x > np.min(next_x))) - 1
                    next_index = np.max(np.argwhere(next_x <= np.max(x))) + 1
                    arrays = [x[max_index:], next_x[:next_index]]
                    joined = np.concatenate(arrays)
                    try:
                        min_step = np.min([np.diff(arr) for arr in arrays])
                        rounded = np.round(joined, min_step)
                    except ValueError:
                        rounded = joined
                    x_new += [np.concatenate([x[:max_index], np.unique(rounded)])]
                elif np.min(x) < np.max(prev_x):
                    min_index = np.max(np.argwhere(x <= np.max(prev_x))) + 1
                    x_new += [x[min_index:]]
                else:
                    x_new += [x]
        x_new = np.concatenate(x_new)
    else:
        raise ValueError("Unrecognised interpolation %s. Needs to be one of: 'even', 'same'" % interpolation)


    interpolated_datasets = [interp1d(x, y, bounds_error=False) for x, y in zip(x_sets, y_sets)]
    y_new = np.nanmean([f(x_new) for f in interpolated_datasets], 0)
    return x_new, y_new


# Tests
def test_remove_spikes():
    _x, _y = [np.linspace(-10, 10, idx) for idx in [100, 200]]
    x, y = np.meshgrid(_x, _y)
    base_img = np.random.random(x.shape)
    base_img += 10*np.exp(-(x - 2) ** 2 / (2 * 3 ** 2) - (y - 2) ** 2 / (2 * 4 ** 2))
    base_img += 6*np.exp(-(x + 2) ** 2 / (2 * 6 ** 2) - (y + 3) ** 2 / (2 * 10 ** 2))

    nois_img = np.copy(base_img)
    # for x0, a in zip(np.random.random(10) * 100 + 750, np.random.random(10) * 100 + 2):
    #     nois_img += lorentzianNd((x, y), x0, a, 0.1)
    nois_img += lorentzianNd((_x, _y), 2, 1, 0.1)
    nois_img += 100*np.exp(-(x) ** 2 / (2 * 0.1 ** 2) - (y) ** 2 / (2 * 0.1 ** 2))
    nois_img += 100*np.exp(-(x-5) ** 2 / (2 * 0.1 ** 2) - (y+5) ** 2 / (2 * 0.1 ** 2))

    _x = np.linspace(750, 850, 1340)
    base_spectra = np.random.random(_x.shape)
    base_spectra += gaussianNd(_x, 780, 10, 3)
    base_spectra += lorentzianNd(_x, 800, 10, 3)
    base_spectra += lorentzianNd(_x, 830, 10, 10)
    noisy_spectra = np.copy(base_spectra)
    for x0, a in zip(np.random.random(10) * 100 + 750, np.random.random(10) * 20 + 2):
        noisy_spectra += lorentzianNd(_x, x0, a, 0.1)
    fig, axs = plt.subplots(2, 6)
    axs[0, 0].imshow(base_img)
    axs[0, 1].imshow(nois_img)
    axs[0, 2].imshow(remove_spikes(nois_img))
    axs[0, 3].imshow(remove_spikes(nois_img, 2))
    axs[0, 4].imshow(remove_spikes(nois_img, smooth=5))
    axs[0, 5].imshow(remove_spikes(nois_img, 1, 5, 100))
    axs[1, 0].plot(base_spectra)
    axs[1, 1].plot(noisy_spectra)
    axs[1, 2].plot(remove_spikes(noisy_spectra))
    axs[1, 3].plot(remove_spikes(noisy_spectra, 2))
    axs[1, 4].plot(remove_spikes(noisy_spectra, smooth=5))
    axs[1, 5].plot(remove_spikes(noisy_spectra, 1, 5, 100))

def test_remove_outliers():
    _x, _y = [np.linspace(-10, 10, idx) for idx in [100, 200]]
    x, y = np.meshgrid(_x, _y)
    base_img = np.random.random(x.shape)
    base_img += 10*np.exp(-(x - 2) ** 2 / (2 * 3 ** 2) - (y - 2) ** 2 / (2 * 4 ** 2))
    base_img += 6*np.exp(-(x + 2) ** 2 / (2 * 6 ** 2) - (y + 3) ** 2 / (2 * 10 ** 2))

    nois_img = np.copy(base_img)
    nois_img += lorentzianNd((_x, _y), 2, 1, 0.1)
    nois_img += 100*np.exp(-(x) ** 2 / (2 * 0.1 ** 2) - (y) ** 2 / (2 * 0.1 ** 2))
    nois_img += 100*np.exp(-(x-5) ** 2 / (2 * 0.1 ** 2) - (y+5) ** 2 / (2 * 0.1 ** 2))

    _x = np.linspace(750, 850, 1340)
    base_spectra = np.random.random(_x.shape)
    base_spectra += gaussianNd(_x, 780, 10, 3)
    base_spectra += lorentzianNd(_x, 800, 10, 3)
    base_spectra += lorentzianNd(_x, 830, 10, 10)
    noisy_spectra = np.copy(base_spectra)
    for x0, a in zip(np.random.random(10) * 100 + 750, np.random.random(10) * 20 + 2):
        noisy_spectra += lorentzianNd(_x, x0, a, 0.1)
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(base_img)
    axs[0, 1].imshow(nois_img)
    axs[0, 2].imshow(remove_outliers(nois_img))
    axs[0, 3].imshow(remove_outliers(nois_img, 0))
    axs[1, 0].plot(base_spectra)
    axs[1, 1].plot(noisy_spectra)
    axs[1, 2].plot(remove_outliers(noisy_spectra))

def test_stitch_datasets():
    x = [np.linspace(-2, -1, 11), np.linspace(0, 5, 11), np.linspace(2.5, 7, 33),
         np.linspace(8, 13, 55), np.linspace(10, 15, 68), np.linspace(14, 19, 74),
         np.linspace(20, 25, 1000), np.linspace(21, 26, 768), np.linspace(22, 27, 31)]
    y = [(idx % 3 +np.random.random(len(_x))) for idx, _x in enumerate(x)]

    fig, axs = plt.subplots(1, 2)
    for ax, interpolation in zip(axs, ['even', 'same']):
        [ax.plot(_x, _y) for _x, _y in zip(x, y)]
        ax.plot(*stitch_datasets(x, y, interpolation), '--')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test_remove_spikes()
    test_remove_outliers()
    plt.show()