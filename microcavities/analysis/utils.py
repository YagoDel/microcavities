# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
from nplab.utils.log import create_logger
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from microcavities.utils import around_step
from microcavities.utils.functools import lorentzianNd, gaussianNd
import collections

LOGGER = create_logger('analysis.utils')


def centroids(array, axis=-1):
    """Reduce an array along an axis, giving the index centroid along that axis"""
    axes_array = [np.arange(x) for x in array.shape]
    mesh = np.meshgrid(*axes_array, indexing='ij')
    img = mesh[axis]
    return np.ma.average(img, axis, array)


def remove_cosmicrays(image, standard_deviations=5, sigma=30, max_iterations=10):
    """Removes cosmic rays spikes in an image

    Iteratively replaces the regions in an image that are more than standard_deviations away from the mean noise by the
    gaussian_filter'ed image

    :param image: 2D array
    :param standard_deviations: float
    :param sigma: float. Standard deviation for Gaussian filter. To be passed to scipy.ndimage.filters.gaussian_filter
    :param max_iterations:
    :return:
    """
    cleaned = np.copy(image)  # prevent modification in place

    for i in range(max_iterations):
        noise_spectrum = cleaned / gaussian_filter(cleaned, sigma)  # normalised array
        mean_noise = noise_spectrum.mean()  # tends to -> 1 as iterations go on
        noise_level = np.std(noise_spectrum)
        spikes = np.nonzero(noise_spectrum > mean_noise + (standard_deviations * noise_level))  # mask of the spikes
        cleaned[spikes] = gaussian_filter(cleaned, sigma)[spikes]  # replace the spikes with the filtered image
        if len(spikes[0]) == 0:
            break
    return cleaned


def remove_outliers(x, axis=None, bar=1.5, side='both', method='IQR', return_mask=False):
    """Uses the IQR or zscore method
    From https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

    :param x: array
    :param axis: axis to remove along
    :param bar: float
    :param side: str
        Whether to remove outliers lower than the threshold ('lt'), greater than the threshold ('gt') or 'both'
    :param method: str. Either IQR or zscore
    :param return_mask: bool
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
    else:
        raise NotImplemented('method = %s' % method)
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


def stitch_datasets(x_sets, y_sets, interpolation='even'):
    """Stitching datasets

    Useful for creating a spectra for a large wavelength range from individual small range images

    TODO: allow for more complicated stitching than just averaging: previous, next, or a smooth transition from one to the next

    :param x_sets: iterable of iterables
    :param y_sets: iterable of iterables
    :param interpolation: str. Currently just a way of selecting x values
        'even'  -  selects a range of x-values that is evenly spaced between the minimum and the maximum x values
        'same'  -  selects an range of x-values that follows x_sets but removes the overlaps.
    :return:
    """

    # Choosing an array of x-values
    if interpolation == 'even':  # creates an x axis of uniformly distributed values
        x_step = [np.diff(x)[0] for x in x_sets]
        x_new = np.arange(np.min(np.concatenate(x_sets)),
                          np.max(np.concatenate(x_sets))+np.min(x_step)/2,
                          np.min(x_step))
    elif interpolation == 'same':  # creates an x axis stitched from the given axes
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

    # Create interpolated functions for each set of x and y axes. Out of bounds are set to np.nan
    interpolated_datasets = [interp1d(x, y, bounds_error=False) for x, y in zip(x_sets, y_sets)]

    # Create the stitched dataset by averaging the interpolated values between different datasets
    y_new = np.nanmean([f(x_new) for f in interpolated_datasets], 0)
    return x_new, y_new


def find_smooth_region(data, threshold=0.1):
    """Returns the boundary indices of the smooth region in data

    Smoothness is defined as a fraction of the min to max variation, given by
    the threshold parameter

    :param 1d array data:
    :param float threshold: percentage of the min to max variation below which
                            a signal is considered to be smooth
    :return:
    """
    # First normalise the data to go from 0 to 1
    data = np.array(data, np.float)
    data -= np.min(data)
    data /= np.max(data)

    # Then find the indices where the variation between adjacent points is
    # larger than the desired threshold
    diff = np.diff(data)
    noise_indices = [0]
    noise_indices += list(np.argwhere(np.abs(diff) > threshold).flatten())
    noise_indices += [len(data)]

    # If there is a flat region in the line, there will be a large gap in the
    # noise_indices. To find the gap, we find the location of the maximum
    # variation between noise_indices
    idx = np.argmax(np.abs(np.diff(noise_indices)))

    # The boundary of the smooth region is given by noise_indices at idx
    boundaries = noise_indices[idx] + 1, noise_indices[idx + 1] + 1
    return boundaries, data[boundaries[0]:boundaries[1]]


def guess_peak(data, xaxis=None, width_lims=None, background_percentile=5):
    """Peak property guessing

    Guesses the background, peak height, peak position and FHWM. Typically used to initialise a fitting procedure
    This will get confused if there's more than one peak with maxima above half the maximum in the data.

    :param data: 1D array
    :param xaxis: None or 1D array
    :param width_lims: None or two tuple of (min, max) width limits
    :param background_percentile: float. To be passed to np.percentile
    :return:
    """
    DEBUG = False  # simply used for plotting the results, in case needed in the future

    data = np.copy(data)
    if xaxis is None:
        xaxis = list(range(len(data)))

    # Finds the peak position
    center_idx = np.argmax(data)
    center = xaxis[center_idx]

    # Finds and removes the background
    bkg = np.percentile(data, background_percentile)
    data -= bkg

    # Finds the peak FWHM
    if width_lims is None:
        # Default width limits is that the peak cannot be sharper than 3 xaxis steps, and it cannot be wider than half
        # the xaxis range
        width_lims = (3 * np.abs(np.mean(np.diff(xaxis))), 0.5 * (np.max(xaxis)-np.min(xaxis)))
    assert len(width_lims) == 2
    minima_indices = np.argsort(np.abs(data - data[center_idx] / 2))  # finds the indices closest to half maximum
    assert len(minima_indices) > 1  # there need to be more than two minima
    first_minimum = minima_indices[0]  # first index is arbitrarily chosen
    widths = np.abs((xaxis[minima_indices]-xaxis[first_minimum]))/2  # peak widths from that first index
    for width in widths:  # find the first peak that is larger than the minima
        if width > width_lims[0]:
            break
    if width < width_lims[0]:
        LOGGER.warn('Peak FWHM is smaller than the limit')
        width = width_lims[0]
    if width > width_lims[1]:
        LOGGER.warn('Peak FWHM is larger than the limit')
        width = width_lims[1]

    # Finds the peak amplitude (assuming a Lorentzian shape)
    ampl = np.pi * width * data[center_idx]

    if DEBUG:
        print(width_lims)
        fig, ax = create_axes()
        _y = np.abs(data - data[center_idx] / 2)
        ax.plot(xaxis, _y)
        ax.vlines([xaxis[first_minimum], xaxis[first_minimum]+width, xaxis[first_minimum]-width],
                  np.min(_y), np.max(_y), 'r')
        plt.figure()
    return dict(amplitude=ampl, sigma=width, center=center, background=bkg)


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
    axs[0, 2].imshow(remove_cosmicrays(nois_img))
    axs[0, 3].imshow(remove_cosmicrays(nois_img, 2))
    axs[0, 4].imshow(remove_cosmicrays(nois_img, sigma=5))
    axs[0, 5].imshow(remove_cosmicrays(nois_img, 1, 5, 100))
    axs[1, 0].plot(base_spectra)
    axs[1, 1].plot(noisy_spectra)
    axs[1, 2].plot(remove_cosmicrays(noisy_spectra))
    axs[1, 3].plot(remove_cosmicrays(noisy_spectra, 2))
    axs[1, 4].plot(remove_cosmicrays(noisy_spectra, sigma=5))
    axs[1, 5].plot(remove_cosmicrays(noisy_spectra, 1, 5, 100))

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