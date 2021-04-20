# -*- coding: utf-8 -*-

import re
import ast
import numpy as np
from scipy.ndimage import gaussian_filter
from microcavities.utils.functools import lorentzianNd, gaussianNd


def SortingKey(pattern):
    """
    To be used to sort the names of folders, assuming the folder names follow the convention 'ParamName=ParamValue'
    or 'ParamName=ParamValue-ParamIteration' if there is more than one dataset with the same parameter value.
    Sorting is done in order of ParamValue, and then ParamIteration
    Args:
        pattern: name of the folder
    Returns:
        Either a single value or a two-tuple that is used for the sort
    """
    splitpat = pattern.split('=')
    returnval = (bool(len(splitpat) - 1),)
    if len(splitpat) > 1:
        if re.match('.\d*-\d+|.*0\.\d*-\d+', splitpat[1]):
            returnval += tuple([np.real(ast.literal_eval(x)) for x in re.findall('(.+)-(\d+)', splitpat[1])[0]])
        elif re.match('\(\d+\.*\d*, *\d+.*\d*\)', splitpat[1]):
            returnval += tuple([np.real(ast.literal_eval(x)) for x in re.findall('(\d+\.*\d*),(\d+\.*\d*)', splitpat[1])[0]])
        else:
            val = ast.literal_eval(splitpat[1])
            returnval += (np.sign(np.real(val)) * np.abs(val),)
    return returnval


def fit_function(name):
    """
    Given a name, return a lmfit Model with a guess method implemented
    :param name:
    :return:
    """


def normalize(array, percentiles=(0, 100), axis=None, cut=False):
    mn = np.percentile(array, percentiles[0], axis=axis, keepdims=True)
    mx = np.percentile(array, percentiles[1], axis=axis, keepdims=True)
    normed = (array - mn) / (mx - mn)
    if cut:
        normed[normed > 1] = 1
        normed[normed < 0] = 0
    return normed


def centroids(array, axis=-1):
    """Reduce an array along an axis, giving the index centroid along that axis"""
    axes_array = [np.arange(x) for x in array.shape]
    mesh = np.meshgrid(*axes_array, indexing='ij')
    img = mesh[axis]
    return np.average(img, axis, array)


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_remove_spikes()
    plt.show()