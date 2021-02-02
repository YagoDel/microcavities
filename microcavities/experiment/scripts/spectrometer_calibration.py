# -*- coding: utf-8 -*-
"""
Two different approaches:
- If many peaks of the light-source appear in the same frame, then you can identify the sequence of peaks and figure out
what the correspond to in the known emission by doing ratios of energy differences (not implemented yet)
- If only few peaks appear in the same frame, then you need to have an idea of what the central wavelength is and of
what the range of wavelengths is in the frame
"""


import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from microcavities.utils import square, get_data_path
from microcavities.analysis.utils import normalize
import h5py
import os

KEY = 'ANALYSIS'  # 'DATA_TAKING'  #
grating = 1714  # 1800  # 1200  #
calibration_path = get_data_path('calibrations')
file_name = 'andor_table2.h5'  # 'spectrometer_07_2020.h5'
if KEY == 'DATA_TAKING':
    camera_pointer = pvcam
    spectrometer_pointer = spectrometer

# From http://www.astrosurf.com/buil/us/spe2/hresol4.htm
all_peaks = np.asarray(
    [3417.9035, 3472.5711, 3515.1900, 3593.5263, 3600.1691, 4488.0926, 4636.125, 4837.3139, 5005.1587, 5031.3504,
     5104.7011, 5113.6724, 5144.9384, 5188.6122, 5330.7775, 5341.0938, 5360.0121, 5400.5617, 5562.7662, 5656.5664,
     5689.8163, 5719.2248, 5748.2985, 5764.4188, 5804.4496, 5820.1558, 5852.4878, 5881.895, 5944.8342, 5975.534,
     6029.9971, 6074.3377, 6096.1631, 6128.4499, 6143.0626, 6163.5939, 6217.2812, 6266.495, 6304.789, 6334.4278,
     6382.9917, 6402.246, 6506.5281, 6532.8822, 6598.9529, 6678.2764, 6717.043, 6929.4673, 7024.0504, 7032.4131,
     7173.9381, 7245.1666, 7438.899, 7488.8712, 7535.7739, 8136.4057, 8300.3263, 8377.6065, 8495.3598, 8591.2583,
     8634.647, 8654.3831, 8655.522, 8679.493, 8681.921, 8704.111, 8771.656, 8780.621, 8783.75, 8830.907, 8853.867,
     8919.5007, 9148.672, 9201.759, 9300.853, 9326.507, 9425.379, 9486.68, 9534.163, 9665.424, 10798.12, 10844.54,
     11143.02])
all_peaks /= 10
distances = np.diff(all_peaks)
width = np.min(distances)/5

with h5py.File(os.path.join(calibration_path, file_name), 'a') as df:
    if KEY == 'DATA_TAKING':
        full_wavelengths = np.linspace(np.min(all_peaks), np.max(all_peaks), 100001)
        full_spectra = np.zeros(100001)
        for peak in all_peaks:
            full_spectra += np.exp(-(full_wavelengths - peak) ** 2 / (2 * width ** 2))
        full_spectrum = np.array([full_wavelengths, full_spectra])
        df.create_dataset('full_spectrum', full_spectrum.shape, full_spectrum.dtype, full_spectrum)
    elif KEY == 'ANALYSIS':
        full_spectrum = df['full_spectrum'][...]
        full_wavelengths = full_spectrum[0]
        full_spectra = full_spectrum[1]


def measure_wavelength_range(wavelength_step=0.1, *args, **kwargs):
    """
    Wherever you measure it, it's only really valid in that wavelength region (that's the whole point of the calibration)
    :param wavelength_step:
    :param args:
    :param kwargs:
    :return:
    """
    if 'distance' not in kwargs:
        kwargs['distance'] = 5
    img0 = camera_pointer.raw_image(False, True)
    spec0 = normalize(np.mean(img0, 0))
    wvl0 = spectrometer_pointer.wavelength
    spectrometer_pointer.wavelength = wvl0 + wavelength_step
    img1 = camera_pointer.raw_image(False, True)
    spec1 = normalize(np.mean(img1, 0))
    indx_peaks1 = find_peaks(spec0, *args, **kwargs)[0]
    indx_peaks2 = find_peaks(spec1, *args, **kwargs)[0]
    print(indx_peaks1, indx_peaks2, indx_peaks1 - indx_peaks2)
    px_to_wvl = np.mean(indx_peaks1 - indx_peaks2)/wavelength_step
    return img0.shape[1] / px_to_wvl


def get_center_wavelengths(wvl_range, n_peaks=3):
    peak_diff = np.abs(all_peaks[:-n_peaks] - all_peaks[n_peaks:])
    indexes = np.argwhere(peak_diff < wvl_range)
    return np.mean([all_peaks[:-n_peaks][indexes], all_peaks[n_peaks:][indexes]], 0)


def count_peaks(wvl_min, wvl_max, margin=0.0, resolution=0.0):
    """
    If any peak in the margin, return 0
    :param wvl_min:
    :param wvl_max:
    :param margin:
    :param resolution:
    :return:
    """
    n_margins = np.sum(np.logical_and(all_peaks > wvl_min - resolution, all_peaks < wvl_min+margin)) + \
                np.sum(np.logical_and(all_peaks < wvl_max + resolution, all_peaks > wvl_max-margin))
    if n_margins == 0:
        return np.sum(np.logical_and(all_peaks > wvl_min+margin, all_peaks < wvl_max-margin))
    else:
        return 0


def rolling_window(array, window):
    """From https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html"""
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def get_central_wavelengths(wavelength_range, margin=None, resolution=0.01):
    if margin is None:
        margin = 0.1 * wavelength_range
    _doublets = np.array([full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution)],
                         full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution) + 1]])[..., 0]
    doublets = np.mean(_doublets, 0)
    step = int(wvl_range / np.diff(full_spectrum[0])[0])
    rolled_ranges = np.array([full_wavelengths[:-step], full_wavelengths[step:]]).transpose()
    center_roll = np.mean(rolled_ranges, 1)
    n_peaks = np.array([count_peaks(x[0], x[1], margin, resolution) for x in rolled_ranges])
    n_peaks[n_peaks < 2] = 0
    n_peak_change = np.diff(n_peaks)
    _indexes = np.nonzero(n_peak_change)[0]
    tst = np.diff(n_peak_change[_indexes])
    mask1 = tst < -1
    _indexes2 = _indexes[:-1][mask1]
    _indexes3 = _indexes[1:][mask1]
    regions = np.array([center_roll[_indexes2], center_roll[_indexes3]]).transpose()
    _indexes4 = np.logical_not([np.any(np.logical_and(doublets > x[0], doublets < x[1])) for x in regions])
    return np.mean(regions[_indexes4], 1)


def plot_selected_spectra(center_wavelengths, wvl_range):
    a, b = square(len(center_wavelengths))
    # The following figure allows you to decide which center_wavelengths to use for fitting and what thresholds you want
    # to use to find peaks
    fig, axs = plt.subplots(a, b)
    axs = np.array([axs])

    for idx in range(len(center_wavelengths)):
        ax = axs.flatten()[idx]
        wvl = center_wavelengths[idx]
        sel = np.logical_and(full_wavelengths > wvl - wvl_range/2, full_wavelengths < wvl + wvl_range/2)
        ax.plot(full_wavelengths[sel], full_spectra[sel], '--')
    return fig, axs


measured_range = measure_wavelength_range()
plot_selected_spectra(get_central_wavelengths(measured_range), measured_range)

# plt.plot(*full_spectrum, 'r', alpha=0.7)
# plt.plot(center_roll, n_peaks, 'b')
# plt.vlines(center_wavelengths, -0.5, 0.5, color='g')
#
# plt.vlines(center_roll[_indexes2], -0.5, 0.5, color='g')
# plt.vlines(center_roll[_indexes3], -0.5, 0.5, color='g', ls='--')
#
# plt.vlines(full_wavelengths[_indexes2], -0.5, 0.5)

# n_peak_max =
# mask = peaks >= 2
# _diff = np.diff(mask)
# label = np.concatenate(([0], ((np.cumsum(_diff)-1) // 2) * mask[1:]))
# tst = np.maximum.reduceat(np.mean(rolled_ranges, 1), label)
# tst = np.add.reduceat(np.mean(rolled_ranges, 1), label)
# np.max()
# plt.plot(*full_spectrum, 'r', alpha=0.7)
# plt.plot(np.mean(rolled_ranges, 1), peaks, 'b')
# plt.plot(np.mean(rolled_ranges, 1), np.concatenate(([0], np.cumsum(np.diff(peaks)))))
# plt.plot(np.mean(rolled_ranges, 1), -label)
# selected_ranges = rolled_ranges[np.argwhere(peaks >= 2)[:, 0]]

# rolled_wavelengths = rolling_window(full_spectrum[0], int(8 / np.diff(full_spectrum[0])[0]))
#
# rolled_spectrum = rolling_window(full_spectrum[1], 8 / np.diff(full_spectrum[0])[0])
#
# # TODO: WHAT IS GOING ON WITH THE EDGES? It's not quite working as expected
# # TODO: Exclude doublets according to resolution
# def get_center_wavelengths(wvl_range, wvl_range_error=None, n_peak_limits=None):
#     if wvl_range_error is None:
#         wvl_range_error = wvl_range * 0.2
#     if n_peak_limits is None:
#         n_peak_limits = (2, np.inf)
#     peak_diff = np.abs(np.diff(all_peaks))
#     # Find the locations in the spectrum that have no peaks
#     # indexes = np.argwhere(peak_diff > resolution)
#     empty_regions = np.array([all_peaks[:-1], all_peaks[1:]]).transpose()
#     empty_locations = np.mean(empty_regions, 1)  # (all_peaks[1:]+all_peaks[:-1])[indexes] / 2
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(*full_spectrum)
#     ax.vlines(empty_locations, -0.1, 0.1, 'r')
#     [ax.fill_betweenx([-0.5, 1.5], x[0]+wvl_range_error*0.1, x[1]-wvl_range_error*0.1,
#                       color='r', ec='none', alpha=0.1) for x in empty_regions]
#     # Choose the ones that don't have a peak within the wvl_range_error
#     indexes = np.argwhere(peak_diff > wvl_range_error / 2)[:, 0]
#     ax.vlines(empty_locations[indexes], -0.2, 0.2, 'g')
#     # Of those, choose the ones that also don't have a peak within +wvl_range another empty region in the +wvl_range region
#     n_peaks_right = np.array([np.sum(np.logical_and(all_peaks > x - wvl_range_error/2, all_peaks < x + wvl_range_error/2)) for x in empty_locations[indexes] + wvl_range])
#     n_peaks_left = np.array([np.sum(np.logical_and(all_peaks > x - wvl_range_error/2, all_peaks < x + wvl_range_error/2)) for x in empty_locations[indexes] - wvl_range])
#     mask1 = np.argwhere(n_peaks_right == 0)
#     mask2 = np.argwhere(n_peaks_left == 0)
#     mask = np.unique(np.append(mask1.flatten(), mask2.flatten()))
#     indexes2 = indexes[mask]
#     ax.vlines(empty_locations[indexes2], -0.2, 0.2, 'g')
#
#     # mask = np.array([np.any([np.logical_and(x > y[0], x < y[1]) for y in empty_regions[indexes]])
#     #                  for x in empty_locations[indexes] + wvl_range])
#     # indexes_lower = indexes[np.argwhere(mask)[:, 0]]
#     # ax.vlines(empty_locations[indexes_lower], -0.3, 0.3, 'y')
#     # indexes_higher = np.array([np.argmin(np.abs(empty_locations - wvl_range - x)) for x in empty_locations[indexes_lower]])
#     # ax.vlines(empty_locations[indexes_higher], -0.5, 0.5, 'y', ls='--')
#     # clean_regions = np.array([empty_locations[indexes_lower], empty_locations[indexes_higher]]).transpose()
#     # clean_centers = np.mean(clean_regions, 1)
#     # print(clean_centers.shape, clean_regions.shape)
#     # [ax.fill_betweenx([-1, 2], x[0]+wvl_range_error*0.1, x[1]-wvl_range_error*0.1,
#     #                   color='g', ec='none', alpha=0.1) for x in clean_regions]
#     # Of those, choose the ones that have the correct number of spectrum peaks
#     n_peaks_per_region = np.array([np.sum(np.logical_and(all_peaks > x - wvl_range/2, all_peaks < x + wvl_range/2)) for x in clean_centers])
#     mask = np.logical_and(n_peaks_per_region >= n_peak_limits[0], n_peaks_per_region <= n_peak_limits[1])
#     spectrometer_wavelengths = clean_centers[np.argwhere(mask)[:, 0]]
#
#     # spectrometer_regions = np.array([empty_locations[indexes3], empty_locations[indexes3] + wvl_range])
#     # spectrometer_wavelengths = np.mean(spectrometer_regions, 0)
#     fig, axs = plot_selected_spectra(spectrometer_wavelengths, wvl_range)
#     # [ax.vlines(empty_locations[indexes3][idx], -0.1, 1.1) for idx, ax in enumerate(axs.flatten())]
#     # [ax.vlines(empty_locations[indexes3][idx], -0.1, 1.1) for idx, ax in enumerate(axs.flatten())]
#
#
#
#     # indexes4 = np.array([np.logical_and(x > y[0], x < y[1]) for x, y in zip(empty_locations, empty_regions)])
#     # indexes_array = np.array([[np.logical_and(x > y[0], x < y[1]) for y in empty_regions] for x in empty_locations + wvl_range])
#     # indexes4 = np.argwhere(np.sum(indexes_array, 1))
#     # falls_in_between = np.array([np.any([np.logical_and(x > y[0], x < y[1]) for y in empty_regions]) for x in empty_locations + wvl_range])
#     # indexes4 = np.argwhere(falls_in_between)
#     # separated_regions = empty_locations[indexes4] + wvl_range / 2
#     # peaks_in_between = [n_peak_limits[1] > np.sum(np.logical_and(all_peaks > x-wvl_range/2, all_peaks < x + wvl_range/2)) > n_peak_limits[0] for x in separated_regions]
#     # return separated_regions[peaks_in_between]
#     # # Distance between the empty location + wvl_range and the nearest empty location
#     # _dist = np.array([np.min(np.abs((x + wvl_range)-empty_locations)) for x in empty_locations])
#     # indexes2 = np.argwhere(_dist < wvl_range_error)
#     # _dist2 = np.array([np.min(np.abs((x + wvl_range)-all_peaks)) for x in empty_locations])
#     # indexes3 = np.argwhere(_dist2 > wvl_range_error)
#     # return empty_locations[indexes3]
# get_center_wavelengths(8, 10);
# center_wavelengths = get_center_wavelengths(8, 10)


if grating == 1200:
    wvl_range = 13
    center_wavelengths = np.array([600, 615, 630, 660, 700, 713, 750, 822, 830, 840,
                                   599, 614, 629, 659, 699, 712, 749, 821, 829, 839,
                                   602, 617, 631, 661, 701, 714, 751, 823, 831, 841])
elif grating in 1800:
    wvl_range = 7.5
    center_wavelengths = np.array([572, 600, 608, 614, 616, 628, 630, 632, 660, 662, 748, 750, 752, 834])
elif grating in 1714:
    wvl_range = 8.5
    center_wavelengths = np.array([600, 608, 614, 616, 628, 630, 632, 660, 662, 748, 750, 834])
    thresholds = [0.1, 0.02, 0.018, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


def threshold_image(imgs=None):
    if imgs is None:
        with h5py.File(os.path.join(calibration_path, file_name), 'r') as dfile:
            imgs = []
            for wvl in center_wavelengths:
                imgs += [dfile['grating=%d/spectrometer_wvl=%d' % (grating, wvl)]]
            imgs = np.array(imgs)

    a, b = square(len(center_wavelengths))
    # The following figure allows you to decide which center_wavelengths to use for fitting and what thresholds you want
    # to use to find peaks
    fig, axs = plt.subplots(a, b)
    axs = np.array([axs])

    for idx, ax in enumerate(axs.flatten()):
        wvl = center_wavelengths[idx]
        spectra = np.mean(imgs[idx], 0)
        spectra -= np.min(spectra)
        spectra /= np.max(spectra)
        wavelengths = np.linspace(wvl - wvl_range, wvl + wvl_range, len(spectra))
        ax.plot(wavelengths, spectra)
        ax.set_title(center_wavelengths[idx])
        sel = np.logical_and(full_wavelengths > wvl - wvl_range, full_wavelengths < wvl + wvl_range)
        ax.plot(full_wavelengths[sel], full_spectra[sel], '--')


# DATA TAKING
if KEY == 'DATA_TAKING':
    imgs = []
    for wvl in center_wavelengths:
        spectrometer_pointer.wavelength = wvl
        img = camera_pointer.raw_image()
        if np.max(np.mean(img, 0)) > 60000:
            _idx = 0
            while np.max(np.mean(img, 0)) > 60000:
                exp = camera_pointer.exposure
                camera_pointer.exposure = exp / 2
                img = camera_pointer.raw_image()
                print(_idx)
                if _idx > 10:
                    break
                _idx += 1
        maxval = np.max(np.mean(img, 0))
        exp = camera_pointer.exposure
        if 60000 * exp / maxval < 30:
            camera_pointer.exposure = 60000 * exp / maxval
        else:
            print('Max exposure. wvl=%d' % wvl)
            camera_pointer.exposure = 30
        img = camera_pointer.raw_image()

        imgs += [img]
    imgs = np.array(imgs)

    threshold_image(imgs)

    # selected_wavelengths = [600, 615, 630, 660, 700, 713, 750, 822, 830, 840]
    selected_wavelengths = center_wavelengths
    if grating == 1200:
        thresholds = [0.02, 0.02, 0.1, 0.1, 0.02, 0.02, 0.15, 0.3, 0.35, 0.36] * 3
    elif grating == 1800:
        thresholds = [0.03, 0.1, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.6, 0.1]

    with h5py.File(os.path.join(calibration_path, file_name), 'a') as df:
        for wvl, thrs in zip(selected_wavelengths, thresholds):
            idx = np.argwhere(center_wavelengths == wvl)[0][0]
            img = imgs[idx]
            try:
                dset = df.create_dataset('grating=%d/spectrometer_wvl=%d' % (grating, wvl), img.shape, img.dtype, img)
            except:
                dset = df['spectrometer_wvl=%d' % wvl]
            dset.attrs.create('threshold', thrs)

# ANALYSIS
elif KEY == 'ANALYSIS':
    selected_wavelengths = center_wavelengths
    with h5py.File(os.path.join(calibration_path, file_name), 'r') as df:
        thresholds = []
        imgs = []
        for wvl in selected_wavelengths:
            dset = df['grating=%d/spectrometer_wvl=%d' % (grating, wvl)]
            imgs += [dset[...]]
            thresholds += [dset.attrs['threshold']]
        imgs = np.array(imgs)
        thresholds = np.array(thresholds)
    pixels = np.arange(imgs.shape[-1]) - imgs.shape[-1]/2

    fits = []
    found_peaks = []
    for wvl, thrs in zip(selected_wavelengths, thresholds):
        idx = np.argwhere(center_wavelengths == wvl)[0][0]
        spectra = np.mean(imgs[idx], 0)
        spectra -= np.min(spectra)
        spectra /= np.max(spectra)
        indx_peaks = find_peaks(spectra, thrs, distance=5)[0]
        pixel_peaks = pixels[indx_peaks]

        sel2 = np.logical_and(all_peaks > wvl - wvl_range, all_peaks < wvl + wvl_range)
        known_peaks = all_peaks[sel2]

        fits += [np.polyfit(pixel_peaks, known_peaks-wvl, 1)]

        wavelengths = np.linspace(wvl - wvl_range, wvl + wvl_range, len(spectra))
        found_peaks += [indx_peaks]
    fits = np.array(fits)
    found_peaks = np.array(found_peaks)

    # PLOT TO CHECK THAT THE FOUND PEAKS ARE WHERE YOU THINK THEY ARE
    a, b = square(len(selected_wavelengths))
    fig, axs = plt.subplots(a, b)
    for idx, ax in enumerate(axs.flatten()):
        idx2 = np.argwhere(center_wavelengths == selected_wavelengths[idx])[0][0]
        wvl = selected_wavelengths[idx]
        spectra = np.mean(imgs[idx2], 0)
        spectra -= np.min(spectra)
        spectra /= np.max(spectra)
        wavelengths = np.linspace(wvl - wvl_range, wvl + wvl_range, len(spectra))
        ax.set_title(center_wavelengths[idx2])
        ax.plot(wavelengths, spectra)

        sel = np.logical_and(full_wavelengths > wvl - wvl_range, full_wavelengths < wvl + wvl_range)
        ax.plot(full_wavelengths[sel], full_spectra[sel], '--')

        ax.plot(wavelengths[found_peaks[idx]], spectra[found_peaks[idx]], 'x')


    # CHECKING THAT THE GRATING IS LINEAR (if physics makes sense, that's always the case)
    a, b = square(len(selected_wavelengths))
    fig, axs = plt.subplots(a, b)
    for idx, ax in enumerate(axs.flatten()):
        ax.set_title(selected_wavelengths[idx])
        indx_peaks = found_peaks[idx]
        wvl = selected_wavelengths[idx]
        pixel_peaks = pixels[indx_peaks]

        sel2 = np.logical_and(all_peaks > wvl - wvl_range, all_peaks < wvl + wvl_range)
        known_peaks = all_peaks[sel2]
        ax.plot(pixel_peaks, known_peaks-wvl, '.')
        ax.plot(pixel_peaks, np.poly1d(fits[idx])(pixel_peaks), '--')

    fig, axs = plt.subplots(1, 2)
    indx = np.argsort(selected_wavelengths)
    if grating == 1800:
        fit_fit = np.polyfit(selected_wavelengths[indx][-6:], fits[:, 0][indx][-6:], 1)
        newx = selected_wavelengths[indx][-6:]
        newy = np.poly1d(fit_fit)(newx)
    elif grating == 1200:
        fit_fit = np.polyfit(selected_wavelengths[indx][-16:], fits[:, 0][indx][-16:], 1)
        newx = selected_wavelengths[indx][-16:]
        newy = np.poly1d(fit_fit)(newx)
    elif grating == 1714:
        fit_fit = np.polyfit(selected_wavelengths[indx], fits[:, 0][indx], 1)
        newx = selected_wavelengths[indx]
        newy = np.poly1d(fit_fit)(newx)

    axs[0].plot(selected_wavelengths[indx], fits[:, 0][indx], '.-')
    axs[0].plot(newx, newy, '--', label='%g\n%g' % tuple(fit_fit))
    axs[0].legend()
    axs[1].plot(selected_wavelengths[indx], fits[:, 1][indx], '.-')
    axs[0].set_title('Slope')
    axs[1].set_title('Offset correction')
    [ax.set_xlabel('Center wavelength') for ax in axs]
    fig.savefig(os.path.join(calibration_path, file_name.rstrip('.h5') + '%d.png' % grating), dpi=1200, bbox_inches='tight')
