# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from microcavities.utils import square, get_data_path
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


def get_center_wavelengths(wvl_range, n_peaks=3):
    peak_diff = np.abs(all_peaks[:-n_peaks] - all_peaks[n_peaks:])
    indexes = np.argwhere(peak_diff < wvl_range)
    return np.mean([all_peaks[:-n_peaks][indexes], all_peaks[n_peaks:][indexes]], 0)


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
