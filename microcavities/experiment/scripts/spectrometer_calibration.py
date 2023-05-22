# -*- coding: utf-8 -*-
"""
Two different approaches:
- If many peaks of the light-source appear in the same frame, then you can identify the sequence of peaks and figure out
what the correspond to in the known emission by doing ratios of energy differences (not implemented yet)
- If only few peaks appear in the same frame, then you need to have an idea of what the central wavelength is and of
what the range of wavelengths is in the frame

# TODO: extend using https://github.com/jveitchmichaelis/rascal
"""


import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from microcavities.utils import square, get_data_path
from microcavities.analysis.utils import normalize
import h5py
import os
from nplab.utils.gui import get_qt_app

app = get_qt_app()
KEY = 'ANALYSIS'  #'DATA_TAKING'  #
grating = (2, 1200)  # 1714  # (1, 1800)  #
calibration_path = get_data_path('calibrations')
file_name = 'table1_andor_02_2022.h5'  # 'andor2_table2.h5'  # 'spectrometer_07_2020.h5'

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


if KEY == 'DATA_TAKING':
    camera_pointer = andor_acton
    spectrometer_pointer = andor_acton
    camera_pointer.Exposure = 3
    spectrometer_pointer.wavelength = 813

    with h5py.File(os.path.join(calibration_path, file_name), 'a') as df:
        if 'full_spectrum' not in df:
            full_wavelengths = np.linspace(np.min(all_peaks), np.max(all_peaks), 100001)
            full_spectra = np.zeros(100001)
            for peak in all_peaks:
                full_spectra += np.exp(-(full_wavelengths - peak) ** 2 / (2 * width ** 2))
            full_spectrum = np.array([full_wavelengths, full_spectra])
            df.create_dataset('full_spectrum', full_spectrum.shape, full_spectrum.dtype, full_spectrum)
        else:
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
        default_kwargs = dict(distance=5, height=0.5)
        kwargs = {**default_kwargs, **kwargs}
        img0 = camera_pointer.raw_image(False, True)
        spec0 = normalize(np.mean(img0, 0))
        wvl0 = spectrometer_pointer.wavelength
        spectrometer_pointer.wavelength = wvl0 + wavelength_step
        img1 = camera_pointer.raw_image(False, True)
        app.processEvents()
        spec1 = normalize(np.mean(img1, 0))
        indx_peaks1 = find_peaks(spec0, *args, **kwargs)[0]
        indx_peaks2 = find_peaks(spec1, *args, **kwargs)[0]
        print(indx_peaks1, indx_peaks2)
        print(indx_peaks1 - indx_peaks2)
        px_to_wvl = np.mean(indx_peaks1 - indx_peaks2)/wavelength_step
        return img0.shape[1] / px_to_wvl


    measured_range = measure_wavelength_range()


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


    def get_central_wavelengths(wavelength_range, margin=None, resolution=0.01, min_peaks=2, max_peaks=4):
        """Finds central wavelengths of spectrum that have more than one peak separated by clearly"""
        if margin is None:
            margin = 0.1 * wavelength_range
        _doublets = np.array([full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution)],
                             full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution) + 1]])[..., 0]
        doublets = np.mean(_doublets, 0)
        step = int(wavelength_range / np.diff(full_spectrum[0])[0])
        rolled_ranges = np.array([full_wavelengths[:-step], full_wavelengths[step:]]).transpose()
        center_roll = np.mean(rolled_ranges, 1)
        n_peaks = np.array([count_peaks(x[0], x[1], margin, resolution) for x in rolled_ranges])
        n_peaks[n_peaks < min_peaks] = 0
        n_peaks[n_peaks > max_peaks] = 0
        n_peak_change = np.diff(n_peaks)
        _indexes = np.nonzero(n_peak_change)[0]
        tst = np.diff(n_peak_change[_indexes])
        mask1 = tst < -1
        _indexes2 = _indexes[:-1][mask1]
        _indexes3 = _indexes[1:][mask1]
        regions = np.array([center_roll[_indexes2], center_roll[_indexes3]]).transpose()
        _indexes4 = np.logical_not([np.any(np.logical_and(doublets > x[0], doublets < x[1])) for x in regions])
        return np.mean(regions[_indexes4], 1)


    def plot_selected_spectra(center_wavelengths, wavelength_range):
        a, b = square(len(center_wavelengths))
        # The following figure allows you to decide which center_wavelengths to use for fitting and what thresholds you want
        # to use to find peaks
        fig, axs = plt.subplots(a, b)
        axs = np.array([axs])

        for idx in range(len(center_wavelengths)):
            ax = axs.flatten()[idx]
            wvl = center_wavelengths[idx]
            sel = np.logical_and(full_wavelengths > wvl - wavelength_range/2, full_wavelengths < wvl + wavelength_range/2)
            ax.plot(full_wavelengths[sel], full_spectra[sel], '--')
        return fig, axs


    # if grating == 1200:
    #     wvl_range = 11.4  # 13
    #     center_wavelengths = np.array([600, 615, 630, 660, 700, 713, 750, 822, 830, 840,
    #                                    599, 614, 629, 659, 699, 712, 749, 821, 829, 839,
    #                                    602, 617, 631, 661, 701, 714, 751, 823, 831, 841])
    # elif grating in 1800:
    #     wvl_range = 7.5
    #     center_wavelengths = np.array([572, 600, 608, 614, 616, 628, 630, 632, 660, 662, 748, 750, 752, 834])
    # elif grating in 1714:
    #     wvl_range = 8.5
    #     center_wavelengths = np.array([600, 608, 614, 616, 628, 630, 632, 660, 662, 748, 750, 834])
    #     thresholds = [0.1, 0.02, 0.018, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # else:
    wvl_range = measured_range
    cntr_wlvs = get_central_wavelengths(measured_range)
    cntr_wlvs = cntr_wlvs[600 < cntr_wlvs]
    cntr_wlvs = cntr_wlvs[cntr_wlvs < 875]
    # center_wavelengths = center_wavelengths[::3]


    def threshold_image(imgs=None, center_wavelengths=None):
        if imgs is None:
            with h5py.File(os.path.join(calibration_path, file_name), 'r') as dfile:
                imgs = []
                for wvl in center_wavelengths:
                    imgs += [dfile['grating=%d/spectrometer_wvl=%d' % (grating[1], wvl)]]
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
            wavelengths = np.linspace(wvl - wvl_range/2, wvl + wvl_range/2, len(spectra))
            ax.plot(wavelengths, spectra)
            ax.set_title(center_wavelengths[idx])
            sel = np.logical_and(full_wavelengths > wvl - wvl_range/2, full_wavelengths < wvl + wvl_range/2)
            ax.plot(full_wavelengths[sel], full_spectra[sel], '--')


    plot_selected_spectra(cntr_wlvs, wvl_range)

    imgs = []
    for wvl in cntr_wlvs:
        spectrometer_pointer.wavelength = wvl
        # img = camera_pointer.raw_image()
        # if np.max(np.mean(img, 0)) > 60000:
        #     _idx = 0
        #     while np.max(np.mean(img, 0)) > 60000:
        #         exp = camera_pointer.Exposure
        #         camera_pointer.Exposure = exp / 2
        #         img = camera_pointer.raw_image(False, True)
        #         app.processEvents()
        #         print(_idx)
        #         if _idx > 10:
        #             break
        #         _idx += 1
        # maxval = np.max(np.mean(img, 0))
        # exp = camera_pointer.Exposure
        # if 60000 * exp / maxval < 30:
        #     camera_pointer.Exposure = 60000 * exp / maxval
        # else:
        #     print('Max exposure. wvl=%d' % wvl)
        #     camera_pointer.Exposure = 30
        # img = camera_pointer.raw_image(False, True)
        img = camera_pointer.auto_exposure()
        app.processEvents()
        imgs += [img]
    imgs = np.array(imgs)

    # Manually selecting which wavelengths will work. Criteria: too many or two few peaks compared to what is expected
    threshold_image(imgs, cntr_wlvs)
    selected_wavelengths = cntr_wlvs

    with h5py.File(os.path.join(calibration_path, file_name), 'a') as df:
        group = df.create_group('grating=%s' % (grating, ))
        group.attrs.create('wavelength_range', wvl_range)
        for indx, wvl in enumerate(selected_wavelengths):
            idx = np.argwhere(cntr_wlvs == wvl)[0][0]
            img = imgs[idx]
            try:
                dset = group.create_dataset('spectrometer_wvl=%d' % wvl, img.shape, img.dtype, img)
            except:
                print('Failed saving %g' % wvl)
elif KEY == 'ANALYSIS':
    selected_wavelengths = None
    default_threshold = 0.07
    with h5py.File(os.path.join(calibration_path, file_name), 'r') as df:
        full_spectrum = df['full_spectrum'][...]
        full_wavelengths = full_spectrum[0]
        full_spectra = full_spectrum[1]

        group = df['grating=%s' % (grating, )]
        if selected_wavelengths is None:
            keys = [key for key in group.keys() if key.startswith('spectrometer_wvl')]
            selected_wavelengths = np.array([float(wvl.strip('spectrometer_wvl=')) for wvl in keys])

        # thresholds = []
        imgs = []
        for wvl in selected_wavelengths:
            dset = group['spectrometer_wvl=%d' % (wvl, )]
            imgs += [dset[...]]
            # if 'threshold' in dset.attrs:
            #     thresholds += [dset.attrs['threshold']]
            # else:
            #     thresholds += [default_threshold]
        imgs = np.array(imgs)
        # thresholds = np.array(thresholds)

        wvl_range = group.attrs['wavelength_range']
    pixels = np.arange(imgs.shape[-1]) - imgs.shape[-1]/2

    def recursive_find_peaks(n_target_peaks, spectra, threshold=default_threshold, recursion_index=1, max_recursion=100):
        indx_peaks = find_peaks(spectra, threshold, distance=5)[0]
        if recursion_index > max_recursion:
            return indx_peaks
        if len(indx_peaks) == n_target_peaks:
            return indx_peaks
        elif len(indx_peaks) <= n_target_peaks:
            return recursive_find_peaks(n_target_peaks, spectra, threshold - default_threshold / recursion_index,
                                        recursion_index+1, max_recursion)
        else:
            return recursive_find_peaks(n_target_peaks, spectra, threshold + default_threshold / recursion_index,
                                        recursion_index + 1, max_recursion)


    fits = []
    residuals = []
    found_peaks = []
    # idx = -1
    for idx, wvl in enumerate(selected_wavelengths):
        sel2 = np.logical_and(all_peaks > wvl - wvl_range/2, all_peaks < wvl + wvl_range/2)
        known_peaks = all_peaks[sel2]

        # idx = np.argwhere(cntr_wlvs == wvl)[0][0]
        # idx += 1
        spectra = np.mean(imgs[idx], 0)
        spectra -= np.min(spectra)
        spectra /= np.max(spectra)
        indx_peaks = recursive_find_peaks(len(known_peaks), spectra)
        # indx_peaks = find_peaks(spectra, default_threshold, distance=5)[0]
        pixel_peaks = pixels[indx_peaks]

        if len(pixel_peaks) == len(known_peaks):
            _values, _residuals, _, _, _ = np.polyfit(pixel_peaks, known_peaks-wvl, 1, full=True)
            if len(_residuals) == 0:
                residuals += [np.nan]
                fits += [_values]
            else:
                residuals += [_residuals[0]]
                if _residuals[0] < 0.5:
                    fits += [_values]
                else:
                    fits += [(np.nan, np.nan)]
        else:
            fits += [(np.nan, np.nan)]
            residuals += [np.nan]
        wavelengths = np.linspace(wvl - wvl_range/2, wvl + wvl_range/2, len(spectra))
        found_peaks += [indx_peaks]
    fits = np.array(fits)
    found_peaks = np.array(found_peaks)

    # PLOT TO CHECK THAT THE FOUND PEAKS ARE WHERE YOU THINK THEY ARE
    a, b = square(len(selected_wavelengths))
    fig, axs = plt.subplots(a, b)
    for idx, ax in enumerate(axs.flatten()):
        idx2 = idx #np.argwhere(center_wavelengths == selected_wavelengths[idx])[0][0]
        wvl = selected_wavelengths[idx]
        spectra = np.mean(imgs[idx2], 0)
        spectra -= np.min(spectra)
        spectra /= np.max(spectra)
        wavelengths = np.linspace(wvl - wvl_range/2, wvl + wvl_range/2, len(spectra))
        ax.set_title(selected_wavelengths[idx2])
        ax.plot(wavelengths, spectra)

        sel = np.logical_and(full_wavelengths > wvl - wvl_range/2, full_wavelengths < wvl + wvl_range/2)
        ax.plot(full_wavelengths[sel], full_spectra[sel], '--')

        ax.plot(wavelengths[found_peaks[idx]], spectra[found_peaks[idx]], 'x')

    # CHECKING THAT THE GRATING IS LINEAR (if nothing is wrong, each plot should be perfectly linear)
    a, b = square(len(selected_wavelengths))
    fig, axs = plt.subplots(a, b)
    for idx, ax in enumerate(axs.flatten()):
        ax.set_title('%.2f %g' % (selected_wavelengths[idx], residuals[idx]))
        indx_peaks = found_peaks[idx]
        wvl = selected_wavelengths[idx]
        pixel_peaks = pixels[indx_peaks]

        sel2 = np.logical_and(all_peaks > wvl - wvl_range/2, all_peaks < wvl + wvl_range/2)
        known_peaks = all_peaks[sel2]
        if len(pixel_peaks) == len(known_peaks):
            ax.plot(pixel_peaks, known_peaks-wvl, '.')
            ax.plot(pixel_peaks, np.poly1d(fits[idx])(pixel_peaks), '--')

    fig, axs = plt.subplots(1, 2)
    indx = np.argsort(selected_wavelengths)
    # if grating == 1800:
    #     indices = slice(-6, -1)
    #     # fit_fit = np.polyfit(selected_wavelengths[indx][-6:], fits[:, 0][indx][-6:], 1)
    #     # newx = selected_wavelengths[indx][-6:]
    # elif grating == 1200:
    #     indices = slice(-16, -1)
    #     # fit_fit = np.polyfit(selected_wavelengths[indx][-16:], fits[:, 0][indx][-16:], 1)
    #     # newx = selected_wavelengths[indx][-16:]
    # elif grating == 1714:
    #     indices = slice(len(selected_wavelengths))
    #     # fit_fit = np.polyfit(selected_wavelengths[indx], fits[:, 0][indx], 1)
    #     # newx = selected_wavelengths[indx]
    # else:
    indices = slice(len(selected_wavelengths))
        # fit_fit = np.polyfit(selected_wavelengths[indx], fits[:, 0][indx], 1)
        # newx = selected_wavelengths[indx]

    # indices = np.logical_and(selected_wavelengths[indx] > 650, selected_wavelengths[indx] < 850)
    _wvls = selected_wavelengths[indx][indices]
    newx = np.linspace(_wvls.min(), _wvls.max(), 101)

    _fits_slope = fits[:, 0][indx][indices]
    mask = np.logical_not(np.isnan(_fits_slope))
    order = 5
    fit_slope = np.polyfit(_wvls[mask], _fits_slope[mask], order)
    newy_slope = np.poly1d(fit_slope)(newx)
    label_slope = ('\n'.join(['%g']*(order+1))) % tuple(fit_slope)

    _fits_offset = fits[:, 1][indx][indices]
    mask = np.logical_not(np.isnan(_fits_offset))
    order = 1
    fit_offset = np.polyfit(_wvls[mask], _fits_offset[mask], order)
    newy_offset = np.poly1d(fit_offset)(newx)
    label_offset = ('\n'.join(['%g']*(order+1))) % tuple(fit_offset)

    axs[0].plot(selected_wavelengths[indx], fits[:, 0][indx], '.-')
    axs[0].plot(newx, newy_slope, '--', label=label_slope)
    axs[0].legend()
    axs[1].plot(selected_wavelengths[indx], fits[:, 1][indx], '.-')
    axs[1].plot(newx, newy_offset, '--', label=label_offset)
    axs[1].legend()
    axs[0].set_title('Slope')
    axs[1].set_title('Offset correction')
    [ax.set_xlabel('Center wavelength') for ax in axs]
    # fig.savefig(os.path.join(calibration_path, file_name.rstrip('.h5') + '%d.png' % grating), dpi=1200, bbox_inches='tight')

    with h5py.File(os.path.join(calibration_path, file_name), 'a') as df:
        group = df.create_group('grating=%s_analysis' % (grating, ))
        group.create_dataset('dispersion', data=fit_slope)
        group.create_dataset('offset', data=fit_offset)

    def make_json(file_path):
        with h5py.File(os.path.join(calibration_path, file_path), 'r') as df:
            gratings = []
            for key in df.keys():
                if key.startswith('grating='):
                    if key.endswith('_analysis'):
                        gratings += [eval(key.strip('grating=').strip('_analysis'))]
            _keys = list(df['grating=%s' % gratings[0][1]].keys())
            data = df['grating=%s/%s' % (gratings[0][1], _keys[0])][...]
            json_dict = dict(detector_size=data.shape[-1], dispersion=dict(), offset=dict())
            for key in gratings:
                group = df['grating=%s_analysis' % (key, )]
                for name in ['dispersion', 'offset']:
                    json_dict[name][str(key[0])] = list(group[name][...])
        return json_dict

