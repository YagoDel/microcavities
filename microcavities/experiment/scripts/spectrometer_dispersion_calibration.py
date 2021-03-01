import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import gridspec
from microcavities.utils import square
from microcavities.analysis.utils import normalize
from nplab.utils.gui import get_qt_app


XeNe_peaks = np.asarray(
    [3417.9035, 3472.5711, 3515.1900, 3593.5263, 3600.1691, 4488.0926, 4636.125, 4837.3139, 5005.1587, 5031.3504,
     5104.7011, 5113.6724, 5144.9384, 5188.6122, 5330.7775, 5341.0938, 5360.0121, 5400.5617, 5562.7662, 5656.5664,
     5689.8163, 5719.2248, 5748.2985, 5764.4188, 5804.4496, 5820.1558, 5852.4878, 5881.895, 5944.8342, 5975.534,
     6029.9971, 6074.3377, 6096.1631, 6128.4499, 6143.0626, 6163.5939, 6217.2812, 6266.495, 6304.789, 6334.4278,
     6382.9917, 6402.246, 6506.5281, 6532.8822, 6598.9529, 6678.2764, 6717.043, 6929.4673, 7024.0504, 7032.4131,
     7173.9381, 7245.1666, 7438.899, 7488.8712, 7535.7739, 8136.4057, 8300.3263, 8377.6065, 8495.3598, 8591.2583,
     8634.647, 8654.3831, 8655.522, 8679.493, 8681.921, 8704.111, 8771.656, 8780.621, 8783.75, 8830.907, 8853.867,
     8919.5007, 9148.672, 9201.759, 9300.853, 9326.507, 9425.379, 9486.68, 9534.163, 9665.424, 10798.12, 10844.54,
     11143.02])
XeNe_peaks /= 10


class ReferenceLamp(object):
    def __init__(self, peaks):
        self.reference_peaks = peaks
        self._spectrum = None

        distances = np.diff(self.reference_peaks)
        width = np.min(distances) / 5
        self.wavelength_range = [np.min(self.reference_peaks) - 5*width, np.max(self.reference_peaks) + 5*width]

    @property
    def spectrum(self):
        if self._spectrum is None:
            self._spectrum = np.array(self.make_spectrum())
        return self._spectrum

    def _find_doublets(self, resolution=0.01):
        # wavelengths, spectrum = self.spectrum
        indices = np.argwhere(np.diff(self.reference_peaks) < resolution)
        doublet_pairs = np.array([self.reference_peaks[indices],
                                  self.reference_peaks[indices + 1]])[..., 0]
        doublet_center = np.mean(doublet_pairs, 0)
        return doublet_center

    def _count_peaks(self, wvl_min, wvl_max, margin=0.0, resolution=0.0):
        """
        If any peak in the margin, return 0
        :param wvl_min:
        :param wvl_max:
        :param margin:
        :param resolution:
        :return:
        """
        n_margins = np.sum(np.logical_and(self.reference_peaks > wvl_min - resolution,
                                          self.reference_peaks < wvl_min + margin)) + \
                    np.sum(np.logical_and(self.reference_peaks < wvl_max + resolution,
                                          self.reference_peaks > wvl_max - margin))
        if n_margins == 0:
            return np.sum(np.logical_and(self.reference_peaks > wvl_min + margin, self.reference_peaks < wvl_max - margin))
        else:
            return 0

    def get_relevant_central_wavelengths(self, wavelength_width, wavelength_range=None, margin=None, resolution=0.01):
        """Finds central wavelengths of spectrum that have more than one peak separated by clearly"""
        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        if margin is None:
            margin = 0.1 * wavelength_width
        # _doublets = np.array([full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution)],
        #                      full_wavelengths[np.argwhere(np.diff(all_peaks) < resolution) + 1]])[..., 0]
        # doublets = np.mean(_doublets, 0)

        peak_distances = np.diff(self.reference_peaks)
        step = np.min(peak_distances[peak_distances > resolution]) / 5  # Fairly arbitrary, just need to make sure we cover enough of the spectra
        _stepped_range = np.arange(wavelength_range[0], wavelength_range[1]-wavelength_width, step)
        # print(step, _stepped_range.shape)
        rolled_ranges = np.array([_stepped_range, _stepped_range + wavelength_width])
        # plt.figure(); plt.plot(rolled_ranges[0]); plt.plot(rolled_ranges[1])
        # print(rolled_ranges.shape, rolled_ranges[:5])
        # [plt.axvspan(x-wavelength_width, x+wavelength_width, color='r') for x in centers]

        # wavelengths, reference_spectrum = self.make_spectrum(wavelength_range)
        # step = int(wavelength_width / np.diff(wavelengths)[0])
        # rolled_ranges = np.array([reference_spectrum[:-step], reference_spectrum[step:]]).transpose()
        # center_roll = np.mean(rolled_ranges, 1)

        # Now that we have some rolled_ranges, we want to know how many peaks per range there are
        peaks_per_range = np.array([self._count_peaks(x[0], x[1], margin, resolution) for x in rolled_ranges.transpose()])
        # smooth_window = np.ones(20)
        # # smoothened = convolve(peaks_per_range, smooth_window, 'valid')
        # sos = signal.butter(10, step/20, 'lp', fs=step, output='sos')
        # filtered = signal.sosfilt(sos, peaks_per_range)
        #
        # plt.figure(); plt.plot(peaks_per_range); plt.plot(filtered)

        # print(peaks_per_range.shape)
        # plt.plot(np.mean(rolled_ranges, 0), peaks_per_range)
        # Ignore rolled_ranges that only have one peak
        peaks_per_range[peaks_per_range < 2] = 0
        # Find which of the rolled_ranges are local maxima in the number of peaks. To do that, we find the regions
        # bound by an increase and a decrease of the number of peaks
        n_peak_change = np.diff(peaks_per_range)
        _indexes = np.nonzero(n_peak_change)[0]  # indices where the number of peaks changes
        # print(_indexes.shape)
        _indexes = np.append(_indexes[0], _indexes[1:][np.diff(_indexes) > 5])  # makes sure transitions don't happen too near each other
        # print(_indexes.shape)
        _peak_change = np.diff(n_peak_change[_indexes])  # essentially the second derivative
        # plt.figure(); plt.plot(peaks_per_range)
        # plt.figure()
        _dummy = np.logical_and(self.spectrum[1] > wavelength_range[0], self.spectrum[1] < wavelength_range[1])
        # plt.plot(self.spectrum[0][_dummy], self.spectrum[1][_dummy])
        # plt.plot(np.mean(rolled_ranges, 0), peaks_per_range); plt.plot(np.mean(rolled_ranges, 0)[:-1], n_peak_change)
        # plt.plot(np.mean(rolled_ranges, 0)[_indexes][:-1], _peak_change)
        mask1 = _peak_change < -1
        # Now we find the center of these local maxima
        _indexes_left = _indexes[:-1][mask1]
        _indexes_right = _indexes[1:][mask1]
        region_centers = np.mean([rolled_ranges[0][_indexes_left], rolled_ranges[1][_indexes_right]], 0)
        # regions = np.array([center_roll[_indexes_left], center_roll[_indexes_right]]).transpose()
        # Ensure no doublets are in these
        doublets = self._find_doublets(resolution)
        _no_doublets = np.logical_not([np.any(np.logical_and(doublets > x - wavelength_width/2, doublets < x + wavelength_width/2)) for x in region_centers])
        return region_centers[_no_doublets]

    def make_spectrum(self, wavelength_range=None, resolution=None):
        distances = np.diff(self.reference_peaks)
        width = np.min(distances) / 5

        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        if resolution is None:
            resolution = width / 10
        wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)
        spectrum = np.zeros(len(wavelengths))
        for peak in self.reference_peaks:
            spectrum += np.exp(-(wavelengths - peak) ** 2 / (2 * width ** 2))
        return wavelengths, spectrum

    def _plot_found_doublets(self, resolution=0.01):
        doublets = self._find_doublets(resolution)
        plt.figure()
        plt.plot(*self.spectrum)
        [plt.axvline(x, color='r') for x in doublets]

    def _plot_relevant_regions(self, wavelength_width, wavelength_range=None, margin=None, resolution=0.01):
        centers = self.get_relevant_central_wavelengths(wavelength_width, wavelength_range, margin, resolution)
        plt.figure()
        plt.plot(*self.spectrum)
        [plt.axvspan(x-wavelength_width/2, x+wavelength_width/2, color='r', alpha=0.1) for x in centers]


def auto_exposed_capture(camera, thresholds=(20000, 50000), saturation=60000, percentile=99.999, max_exposure=30, fast_exposure=0.5,
                         reduction_fraction=10, depth=0, exposure_property_name='Exposure', app=None):
    if depth > 10:
        raise ValueError('Auto-exposure failed after 10 recursions')
    if app is None:
        app = get_qt_app()
    if fast_exposure is not None:
        exp = getattr(camera, exposure_property_name)
        app.processEvents()
        if exp > fast_exposure:
            setattr(camera, exposure_property_name, fast_exposure)
            app.processEvents()

    spectra = camera.raw_image(False, True)
    app.processEvents()
    max_value = np.percentile(spectra, percentile)
    # print(max_value)

    if depth == -1:
        # This handles the special case where the exposure is maximum
        return spectra
    elif max_value > thresholds[1]:
        exp = getattr(camera, exposure_property_name)
        app.processEvents()
        if max_value >= saturation:
            new_exposure = exp / reduction_fraction
        else:
            new_exposure = (thresholds[1] - 0.1*np.diff(thresholds)) * exp / max_value
        setattr(camera, exposure_property_name, new_exposure)
        app.processEvents()
        return auto_exposed_capture(camera, thresholds, saturation, percentile, max_exposure, None, reduction_fraction, depth+1, exposure_property_name, app)
    elif max_value < thresholds[0]:
        exp = getattr(camera, exposure_property_name)
        app.processEvents()
        new_exposure = np.mean(thresholds) * exp / max_value
        if new_exposure > max_exposure:
            print('Required exposure %g s is more than the maximum %g s. Setting to maximum.' % (new_exposure, max_exposure))
            # camera.Exposure = max_exposure
            setattr(camera, exposure_property_name, max_exposure)
            app.processEvents()
            depth = -2
        else:
            # camera.Exposure = new_exposure
            setattr(camera, exposure_property_name, new_exposure)
            app.processEvents()
        return auto_exposed_capture(camera, thresholds, saturation, percentile, max_exposure, None, reduction_fraction, depth+1, exposure_property_name, app)
    else:
        return spectra


class SpectrometerDispersion(object):
    def __init__(self, spectrometer, camera, lamp=None, app=None):
        self.spectrometer = spectrometer
        self.camera = camera
        if app is None:
            app = get_qt_app()
        self.app = app
        self._wavelength_range = None

        if lamp is None:
            self.lamp = ReferenceLamp(XeNe_peaks)
            self.lamp.spectrum
        else:
            self.lamp = lamp

    def _measure(self, wavelength=None):
        if wavelength is not None:
            self.spectrometer.wavelength = wavelength
        spectra = auto_exposed_capture(self.camera)
        self.app.processEvents()
        return spectra

    def measure(self, wvl_limits=None, wvls=None, wvl_step=None):
        if wvls is None:
            if wvl_step is None:
                wvl_step = 0.8 * self.wavelength_range
            measured_wavelengths = np.arange(*wvl_limits, step=wvl_step)
        else:
            if wvl_limits is not None:
                measured_wavelengths = wvls[np.logical_and(wvls > wvl_limits[0], wvls < wvl_limits[1])]
            else:
                measured_wavelengths = wvls
        spectra = []
        for wvl in measured_wavelengths:
            spectra += [self._measure(wvl)]
        return np.array(spectra), measured_wavelengths

    def estimate_dispersion(self, wavelength_step=0.1, depth=0, *args, **kwargs):
        """
        Wherever you measure it, it's only really valid in that wavelength region (that's the whole point of the calibration)
        :param wavelength_step:
        :param args:
        :param kwargs:
        :return:
        """
        if 'distance' not in kwargs:
            kwargs['distance'] = 5
        if 'height' not in kwargs:
            kwargs['height'] = 0.1
        spec0 = normalize(self._measure())
        wvl0 = self.spectrometer.wavelength
        spec1 = normalize(self._measure(wvl0 + wavelength_step))
        indx_peaks1 = find_peaks(spec0, *args, **kwargs)[0]
        indx_peaks2 = find_peaks(spec1, *args, **kwargs)[0]
        # print(len(indx_peaks1), len(indx_peaks2))
        # print(indx_peaks1, indx_peaks2)
        if len(indx_peaks1) != len(indx_peaks2):
            if depth > 4:
                print('Too many recursions. Find a spectrum area with fewer peaks, or reduce the wavelength step')
                return np.nan
            self.estimate_wavelength_range(wavelength_step, depth + 1, *args, **kwargs)
        # print(indx_peaks1, indx_peaks2, indx_peaks1 - indx_peaks2)
        px_to_wvl = np.mean(indx_peaks1 - indx_peaks2)/wavelength_step
        return 1 / px_to_wvl, spec0.shape[0]

    def estimate_wavelength_range(self, wavelength_step=0.1, *args, **kwargs):
        return np.prod(self.estimate_dispersion(wavelength_step, 0, *args, **kwargs))

    @property
    def wavelength_range(self):
        if self._wavelength_range is None:
            self._wavelength_range = self.estimate_wavelength_range()
        return self._wavelength_range

    def auto_match_peaks(self, wavelength, spectra, lamp=None, depth=0, height_step=None, *args, **kwargs):
        if lamp is None:
            lamp = self.lamp
        n_known_peaks = lamp._count_peaks(wavelength.min(), wavelength.max())
        if depth > 10:
            return [np.nan] * n_known_peaks

        if 'distance' not in kwargs:
            kwargs['distance'] = 5
        if 'height' not in kwargs:
            kwargs['height'] = 0.5
        measured_peaks = find_peaks(normalize(spectra), *args, **kwargs)[0]
        print(kwargs['height'])
        print(len(measured_peaks), n_known_peaks)

        if len(measured_peaks) > n_known_peaks:
            if height_step is None:
                height_step = 0.25
            kwargs['height'] = kwargs['height'] + height_step
            return self.auto_match_peaks(wavelength, spectra, lamp, depth+1, height_step/2, *args, **kwargs)
        elif len(measured_peaks) < n_known_peaks:
            if height_step is None:
                height_step = 0.25
            kwargs['height'] = kwargs['height'] - height_step
            return self.auto_match_peaks(wavelength, spectra, lamp, depth + 1, height_step/2, *args, **kwargs)
        else:
            return measured_peaks

    def _single_measure_dispersion(self, wavelength, spectra=None, plot=None):
        if spectra is None:
            spectra = self._measure(wavelength)
        wvls = np.linspace(wavelength - self.wavelength_range / 2,
                           wavelength + self.wavelength_range / 2,
                           spectra.shape[0])
        fitted_indices = self.auto_match_peaks(wvls, spectra)
        _indxs2 = np.logical_and(self.lamp.reference_peaks > wavelength - self.wavelength_range / 2,
                                 self.lamp.reference_peaks < wavelength + self.wavelength_range / 2)
        lamp_peaks = self.lamp.reference_peaks[_indxs2] - wavelength
        meas_peaks = wvls[fitted_indices] - wavelength
        m = np.polyfit(lamp_peaks, meas_peaks, 1)

        if plot is not None:
            if plot is True:
                _, ax = plt.subplots(1, 1)
            else:
                ax = plot

            ax.plot(wvls, normalize(spectra))
            ax.plot(wvls[fitted_indices], normalize(spectra)[fitted_indices], 'x')

            ref_spectrum = self.lamp.spectrum
            indxs = np.logical_and(ref_spectrum[0] > wavelength - self.wavelength_range / 2,
                                   ref_spectrum[0] < wavelength + self.wavelength_range / 2)
            x = ref_spectrum[0][indxs]
            y = ref_spectrum[1][indxs]
            ax.plot(x, normalize(y))
        else:
            ax = None
        return m, lamp_peaks, meas_peaks, fitted_indices, ax

    def measure_dispersion(self, central_wavelengths=None, wavelength_limits=None, max_measurements=10, plot=False):
        if wavelength_limits is None:
            wavelength_limits = (self.lamp.reference_peaks.min(), self.lamp.reference_peaks.max())
        if central_wavelengths is None:
            central_wavelengths = self.lamp.get_relevant_central_wavelengths(self.wavelength_range, wavelength_limits)
        if len(central_wavelengths) > max_measurements:
            central_wavelengths = np.random.choice(central_wavelengths, max_measurements, False)

        if plot:
            rows, cols = square(len(central_wavelengths))
            fig = plt.figure(figsize=(cols*2*2, rows*2))
            gs0 = gridspec.GridSpec(1, 2)
            gs1 = gridspec.GridSpecFromSubplotSpec(rows, cols, gs0[0], hspace=0.01, wspace=0.01)
            gs2 = gridspec.GridSpecFromSubplotSpec(rows, cols, gs0[1], hspace=0.01, wspace=0.01)
            axs = []
        lamp_peaks = []
        measured_peaks = []
        dispersions = []
        for indx, wvl in enumerate(central_wavelengths):
            if plot:
                r = indx // cols
                c = indx % cols

                if rows > 1 and cols > 1:
                    ax = plt.subplot(gs1[r, c])
                    ax2 = plt.subplot(gs2[r, c])
                elif rows > 1:
                    ax = plt.subplot(gs1[r])
                    ax2 = plt.subplot(gs2[r])
                elif cols > 1:
                    ax = plt.subplot(gs1[c])
                    ax2 = plt.subplot(gs2[c])
                else:
                    ax = plt.subplot(gs1)
                    ax2 = plt.subplot(gs2)
                axs += [[ax, ax2]]
            else:
                ax = None
                ax2 = None

            m, _lamp_peaks, _meas_peaks, fitted_indices, ax = self._single_measure_dispersion(wvl, plot=ax)

            if _lamp_peaks is not None and plot:
                ax2.plot(_lamp_peaks, _meas_peaks, '.-')
                ax2.text(0.5, 0.99, str(m), transform=ax2.transAxes, va='top', ha='center')

        # # for r in range(rows):
        # #     for c in range(cols):
        #     if rows > 1 and cols > 1:
        #         ax = plt.subplot(gs1[r, c])
        #         ax2 = plt.subplot(gs2[r, c])
        #     elif rows > 1:
        #         ax = plt.subplot(gs1[r])
        #         ax2 = plt.subplot(gs2[r])
        #     elif cols > 1:
        #         ax = plt.subplot(gs1[c])
        #         ax2 = plt.subplot(gs2[c])
        #     else:
        #         ax = plt.subplot(gs1)
        #         ax2 = plt.subplot(gs2)

            # indx = r*cols + c
            # ax, _lamp_peaks, _meas_peaks, m = self._single_compare_to_reference(data[indx], central_wavelengths[indx], ax)

            # if _lamp_peaks is not None:
            #     ax2.plot(_lamp_peaks, _meas_peaks, '.-')
            #     ax2.text(0.5, 0.99, str(m), transform=ax2.transAxes, va='top', ha='center')
            lamp_peaks += [_lamp_peaks]
            measured_peaks += [_meas_peaks]
            dispersions += [m]

        if plot:
            fig, axs3 = plt.subplots(1, 2)
            axs3[0].plot(central_wavelengths, np.array(dispersions)[:, 0], '.-')
            axs3[1].plot(central_wavelengths, np.array(dispersions)[:, 1], '.-')
        return np.array(dispersions), np.array(lamp_peaks), np.array(measured_peaks)




#
#
# x = np.linspace(640 - cal.wavelength_range/2, 640 + cal.wavelength_range/2, 2048)
# y = cal._measure(640)
# indxs = cal.auto_match_peaks(x, y)
#
#
# plt.plot(x, normalize(y))
# plt.plot(x[indxs], normalize(y)[indxs], 'x')
# _indxs = np.logical_and(cal.lamp.spectrum[0] > 640 - cal.wavelength_range / 2,
#                         cal.lamp.spectrum[0] < 640 + cal.wavelength_range / 2)
# x2 = cal.lamp.spectrum[0][_indxs]
# y2 = cal.lamp.spectrum[1][_indxs]
# plt.plot(x2, normalize(y2))
#
# _indxs2 = np.logical_and(cal.lamp.reference_peaks > 640 - cal.wavelength_range / 2,
#                         cal.lamp.reference_peaks < 640 + cal.wavelength_range / 2)
# plt.figure()
# plt.plot(cal.lamp.reference_peaks[_indxs2], x[indxs], '.-')
#
# cal.compare_to_reference([640])
# cal.compare_to_reference(wvls[4:13])
# cal.estimate_wavelength_range()




