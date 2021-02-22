import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from microcavities.utils import square, get_data_path
from microcavities.analysis.utils import normalize
import h5py
import os

camera_pointer = pvcam
spectrometer_pointer = spectrometer
minimum_wavelength = 300
maximum_wavelength = 800
wvl_range = None


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
tst = ReferenceLamp(XeNe_peaks)
# tst._plot_found_doublets(0.2)
tst._plot_relevant_regions(5)


class SpectrometerDispersion(object):
    def __init__(self, spectrometer, camera):
        self.spectrometer = spectrometer
        self.camera = camera

    def _measure(self, wavelength):
        self.spectrometer.wavelength = wavelength
        return auto_exposed_capture(self.camera)

    def measure_continuous_range(self, min_wvl, max_wvl, wvl_step=None):
        if wvl_step is None:
            print("Estimating wavelength step from current peaks")
            wvl_step = self.estimate_wavelength_range()
        measured_wavelengths = np.arange(min_wvl, max_wvl, wvl_step)
        spectra = []
        for wvl in measured_wavelengths:
            self._measure(wvl)
        return spectra, measured_wavelengths

    def estimate_wavelength_range(self, wavelength_step=0.1, depth=0, *args, **kwargs):
        """
        Wherever you measure it, it's only really valid in that wavelength region (that's the whole point of the calibration)
        :param wavelength_step:
        :param args:
        :param kwargs:
        :return:
        """
        if 'distance' not in kwargs:
            kwargs['distance'] = 5
        img0 = self.camera.raw_image(False, True)
        spec0 = normalize(np.mean(img0, 0))
        wvl0 = self.spectrometer.wavelength
        self.spectrometer.wavelength = wvl0 + wavelength_step
        img1 = camera_pointer.raw_image(False, True)
        spec1 = normalize(np.mean(img1, 0))
        indx_peaks1 = find_peaks(spec0, *args, **kwargs)[0]
        indx_peaks2 = find_peaks(spec1, *args, **kwargs)[0]
        if len(indx_peaks1) != len(indx_peaks2):
            if depth > 4:
                print('Too many recursions. Find a spectrum area with fewer peaks, or reduce the wavelength step')
                return np.nan
            self.estimate_wavelength_range(wavelength_step, depth + 1, *args, **kwargs)
        # print(indx_peaks1, indx_peaks2, indx_peaks1 - indx_peaks2)
        px_to_wvl = np.mean(indx_peaks1 - indx_peaks2)/wavelength_step
        return img0.shape[1] / px_to_wvl

    @property
    def wavelength_range(self):
        if self._wavelength_range is None:
            self._wavelength_range = self.estimate_wavelength_range()
        return self._wavelength_range

    def compare_to_reference(self, central_wavelengths, wavelength_range=None, data=None, lamp=None):
        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        if data is None:
            data = []
            for wvl in central_wavelengths:
                data += [self._measure(wvl)]
            data = np.array(data)
        if lamp is None:
            lamp = ReferenceLamp(XeNe_peaks)
        spectrum = lamp.spectrum

        rows, cols = square(len(central_wavelengths))
        fig, axs = plt.subplots(rows, cols, gridspec_kw=dict(0.01, 0.01))
        for r in range(rows):
            for c in range(cols):
                indx = r*cols + c
                y = data[indx]
                c_wvl = central_wavelengths[indx]
                x = np.linspace(c_wvl - wavelength_range/2, c_wvl + wavelength_range/2, y.shape)
                axs[r, c].plot(x, y)

                indxs = np.logical_and(spectrum[0] > c_wvl - wavelength_range/2,
                                       spectrum[0] < c_wvl + wavelength_range/2)
                x = spectrum[0][indxs]
                y = spectrum[1][indxs]
                axs[r, c].plot(x, y)
        return fig, axs


def auto_exposed_capture(camera, thresholds=(50000, 60000), percentile=100, max_exposure=30, reduction_fraction=10, depth=0):
    if depth > 10:
        raise RecursionError('Auto-exposure failed after 10 recursions')
    img = camera.raw_image()
    spectra = np.mean(img, 0)
    max_value = np.percentile(spectra, percentile)
    if depth == -1:
        # This handles the special case where the exposure is maximum
        return spectra
    elif max_value > thresholds[1]:
        return auto_exposed_capture(camera, thresholds, max_exposure, reduction_fraction, depth+1)
    elif max_value < thresholds[0]:
        exp = camera.exposure
        new_exposure = np.mean(thresholds) * exp / max_value
        if new_exposure > max_exposure:
            print('Required exposure %g s is more than the maximum %g s. Setting to maximum.' % (new_exposure, max_exposure))
            camera.exposure = max_exposure
            depth = -2
        else:
            camera.exposure = new_exposure
        return auto_exposed_capture(camera, thresholds, max_exposure, reduction_fraction, depth+1)
    else:
        return spectra



