# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, uic
from nplab.utils.log import create_logger
from nplab.utils.show_gui_mixin import ShowGUIMixin
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from lmfit.models import LorentzianModel, ConstantModel
import pyqtgraph as pg
import pymsgbox
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LOGGER = create_logger("analysis_functions")


cdict = {'red': [(0.0, 0.0, 1.0),
                 (0.25, 0.0, 0.0),
                 (0.5, 1.0, 1.0),
                 (0.75, 1.0, 1.0),
                 (1.0, 0.0, 0.0)],
         'green': [(0.0, 0.0, 1.0),
                   (0.25, 0.4, 0.4),
                   (0.5, 1.0, 1.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)],
         'blue': [(0.0, 1.0, 1.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]}
mycmap = LinearSegmentedColormap('Michael', cdict, 256)
plt.register_cmap(cmap=mycmap)


def gui_checkplot():
    plt.show()
    reply = pymsgbox.confirm('Are you happy to continue?',
                             'Check plot',
                             ["Yes", 'No'])
    if reply == 'No':
        raise RuntimeError("Unsatisfactory plotting")


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


def guess_peak(data, xaxis=None):

    # if x is None:
    #     x = np.arange(len(data))
    #
    # # Fitting a (single) Lorentzian to the spectra
    # model = LorentzianModel() + ConstantModel()

    # Guessing initial parameters for a fit
    if xaxis is None:
        xaxis = list(range(len(data)))
    center_idx = np.argmax(data)
    center = xaxis[center_idx]
    bkg = np.percentile(data, 10)  # np.mean(data[10:100])

    minima = np.argsort(np.abs(data - bkg - (data[center_idx] - bkg) / 2))
    minimum_1 = minima[0]
    for dum in minima[1:]:
        if np.abs(dum - minimum_1) > 5:
            minimum_2 = dum
            break
    width = np.min([np.abs(xaxis[minimum_1] - xaxis[minimum_2]) / 2, 0.001])
    ampl = np.pi * width * (data[center_idx] - bkg)

    return dict(amplitude=ampl, sigma=width, center=center, background=bkg)


# DISPERSION ANALYSIS FUNCTIONS
# Functions to be used on k-space spectra images

def find_k0(image):
    quad_fit = fit_quadratic_dispersion(image)
    fitted_k0_pixel = - quad_fit[1] / (2 * quad_fit[0])
    return fitted_k0_pixel


def find_mass(image, energy, wavevector, plotting=None):
    # Finding the mass (in physical units) by quadratically fitting the bottom
    # of the dispersion. We do not want to include high-k values in the fitting
    # so we limit the fitting to at most a 141 pixel wide region, which we
    # found phenomenologically
    hbar = 0.658  # in meV*ps
    c = 300  # in um/ps
    mass_conversion_factor = (hbar * c) ** 2  # for meV / c**2 units
    mass_conversion_factor /= 10 ** 9  # for MeV / c**2
    mass_conversion_factor /= 0.511  # for ratio with free electron mass

    quad_fit = fit_quadratic_dispersion(image, energy, wavevector, plotting)

    a = np.abs(quad_fit[0])  # meV * um**2
    mass = 1 / (2 * a)
    mass *= mass_conversion_factor
    mass = hbar**2 / (2 * a)  # (meV*ps)**2 / meV * um**2  = meV*ps**2 / um**2
    mass *= c**2  # for meV / c**2 units
    mass /= 10**9  # for MeV / c**2
    mass /= 0.511  # for ratio with free electron mass

    return mass


def fit_quadratic_dispersion(image, energy=None, wavevector=None, plotting=None, max_fit_k=None):

    if wavevector is None:
        wavevector = np.arange(image.shape[0])
    if energy is None:
        energy = np.arange(image.shape[1])
    if max_fit_k is None:
        max_fit_idx = image.shape[0]//4
    else:
        max_fit_idx = int(max_fit_k / np.mean(np.diff(wavevector)))

    # Smoothens the image to find the location of of the dispersion curve by simple peak finding
    peak_pos = np.argmax(gaussian_filter(image, 1), 1)
    energies = energy[peak_pos]
    # low SNR means not all the peak finding will correspond to the dispersion. We get around that by looking at the
    # largest smooth region that is still quadratic.
    smooth_idx, region = find_smooth_region(peak_pos)
    k0_idx = int(np.mean(smooth_idx))
    if np.diff(smooth_idx) > max_fit_idx:
        fitting_x = wavevector[k0_idx - max_fit_idx//2:k0_idx + max_fit_idx//2]
        fitting_y = energies[k0_idx - max_fit_idx//2:k0_idx + max_fit_idx//2]
    else:
        fitting_x = wavevector[smooth_idx[0]:smooth_idx[1]]
        fitting_y = energies[smooth_idx[0]:smooth_idx[1]]

    quad_fit = np.polyfit(fitting_x, fitting_y, 2)
    if plotting is not None:
        try:
            fig, axs = plotting
        except:
            fig, axs = plt.subplots(1, 2, figsize=(7, 6))
        axs[0].imshow(image.transpose())
        axs[1].plot(wavevector, energies)
        axs[1].plot(fitting_x, np.poly1d(quad_fit)(fitting_x))
    return quad_fit


def dispersion(image, k_axis=None, energy_axis=None, plotting=True,
               known_sample_parameters=None):
    """Finds polariton energy and mass. If possible, also finds detuning.

    If given, energies should be in meV and wavevectors in inverse micron.

    Parameters
    ----------
    image : 2D np.ndarray
        Dispersion PL data. First axis is k (order irrelevant), second axis is
        energy (high-energy at small pixels).
    k_axis : 1D np.ndarray or float
        optional (the default is None). If a float, the pixel to wavevector
        calibration. If a 1d array, the wavevector values. If not given,
        results will be in "pixel" units.
    energy_axis : 1D np.ndarray
        optional (the default is None). 1d array of energy values. If not
        given, results will be in "pixel" units.
    plotting : bool
        Whether to pop-up a GUI for checking fitting is working correctly (the
        default is True).
    known_sample_parameters : dict
        Should contain at least two keys: exciton_energy and coupling. They
        will be used for returning a detuning (the default is None).

    Returns
    -------
    tuple
        Energy, lifetime, mass.
    list
        Updated args that can be passed to the next function call
    dict
        Updated kwargs that can be passed to the next function call

    """
    hbar = 0.658  # in meV*ps
    # c = 300  # in um/ps

    if energy_axis is None:
        energy_axis = np.arange(image.shape[1])
    elif np.mean(energy_axis) < 100:
        # Transforming it into meV
        energy_axis *= 1e3

    fitted_k0_pixel = find_k0(image)
    LOGGER.debug('Center pixel: %d' % fitted_k0_pixel)

    k_pixels = np.arange(image.shape[0])
    if k_axis is None or isinstance(k_axis, float):
        k0_idx = int(fitted_k0_pixel)
        if k_axis is None:
            k_axis = k_pixels - fitted_k0_pixel
        else:
            k_axis = (k_pixels - fitted_k0_pixel) * k_axis
    else:
        k0_idx = int(np.argmin(np.abs(k_axis)))
        if np.abs(k0_idx - fitted_k0_pixel) > 3:
            LOGGER.warn("Fitted bottom of the dispersion occurs at k=%g not at k=0" % k_axis[int(fitted_k0_pixel)])

    k0_spectra = image[k0_idx]
    model = LorentzianModel() + ConstantModel()
    my_guess = guess_peak(k0_spectra, energy_axis)
    params_guess = model.make_params(sigma=my_guess['sigma'], center=my_guess['center'],
                                     amplitude=my_guess['amplitude'], c=my_guess['background'])
    result = model.fit(k0_spectra, params_guess, x=energy_axis)

    if plotting:
        fig, axs = plt.subplots(1, 3, figsize=(7, 6))
        vmin = np.percentile(image, 0.1)
        vmax = np.percentile(image, 99.9)
        axs[0].imshow(image.transpose(), vmin=vmin, vmax=vmax)
        axs[2].plot(energy_axis, k0_spectra)
        axs[2].plot(energy_axis, result.init_fit, '--')
        axs[2].plot(energy_axis, result.best_fit)
        mass = find_mass(image, energy_axis, k_axis, (fig, (axs[:2])))

        # gui_checkplot()

    energy = result.best_values['center']  # * 1e-3  # energy_axis[int(result.best_values['center'])]
    lifetime = hbar / (2 * result.best_values['sigma'])

    results = (energy, lifetime, mass)
    args = ()  # (k_axis, energy_axis, plotting, known_sample_parameters)
    kwargs = dict(plotting=plotting)

    if known_sample_parameters is not None:
        known_sample_parameters['polariton_mass'] = mass
        try:
            exciton_fraction, _ = hopfield_coefficients(**known_sample_parameters)
            results += (exciton_fraction, )
        except Exception as e:
            print(e)
            pass
    return results, args, kwargs


def hopfield_coefficients(rabi_splitting=None, detuning=None, exciton_energy=None, photon_energy=None,
                          exciton_mass=None, photon_mass=None, polariton_mass=None):
    if rabi_splitting is not None:
        if detuning is None:
            detuning = exciton_energy - photon_energy
        exciton_fraction = 0.5 * (1 + detuning / np.sqrt(detuning**2 + rabi_splitting**2))
    elif exciton_mass is not None:
        exciton_fraction = (exciton_mass * (photon_mass - polariton_mass)) / (polariton_mass * (photon_mass - exciton_mass))
    else:
        raise ValueError('Need to give either energies or masses')
    photon_fraction = 1 - exciton_fraction
    return exciton_fraction, photon_fraction

# IMAGE ANALYSIS
# Functions to be used on real- and k-space images

class roi2d_GUI(QtWidgets.QMainWindow):
    """ Multi region-of-interest GUI

    Base class for creating GUIs for analysing images where you want to create
    multiple ROIs and extract information about those ROIs
    """

    def __init__(self, images, results, **kwargs):
        super(roi2d_GUI, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'roi_gui.ui'), self)

        self.images = images
        self.results = results
        self.kwargs = dict(kwargs)

        self.spinBoxNoROI.valueChanged.connect(self._make_ROIs)
        self.pushButtonSetROI.clicked.connect(self.SetROIs)
        self.pushButtonAutoMove.clicked.connect(self._move_ROIs)
        self.pushButtonNewImage.clicked.connect(self.new_image)

        if len(self.images.shape) <= 2:
            self.pushButtonNewImage.hide()
            self._current_image = self.images
            self.graphicsView.setImage(self.images)

        self.rois = []

    def _make_single_ROI(self, pen):
        print("ROI type: ", self.comboBox_roitype.text())
        if self.comboBox_roitype.text() == 'Square':
            roi = pg.ROI([50, 10], [3, 3], pen=pen)
            roi.addScaleHandle([1, 0.5], [0.5, 0.5])
            roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            roi.addRotateHandle([0, 0], [0.5, 0.5])
            return roi
        if self.comboBox_roitype.text() == 'Round':
            return pg.CircleROI([10, 10], [3, 3], pen=pen)

    def _make_ROIs(self):
        n_rois = self.spinBoxNoROI.value()

        for roi in self.rois:
            self.graphicsView.getView().removeItem(roi)

        self.rois = []
        for n in range(n_rois):
            pen = pg.mkPen(n)
            self.rois += [self._make_single_ROI(pen)]

        for roi in self.rois:
            self.graphicsView.getView().addItem(roi)

    def SetROIs(self):
        affineSliceParams = []
        for roi in self.rois:
            affineSliceParams += [roi.getAffineSliceParams(self._current_image,
                                                           self.graphicsView.getImageItem())]

        self.results['affineSliceParams'] = affineSliceParams

    def _move_ROIs(self):
        pass

    def new_image(self):
        self._index = np.random.randint(0, self.images.shape[0])
        self._current_image = self.images[self._index]
        self.graphicsView.setImage(self.images[self._index])


class roi2d(ShowGUIMixin):
    def __init__(self, images):
        super(roi2d, self).__init__()

        self.images = images
        self.results = dict()

    def get_qt_ui(self):
        return roi2d_GUI(self.images, self.results)


# def roi2d(data, apply_function, roi_params=None, axes=-1):
#
#     if roi_params is None:
#         # popup a ROI and allow the user to move it
#         results = dict()
#         app = get_qt_app()
#         gui = roi2d_GUI(data, results)
#         gui.show()
#         app.exec_()
#
#         roi_params = results['affineSliceParams']
#     #     popup a window with a plot and a button. Plot a random set of data and allow the button to change it.
#     #     Also add a checkbox to the GUI to either lock the ROI to image pixels or not
#     #
#     # iterate over data, select a roi and apply_function
#     # roid = pyqtgraph.affineSlice(data, shape, origin, vectors, axes, order=1, returnCoords=False, **kargs)
#     # np.apply_over_axes(apply_function, roid, axes)
#
#     return 1


def roi_intensity(data, roi_params=None):
    return np.sum(data), [roi_params], {}
