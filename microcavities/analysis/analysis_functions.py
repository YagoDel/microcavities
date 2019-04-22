# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter
from lmfit.models import LorentzianModel, GaussianModel, ConstantModel
from scipy.optimize import fsolve
import pymsgbox
import matplotlib.pyplot as plt


def gui_checkplot():
    reply = pymsgbox.confirm('Are you happy to continue?', 'Check plot', ["Yes", 'No'])
    if reply == 'No':
        raise RuntimeError("Unsatisfactory plotting")

def roi(data, apply_function, roi_params=None, axes=-1):

    # if roi_params is None:
    #     popup a window with a plot and a button. Plot a random set of data and allow the button to change it.
    #     popup a ROI and allow the user to move it. 1D or 2D
    #     Also add a checkbox to the GUI to either lock the ROI to image pixels or not
    #
    # iterate over data, select a roi and apply_function
    # roid = pyqtgraph.affineSlice(data, shape, origin, vectors, axes, order=1, returnCoords=False, **kargs)
    # np.apply_over_axes(apply_function, roid, axes)

    return 1


def roi_intensity(data, roi_params=None):
    return np.sum(data), [roi_params], {}


def fit_gaussian(data):
    return np.sum(data, -1), [], {}


def dispersion_k0(image, k_axis=None, energy_axis=None, plotting=False,
                  known_sample_parameters=None):
    """Finds polariton energy and mass. If possible, also finds detuning.

    If given, energies should be in meV and wavevectors in inverse micron.

    :param ndarray image: 2d array. Backgrounded dispersion PL data. First axis is k (order irrelevant), second axis is energy (high-energy at small pixels).
    :param ndarray or float k_axis: optional. If a float, the pixel to wavevector calibration. If a 1d array, the wavevector values. If not given, results will be in "pixel" units. Defaults to None.
    :param tuple energy_axis: optional. 1d array of energy values. If not given, results will be in "pixel" units. Defaults to None.
    :param bool plotting: whether to pop-up a GUI for checking fitting is working correctly. Defaults to False.
    :param dict known_sample_parameters: if given, should contain at least two keys: exciton_energy and coupling. They will be used for returning a detuning. Defaults to None.
    :return: Description of returned object.
    :rtype: type
    """

    hbar = 0.658 # in meV*ps
    c = 300  # in um/ps

    if energy_axis is None:
        energy_axis = np.arange(image.shape[1])[::-1]

    # peak_pos gives the energy of the maximal emission as a function of k
    peak_pos = []
    for row in image:
        peak_pos += [np.average(energy_axis, weights=row)]
    peak_pos = np.array(peak_pos)
    peak_pos[peak_pos < 0] = 0
    peak_pos[peak_pos > image.shape[1]] = 0

    if k_axis is None or isinstance(k_axis, float):
        # peak_pos is generally noisy, so filter it and find an approximate k=0 value by finding the minimum energy
        filtered = savgol_filter(peak_pos, 11, 2)
        approx_k0 = np.argmin(filtered)
        # Fit a quadratic around the approximate k=0 and find a more accurate value for the k=0 pixel
        pixels = np.arange(image.shape[0])
        fitting_data = peak_pos[approx_k0 - 80 : approx_k0 + 80]
        fitting_x =  pixels[approx_k0 - 80 : approx_k0 + 80]
        quad_fit = np.polyfit(fitting_x, fitting_data, 2)
        fitted_k0 = - quad_fit[1] / (2 * quad_fit[0])

        if plotting:
            fig, ax = plt.subplots(1, 1)
            ax.plot(pixels, peak_pos)
            ax.plot(fitting_x, np.poly1d(quad_fit)(fitting_x))
            gui_checkplot()

        if k_axis is None:
            k_axis = pixels - fitted_k0
        else:
            k_axis = (pixels - fitted_k0) * k_axis


    k0_idx = int(np.argmin(np.abs(k_axis)))
    k0_spectra = image[k0_idx]

    # Finding the mass
    fitting_x = k_axis[k0_idx-100:k0_idx+100]
    fitting_y = peak_pos[k0_idx-100:k0_idx+100]
    quad_fit = np.polyfit(fitting_x, fitting_y, 2)
    a = np.abs(quad_fit[0])  # meV * um**2
    mass = hbar**2 / (2 * a)  # (meV * ps)**2 / meV * um**2  = meV * ps**2 / um **2
    mass *= c**2  # for meV / c**2 units
    mass /= 10**9  # for MeV / c**2
    mass /= 0.511  # for ratio with free electron mass

    # Fitting a (single) Lorentzian to the spectra
    model = LorentzianModel() + ConstantModel()

    # Guessing initial parameters for the fit
    center = np.argmax(k0_spectra)
    bkg = np.mean(k0_spectra[10:100])

    minima = np.argsort(np.abs(k0_spectra - k0_spectra[496] / 2))
    minimum_1 = minima[0]
    for dum in minima[1:]:
        if np.abs(dum - minimum_1) > 5:
            minimum_2 = dum
            break
    width = np.abs(minimum_1 - minimum_2) / 2
    ampl = np.pi * width * (k0_spectra[center] - bkg)

    # Using guess and plotting (if wanted)
    params_guess = model.make_params(sigma=width, center=center, amplitude=ampl, c=bkg)
    result = model.fit(k0_spectra, params_guess, x=energy_axis)
    if plotting:
        fig, ax = plt.subplots(1, 1)
        ax.plot(energies, k0_spectrum)
        ax.plot(energies, result.init_fit)
        ax.plot(energies, result.best_fit)
        gui_checkplot()

    energy = energy_axis[int(result.best_values['center'])]
    lifetime = hbar / energy_axis[int(result.best_values['sigma'])]

    return energy, lifetime, mass
