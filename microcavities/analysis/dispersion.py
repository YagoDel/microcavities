# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
from microcavities.experiment.utils import spectrometer_calibration, magnification
from scipy.ndimage import gaussian_filter
from nplab.utils.log import create_logger
from microcavities.analysis.utils import find_smooth_region
from lmfit.models import LorentzianModel, ConstantModel

LOGGER = create_logger("dispersion")


def guess_peak(data, xaxis=None, width_lims=(5, 0.001)):
    """Peak property guessing

    Guesses the background, peak height, peak position and FHWM. Used to initialise a fitting procedure

    :param data:
    :param xaxis:
    :param width_lims:
    :return:
    """
    # Guessing initial parameters for a fit
    if xaxis is None:
        xaxis = list(range(len(data)))
    center_idx = np.argmax(data)
    center = xaxis[center_idx]
    bkg = np.percentile(data, 10)  # np.mean(data[10:100])

    minima = np.argsort(np.abs(data - bkg - (data[center_idx] - bkg) / 2))
    minimum_1 = minima[0]
    minimum_2 = -1
    for dum in minima[1:]:
        if np.abs(dum - minimum_1) > width_lims[0]:
            minimum_2 = dum
            break
    width = np.min([np.abs(xaxis[minimum_1] - xaxis[minimum_2]) / 2, width_lims[1]])
    ampl = np.pi * width * (data[center_idx] - bkg)

    return dict(amplitude=ampl, sigma=width, center=center, background=bkg)


def fit_energy(spectra, energy_axis, model=None, guess_kwargs=None):
    """

    :param spectra: 1D array
    :param energy_axis:
    :param model: lmfit.Model. Needs to have a sigma, center, amplitude and c parameters
    :param guess_kwargs: dict.
    :return:
    """
    if model is None:
        model = LorentzianModel() + ConstantModel()
    if guess_kwargs is None: guess_kwargs = dict()
    my_guess = guess_peak(spectra, energy_axis, **guess_kwargs)
    params_guess = model.make_params(sigma=my_guess['sigma'], center=my_guess['center'],
                                     amplitude=my_guess['amplitude'], c=my_guess['background'])
    return model.fit(spectra, params_guess, x=energy_axis)


def make_dispersion_axes(image, wavevector_axis=('rotation_pvcam', 'k_space'),
                         energy_axis=('rotation_acton', 780, '2'), energy_roi=None):
    # Getting energy_axis from calibration files, if not given
    if energy_roi is None: energy_roi = (0, image.shape[1])
    if energy_axis is None:
        energy_axis = np.arange(image.shape[1])
    elif image.shape[1] != len(energy_axis):
        try:
            wvls = spectrometer_calibration(*energy_axis)
        except Exception as e:
            LOGGER.debug('Failed energy_axis: ', energy_axis)
            raise e
        energy_axis = 1240000 / wvls  # default units are meV
    energy_axis = energy_axis[energy_roi[0]:energy_roi[1]]

    # Getting wavevector_axis from calibration files, if not given
    k0 = find_k0(image)
    if wavevector_axis is None:
        wavevector_axis = np.arange(image.shape[0]) - k0
    elif image.shape[0] != len(wavevector_axis):
        mag = magnification(*wavevector_axis)[0]
        mag *= 1e-6  # default units are micron
        wavevector_axis = (np.arange(image.shape[0], dtype=np.float) - k0) * mag

        # Safety check to make sure that the fitted k~0 is near the given k~0
        if np.abs(k0 - np.argmin(np.abs(wavevector_axis))) > 3:
            LOGGER.warn("Fitted bottom of the dispersion occurs at k=%g not at k=0" % wavevector_axis[int(k0)])

    return wavevector_axis, energy_axis


def fit_quadratic_dispersion(image, energy=None, wavevector=None, plotting=None, max_fit_k=None):
    """Quadratic fit of a dispersion image. Used in find_mass and find_k0

    Finds peaks along the momentum axis and then fits a quadratic to it

    :param image: 2D np.ndarray
        Dispersion PL data. First axis is k (order irrelevant), second axis is energy (high-energy at small pixels).
    :param energy: 1D np.ndarray
        optional (the default is None). 1d array of energy values. If not given, results will be in "pixel" units.
    :param wavevector: 1D np.ndarray or float
        optional (the default is None). If not given, results will be in "pixel" units.
    :param plotting:
    :param max_fit_k:
    :return:
    """
    if wavevector is None:
        wavevector = np.arange(image.shape[0])
    if energy is None:
        energy = np.arange(image.shape[1])
    if max_fit_k is None:
        max_fit_idx = image.shape[0]//4  # works for most dispersions
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
            fig, ax = plotting
        except:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        pcolormesh(image.transpose(), ax, wavevector, energy, diverging=False, cbar=False, cmap='Greys')
        ax.plot(wavevector, energies, '--', lw=0.7)
        ax.plot(fitting_x, np.poly1d(quad_fit)(fitting_x), '--', lw=1)
    return quad_fit


def find_k0(image):
    """Finds the pixel value of the k~0 polariton by finding the minima in the dispersion curve

    :param image: 2D np.ndarray
        Dispersion PL data. First axis is k (order irrelevant), second axis is energy (high-energy at small pixels).
    :return:
    """
    quad_fit = fit_quadratic_dispersion(image)
    fitted_k0_pixel = - quad_fit[1] / (2 * quad_fit[0])
    return fitted_k0_pixel


def find_mass(image, energy_axis=('rotation_acton', 780, '2'), wavevector_axis=('rotation_pvcam', 'k_space'), plotting=None):
    """Measuring the mass (in physical units) by quadratically fitting the bottom of a dispersion

    :param image: 2D np.ndarray
        Dispersion PL data. First axis is k (order irrelevant), second axis is energy (high-energy at small pixels).
    :param energy_axis: 1D np.ndarray
        optional (the default is None). 1d array of energy values. If not given, results will be in "pixel" units.
    :param wavevector_axis: 1D np.ndarray or float
        optional (the default is None). If not given, results will be in "pixel" units.
    :param plotting: bool
    :return:
    """

    hbar = 0.658  # in meV*ps
    c = 300  # in um/ps

    quad_fit = fit_quadratic_dispersion(image, energy_axis, wavevector_axis, plotting)

    a = np.abs(quad_fit[0])  # meV * um**2
    mass = hbar**2 / (2 * a)  # (meV*ps)**2 / meV * um**2  = meV*ps**2 / um**2
    mass *= c**2  # for meV / c**2 units
    mass /= 10**9  # for MeV / c**2
    mass /= 0.511  # for ratio with free electron mass

    return mass


def dispersion(image, k_axis=None, energy_axis=None, plotting=True,
               known_sample_parameters=None):
    """Finds polariton energy, mass and lifetime. If possible, also finds detuning.

    If given, energies should be in eV or meV and wavevectors in inverse micron.
    Can fail if the upper polariton is also present in image.

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
    k_axis, energy_axis = make_dispersion_axes(image, k_axis, energy_axis)

    # Fitting the linewidth
    k0_spectra = image[np.argmin(np.abs(k_axis))]
    result = fit_energy(k0_spectra, energy_axis)
    if plotting:
        fig, axs = plt.subplots(1, 2, figsize=(7, 6))
        axs[1].plot(energy_axis, k0_spectra)
        axs[1].plot(energy_axis, result.init_fit, '--')
        axs[1].plot(energy_axis, result.best_fit)
        _plotting = (fig, (axs[0]))
    else:
        _plotting = None

    # Fitting the mass
    mass = find_mass(image, energy_axis, k_axis, _plotting)

    # Getting return values in physically useful units
    energy = result.best_values['center']
    hbar = 0.658  # in meV*ps
    lifetime = hbar / (2 * result.best_values['sigma'])

    results = (energy, lifetime, mass)
    args = ()
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


def _up_low_polariton(exciton, photon, rabi_splitting):
    average_energy = (exciton + photon) / 2
    detuning = exciton - photon
    offset = np.sqrt(detuning**2 + rabi_splitting**2) / 2

    up = average_energy + offset
    low = average_energy - offset
    return low, up


def exciton_photon_dispersions(k_axis, photon_energy, rabi_splitting, photon_mass, exciton_energy, exciton_mass,
                               k_offset=0, for_fit=True):
    hbar = 0.658  # in meV*ps
    c = 300  # in um/ps

    exciton_mass *= (0.511 * 1e9) / c**2
    photon_mass *= (0.511 * 1e9) / c**2

    exciton_dispersion = exciton_energy + (hbar * (k_axis+k_offset))**2 / (2*exciton_mass)
    photon_dispersion = photon_energy + (hbar * (k_axis+k_offset))**2 / (2*photon_mass)

    lower_p, upper_p = _up_low_polariton(exciton_dispersion, photon_dispersion, rabi_splitting)
    if for_fit:
        return lower_p, upper_p
    else:
        return lower_p, upper_p, exciton_dispersion, photon_dispersion
