# -*- coding: utf-8 -*-
"""Utility functions to analyse low power dispersion images"""

from microcavities.analysis import *
from microcavities.analysis.utils import guess_peak
from microcavities.utils import depth
from microcavities.utils.polariton_characterisation import *
from microcavities.experiment.utils import spectrometer_calibration, magnification, load_calibration_file
from scipy.ndimage import gaussian_filter
from nplab.utils.log import create_logger
from microcavities.analysis.utils import find_smooth_region
from lmfit.models import LorentzianModel, ConstantModel
from functools import partial
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy

LOGGER = create_logger("dispersion")


def _up_low_polariton(exciton, photon, rabi_splitting):
    average_energy = (exciton + photon) / 2
    detuning = exciton - photon
    offset = np.sqrt(detuning**2 + rabi_splitting**2) / 2

    up = average_energy + offset
    low = average_energy - offset
    return low, up


def exciton_photon_dispersions(k_axis, photon_energy, rabi_splitting, photon_mass, exciton_energy, exciton_mass,
                               k_offset=0, for_fit=True, **kwargs):
    exciton_mass *= m_e
    photon_mass *= m_e

    exciton_dispersion = exciton_energy + (hbar * (k_axis+k_offset))**2 / (2*exciton_mass)
    photon_dispersion = photon_energy + (hbar * (k_axis+k_offset))**2 / (2*photon_mass)

    lower_p, upper_p = _up_low_polariton(exciton_dispersion, photon_dispersion, rabi_splitting)
    if for_fit:
        return lower_p, upper_p
    else:
        return lower_p, upper_p, exciton_dispersion, photon_dispersion


# Low-energy k~0 dispersion fitting

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


def make_dispersion_axes(image=None, k0=None, wavevector_axis=('rotation_pvcam', 'k_space'),
                         energy_axis=('rotation_acton', 780, '2'), energy_roi=None):
    # Getting energy_axis from calibration files, if not given
    if image is not None and energy_axis is None:
        # This is simply used for situations when we don't care about the axis
        energy_axis = np.arange(image.shape[1])
    else:
        try:
            wvls = spectrometer_calibration(*energy_axis)
            energy_axis = 1240000 / wvls  # default units are meV
        except Exception as e:
            pass
    if energy_roi is None: energy_roi = (0, len(energy_axis))
    energy_axis = energy_axis[energy_roi[0]:energy_roi[1]]

    # Getting wavevector_axis from calibration files, if not given
    if k0 is None:
        assert image is not None
        k0 = find_k0(image, plotting=False)
    if image is not None and wavevector_axis is None:
        wavevector_axis = np.arange(image.shape[0]) - k0
    else:  # image.shape[0] != len(wavevector_axis):
        try:
            calibration = load_calibration_file(wavevector_axis[0])
            mag = magnification(*wavevector_axis)[0]
            mag *= 1e-6  # default units are micron
            wavevector_axis = (np.arange(calibration['detector_shape'][1], dtype=np.float) - k0) * mag
        except TypeError as e:
            pass
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
    fig, ax = create_axes(plotting)
    if ax is not None:
        pcolormesh(image.transpose(), ax, wavevector, energy, diverging=False, cbar=False, cmap='Greys')
        ax.plot(wavevector, energies, '--', lw=0.7)
        ax.plot(fitting_x, np.poly1d(quad_fit)(fitting_x), '--', lw=1)
    return quad_fit


def find_k0(image, *args, **kwargs):
    """Finds the pixel value of the k~0 polariton by finding the minima in the dispersion curve

    :param image: 2D np.ndarray
        Dispersion PL data. First axis is k (order irrelevant), second axis is energy (high-energy at small pixels).
    :return:
    """
    quad_fit = fit_quadratic_dispersion(image, *args, **kwargs)
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

    quad_fit = fit_quadratic_dispersion(image, energy_axis, wavevector_axis, plotting)

    a = np.abs(quad_fit[0])  # meV * um**2
    mass = hbar**2 / (2 * a)  # (meV*ps)**2 / meV * um**2  = meV*ps**2 / um**2
    mass /= m_e  # for ratio with free electron mass

    return mass


def dispersion(image, k_axis=None, energy_axis=None, plotting=None, fit_kwargs=None):
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
    fit_kwargs : dict
        To be passed to fit_dispersion for extracting the parameters of the two oscillator model

    Returns
    -------
    tuple
        Energy, lifetime, mass.
    list
        Updated args that can be passed to the next function call
    dict
        Updated kwargs that can be passed to the next function call

    """
    k_axis, energy_axis = make_dispersion_axes(image, None, k_axis, energy_axis)

    # Fitting the linewidth
    k0_spectra = normalize(image[np.argmin(np.abs(k_axis))])
    result = fit_energy(k0_spectra, energy_axis)

    fig, axs = create_axes(plotting, (1, 2))

    if axs is not None:
        axs[1].plot(energy_axis, k0_spectra, '-', energy_axis, result.init_fit, '--', energy_axis, result.best_fit, '-')
        axs[1].set_xlim(result.best_values['center'] - 10*result.best_values['sigma'],
                        result.best_values['center'] + 10*result.best_values['sigma'])
        axs[1].text(0.95, 0.95, 'FWHM = %.2g meV' % (2*result.best_values['sigma']), transform=axs[1].transAxes,
                    ha='right', va='top')
        plotting = (fig, (axs[0]))  # to pass axes to find_mass

    # Fitting the mass
    mass = find_mass(image, energy_axis, k_axis, False)

    # Fitting the Rabi splitting and detuning
    if fit_kwargs is None: fit_kwargs = dict()
    defaults = dict(mode='lp', least_squares_kw=dict(max_nfev=5e4))
    fit_kwargs = {**defaults, **fit_kwargs}
    final_params, parameter_errors, res = fit_dispersion(image, k_axis, energy_axis, plotting, **fit_kwargs)
    LOGGER.debug('Final fit parameters: %s' % final_params)

    if res.success:
        exciton_fraction, _ = hopfield_coefficients(final_params['rabi_splitting'],
                                                    final_params['photon_energy'] - final_params['exciton_energy'])
        # Getting return values in physically useful units
        energy = result.best_values['center']
        lifetime = 2 * np.pi * hbar / (2 * result.best_values['sigma'])  # in ps

        final_params['polariton_energy'] = energy
        final_params['polariton_lifetime'] = lifetime
        final_params['polariton_mass'] = mass
        final_params['exciton_fraction'] = exciton_fraction

    return final_params


# Full dispersion fitting
def fit_dispersion(image, k_axis, energy_axis, plotting=False, known_sample_parameters=None, mode='both',
                   find_bands_kwargs=None, starting_fit_parameters=None, least_squares_kw=None):
    """
    :param image: 2D array. 1st axis is momentum, 2nd axis is energy
    :param k_axis: 1D array
    :param energy_axis: 1D array
    :param plotting: bool
    :param known_sample_parameters:
    :param mode: str
        One of 'both' or 'lp', whether to perform the fit on just the lower polariton or both polariton branches
    :param find_bands_kwargs: dict. To be passed to find_bands
    :param starting_fit_parameters: dict
        - photon_energy in meV
        - rabi_splitting in meV
        - photon_mass in m_e (free electron mass)
        - exciton_energy in meV
        - exciton_mass in m_e (free electron mass)
        - k_offset in um-1
    :param least_squares_kw: dict. Keys same as starting_fit_parameters

    :return:
    """
    LOGGER.debug('Call fit_dispersion: \n\t%s' % find_bands_kwargs)

    if find_bands_kwargs is None: find_bands_kwargs = dict()
    if least_squares_kw is None: least_squares_kw = dict()
    if known_sample_parameters is None: known_sample_parameters = dict()
    if starting_fit_parameters is None: starting_fit_parameters = dict()

    n_clusters = 2   # by default try to fit both upper and lower polariton
    if mode == 'lp': n_clusters = 1

    # Default find_band parameters that work for most dispersion images. Might need optimizing
    defaults = dict(direction='y',
                    find_peak_kwargs=dict(height=np.percentile(image, 90), distance=500),
                    clustering_kwargs=dict(min_cluster_size=40, min_cluster_distance=10,
                                           agglom_kwargs=dict(n_clusters=n_clusters, linkage='single')),
                    xaxis=k_axis, yaxis=energy_axis)
    for key, value in defaults.items():
        if key in find_bands_kwargs:
            find_bands_kwargs[key] = {**value, **find_bands_kwargs[key]}
        else:
            find_bands_kwargs[key] = value
    bands = find_bands(image, **find_bands_kwargs)

    lp_energy_guess = np.percentile(bands[0][:, 1], 5)
    if len(bands) > 1:
        up_energy_guess = np.percentile(bands[1][:, 1], 5)
    else:
        up_energy_guess = lp_energy_guess + 10
    default_start_values = dict(photon_energy=lp_energy_guess, exciton_energy=up_energy_guess,
                                rabi_splitting=10, photon_mass=2e-5, exciton_mass=0.35, k_offset=0)
    starting_fit_parameters = {**default_start_values, **starting_fit_parameters}
    LOGGER.info('[fit_dispersion] Default start values for fit: %s' % (starting_fit_parameters, ))

    # Handling known parameters
    parameter_names = ['photon_energy', 'rabi_splitting', 'photon_mass', 'exciton_energy', 'exciton_mass', 'k_offset']
    scales = [0.01, 0.1, 1e-6, 0.01, 0.01, 0.01]  # parameters have very different orders of magnitude scaling

    dispersion_parameters = dict()
    unknown_parameters = []
    least_square_scale = []
    least_square_start = []
    for key, sc in zip(parameter_names, scales):
        st = starting_fit_parameters[key]
        if key in known_sample_parameters.keys():
            exec('%s=%g' % (key, known_sample_parameters[key]), dispersion_parameters)
        else:
            unknown_parameters += [key]
            least_square_scale += [sc]
            least_square_start += [st]
    exec_string = ','.join(unknown_parameters)

    # Create cost functions for least square function fitting
    if mode == 'both':
        def cost_function(x, experiment_bands):
            exec('%s=%s' % (exec_string, list(x)), dispersion_parameters)

            # Least squares for the lower polariton band
            lower_polariton, _ = exciton_photon_dispersions(experiment_bands[0][:, 0],
                                                            *[dispersion_parameters[key] for key in parameter_names])
            lower_cost = np.sum(np.abs(lower_polariton - experiment_bands[0][:, 1]) ** 2)

            # Least squares for the upper polariton band
            _, upper_polariton = exciton_photon_dispersions(experiment_bands[1][:, 0],
                                                            *[dispersion_parameters[key] for key in parameter_names])
            upper_cost = np.sum(np.abs(upper_polariton - experiment_bands[1][:, 1]) ** 2)
            return lower_cost + upper_cost
    elif mode == 'lp':
        def cost_function(x, experiment_bands):
            exec('%s=%s' % (exec_string, list(x)), dispersion_parameters)

            lower_polariton, _ = exciton_photon_dispersions(experiment_bands[0][:, 0],
                                                            *[dispersion_parameters[key] for key in parameter_names])
            lower_cost = np.sum(np.abs(lower_polariton - experiment_bands[0][:, 1]) ** 2)
            return lower_cost

    res = least_squares(partial(cost_function, experiment_bands=bands), least_square_start,
                        x_scale=least_square_scale, **least_squares_kw)

    # parsing fit values
    if res.success:
        final_params = [dispersion_parameters[key] for key in parameter_names]
    else:
        LOGGER.warn('Failed fit %s' % starting_fit_parameters)
        final_params = []
        for key, sc, st in zip(parameter_names, scales, starting_fit_parameters):
            if key in known_sample_parameters:
                final_params += [known_sample_parameters[key]]
            else:
                final_params += [starting_fit_parameters[key]]

    final_params = {key: value for key, value in zip(parameter_names, final_params)}

    # parsing fit errors, copying from scipy.optimize.curve_fit
    # todo: FIX. this currently gives values that are much smaller than makes physical sense
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    parameter_errors = np.sqrt(np.diag(pcov))

    if plotting:
        fig, ax = plot_dispersion(plotting, k_axis, energy_axis, image, final_params, bands)
        if not res.success:
            ax.text(0.5, 0.98, 'Failed two-mode fit', va='top', ha='center', transform=ax.transAxes)
    return final_params, parameter_errors, res


def plot_dispersion(axes, k_axis, energy_axis, image=None, fit_params=None, bands=None):
    fig, ax = create_axes(axes)
    if image is not None:
        imshow(image.transpose(), ax, xaxis=k_axis, yaxis=energy_axis, diverging=False, cbar=False, norm=LogNorm())
    if bands is not None:
        [ax.plot(*band.transpose()) for band in bands]
    if fit_params is not None:
        new_k = np.linspace(k_axis.min(), k_axis.max(), 101)
        lower, upper, exciton, photon = exciton_photon_dispersions(new_k, **fit_params, for_fit=False)
        [ax.plot(new_k, y, color=c, alpha=0.3, lw=3) for y, c in zip([lower, upper], ['darkviolet', 'darkorange'])]
        [ax.plot(new_k, y, color='k', alpha=0.3, lw=3, ls='--') for y in [exciton, photon]]

        ax.text(0.5, 0.95,
                u'$\Omega$ = %.2g meV\n$\Delta$ = %.2g meV' % (fit_params['rabi_splitting'],
                                                               fit_params['photon_energy'] - fit_params['exciton_energy']),
                transform=ax.transAxes, ha='center', va='top', backgroundcolor=(1, 1, 1, 0.7))
    return fig, ax


def cluster_points(points, fig_ax=None, axis_limits=None, agglom_kwargs=None, noise_cluster_size=5,
                   min_cluster_distance=10, min_cluster_size=30, scale=None, shear=None):
    """
    Currently just a wrapper of sklearn.cluster.AgglomerativeClustering with additional filtering of clusters

    :param points:
    :param fig_ax: pyplot.Figure or pyplot.Axes or None
    :param axis_limits: tuple
        Either a two-tuple (e.g. [-1, 1]) or a two-tuple of two-tuples (e.g. [[-1, 1], [800, 850]]), indicating either
        the limits on a single axis, or both axis
    :param agglom_kwargs:
    :param noise_cluster_size:
    :param min_cluster_distance:
    :param min_cluster_size:
    :return:
    """
    LOGGER.debug('Call cluster_points: \n\t%s\n\t%s\n\t%s\n\t%s' % (agglom_kwargs, noise_cluster_size, min_cluster_distance, min_cluster_size))

    if axis_limits is not None:  # Removing points outside the desired axis_limits
        if depth(axis_limits) == 1:  # If only one tuple, apply to y-direction
            mask = np.logical_and(points[:, 1] < axis_limits[1],
                                  points[:, 1] > axis_limits[0])
            points = points[mask]
        else:  # if two tuples, apply each to it's corresponding direction
            for idx, _axis_limits in enumerate(axis_limits):
                if _axis_limits is not None:
                    mask = np.logical_and(points[:, idx] < _axis_limits[1],
                                          points[:, idx] > _axis_limits[0])
                    points = points[mask]

    """Clustering"""
    if shear is not None:
        points = np.dot(points, [[1, shear], [0, 1]])
    if scale is not None:
        points = np.dot(points, [[scale, 0], [0, 1]])

    if agglom_kwargs is None: agglom_kwargs = dict()
    defaults = dict(n_clusters=2, distance_threshold=None,
                    compute_full_tree=True, linkage='single')
    kwargs = {**defaults, **agglom_kwargs}

    model = AgglomerativeClustering(**kwargs)
    clusters = model.fit(points)
    label_history = [('first labels', deepcopy(clusters.labels_))]
    labels = clusters.labels_

    if shear is not None:
        points = np.dot(points, [[1, -shear], [0, 1]])
    if scale is not None:
        points = np.dot(points, [[1/scale, 0], [0, 1]])

    """Filtering irrelevant clusters"""
    if noise_cluster_size is not None:
        # Removes clusters smaller than a threshold
        _labels = list(labels)
        counts = np.array([_labels.count(x) for x in _labels])
        mask = counts > noise_cluster_size
        labels[~mask] = -1
        label_history += [('Noise filtering', deepcopy(labels))]

    if min_cluster_distance is not None and model.n_clusters_ > 1:
        # Clusters clusters that are closer than some threshold are merged
        # todo: don't use cluster_centers to determine whether to merge a cluster or not
        cluster_centers = np.array([np.mean(points[labels == l], 0) for l in range(model.n_clusters_)])
        cluster_indices = np.arange(len(cluster_centers))
        _mask = ~np.isnan(np.mean(cluster_centers, 1))
        if len(cluster_centers[_mask]) > 1:
            _model = AgglomerativeClustering(n_clusters=None, distance_threshold=min_cluster_distance,
                                             compute_full_tree=True, linkage='single')
            clusters_of_clusters = _model.fit(cluster_centers[_mask])
            new_labels = np.copy(labels)
            for idx, new_label in zip(cluster_indices[_mask], clusters_of_clusters.labels_):
                new_labels[labels == idx] = new_label
            labels = new_labels
        label_history += [('Proximity clustering', deepcopy(labels))]

    if min_cluster_size is not None:
        # Removes clusters smaller than a threshold
        _labels = list(labels)
        counts = np.array([_labels.count(x) for x in _labels])
        mask = counts > min_cluster_size
        labels[~mask] = -1
        label_history += [('Removing small clusters', deepcopy(labels))]

    masked_clusters = [points[labels == l] for l in np.unique(labels) if l >= 0]

    if fig_ax is not None:
        a, b = square(len(label_history))
        fig, axs = plt.subplots(a, b, num='clustering')
        for idx, ax in enumerate(axs.flatten()):
            ax.scatter(*points.transpose(), c=label_history[idx][1])
            ax.set_title(label_history[idx][0])

    return masked_clusters


def find_bands(image, plotting=None, direction='both', find_peak_kwargs=None, clustering_kwargs=None,
               xaxis=None, yaxis=None, max_number_of_peaks=5e3):
    """Find peaks in image, and cluster them into bands

    :param image: 2D array
    :param plotting: bool
    :param direction: str. Which direction to find peaks along in image
    :param find_peak_kwargs: dict. To be passed to _find_peaks
    :param clustering_kwargs: dict. To be passed to cluster_points
    :param xaxis:
    :param yaxis:
    :param max_number_of_peaks: int
        If the number of peaks from find_peaks is more than this, avoid clustering algorithm, since it takes too long
    :return:
    """
    assert direction in ['both', 'x', 'y']
    LOGGER.debug('Call find_bands: \n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s' % (plotting, direction, find_peak_kwargs, clustering_kwargs, xaxis, yaxis, max_number_of_peaks))

    # Smoothened find_peaks
    def _find_peaks(x, savgol_kwargs=None, *args, **kwargs):
        """Simple extension of find_peaks to give more than single-pixel accuracy"""
        if savgol_kwargs is None: savgol_kwargs = dict()
        savgol_kwargs = {**dict(window_length=5, polyorder=3), **savgol_kwargs}

        smoothened = savgol_filter(x, **savgol_kwargs)
        sampling = interp1d(range(len(smoothened)), smoothened, 'quadratic')
        new_x = np.linspace(0, len(smoothened) - 1, len(smoothened) * 10)
        new_y = sampling(new_x)
        results = find_peaks(new_y, *args, **kwargs)
        return new_x[results[0]], results[1]

    # One set of kwargs for each direction
    if find_peak_kwargs is None:
        find_peak_kwargs = [dict(), dict()]
    elif type(find_peak_kwargs) == dict:
        find_peak_kwargs = [find_peak_kwargs, find_peak_kwargs]
    if clustering_kwargs is None: clustering_kwargs = dict()

    peaks = []
    if direction in ['both', 'x']:
        # Find peaks along each column
        for idx, x in enumerate(image.transpose()):
            pks = _find_peaks(x, **find_peak_kwargs[0])[0]
            peaks += [(pk, idx) for pk in pks]
    if direction in ['both', 'y']:
        # Find peaks along each row
        for idx, x in enumerate(image):
            pks = _find_peaks(x, **find_peak_kwargs[1])[0]
            peaks += [(idx, pk) for pk in pks]
    peaks = np.asarray(peaks, dtype=float)
    assert len(peaks) < max_number_of_peaks

    # Cluster points as pixels
    clusters = cluster_points(peaks, plotting, **clustering_kwargs)
    # Transform pixels to axis units
    if xaxis is None: xaxis = np.arange(image.shape[0])
    if yaxis is None: yaxis = np.arange(image.shape[1])
    xfunc = partial(np.interp, xp=np.arange(len(xaxis)), fp=xaxis)
    yfunc = partial(np.interp, xp=np.arange(len(yaxis)), fp=yaxis)
    clusters = [np.transpose([xfunc(cluster[:, 0]), yfunc(cluster[:, 1])]) for cluster in clusters]

    if plotting is not None:
        fig, ax = create_axes(plotting)
        imshow(image.transpose(), ax, xaxis=xaxis, yaxis=yaxis, cbar=False, diverging=False, norm=LogNorm())
        for cluster in clusters: ax.plot(*cluster.transpose(), 'r.', alpha=1, ms=0.2)
    return clusters

