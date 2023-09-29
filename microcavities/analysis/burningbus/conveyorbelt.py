# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter
# from microcavities.simulations.quantum_box import *
from microcavities.utils import apply_along_axes, random_choice
from microcavities.analysis import *
from microcavities.analysis.phase_maps import low_pass
from microcavities.analysis.utils import remove_outliers
from microcavities.analysis.interactive import InteractiveBase, InteractiveBaseUi
from microcavities.analysis.dispersion import *
# from microcavities.experiment.utils import magnification, spectrometer_calibration
from microcavities.utils.HierarchicalScan import get_data_from_yamls
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import expm
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from nplab.utils.log import create_logger
from nplab.utils.gui import QtWidgets, get_qt_app
from itertools import combinations
from copy import deepcopy
from functools import partial
from tqdm import tqdm
import h5py
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter, ScalarFormatter, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from microcavities.utils.HierarchicalDatastructure import h5py_get_data

plt.style.use(os.path.join(os.path.dirname(get_data_path('')), 'Papers/Conveyor/python/paper_style.mplstyle'))
plt.rcParams["pgf.texsystem"] = "pdflatex"
raw_data_path = get_data_path('2023_09_07/raw_data.h5')
analysis_directory = get_data_path('2023_09_15')
laser_separations = np.load(get_data_path('2023_09_07/laser_separation.npy'), allow_pickle=True).item()
aom_powers = np.linspace(1, 111, 12)
power_axis = aom_powers / 100  # power_density(aom_powers, laser_separations[group_name], ellipse_axis=(120, 40))
power_units = '$P_{th}$'  # '%sW%sm$^{-2}$' % (greek_alphabet['mu'], greek_alphabet['mu'])
norm_power = colors.Normalize(vmin=power_axis[0] - 2 * np.diff(power_axis)[0], vmax=power_axis[-1])
norm_laserangle = colors.Normalize(vmin=0.28, vmax=0.8)
frequencies_axis = np.linspace(-6, 6, 13)
fit_parameters = np.load(get_data_path('2023_09_07/low_power_fitting_params_all.npy'), allow_pickle=True).item()

density_cmap = 'BlueYellowRed'
label_colour = (216 / 255., 220 / 255., 214 / 255., 0.9)
plotlabel_bbox_kwargs = dict(boxstyle='round,pad=0.1', fc=label_colour, ec='none')
plot_kwargs_experimental_bands = dict(color='xkcd:silver', ls='-', alpha=1, lw=1)
plot_kwargs_experimental_bands_fits = dict(color='#88918d', ls='-.', alpha=0.9, lw=1.5)
plot_kwargs_theory_bands = dict(color='k', ls='--', alpha=0.7)
color_dictionary = dict(fig1_k0='#6e017f', fig1_k1='#003366', fig1_J='#089114')
letter_label_weight = 'bold'
letter_position = (-0.06, 1.01)
colorbar_font_size = 8
letter_kw = dict(weight='bold', fontsize=9, ha='right', va='bottom')


LOGGER = create_logger('Fitting')
LOGGER.setLevel('WARN')

"""UTILITY FUNCTIONS"""


class dummy_formatter(ScalarFormatter):
    def __init__(self, offset, *args, **kwargs):
        super(dummy_formatter, self).__init__(*args, **kwargs)
        self.set_useOffset(offset)
        self.format = '%g'

    def format_data(self, value):
        return '%g' % value


def normalize_energy_axis(ax, scan_number=None, energy=None):
    """Normalizes the energy axis to have zero energy at the LP energy"""
    if scan_number is not None:
        _polariton_energy = fit_parameters['Scan%d' % scan_number]['polariton_energy']
    elif energy is not None:
        _polariton_energy = energy
    else:
        raise ValueError('Need to give a scan_number or an energy')
    _formatter = dummy_formatter(_polariton_energy, True, None, True)
    _formatter._offset_threshold = 2
    _formatter.format = '%.1g'
    ax.yaxis.set_major_formatter(_formatter)
    return _polariton_energy


def power_density_angled(power, angle, area=None):
    """Microcavity surface power density of a laser at an angle

    Basically simply projects the area of the laser (which is the cross-sectional area of the beam) on to the plane of
    the microcavity

    :param power:
    :param angle:
    :param area:
    :return:
    """
    area_angled = area / np.cos(angle)
    return power / area_angled


def area_of_ellipse(major_axis, minor_axis):
    return np.pi * major_axis * minor_axis


def power_density(power, wavevector, laser_wavelength=0.75, ellipse_axis=(100, 20)):
    """Utility function just for this project"""
    _mom = 2 * np.pi / laser_wavelength
    angle_of_incidence = np.abs(wavevector) / _mom  # approximately true for small angles
    return power_density_angled(power, angle_of_incidence, area_of_ellipse(*ellipse_axis),)


def get_experimental_axes(scan_number):
    if scan_number == 3:
        roi = (1228, 1615, 48, 465)
    elif scan_number == 4:
        roi = (1260, 1624, 57, 439)
    elif scan_number == 5:
        roi = (1297, 1603, 80, 430)
    elif scan_number == 7:
        roi = (1278, 1548, 82, 415)
    elif scan_number == 8:
        roi = (1274, 1551, 88, 408)
    elif scan_number == 9:
        roi = (1306, 1524, 113, 383)
    elif scan_number == 10:
        roi = (1233, 1563, 118, 392)
    elif scan_number == 11:
        roi = (1288, 1583, 98, 421)
    else:
        raise ValueError('No ROI given for scan: %d' % scan_number)

    with h5py.File(raw_data_path, 'r') as df:
        img = df['Scan%d/characterisation/low_power' % scan_number][...]

    _k, _e = make_dispersion_axes(img, None, ('rotation_acton', 'k_space'), ('rotation_acton', 806, '1'))
    k = _k[roi[2]:roi[3] + 1]
    e = _e[roi[0]:roi[1] + 1]

    if scan_number == 7:
        k += 0.12
    return k, e


def plot_single_dispersion_image(ax, scan_number, indices):
    """Plotting a single dispersion image"""
    # Getting experimental data and simulated parameters
    images, variables = h5py_get_data(raw_data_path, 'Scan%d' % scan_number, 'img')
    bands = np.load(get_data_path('%s/bands/Scan%d.npy' % (analysis_directory, scan_number)), allow_pickle=True)
    # laser_separations = np.load(get_data_path('2023_09_07/laser_separation.npy'), allow_pickle=True).item()
    k, e = get_experimental_axes(scan_number)

    # Getting the theory bands
    with h5py.File(get_data_path('2023_09_07/analysed_data.h5'), 'r') as df:
        group = df['Scan%d' % scan_number]
        fitted_depths = group['fitted_depths'][...]
        fitted_backgrounds = group['fitted_backgrounds'][...]

    fit_params = np.load(get_data_path('2023_09_07/low_power_fitting_params.npy'), allow_pickle=True).item()
    laser_separation = laser_separations['Scan%d' % scan_number]
    period = 2 * np.pi / laser_separation
    hamiltonian_kwargs = dict(rabi=fit_params['rabi_splitting'],
                              detuning=(fit_params['photon_energy'] - fit_params['exciton_energy']),
                              mass_photon=fit_params['photon_mass'], mass_exciton=fit_params['exciton_mass'])
    theory_energies, theory_k_axis = run_simulations(hamiltonian_conveyor_k, [fitted_depths[indices]], [period],
                                                     hamiltonian_kwargs, disable_output=True)

    # Plotting
    imshow(images[indices].transpose(), ax, diverging=False, cbar=False, cmap=density_cmap, norm=LogNorm(),
           xaxis=k, yaxis=e)
    [ax.plot(*b.transpose(), **plot_kwargs_experimental_bands) for b in bands[indices]]
    [ax.plot(theory_k_axis, t_e.real, **plot_kwargs_theory_bands) for t_e in (theory_energies[:, :10].transpose() + fitted_backgrounds[indices])]
    ax.set_ylim(e.min(), e.max())
    ax.set_xlim(k.min(), k.max())
    _polariton_energy = normalize_energy_axis(ax, scan_number)
    ax.set_yticks(_polariton_energy + np.linspace(1, 3, 6))

    ax.text(0.5, 0.98,
            '%s$k_{\mathrm{laser}}$=%.2g %sm$^{-1}$\n$P_m$=%g %s' % (greek_alphabet['Delta'],
                                                                     laser_separations['Scan%d' % scan_number],
                                                                     greek_alphabet['mu'], power_axis[indices[1]],
                                                                     power_units),
            fontsize=9, transform=ax.transAxes, ha='center', va='top', bbox=plotlabel_bbox_kwargs)


def plot_potentialdepth_vs_frequency(ax, scan_number, v_depth_lims=None):
    color_values = cm.get_cmap('Oranges')(norm_power(power_axis))

    group_name = 'Scan%d' % scan_number
    with h5py.File(get_data_path('2023_09_07/analysed_data.h5'), 'r') as df:
        if group_name in df:
            fitted_depths = df[group_name + '/fitted_depths'][...]
            f = remove_outliers(fitted_depths, axis=-1)
            if v_depth_lims is not None:
                f[f > v_depth_lims[1]] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                [plot_fill(f, ax, xaxis=frequencies_axis, color=c) for f, c in
                 zip(np.transpose(f, (1, 2, 0)), color_values)]
    # label_axes(ax, '%sf [GHz]' % greek_alphabet['Delta'],  # 'V$_{\mathrm{eff}}$ [meV]',
    #            letter='(e)', letter_kw=letter_kw,
    #            letter_position=(0.05, letter_position[1]), )  # , ylabel_kw=dict(labelpad=label_offset))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.text(0.5, 0.99, '%s$k_{\mathrm{laser}}$=%.2g%sm$^{-1}$' % (
    greek_alphabet['Delta'], laser_separations['Scan%d' % scan_number], greek_alphabet['mu']),
            fontsize=8, ha='center', va='top', transform=ax.transAxes, bbox=plotlabel_bbox_kwargs)
    if v_depth_lims is not None:
        ax.set_ylim(*v_depth_lims)


"""EXPERIMENTAL ANALYSIS"""


def conveyor_find_bands(energy_width_limit=0.5, energy_range=None, momentum_asymmetry_limit=0.5, flatness_threshold=None, *args, **kwargs):
    """Wrapper for microcavities.analysis.dispersion.find_bands

    Adds filtering to avoid band clusters that are very wide in energy and/or have a very asymmetric distribution wrt
    the momentum axis

    :param energy_width_limit:
    :param energy_range:
    :param momentum_asymmetry_limit:
    :param args:
    :param kwargs:
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs['return_indices'] = True

        bands, bands_indices = find_bands(*args, **kwargs)

        # Creating a mask to remove wide bands
        band_energy_range = np.array([(np.percentile(b[:, 1], 90) - np.percentile(b[:, 1], 10)) for b in bands])
        mask1 = band_energy_range < energy_width_limit
        LOGGER.debug('[conveyor_find_bands] Filtering wide bands: %s \t mask1: %s' % (band_energy_range, mask1))

        # Creating a mask to remove bands that are asymmetric wrt the k-axis
        band_momentum_average = np.array([np.mean(b[:, 0]) for b in bands])
        mask2 = np.abs(band_momentum_average) < momentum_asymmetry_limit
        LOGGER.debug('[conveyor_find_bands] Filtering by k_avg: %s \t mask2: %s' % (band_momentum_average, mask2))

        # Creating a mask to remove bands that are not in the expected energy_range
        if energy_range is not None:
            average_band_energy = np.array([np.percentile(b[:, 1], 50) for b in bands])

            if energy_range[0] < 0:
                band_intensities = [args[0][np.asarray(b, dtype=int)[:, 0], np.asarray(b, dtype=int)[:, 1]] for b in
                                    bands_indices]
                band_intensity = [np.sum(i) for i in band_intensities]
                max_intensity_band = np.argmax(band_intensity)
                energy_of_max_intensity_band = average_band_energy[max_intensity_band]
                _mask1 = energy_range[1] + energy_of_max_intensity_band > average_band_energy
                _mask2 = average_band_energy > energy_range[0] + energy_of_max_intensity_band
            else:
                _mask1 = energy_range[1] > average_band_energy
                _mask2 = average_band_energy > energy_range[0]
            mask3 = np.logical_and(_mask1, _mask2)
        else:
            mask3 = np.array([True] * len(bands))
        LOGGER.debug('[conveyor_find_bands] Filtering by energy range: %s' % (mask2,))

        # Creating a mask to remove bands that are not flat enough
        if flatness_threshold is not None:
            mask4 = []
            edgyness = []
            for b in bands:
                _diff = np.diff(b, axis=0)
                _edgy = np.sqrt(np.mean(np.abs(_diff[:, 1] / _diff[:, 0]) ** 2))
                edgyness += [_edgy]
                if _edgy > flatness_threshold:
                    mask4 += [False]
                else:
                    mask4 += [True]
        else:
            edgyness = None
            mask4 = np.array([True] * len(bands))
        LOGGER.debug('[conveyor_find_bands] Filtering by flatness_threshold: %s \t mask4: %s' % (edgyness, mask4))

        # Combining masks and removing bands
        _mask = np.logical_and(mask1, mask2)
        _mask2 = np.logical_and(_mask, mask3)
        mask = np.logical_and(_mask2, mask4)

        new_bands = []
        for b, m in zip(bands, mask):
            if m:
                new_bands += [b]

        # Ordering by energy
        average_band_energy = np.array([np.percentile(b[:, 1], 50) for b in new_bands])
        indices = np.argsort(average_band_energy)
        new_bands = [new_bands[index] for index in indices]

        return new_bands


def analyse_bands_single(bands, n_bands=None, k_range_fit=None, plotting=False):
    if n_bands is None:
        n_bands = len(bands)
    assert n_bands <= len(bands)

    if len(bands) == 0:
        return np.array([np.nan]), np.array([np.nan])

    _fits = []
    energies = []
    tilts = []
    for idx in range(n_bands):
        band = bands[idx]
        if k_range_fit is not None:
            mask = k_range_fit > np.abs(band[:, 0])
            band = band[mask]
            LOGGER.debug('Excluding %d points in band %d from linear fit' % (np.sum(~mask), idx))
        if len(band) > 0:
            fit = np.polyfit(band[:, 0], band[:, 1], 1)
            # print(fit, fit[0])
            # func = np.poly1d(fit)
            k0_energy = np.poly1d(fit)(0)  # func(self.config['k_masking']['k0'])
            tilt = fit[0]
            # # print(np.min(band[:, 1]), np.max(band[:, 1]), self.configuration['energy_axis'])
            # inverse_fit = np.polyfit(self.k_inverse(band[:, 0]), self.e_inverse(band[:, 1]), 1)
            # vectors, start_slice, end_slice = self._calculate_slice(inverse_fit, band_width)
            # slice, coords = pg.affineSlice(self.image, end_slice, start_slice, vectors, (0, 1),
            #                                returnCoords=True)
        else:
            k0_energy, tilt, fit = np.nan, np.nan, np.nan
            # k0_energy, tilt, slice, coords, fit = _return_NaN()

        # _coords += [coords]
        _fits += [fit]
        energies += [k0_energy]
        tilts += [tilt]
        # slices += [slice]

    if plotting:
        fig = plt.figure(num='analyse_bands')
        gs = gridspec.GridSpec(1, 2, fig)
        ax0 = plt.subplot(gs[0])
        imshow(self.image, ax0, diverging=False, cmap='Greys', norm=LogNorm(), xaxis=self.k_axis,
               yaxis=self.energy_axis)

        gs1 = gridspec.GridSpecFromSubplotSpec(2 * len(slices), 1, gs[1])
        axs = gs1.subplots()
        for idx, _fit, energ, tilt, slice in zip(range(len(_fits)), _fits, energies, tilts, slices):
            color = cm.get_cmap('viridis', len(_fits))(idx)
            # func = np.poly1d(_fit)
            # x_points = self.k_func([0, self.image.shape[1] - 1])
            # ax0.plot(x_points, func(x_points))
            if not np.isnan(energ):
                ax0.plot(self.k_func(_coords[idx][1].flatten()),
                         self.energy_func(_coords[idx][0].flatten()), '.',
                         ms=0.3, color=color)
                try:
                    imshow(slice.transpose(), axs[-1 - 2 * idx], cbar=False, diverging=False)
                    y = np.sum(slice, 1)
                    axs[-2 - 2 * idx].semilogy(y, color=color)
                    [axs[-2 - 2 * idx].axvline(x) for x in find_peaks(-y, width=6, distance=6, prominence=0.01)[0]]
                    colour_axes(axs[-1 - 2 * idx], color)
                    colour_axes(axs[-2 - 2 * idx], color)
                except TypeError:
                    imshow(slice.transpose(), axs, cbar=False, diverging=False)

    return np.array(energies), np.array(tilts) #, np.array(slices)


"""SCHRODINGER EQUATION SIMULATIONS"""


def analytical_coupling_strength(potential_depth, lattice_period, mass):
    return 4 * potential_depth * np.exp(-np.sqrt(2*mass*potential_depth)*lattice_period/hbar)


def calculate_chern_number(hamiltonian, momentum_range, time_period, n_points=100, band_number=3, hamiltonian_kw=None):
    """Calculates the Chern number

    Numerically evaluates band differentials in momentum and time to extract the Berry curvature and then sums over the
    whole 2D Brillouin zone to get the Chern number

    :param hamiltonian: function
    :param momentum_range: float
    :param time_period: float
    :param n_points: int
    :param band_number: int
    :param hamiltonian_kw: dict or None
    :return:
    """
    # Choosing a differential size that is smaller than the step size along the k and t dimensions
    delta_k = momentum_range / (100*n_points)
    delta_t = time_period / (100*n_points)

    # Defining the Hamiltonian function to that it has only two parameters: k, t
    f = np.abs(1/time_period)
    if hamiltonian_kw is None: hamiltonian_kw = dict()
    _hamiltonian = partial(hamiltonian, delta_k=momentum_range, frequency=f, **hamiltonian_kw)

    # Looping over the Brillouin zones and one full time period
    k_range = np.arange(-momentum_range / 2, momentum_range / 2, momentum_range / n_points)
    t_range = np.arange(0, time_period, time_period / n_points)

    berry_curvature = []
    for kx in tqdm(k_range, 'Brillouin zone sum'):
        _berry_curvature = []
        for t in t_range:
            # Band wavefunction evaluated at four points (k, t), (k+dk, t), (k, t+dt), (k+dk, k+dt)
            vectors = []
            for _t in [t, t + delta_t]:
                for _k in [kx, kx + delta_k]:
                    h = _hamiltonian(_k, _t)
                    eigenvalue, eigenvector = np.linalg.eig(h)
                    vector = eigenvector[:, np.argsort(np.real(eigenvalue))[band_number]]  #
                    vectors += [vector]

            # Fixing the gauge of the wavefunctions by making it real in the same BZ
            index = np.argmax(np.abs(vectors[0]))  # BZ index
            vectors = [v * np.exp(- 1j * np.angle(v[index])) for v in vectors]

            # Berry connections as partial differentials wrt k and t
            a_k = np.dot(vectors[0].transpose().conj(), (vectors[1] - vectors[0]) / delta_k)
            a_t = np.dot(vectors[0].transpose().conj(), (vectors[2] - vectors[0]) / delta_t)
            a_k_dt = np.dot(vectors[2].transpose().conj(), (vectors[3] - vectors[2]) / delta_k)
            a_t_dk = np.dot(vectors[1].transpose().conj(), (vectors[3] - vectors[1]) / delta_t)

            # Berry curvature
            _berry_curvature += [(a_t_dk - a_t) / delta_k - (a_k_dt - a_k) / delta_t]
        berry_curvature += [_berry_curvature]
    chern_number = np.sum(berry_curvature) * (momentum_range / n_points) * (time_period / n_points) / (2 * np.pi * 1j)
    return chern_number, np.array(berry_curvature)


"""SIMULATIONS FOR EXPERIMENT"""
from microcavities.simulations.linear.one_d.realspace import *
from microcavities.simulations.linear.one_d.kspace import *


def run_simulations(hamiltonian, depths, periods, hamiltonian_kwargs, backgrounds=0, n_bands=20,
                    disable_output=False, k_axis=None):
    """Run simulations for a grid of depths, periods, backgrounds, and/or masses

    :param hamiltonian:
    :param depths:
    :param periods:
    :param hamiltonian_kwargs:
    :param backgrounds:
    :param n_bands:
    :param disable_output:
    :param k_axis:
    :return:
    """
    try: len(depths)
    except: depths = [depths]
    try: len(periods)
    except: periods = [periods]
    try: len(backgrounds)
    except: backgrounds = [backgrounds]

    if k_axis is None:
        k_axis = np.linspace(-3, 3, 301)

    values = []
    for depth in tqdm(depths, 'run_simulations', disable=disable_output):
        _vals = []
        for period in periods:
            # _valss = []
            # for mass in masses:
            bands, _ = solve_for_krange(k_axis,
                                        partial(hamiltonian, t=0, frequency=0, n_bands=n_bands,
                                                potential_depth=depth, period=period, **hamiltonian_kwargs))
            _values = []
            for background in backgrounds:
                _eig = bands + background
                _values += [_eig]
            _vals += [bands]
            # _vals += [_valss]
        values += [_vals]
    return np.squeeze(values), k_axis


def make_theory_interpolator(data_location=None, simulated_all_bands=None, simulated_all_parameters=None):
    if data_location is not None:
        with h5py.File(get_data_path(data_location[0]), 'r') as df:
            dset = df[data_location[1]]
            theory_energies = dset[...]
            theory_k_axis = dset.attrs['k_axis']
            theory_depths = dset.attrs['depths']
            theory_periods = dset.attrs['periods']
            theory_hamiltonian_kwargs = eval(dset.attrs['hamiltonian_kwargs'])
        theory_centers = np.amin(theory_energies, 1)
    else:
        theory_depths = simulated_all_parameters['depths']
        theory_centers = np.amin(simulated_all_bands, 1)

    theory_parameters = dict(hamiltonian=hamiltonian_conveyor_k,
                             hamiltonian_kwargs=theory_hamiltonian_kwargs,
                             depths=theory_depths,
                             periods=[theory_periods])

    all_splittings = np.diff(theory_centers, axis=-1)
    first_splitting = all_splittings[..., 0]
    return interp1d(first_splitting, theory_depths, bounds_error=False, fill_value=np.nan), theory_parameters


def fit_theory_to_bands_single(experimental_bands, experimental_adjust_tilt=False,
                               theory_interpolator=None, simulated_all_bands=None, simulated_all_parameters=None,
                               calculate_bands=True):
    """Fitting function for a single set of experimental parameters

    :param experimental_bands:
    :param theory_interpolator:
    :param simulated_all_bands:
    :param simulated_all_parameters:
    :param experimental_adjust_tilt:
    :return:
    """

    # Experimental data preparation
    energies, tilt = analyse_bands_single(experimental_bands)
    try:
        exper_split = np.diff(energies, axis=-1)[..., 0]
    except:
        return np.nan, np.nan, np.full((301, 82), np.nan), np.full((301, ), np.nan)

    if experimental_adjust_tilt:
        angle = tilt * np.diff(config['k_axis'])[0] / np.diff(config['energy_axis'])[0]
        angle = angle[..., 1:4]  # ignoring the ground state
        if initial_shape == (1, ):
            angle = np.array([angle])

        shape = angle.shape
        if len(shape) == 4:
            angle = remove_outliers(angle, (0, 2, 3))
            angle = np.nanmean(angle, (0, 2, 3))
            angle = np.repeat(angle[np.newaxis, :], shape[0], 0)
            angle = np.repeat(angle[..., np.newaxis], shape[2], 2)
        else:
            angle = remove_outliers(angle, (1, 2))
            angle = np.nanmean(angle, (1, 2))
            angle = np.repeat(angle[..., np.newaxis], shape[1], 1)
        exper_split *= np.cos(angle)

    # Theory data preparation
    if theory_interpolator is None:
        theory_depth_vs_splitting, _ = make_theory_interpolator(None, simulated_all_bands, simulated_all_parameters)
        # theory_depths = simulated_all_parameters['depths']
        # theory_centers = np.amin(simulated_all_bands, 1)
        #
        # all_splittings = np.diff(theory_centers, axis=-1)
        # first_splitting = all_splittings[..., 0]
        # theory_depth_vs_splitting = interp1d(first_splitting, theory_depths, bounds_error=False, fill_value=np.nan)
    else:
        theory_depth_vs_splitting = theory_interpolator

    # Fitting
    fitted_depth = theory_depth_vs_splitting(exper_split)

    # Re-running simulations for the fitted values
    if np.isnan(fitted_depth):
        if simulated_all_bands is None:
            theory_bands = np.full((301, 82), np.nan)
            k_axis = np.full((301, ), np.nan)
        else:
            theory_bands = np.full(simulated_all_bands.shape[1:], np.nan)
            k_axis = np.full((simulated_all_bands.shape[1], ), np.nan)
        fitted_background = np.nan
    else:
        rerun_parameters = dict(simulated_all_parameters)
        rerun_parameters['depths'] = [fitted_depth]
        rerun_parameters['disable_output'] = True
        theory_bands, k_axis = run_simulations(**rerun_parameters)

        min_theory = np.min(theory_bands)
        min_experiment = energies[0]
        fitted_background = min_experiment - min_theory

    return fitted_depth, fitted_background, theory_bands, k_axis


def plot_theory_fit(axes, image, image_axes, experimental_bands=None, theory_bands=None, theory_interpolator=None,
                    simulated_all_parameters=None):
    if experimental_bands is None:  # Finding the bands with default parameters
        experimental_bands = conveyor_find_bands(0.5, None, 0.5, image)

    if theory_bands is None:
        th_depth, th_bkg, th_b, th_k = fit_theory_to_bands_single(experimental_bands, theory_interpolator,
                                                        simulated_all_parameters=simulated_all_parameters)
    else:
        th_k, th_b, th_bkg = theory_bands

    fig, ax = create_axes(axes)
    imshow(image.transpose(), ax, xaxis=image_axes[0], yaxis=image_axes[1], cbar=False, diverging=False,
           cmap=density_cmap)
    [ax.plot(*b.transpose(), 'k.') for b in experimental_bands]
    ax.plot(th_k, th_b + th_bkg, '--')
