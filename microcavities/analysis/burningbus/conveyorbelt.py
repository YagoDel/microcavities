# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
import lmfit
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter
from microcavities.simulations.quantum_box import *
from matplotlib.colors import LogNorm
from microcavities.utils import apply_along_axes, random_choice
from microcavities.analysis.characterisation import *
from microcavities.analysis.phase_maps import low_pass
from microcavities.analysis.utils import remove_outliers
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
import pyqtgraph as pg
from nplab.utils.log import create_logger
from nplab.utils.gui import QtWidgets, get_qt_app
from itertools import combinations
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from microcavities.analysis.interactive import InteractiveBase, InteractiveBaseUi
from pyqtgraph.parametertree import ParameterTree, Parameter
import h5py
LOGGER = create_logger('Fitting')
LOGGER.setLevel('WARN')

collated_data_path = get_data_path('2021_07_conveyorbelt/collated_data.h5')
collated_analysis = get_data_path('2021_07_conveyorbelt/collated_analysis.h5')
spatial_scale = magnification('rotation_pvcam', 'real_space')[0] * 1e6
momentum_scale = magnification('rotation_pvcam', 'k_space')[0] * 1e-6
mu = '\u03BC'
delta = '\u0394'

# List of parameters required for different steps in the analysis for each of the 9 datasets
configurations = [
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.007, prominence=0.00001)),
        clustering=dict(energy_limit=[30, 30], min_cluster_size=15, min_cluster_distance=2.0,
                        noise_cluster_size=0, bandwidth=7.0),
        make_bands=dict(k0=-0.2, k_acceptance=1., bandwidth=1.0),
        analyse_bands=dict(k_range_fit=0.5),
        brillouin_plot=dict()
    ),
    None,
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          # find_peaks=dict(height=0.007, prominence=0.00001),
                          find_peaks=dict(height=0.01, prominence=0.01)),
        clustering=dict(energy_limit=[10, 30], min_cluster_size=30, min_cluster_distance=3.0,
                        noise_cluster_size=3, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=1.0, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.15)
    ),
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.007, prominence=0.00001)),
        clustering=dict(energy_limit=[30, 40], min_cluster_size=15, min_cluster_distance=4.0,
                        noise_cluster_size=0, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=1.0, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.1)
    ),
    dict(
        selected_bands=[],
        data_axes_order=(1, 0, 2, 3, 4),
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.01, prominence=0.001)),
        clustering=dict(AgglomerativeClustering=dict(n_clusters=15, ),
                        energy_limit=[30, 30], min_cluster_size=10, min_cluster_distance=5.0,
                        noise_cluster_size=0, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=0.5, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.12, ylim=(1.548265, 1.54875))
    ),
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.01, prominence=0.001)),
        clustering=dict(energy_limit=[30, 30], min_cluster_size=15, min_cluster_distance=3.0,
                        noise_cluster_size=0, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=0.6, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.11)
    ),
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.005, prominence=0.0002)),
        clustering=dict(energy_limit=[40, 30], min_cluster_size=15, min_cluster_distance=3.0,
                        noise_cluster_size=3, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=0.6, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.1)
    ),
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.005, prominence=0.005)),
        clustering=dict(energy_limit=[30, 30], min_cluster_size=15, min_cluster_distance=3.0,
                        noise_cluster_size=0, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=0.6, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.07, ylim=(1.548, 1.5486))
    ),
    dict(
        selected_bands=[],
        image_preprocessing=dict(),
        peak_finding=dict(peak_width=3.0,
                          find_peaks=dict(height=0.005, prominence=0.008)),
        clustering=dict(energy_limit=[30, 30], min_cluster_size=15, min_cluster_distance=3.0,
                        noise_cluster_size=0, bandwidth=100.0),
        make_bands=dict(k0=-0.1, k_acceptance=0.6, bandwidth=0.0005),
        analyse_bands=dict(k_range_fit=1.5),
        brillouin_plot=dict(k0_offset=-0.08, ylim=(1.54801, 1.54868))
    )
]


def get_experimental_data_base(dataset_index):
    with h5py.File(collated_analysis, 'r') as dfile:
        laser_separations = dfile['laser_separations'][...]
    with h5py.File(collated_data_path, 'r') as dfile:
        dset = dfile['alignment%d/scan' % dataset_index]
        data = dset[...]
        _v = dset.attrs['variables']
        eax = dset.attrs['eaxis']
        kax = dset.attrs['kaxis']
    variables = eval(_v)

    config = configurations[dataset_index]
    if 'data_axes_order' in config:
        data = np.transpose(data, config['data_axes_order'])

    normal_laser_size = np.pi * (20**2)
    _mom = 2*np.pi / 0.805
    angle_of_incidence = np.abs(laser_separations[dataset_index]) / _mom
    laser_size = normal_laser_size * np.sin(angle_of_incidence)
    norm_ax = variables['vwp'] / laser_size

    variables['normalised_power_axis'] = norm_ax
    config['k_axis'] = kax
    config['energy_axis'] = eax
    config['laser_angle'] = laser_separations[dataset_index]

    return data, variables, config


def get_experimental_data(dataset_index):
    data, variables, config = get_experimental_data_base(dataset_index)
    # config = configurations[dataset_index]
    if configurations[dataset_index] is None:
        analyse = False
    else:
        analyse = True

    if analyse:
        cls = FullDatasetModeDetection(data, config)
        cls.configuration['plotting'] = False
        bands = np.load(get_data_path('2021_07_conveyorbelt/bands/dataset%d.npy' % dataset_index), allow_pickle=True)
        analysed_bands = cls.analyse_bands(bands)
    else:
        bands = None
        analysed_bands = None
    return data, bands, config, analysed_bands, variables


def get_selected_tilt_data(dataset_index):
    _, _, config, analysis_results, variables = get_experimental_data(dataset_index)
    mode_tilts = -remove_outliers(analysis_results[1])

    # if dataset_index in [1]:
    #     with h5py.File(get_data_path('2021_07_conveyorbelt/collated_analysed_data.h5'), 'r') as full_file:
    #         dset = full_file['speeds%d' % (dataset_index + 1)]
    #         mode_tilts = -dset[...]
    #         all_freq = dset.attrs['freq']
    #         config = dict(laser_angle=laser_separations[dataset_index])
    # else:
    #     with h5py.File(collated_data_path, 'r') as dfile:
    #         dset = dfile['alignment%d/scan' % dataset_index]
    #         _v = dset.attrs['variables']
    #     variables = eval(_v)
    #     # power_axis = np.array(variables['vwp']) * 1e3
    #     all_freq = np.array(variables['f'])
    #     # if 'power' in variables:
    #     #     cw_power_axis = np.array(variables['power']) * 1e3
    #     # if 'cw' in variables:
    #     #     cw_power_axis = np.array(variables['cw']) * 1e3
    #
    #     data, bands, config, analysis_results, _ = get_experimental_data(dataset_index)
    #
    #     mode_tilts = -remove_outliers(analysis_results[1])
    shape = mode_tilts.shape
    all_freq = np.array(variables['f'])

    # Manually chosen fitting regions (avoiding high-frequency, low-modulation, low-power)
    if dataset_index == 0:
        tst = np.transpose(mode_tilts, (1, 2, 0, 3))
        all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
        selected_modes = tst[:, 4:, -1, 1:3]
        selected_modes = np.reshape(selected_modes, (shape[1], (shape[2] - 4) * 2))
        modes_for_fitting = selected_modes[2:-2]
        fitting_freq = all_freq[2:-2]
    elif dataset_index == 1:
        all_modes = np.copy(mode_tilts)
        selected_modes = all_modes[:, 2:]
        modes_for_fitting = selected_modes[2:]
        fitting_freq = all_freq[2:]
    elif dataset_index == 2:
        all_modes = np.reshape(mode_tilts, (shape[0], shape[1] * shape[2]))
        selected_modes = mode_tilts[:, 4:, 0]
        modes_for_fitting = selected_modes[5:-5]
        fitting_freq = all_freq[5:-5]
        # selected_modes = np.reshape(selected_modes, (shape[0], (shape[1]-4) * shape[2]))
    elif dataset_index == 3:
        all_modes = np.reshape(mode_tilts, (shape[0], shape[1] * shape[2]))
        selected_modes = mode_tilts[:, 3:, 0]
        modes_for_fitting = selected_modes[2:]
        fitting_freq = all_freq[2:]
        # selected_modes = np.reshape(selected_modes, (shape[0], (shape[1]-3) * shape[2]))
    elif dataset_index == 4:
        tst = np.transpose(mode_tilts, (1, 2, 0, 3))
        all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
        selected_modes = mode_tilts[:, :, 2:, 0]
        tst = np.transpose(selected_modes, (1, 2, 0))
        selected_modes = np.reshape(tst, (shape[1], shape[0] * (shape[2] - 2)))
        modes_for_fitting = selected_modes
        fitting_freq = all_freq
    elif dataset_index == 5:
        tst = np.transpose(mode_tilts, (1, 2, 0, 3))
        all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
        selected_modes = mode_tilts[:, :, 2:, :2]
        tst = np.transpose(selected_modes, (1, 2, 0, 3))
        selected_modes = np.reshape(tst, (shape[1], shape[0] * (shape[2] - 2) * (shape[3] - 2)))
        modes_for_fitting = selected_modes[:-2]
        fitting_freq = all_freq[:-2]
    elif dataset_index == 6:
        tst = np.transpose(mode_tilts, (1, 2, 0, 3))
        all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
        selected_modes = mode_tilts[1:, :, 2:, 1:3]
        tst = np.transpose(selected_modes, (1, 2, 0, 3))
        selected_modes = np.reshape(tst, (shape[1], (shape[0] - 1) * (shape[2] - 2) * (2)))
        modes_for_fitting = selected_modes[5:-5]
        fitting_freq = all_freq[5:-5]
    elif dataset_index == 7:
        tst = np.transpose(mode_tilts, (1, 2, 0, 3))
        all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
        selected_modes = mode_tilts[:, :, 2:, :2]
        tst = np.transpose(selected_modes, (1, 2, 0, 3))
        selected_modes = np.reshape(tst, (shape[1], shape[0] * (shape[2] - 2) * 2))
        modes_for_fitting = selected_modes[2:-2]
        fitting_freq = all_freq[2:-2]
    elif dataset_index == 8:
            tst = np.transpose(mode_tilts, (1, 2, 0, 3))
            all_modes = np.reshape(tst, (shape[1], shape[0] * shape[2] * shape[3]))
            selected_modes = mode_tilts[:, :, 2:, :2]
            tst = np.transpose(selected_modes, (1, 2, 0, 3))
            selected_modes = np.reshape(tst, (shape[1], shape[0] * (shape[2] - 2) * 2))
            modes_for_fitting = selected_modes[2:-2]
            fitting_freq = all_freq[2:-2]
    else:
        raise ValueError()
    full_frequency_axis = np.linspace(-9, 9, 19)
    available_frequency = np.argwhere([a in fitting_freq for a in full_frequency_axis]).flatten()
    left_index = np.min(available_frequency)
    right_index = np.max(available_frequency)
    left_append = np.full((left_index, modes_for_fitting.shape[1]), np.nan)
    right_append = np.full((18 - right_index, modes_for_fitting.shape[1]), np.nan)
    modes_for_fitting = np.concatenate([left_append, modes_for_fitting, right_append], 0)

    available_frequency = np.argwhere([a in all_freq for a in full_frequency_axis]).flatten()
    left_index = np.min(available_frequency)
    right_index = np.max(available_frequency)
    left_append = np.full((left_index, all_modes.shape[1]), np.nan)
    right_append = np.full((18 - right_index, all_modes.shape[1]), np.nan)
    all_modes = np.concatenate([left_append, all_modes, right_append], 0)
    left_append = np.full((left_index, selected_modes.shape[1]), np.nan)
    right_append = np.full((18 - right_index, selected_modes.shape[1]), np.nan)
    selected_modes = np.concatenate([left_append, selected_modes, right_append], 0)

    return selected_modes, all_modes, modes_for_fitting, full_frequency_axis, config


# EXPERIMENTAL DATA FITTING

def smoothened_find_peaks(x, *args, **kwargs):
    """Simple extension of find_peaks to give more than single-pixel accuracy"""
    smoothened = savgol_filter(x, 5, 3)
    sampling = interp1d(range(len(smoothened)), smoothened, 'quadratic')
    new_x = np.linspace(0, len(smoothened)-1, len(smoothened)*10)
    new_y = sampling(new_x)
    results = find_peaks(new_y, *args, **kwargs)
    return new_x[results[0]], results[1]


# class FitQHOModes:
#     """
#     TODO: sanity checks that the number of expected lobes for band are there
#     """
#     def __init__(self, image, configuration):
#         # if image is None:
#         #     idx = np.random.randint(_data.shape[0])
#         #     idx2 = np.random.randint(_data.shape[1])
#         #     idx3 = np.random.randint(_data.shape[2])
#         #     print(idx, idx2, idx3)
#         #     image = _data[idx, idx2, idx3]
#         # if configuration is None:
#         #     configuration = config
#         self.config = configuration
#         self.smoothened_image = low_pass(normalize(image), self.config['low_pass_threshold'])
#
#         if 'energy_axis' not in self.config:
#             e_roi = self.config['energy_roi']
#             _wvls = spectrometer_calibration('rotation_acton', 803, '2')[e_roi[1]:e_roi[0]:-1]
#             self.energy_axis = 1240 / _wvls
#         else:
#             self.energy_axis = self.config['energy_axis']
#         self.energy_func = interp1d(range(len(self.energy_axis)), self.energy_axis, bounds_error=False, fill_value=np.nan)
#         self.e_inverse = interp1d(self.energy_axis, range(len(self.energy_axis)))
#         if 'k_axis' not in self.config:
#             self.k_axis = np.arange(self.smoothened_image.shape[1], dtype=np.float) - self.config['k_masking']['k0']
#             self.k_axis *= momentum_scale
#         else:
#             self.k_axis = self.config['k_axis']
#         self.k_func = interp1d(range(len(self.k_axis)), self.k_axis, bounds_error=False, fill_value=np.nan)
#         self.k_inverse = interp1d(self.k_axis, range(len(self.k_axis)))
#         # print(self.energy_axis.shape, self.k_axis.shape, self.smoothened_image.shape, image.shape)
#         # plt.figure()
#         # plt.plot(self.energy_axis)
#         # plt.plot(range(10), self.energy_func(range(10)))
#
#     def _mask_by_momentum(self, peaks, k_range=None):
#         # Eliminates peaks that are outside the given momentum range
#         if k_range is None:
#             if 'k_range' in self.config:
#                 k_range = self.config['k_range']
#             else:
#                 k_range = (np.min(peaks[:, 0])-1, np.max(peaks[:, 0]) + 1)
#         return np.logical_and(peaks[:, 0] > k_range[0], peaks[:, 0] < k_range[1])
#
#     def _mask_by_intensity(self, peak_intensities, threshold=0.01):
#         return peak_intensities > threshold * np.max(peak_intensities)
#
#     def mask_untrapped(self, peaks, min_energy=None):
#         config = self.config['k_masking']
#         k0 = config['k0']
#         slope = config['k_slope']
#         if min_energy is None:
#             min_energy = np.max(peaks[:, 1]) - config['e_offset']
#             LOGGER.debug('mask_untrapped min_energy=%g' % min_energy)
#         mask1 = (peaks[:, 0]-k0)*slope+min_energy > peaks[:, 1]
#         mask2 = -(peaks[:, 0]-k0)*slope+min_energy > peaks[:, 1]
#         mask = np.logical_and(mask1, mask2)
#         # fig, ax = plt.subplots(1, 1)
#         # ax.plot(peaks[:, 0], peaks[:, 1], '.')
#         # ax.plot(peaks[:, 0], (peaks[:, 0]-k0)*slope+min_energy)
#         # ax.plot(peaks[:, 0], -(peaks[:, 0]-k0)*slope+min_energy)
#         return peaks[mask], min_energy
#
#     def find_peaks_1d(self, ax=None, **plot_kwargs):
#         config = self.config['peak_finding']
#         peaks = []
#         for idx, x in enumerate(self.smoothened_image.transpose()):
#             # pks = find_peaks(x, **self.config['find_peaks_kw'])[0]
#             pks = smoothened_find_peaks(x, **config['peaks_1d'])[0]
#             peaks += [(idx, pk) for pk in pks]
#         peaks = np.asarray(peaks, dtype=np.float)
#         if self.config['plotting'] and ax is None:
#             _, ax = plt.subplots(1, 1)
#         if ax is not None:
#             imshow(self.smoothened_image, ax, cmap='Greys', diverging=False, norm=LogNorm(),
#                    xaxis=self.k_axis, yaxis=self.energy_axis)
#             defaults = dict(ms=1, ls='None', marker='.')
#             for key, value in defaults.items():
#                 if key not in plot_kwargs:
#                     plot_kwargs[key] = value
#             ax.plot(self.k_func(peaks[:, 0]), self.energy_func(peaks[:, 1]), **plot_kwargs)
#         return peaks
#
#     def find_peaks_2d(self, ax=None, **plot_kwargs):
#         config = self.config['peak_finding']
#         peaks = peak_local_max(self.smoothened_image, **config['peaks_2d'])
#         if self.config['plotting'] and ax is None:
#             _, ax = plt.subplots(1, 1)
#         if ax is not None:
#             defaults = dict(ms=1, ls='None', marker='.')
#             for key, value in defaults.items():
#                 if key not in plot_kwargs:
#                     plot_kwargs[key] = value
#             imshow(self.smoothened_image, ax, cmap='Greys', diverging=False, norm=LogNorm(),
#                    xaxis=self.k_axis, yaxis=self.energy_axis)
#             ax.plot(self.k_func(peaks[:, 1]), self.energy_func(peaks[:, 0]), **plot_kwargs)
#         return peaks[:, ::-1]
#
#     def make_bands(self, peaks1d=None, peaks2d=None, n_bands=None, ax=None, plot_kwargs=None):
#         """Create bands from 1D and 2D peak fitting
#
#         Algorithm proceeds as follows:
#             Starting from 2D peaks, create chains of nearest-neighbours from the 1D peak fitting:
#                 - The neighbouring distance can be configured
#                 - Excludes points that are outside the expected trapping cone (see mask_untrapped)
#                 - Excludes points that are too far away from the expected mode linewidth (prevents considering chains
#                 that climb up the thermal line)
#             Groups the chains into bands:
#                 - Only considers chains that are long enough (min_band_points)
#                 - If the average energy of the chain is similar to an existing band (min_band_distance), append the two
#                 chains (ensures that different lobes of the same band get grouped correctly. Would fail for heavily
#                 tilted bands with high contrast between lobes)
#             Sanity checks:
#                 - The average momentum of the band needs to be near k0 (k_acceptance). Otherwise, exclude
#                 - Sorts the band by energy
#
#         :param peaks1d:
#         :param peaks2d:
#         :param ax:
#         :return:
#         """
#         if plot_kwargs is None:
#             plot_kwargs = dict()
#         if self.config['plotting'] and ax is None:
#             _, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey='all', sharex='all')
#         elif ax is None:
#             ax0 = None
#             ax1 = None
#             ax2 = None
#         else:
#             try:
#                 ax0, ax1, ax2 = ax
#             except Exception as e:
#                 ax0 = ax
#                 ax1 = ax
#                 ax2 = ax
#         if peaks1d is None:
#             peaks1d = self.find_peaks_1d(ax0, **plot_kwargs)
#         if peaks2d is None:
#             peaks2d = self.find_peaks_2d(ax1, **plot_kwargs)
#         tree = KDTree(peaks1d)
#         groups = tree.query_radius(peaks2d, self.config['peak_finding']['neighbourhood'])
#
#         # First, select 2D peaks that have more than 2 nearest neighbours
#         selected_points = []
#         for idx, g in enumerate(groups):
#             if len(g) > self.config['peak_finding']['min_neighbours']:
#                 selected_points += [peaks2d[idx]]
#         selected_points = np.array(selected_points)
#         # Exclude and estimate the parameters for excluding untrapped 2D peaks
#         # TODO: when calculating the min_energy, add some sort of limits/pre-given estimates
#         selected_points, _min_energy = self.mask_untrapped(selected_points)
#
#         # Get chains of 1D peaks. Starting from the selected 2D points, extract chains of 1D peaks that are sequences of
#         # points that are not too far apart
#         config = self.config['chain_making']
#         kd_tree_radius = config['kd_tree_radius']
#         starting_groups = [peaks1d[g] for g in tree.query_radius(selected_points, kd_tree_radius)]
#         LOGGER.debug('%d starting groups %s, %d selected points for chains' % (len(starting_groups), [len(x) for x in starting_groups], len(selected_points)))
#
#         chains = []
#         for g in starting_groups:
#             max_idx = config['max_iterations']
#             stop = False
#             idx = 0
#             new_g = np.copy(g)
#             while not stop:
#                 if idx > max_idx:
#                     raise RuntimeError('Maximum iterations reached')
#                 start_length = len(new_g)
#                 if start_length == 0:
#                     stop = True
#                 else:
#                     indices = tree.query_radius(new_g, kd_tree_radius)
#                     new_g = [peaks1d[g] for g in indices]
#                     new_g = np.concatenate(new_g)
#                     new_g = np.unique(new_g, axis=0)
#                     end_length = len(new_g)
#                     idx += 1
#                     if start_length == end_length:  # idx > max_idx or
#                         stop = True
#             _len1 = len(new_g)
#             # Exclude untrapped polaritons
#             new_g = self.mask_untrapped(new_g, _min_energy)[0]
#             LOGGER.debug('Found chain of length %d (%d after masking) after %d iterations' % (_len1, len(new_g), idx))
#
#             # Only include chains that are larger than a minimum size (in points)
#             if len(new_g) > config['min_chain_size']:
#                 chains += [new_g]
#         # Order the chains in energy, which prevents issues in which joining chains in different orders can give
#         # different bands
#         average_energies = [np.mean(p, 0)[1] for p in chains]
#         argsort = np.argsort(self.energy_func(average_energies))
#         chains = [chains[idx] for idx in argsort]
#
#         tilt_limits = [config['expected_tilt']-config['tilt_width'], config['expected_tilt']+config['tilt_width']]
#
#         def calculate_tilts(_chain, _chains=None):
#             if _chains is None:
#                 _chains = chains
#             tilts = []
#             for chain in _chains:
#                 k1, e1 = [f(x) for f, x in zip((self.k_func, self.energy_func), np.median(chain, 0))]
#                 k0, e0 = [f(x) for f, x in zip((self.k_func, self.energy_func), np.median(_chain, 0))]
#                 deltak = (k1-k0)
#                 deltae = (e1-e0)
#                 tilts += [deltae/deltak]
#             tilts = np.array(tilts)
#             return tilts
#
#         def join_chains_by_tilt(_chains):
#             _new_chains = []
#             _merged_chains = []
#             for chain_index, chain in enumerate(_chains):
#                 if chain_index in _merged_chains:
#                     continue
#                 LOGGER.debug('Chain_index: %d' % chain_index)
#                 tilts = calculate_tilts(chain, _chains)
#                 LOGGER.debug('Tilts: %s' % tilts)
#                 where = np.nonzero(np.logical_or(
#                     np.logical_and(tilts >= tilt_limits[0], tilts <= tilt_limits[1]),
#                     np.isnan(tilts)))[0]
#                 LOGGER.debug('Where: %s   tilts[where]: %s' % (where, tilts[where]))
#                 _merged_chains += [idx for idx in where]
#                 to_merge = [_chains[idx] for idx in where]
#                 merged_chain = np.concatenate(to_merge)
#                 _new_chains += [np.unique(merged_chain, axis=0)]
#                 LOGGER.debug('old_chains: %d   new_chains: %d' % (len(_chains), len(_new_chains)))
#             if len(_chains) > len(_new_chains):
#                 return join_chains_by_tilt(_new_chains)
#             else:
#                 return _new_chains
#
#         LOGGER.debug('Joining chains by tilt')
#         bands = join_chains_by_tilt(chains)
#
#         def calculate_distances(_chain, _chains):
#             distances = []
#             for chain in _chains:
#                 distances += [np.sqrt(np.sum(np.abs(np.mean(chain, 0) - np.mean(_chain, 0))**2))]
#             return np.array(distances)
#
#         def join_chains_by_distance(_chains, threshold=config['min_chain_separation']):
#             _new_chains = []
#             _merged_chains = []
#             for chain_index, chain in enumerate(_chains):
#                 if chain_index in _merged_chains:
#                     continue
#                 LOGGER.debug('Chain_index: %d' % chain_index)
#                 distances = calculate_distances(chain, _chains)
#                 LOGGER.debug('Distances: %s' % distances)
#                 where = np.nonzero(distances < threshold)[0]
#                 LOGGER.debug('Where: %s   tilts[where]: %s' % (where, distances[where]))
#                 _merged_chains += [idx for idx in where]
#                 to_merge = [_chains[idx] for idx in where]
#                 merged_chain = np.concatenate(to_merge)
#                 _new_chains += [np.unique(merged_chain, axis=0)]
#                 LOGGER.debug('old_chains: %d   new_chains: %d' % (len(_chains), len(_new_chains)))
#             if len(_chains) > len(_new_chains):
#                 return join_chains_by_distance(_new_chains)
#             else:
#                 return _new_chains
#         LOGGER.debug('Joining chains by distance')
#         bands = join_chains_by_distance(bands)
#
#         config = self.config['band_filtering']
#         LOGGER.debug('Full chains: %s' % (str([(np.array(band).shape, np.mean(band, 0), np.percentile(band[:, 1], 10)) for band in bands])))
#         # Clipping bands that snake up the thermal line
#         masks = [band[:, 1] > np.percentile(band[:, 1], 90) - config['max_band_linewidth'] for band in bands]
#         LOGGER.debug('Removing %s points in each band because they fall outside the linewidth' % str([np.sum(mask) for mask in masks]))
#         bands = [g[m] for g, m in zip(bands, masks)]
#
#         # Excluding groups that have average momentum too far from 0
#         k0 = self.config['k_masking']['k0']
#         ka = config['k_acceptance']
#         average_energies = np.array([np.mean(p, 0)[1] for p in bands])
#         masks = np.array([k0-ka < np.mean(g[:, 0]) < k0+ka for g in bands])
#         _excluded_indices = np.arange(len(average_energies))[~masks]
#         LOGGER.debug('Excluding bands by momentum: %s with %s energy %s momentum' % (_excluded_indices,
#                                                                          self.energy_func(average_energies[~masks]),
#                                                                          [self.k_func(np.mean(bands[x], 0)) for x in _excluded_indices]))
#         bands = [g for g, m in zip(bands, masks) if m]
#
#         # Keeping only bands that are larger than some minimum size
#         bands = [band for band in bands if len(band) > config['min_band_size']]
#
#         # Sorting by energy
#         average_energies = [np.mean(p, 0)[1] for p in bands]
#         argsort = np.argsort(self.energy_func(average_energies))
#         bands = [bands[idx] for idx in argsort]
#
#         # Transform into units
#         bands = [np.array([self.k_func(band[:, 0]), self.energy_func(band[:, 1])]).transpose() for band in bands]
#
#         # Make a fixed length tuple if required
#         if n_bands is not None:
#             if n_bands < len(bands):
#                 bands = bands[:n_bands]
#             else:
#                 bands += [np.empty((0, 2))]*(n_bands-len(bands))
#
#         LOGGER.debug('Final bands: %s' % str(list([len(band), np.mean(band[:, 0]), np.mean(band[:, 1])] for band in bands)))
#
#         if ax2 is not None:
#             imshow(self.smoothened_image, ax2, diverging=False, cmap='Greys', norm=LogNorm(), cbar=False,
#                    xaxis=self.k_axis, yaxis=self.energy_axis)
#             defaults = dict(ms=1, ls='None', marker='.')
#             for key, value in defaults.items():
#                 if key not in plot_kwargs:
#                     plot_kwargs[key] = value
#             ax2.plot(self.k_func(selected_points[:, 0]), self.energy_func(selected_points[:, 1]), color='k', **plot_kwargs)
#             for s in bands:
#                 ax2.plot(s[:, 0], s[:, 1], **plot_kwargs)
#         return bands
#
#     def _calculate_slice(self, linear_fit, band_width, image=None):
#         if image is None:
#             image = self.smoothened_image
#         full_func = np.poly1d(linear_fit)
#         slope_func = np.poly1d([linear_fit[0], 0])
#         start_slice = (full_func(0) - band_width / 2, 0)
#         end_slice = (image.shape[1], band_width)
#         v1 = [slope_func(1), 1]
#         v2 = [1, -slope_func(1)]
#         vectors = np.array([v1, v2])
#         vectors = [v / np.linalg.norm(v) for v in vectors]
#         LOGGER.debug('Slice at: %s %s %s' % (vectors, start_slice, end_slice))
#         return vectors, start_slice, end_slice
#
#     def analyse_bands(self, bands=None, n_bands=None, gs=None):
#         # Fit a linear trend to it:
#         #   Return the k0 energy and the tilt
#         # Mode profiles
#         #   Density distribution?
#         # Linewidths?
#         #   Get energy modes at the peaks of the mode profiles?
#         if bands is None:
#             bands = self.make_bands()
#         if n_bands is None:
#             n_bands = len(bands)
#
#         _fits = []
#         _coords = []
#         energies = []
#         tilts = []
#         slices = []
#         for idx in range(n_bands):
#             band_width = 10
#             if idx > len(bands) or bands[idx].size == 0:
#                 energies += [np.nan]
#                 tilts += [np.nan]
#                 slices += [np.full((self.smoothened_image.shape[1], band_width), np.nan)]
#             # elif:
#             #     np.isempty(bands[idx]
#             else: # idx < len(bands):
#                 band = bands[idx]
#                 fit = np.polyfit(band[:, 0], band[:, 1], 1)
#                 # func = np.poly1d(fit)
#                 k0_energy = np.poly1d(fit)(0)  # func(self.config['k_masking']['k0'])
#                 tilt = fit[0]
#                 inverse_fit = np.polyfit(self.k_inverse(band[:, 0]), self.e_inverse(band[:, 1]), 1)
#                 vectors, start_slice, end_slice = self._calculate_slice(inverse_fit, band_width)
#                 slice, coords = pg.affineSlice(self.smoothened_image, end_slice, start_slice, vectors, (0, 1), returnCoords=True)
#                 _coords += [coords]
#                 _fits += [fit]
#                 energies += [k0_energy]
#                 tilts += [tilt]
#                 slices += [slice]
#             # else:
#             #     energies += [np.nan]
#             #     tilts += [np.nan]
#             #     slices += [np.full((self.smoothened_image.shape[1], band_width), np.nan)]
#
#         if self.config['plotting'] and gs is None:
#             fig = plt.figure()
#             gs = gridspec.GridSpec(1, 2, fig)
#         if gs is not None:
#             ax0 = plt.subplot(gs[0])
#             imshow(self.smoothened_image, ax0, diverging=False, cmap='Greys', norm=LogNorm(), xaxis=self.k_axis, yaxis=self.energy_axis)
#             # ax0.imshow(self.smoothened_image, cmap='Greys', norm=LogNorm(), aspect='auto')
#
#             gs1 = gridspec.GridSpecFromSubplotSpec(len(slices), 1, gs[1])
#             axs = gs1.subplots()
#             for idx, _fit, energ, tilt, slice in zip(range(len(_fits)), _fits, energies, tilts, slices):
#                 color = cm.get_cmap('Iris', len(_fits))(idx)
#                 func = np.poly1d(_fit)
#                 x_points = self.k_func([0, self.smoothened_image.shape[1]-1])
#                 ax0.plot(x_points, func(x_points))
#                 ax0.plot(self.k_func(_coords[idx][1].flatten()), self.energy_func(_coords[idx][0].flatten()),
#                          '.', ms=0.3, color=color)
#                 try:
#                     axs[idx].imshow(slice.transpose())
#                     colour_axes(axs[idx], color)
#                 except TypeError:
#                     axs.imshow(slice.transpose())
#
#         return np.array(energies), np.array(tilts), np.array(slices)


def _make_axis_functions(configuration):
    if 'energy_axis' not in configuration:
        e_roi = configuration['energy_roi']
        _wvls = spectrometer_calibration('rotation_acton', 803, '2')[e_roi[1]:e_roi[0]:-1]
        energy_axis = 1240 / _wvls
    else:
        energy_axis = configuration['energy_axis']
    energy_func = interp1d(range(len(energy_axis)), energy_axis, bounds_error=False, fill_value=np.nan)
    e_inverse = interp1d(energy_axis, range(len(energy_axis)))

    if 'k_axis' not in configuration:
        k_axis = np.arange(image.shape[1], dtype=np.float) - configuration['k_masking']['k0']
        k_axis *= momentum_scale
    else:
        k_axis = configuration['k_axis']
    k_func = interp1d(range(len(k_axis)), k_axis, bounds_error=False, fill_value=np.nan)
    k_inverse = interp1d(k_axis, range(len(k_axis)))
    return energy_func, k_func, e_inverse, k_inverse


class SingleImageModeDetection:
    """
    example_config = dict(
        plotting=False, #['make_bands'],  #True,  # ['image_preprocessing', 'peak_finding']
        image_preprocessing=dict(
            normalization_percentiles=[0, 100],
            low_pass_threshold=0.4),
        peak_finding=dict(peak_width=3, savgol_filter=dict(), find_peaks=dict(height=0.007, prominence=0.00001)),
        clustering=dict(#shear=-0.03, #scale=0.01,
                        # AgglomerativeClustering=dict(n_clusters=15, distance_threshold=None,
                        #                              compute_distances=True),
                        energy_limit=30,
                        min_cluster_size=15, min_cluster_distance=3),
        make_bands=dict(k0=-0.2, k_acceptance=1, bandwidth=0.3),
        analyse_bands=dict(k_range_fit=0.5)
    )
    """
    def __init__(self, image, configuration):
        self.configuration = deepcopy(configuration)  #{**configuration}

        if 'plotting' not in configuration:
            self.configuration['plotting'] = False
        self.raw_image = image
        self.image = self._preprocess(image)

        if 'energy_axis' not in self.configuration:
            e_roi = self.configuration['energy_roi']
            _wvls = spectrometer_calibration('rotation_acton', 803, '2')[e_roi[1]:e_roi[0]:-1]
            self.energy_axis = 1240 / _wvls
        else:
            self.energy_axis = self.configuration['energy_axis']
        self.energy_func = interp1d(range(len(self.energy_axis)), self.energy_axis, bounds_error=False, fill_value=np.nan)
        self.e_inverse = interp1d(self.energy_axis, range(len(self.energy_axis)))
        if 'k_axis' not in self.configuration:
            self.k_axis = np.arange(self.image.shape[1], dtype=np.float) - self.configuration['k_masking']['k0']
            self.k_axis *= momentum_scale
        else:
            self.k_axis = self.configuration['k_axis']
        self.k_func = interp1d(range(len(self.k_axis)), self.k_axis, bounds_error=False, fill_value=np.nan)
        self.k_inverse = interp1d(self.k_axis, range(len(self.k_axis)))

    def _to_plot(self, step_name):
        if isinstance(self.configuration['plotting'], bool):
            return self.configuration['plotting']
        elif step_name in self.configuration['plotting']:
            return True
        else:
            return False

    def _preprocess(self, image):
        """
        TODO: low_pass with different amplitudes in different directions? So smoothen k, but keep E sharp
        :param image:
        :return:
        """
        if 'image_preprocessing' in self.configuration:
            if self._to_plot('image_preprocessing'):
                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                imshow(image, axs[0], cbar=False, diverging=False)
            defaults = dict(normalization_percentiles=[0, 100],
                            low_pass_threshold=0.4)
            config = {**defaults, **self.configuration['image_preprocessing']}
            if 'normalization_percentiles' in config:
                image = normalize(image, config['normalization_percentiles'])
                if self._to_plot('image_preprocessing'): imshow(image, axs[1], cbar=False, diverging=False)
            if 'low_pass_threshold' in config:
                image = low_pass(image, config['low_pass_threshold'])
                if self._to_plot('image_preprocessing'): imshow(image, axs[2], cbar=False, diverging=False)
        return image

    @staticmethod
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

    def _calculate_shear(self, points=None):
        if points is None:
            points = self.find_peaks_1d()
        pair_points = np.array(list(combinations(points, 2)))
        vectors = np.squeeze(np.diff(pair_points, axis=1))
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        histogram, bins = np.histogram(angles, 30, [-0.2, 0.2])
        bin_centers = np.mean([bins[:-1], bins[1:]], 0)
        mode = bin_centers[np.argmax(histogram)]

        if self._to_plot('calculate_shear'):
            fig, ax = plt.subplots(1, 1, num='Shear')
            ax.hist(angles, 30, [-0.2, 0.2])
            ax.set_title('mean=%g  median=%g  mode=%g' % (np.mean(angles), np.median(angles), mode))

        shear = - 1 * np.tan(mode) * np.sign(self.configuration['laser_angle'])
        LOGGER.debug('Found shear: %g' % shear)
        return shear

    def find_peaks_1d(self, ax=None, **plot_kwargs):
        config = self.configuration['peak_finding']
        if 'peak_width' in config:
            for key in ['savgol_filter', 'find_peaks']:
                if key not in config:
                    config[key] = dict()
            if 'window_length' not in config['savgol_filter']:
                wlen = 2*int(np.round(config['peak_width']))
                config['savgol_filter']['window_length'] = wlen + (wlen+1) % 2
            if 'width' not in config['find_peaks']:
                config['find_peaks']['width'] = config['peak_width']

        peaks = []
        for idx, x in enumerate(self.image.transpose()):
            pks = self._find_peaks(x, **config['find_peaks'])[0]
            peaks += [(idx, pk) for pk in pks]
        peaks = np.asarray(peaks, dtype=float)

        if self._to_plot('peak_finding') and ax is None:
            _, ax = plt.subplots(1, 1, num='peak_finding')
        if ax is not None:
            imshow(self.image, ax, cmap='Greys', diverging=False, norm=LogNorm(),
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            plot_kwargs = {**dict(ms=1, ls='None', marker='.'), **plot_kwargs}
            ax.plot(self.k_func(peaks[:, 0]), self.energy_func(peaks[:, 1]), **plot_kwargs)
        return peaks

    def cluster(self, points=None, ax=None):
        if points is None:
            points = self.find_peaks_1d()
        config = self.configuration['clustering']

        if 'energy_limit' in config:
            max_energy = np.argmax(np.sum(self.image, -1))
            try:
                e_limits = [max_energy - config['energy_limit'][1],
                            max_energy + config['energy_limit'][0]]
            except:
                e_limits = [max_energy - config['energy_limit'],
                            self.image.shape[-1]]
            mask = np.logical_and(points[:, 1] < e_limits[1],
                                  points[:, 1] > e_limits[0])
            points = points[mask]
        if 'scale' not in config:
            config['scale'] = 1.2 / self.image.shape[0]
        if 'shear' not in config:
            config['shear'] = self._calculate_shear(points)

        shear_matrix = [[1, config['shear']], [0, 1]]
        scaling_matrix = [[config['scale'], 0], [0, 1]]

        sheared = np.dot(points, shear_matrix)
        scaled = np.dot(sheared, scaling_matrix)

        if 'AgglomerativeClustering' in config:
            kwargs = config['AgglomerativeClustering']
        else:
            kwargs = dict()
        # defaults = dict(n_clusters=None, distance_threshold=1, compute_full_tree=True, linkage='single')
        defaults = dict(n_clusters=len(points) // 10, distance_threshold=None,
                        compute_full_tree=True, linkage='single')
        kwargs = {**defaults, **kwargs}

        model = AgglomerativeClustering(**kwargs)
        clusters = model.fit(scaled)
        first_labels = deepcopy(clusters.labels_)
        labels = clusters.labels_

        if 'noise_cluster_size' in config:
            # By first using many clusters, and then removing small clusters, we get rid of noisy regions
            _labels = list(labels)
            counts = np.array([_labels.count(x) for x in _labels])
            mask = counts > config['noise_cluster_size']
            labels[~mask] = -1

        if 'min_cluster_distance' in config and model.n_clusters_ > 1:
            # DON'T USE THE CLUSTER CENTERS
            cluster_centers = np.array([np.mean(scaled[labels == l], 0) for l in range(model.n_clusters_)])
            cluster_indices = np.arange(len(cluster_centers))
            _mask = ~np.isnan(np.mean(cluster_centers, 1))
            if len(cluster_centers[_mask]) > 1:
                _model = AgglomerativeClustering(n_clusters=None, distance_threshold=config['min_cluster_distance'],
                                                 compute_full_tree=True, linkage='single')
                clusters_of_clusters = _model.fit(cluster_centers[_mask])
                new_labels = np.copy(labels)
                for idx, new_label in zip(cluster_indices[_mask], clusters_of_clusters.labels_):
                    new_labels[labels == idx] = new_label
                labels = new_labels

        # if 'noise_cluster_size' in config:
        #     # By first using many clusters, and then removing small clusters, we get rid of noisy regions
        #     _labels = list(labels)
        #     counts = np.array([_labels.count(x) for x in _labels])
        #     mask = counts > config['noise_cluster_size']
        #     labels[~mask] = -1

        if 'min_cluster_size' in config:
            _labels = list(labels)
            counts = np.array([_labels.count(x) for x in _labels])
            mask = counts > config['min_cluster_size']
            labels[~mask] = -1

        if 'bandwidth' in config:
            for label in range(model.n_clusters_):
                _points = scaled[labels == label]
                if len(_points) > 0:
                    width = np.percentile(_points[:, 1], 90) - np.percentile(_points[:, 1], 10)
                    if width > config['bandwidth']:
                        labels[labels == label] = -1
                        average_energy = np.mean(_points[:, 1])
                        LOGGER.debug('Excluding band %d by bandwidth: %s energy %s bandwidth' % (label, average_energy, width))

        LOGGER.debug('Found %d clusters with labels %s' % (len(np.unique(labels[mask])), np.unique(labels[mask])))

        masked_clusters = [points[labels == l] for l in np.unique(labels) if l >= 0]

        if self._to_plot('clustering'):
            fig, axs = plt.subplots(1, 6, num='clustering')
            axs[0].scatter(self.k_func(points[:, 0]), self.energy_func(points[:, 1]), c=first_labels)
            axs[1].scatter(*sheared.transpose(), c=first_labels)
            axs[2].scatter(*scaled.transpose(), c=first_labels)
            axs[2].scatter(*cluster_centers.transpose(), c='k')
            sizes = np.linspace(5, 15, 5)[np.arange(len(clusters_of_clusters.labels_)) % 5]
            axs[3].scatter(*cluster_centers[_mask].transpose(), c=clusters_of_clusters.labels_, s=sizes)
            axs[4].scatter(*points.transpose(), c=labels)
            # axs[-1].scatter(self.k_func(points[:, 0]), self.energy_func(points[:, 1]), c=labels)
            [axs[-1].plot(self.k_func(x[:, 0]), self.energy_func(x[:, 1]), '.') for x in masked_clusters]
            [imshow(self.image, ax, cmap='Greys', diverging=False, norm=LogNorm(), cbar=False,
                    xaxis=self.k_axis, yaxis=self.energy_axis) for ax in axs[[0, -1]]]
        elif ax is not None:
            imshow(self.image, ax, cmap='Greys', diverging=False, norm=LogNorm(), cbar=False,
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            ax.scatter(self.k_func(points[mask, 0]), self.energy_func(points[mask, 1]), c=labels[mask], s=0.1)

        return masked_clusters

    def _make_units(self, points):
        return np.array([self.k_func(points[:, 0]), self.energy_func(points[:, 1])]).transpose()

    def make_bands(self, clusters=None, ax=None):
        # Order clusters by energy
        # Exclude clusters by bandwidth and central momentum
        if clusters is None:
            clusters = [self._make_units(c) for c in self.cluster()]

        config = self.configuration['make_bands']

        average_energies = [np.mean(p[:, 1]) for p in clusters]
        argsort = np.argsort(average_energies)
        bands = [clusters[idx] for idx in argsort]

        if 'k_acceptance' in config:
            # Excluding groups that have average momentum too far from 0
            # TODO: using the mean momentum of a band leads to issues when the band extends towards untrapped polaritons
            #  Two solutions:
            # * Find a way of systematically removing untrapped polaritons
            # * Calculate the symmetry of the band around k0, instead of the mean, e.g. grab the nearest N points to k0 and get the average momentum of those

            k0 = config['k0']
            ka = config['k_acceptance']
            # average_energies = np.array([np.mean(p[:, 1], 0) for p in bands])
            # masks = np.array([k0-ka < np.mean(g[:, 0]) < k0+ka for g in bands])
            masks = []
            for idx, band in enumerate(bands):
                # nearest_points = band[np.abs(band[:, 0]) < k_limit]
                left_points = band[band[:, 0] < 0][:, 0]
                right_points = band[band[:, 0] >= 0][:, 0]
                left_points = left_points[np.argsort(np.abs(left_points))]
                right_points = right_points[np.argsort(np.abs(right_points))]
                n_points = int(np.min([10, np.max([len(b) for b in [left_points, right_points]])]))
                left_right_asymmetry = np.abs(np.mean(-left_points[:n_points]) - np.mean(right_points[:n_points]))
                mask = left_right_asymmetry < ka
                if mask:
                    average_energy = np.mean(band[:, 1])
                    LOGGER.debug('Excluding band %d by momentum: %s energy %s momentum asymmetry' % (idx, average_energy, left_right_asymmetry))
                masks += [mask]
            masks = np.array(masks)
            bands = [g for g, m in zip(bands, masks) if m]

        if 'bandwidth' in config:
            widths = np.array([np.ptp(band[:, 1]) for band in bands])
            mask = widths < config['bandwidth']
            bands = [band for m, band in zip(mask, bands) if m]
            if mask.any():
                for idx in range(len(bands)):
                    if mask[idx]:
                        average_energy = np.mean(bands[idx][:, 1])
                        LOGGER.debug('Excluding band %d by bandwidth: %s energy %s bandwidth' % (idx, average_energy,
                                                                                                 widths[idx]))

        if self._to_plot('make_bands'):
            fig, ax = plt.subplots(1, 1, num='make_bands')
            imshow(self.image, ax, cmap='Greys', diverging=False, norm=LogNorm(), cbar=False,
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            [ax.plot(*band.transpose(), '.') for band in bands]
        elif ax is not None:
            imshow(self.image, ax, cmap='Greys', diverging=False, norm=LogNorm(), cbar=False,
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            [ax.plot(*band.transpose(), '.', ms=0.2) for band in bands]
        return bands

    def _calculate_slice(self, linear_fit, band_width, image=None):
        if image is None:
            image = self.image
        full_func = np.poly1d(linear_fit)
        slope_func = np.poly1d([linear_fit[0], 0])
        start_slice = (full_func(0) - band_width / 2, 0)
        end_slice = (image.shape[1], band_width)
        v1 = [slope_func(1), 1]
        v2 = [1, -slope_func(1)]
        vectors = np.array([v1, v2])
        vectors = [v / np.linalg.norm(v) for v in vectors]
        LOGGER.debug('Slice at: %s %s %s' % (vectors, start_slice, end_slice))
        return vectors, start_slice, end_slice

    def analyse_bands(self, bands=None, n_bands=None, gs=None):
        if bands is None:
            bands = self.make_bands()
        if n_bands is None:
            n_bands = len(bands)

        config = self.configuration['analyse_bands']

        _fits = []
        _coords = []
        energies = []
        tilts = []
        slices = []
        for idx in range(n_bands):
            band_width = 4
            def _return_NaN():
                return np.nan, np.nan, np.full((self.image.shape[1], band_width), np.nan), np.nan, np.nan
            if idx >= len(bands):
                k0_energy, tilt, slice, coords, fit = _return_NaN()
            else:
                band = bands[idx]
                if 'k_range_fit' in config:
                    mask = config['k_range_fit'] > np.abs(band[:, 0])
                    band = band[mask]
                    LOGGER.debug('Excluding %d points in band %d from linear fit' % (np.sum(~mask), idx))
                if len(band) > 0:
                    fit = np.polyfit(band[:, 0], band[:, 1], 1)
                    # func = np.poly1d(fit)
                    k0_energy = np.poly1d(fit)(0)  # func(self.config['k_masking']['k0'])
                    tilt = fit[0]
                    inverse_fit = np.polyfit(self.k_inverse(band[:, 0]), self.e_inverse(band[:, 1]), 1)
                    vectors, start_slice, end_slice = self._calculate_slice(inverse_fit, band_width)
                    slice, coords = pg.affineSlice(self.image, end_slice, start_slice, vectors, (0, 1),
                                                   returnCoords=True)
                else:
                    k0_energy, tilt, slice, coords, fit = _return_NaN()

            _coords += [coords]
            _fits += [fit]
            energies += [k0_energy]
            tilts += [tilt]
            slices += [slice]
            # else:
            #     energies += [np.nan]
            #     tilts += [np.nan]
            #     slices += [np.full((self.smoothened_image.shape[1], band_width), np.nan)]

        if self._to_plot('analyse_bands'):
            fig = plt.figure(num='analyse_bands')
            gs = gridspec.GridSpec(1, 2, fig)
        if gs is not None:
            ax0 = plt.subplot(gs[0])
            imshow(self.image, ax0, diverging=False, cmap='Greys', norm=LogNorm(), xaxis=self.k_axis,
                   yaxis=self.energy_axis)

            gs1 = gridspec.GridSpecFromSubplotSpec(2*len(slices), 1, gs[1])
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
                        imshow(slice.transpose(), axs[-1-2*idx], cbar=False, diverging=False)
                        y = np.sum(slice, 1)
                        axs[-2-2*idx].semilogy(y, color=color)
                        [axs[-2-2*idx].axvline(x) for x in find_peaks(-y, width=6, distance=6, prominence=0.01)[0]]
                        colour_axes(axs[-1-2*idx], color)
                        colour_axes(axs[-2-2*idx], color)
                    except TypeError:
                        imshow(slice.transpose(), axs, cbar=False, diverging=False)

        return np.array(energies), np.array(tilts), np.array(slices)


class FullDatasetModeDetection:
    """
    Ideas:
        Use the frequency series to estimate the shear (instead of doing it on each image)

    More:
        Build manifolds over the parameters
    """
    def __init__(self, dataset, configuration):
        self.configuration = configuration
        # if 'data_axes_order' in configuration:
        #     self.dataset = np.transpose(dataset, configuration['data_axes_order'])
        # else:
        self.dataset = dataset
        # return

    def _choose_random_image(self):
        image, indices = random_choice(self.dataset, tuple(range(len(self.dataset.shape) - 2)), return_indices=True)
        LOGGER.debug('Indices %s' % (indices,))
        return image

    def _cluster_single_image(self, image=None, ax=None):
        if image is None: image = self._choose_random_image()
        cls = SingleImageModeDetection(image, self.configuration)
        return cls.cluster(ax=ax)

    def _make_bands_single_image(self, image=None, ax=None, configuration=None):
        if image is None: image = self._choose_random_image()
        if configuration is None: configuration = self.configuration
        cls = SingleImageModeDetection(image, configuration)
        return cls.make_bands(ax=ax)

    def _analyse_bands_single_image(self, image=None, bands=None, n_bands=None, gs=None):
        if image is None: image = self._choose_random_image()
        cls = SingleImageModeDetection(image, self.configuration)
        return cls.analyse_bands(bands=bands, n_bands=n_bands, gs=gs)

    def _make_subplots(self, name=None):
        iteraxes = self.dataset.shape[:-2]
        if len(iteraxes) == 2:
            _fig, axs = plt.subplots(*iteraxes, sharex=True, sharey=True, figsize=(10, 10), num=name)
            fig = [_fig]
        else:
            axs = []
            fig = []
            for idx in range(iteraxes[0]):
                _fig, _axs = plt.subplots(*iteraxes[1:], sharex=True, sharey=True, figsize=(10, 10), num='%s_%d' % (name, idx))
                axs += [_axs]
                fig += [_fig]
            axs = np.array(axs)
        return fig, axs, iteraxes

    def cluster(self):
        LOGGER.info('FullDatasetModeDetection.cluster')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figs, axs, iteraxes = self._make_subplots('cluster')
            clusters = []
            progress = 0
            for indices in np.ndindex(iteraxes):
                if (indices[0] * 100 / iteraxes[0]) > progress:
                    LOGGER.info('cluster %d %%' % (indices[0] * 100 / iteraxes[0]))
                    progress += 10
                data = self.dataset[indices]
                clusters += [self._cluster_single_image(data, axs[indices])]
            return clusters, figs
            # for idx0 in range(iteraxes[0]):
            #     for idx1 in range(iteraxes[1]):
            #         self._cluster_single_image(self.dataset[idx0, idx1], axs[idx0, idx1])

    def make_bands(self):
        LOGGER.info('FullDatasetModeDetection.make_bands')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figs, axs, iteraxes = self._make_subplots('make_bands')
            bands = []
            progress = 0
            for indices in np.ndindex(iteraxes):
                if (indices[0] * 100 / iteraxes[0]) > progress:
                    LOGGER.info('make_bands %d %%' % (indices[0] * 100 / iteraxes[0]))
                    progress += 10
                data = self.dataset[indices]
                bands += [self._make_bands_single_image(data, axs[indices])]
            bands = np.reshape(bands, iteraxes)
            return bands, figs
            # iteraxes = self.dataset.shape[:2]
            # fig, axs = plt.subplots(*iteraxes, sharex=True, sharey=True, figsize=(10, 10))
            # for idx0 in range(iteraxes[0]):
            #     for idx1 in range(iteraxes[1]):
            #         self._make_bands_single_image(self.dataset[idx0, idx1], axs[idx0, idx1])

    def analyse_bands(self, bands=None, n_bands=None):
        LOGGER.info('FullDatasetModeDetection.analyse_bands')
        with warnings.catch_warnings():
            if bands is None:
                bands, _ = self.make_bands()

            warnings.simplefilter("ignore")
            # _figs, _, iteraxes = self._make_subplots('analyse_bands')
            # [plt.close(fig) for fig in _figs]

            iteraxes = self.dataset.shape[:-2]
            if n_bands is None:
                n_bands = int(np.max([len(bands[indices]) for indices in np.ndindex(iteraxes)]))

            energies = []
            tilts = []
            slices = []
            progress = 0
            for indices in np.ndindex(iteraxes):
                if (indices[0] * 100 / iteraxes[0]) > progress:
                    LOGGER.info('analyse_bands %d %%' % (indices[0] * 100 / iteraxes[0]))
                    progress += 10
                data = self.dataset[indices]
                _bands = bands[indices]
                e, t, s = self._analyse_bands_single_image(data, _bands, n_bands)
                energies += [e]
                tilts += [t]
                slices += [s]
            LOGGER.debug('Reshaping analysed band data')
            energies = np.reshape(energies, iteraxes + (n_bands, ))
            tilts = np.reshape(tilts, iteraxes + (n_bands, ))
            slices = np.reshape(slices, iteraxes + s.shape)

            # iteraxes = self.dataset.shape[:2]
            # # fig, axs = plt.subplots(*iteraxes, sharex=True, sharey=True, figsize=(10, 10))
            # energies = []
            # tilts = []
            # slices = []
            # for idx0 in range(iteraxes[0]):
            #     _energy = []
            #     _tilt = []
            #     _slice = []
            #     for idx1 in range(iteraxes[1]):
            #         e, t, m = self._analyse_bands_single_image(self.dataset[idx0, idx1], n_bands)
            #         _energy += [e]
            #         _tilt += [t]
            #         _slice += [m]
            #     energies += [_energy]
            #     tilts += [_tilt]
            #     slices += [_slice]
            # energies = np.array(energies)
            # tilts = np.array(tilts)
            # slices = np.array(slices)
            # # print(tilts.shape, energies.shape, slices.shape)
            # # print(slices[0,0,0].shape)

            LOGGER.debug('Plotting analysed band data')
            # print(tilts.shape, slices.shape, energies.shape, len(iteraxes))
            if self.configuration['plotting']:
                if len(iteraxes) == 2:
                    fig1, _, _ = subplots(tilts,
                                          partial(imshow, diverging=False, cmap='RdBu', cbar=False, vmin=-1e-4, vmax=1e-4),
                                          (-1,))
                    fig2, _, _ = subplots(normalize(np.nanmean(slices, -1), axis=-1), waterfall, (0, 1))
                    # freq_idx = tilts.shape[0] // 2
                    # waterfall(normalize(np.nanmean(slices[freq_idx, -1], -1), axis=-1))
                    fig3, _, _ = subplots((energies-np.nanmean(energies))*1e3,
                                          partial(imshow, diverging=False, vmin=-1, vmax=1, cmap='viridis', cbar=False),
                                          (-1, ))
                else:
                    fig1, _, _ = subplots(tilts,
                                          partial(imshow, diverging=False, cmap='RdBu', cbar=False, vmin=-1e-4, vmax=1e-4),
                                          (0, -1,))
                    fig2, _, _ = subplots(normalize(np.nanmean(slices[-1], -1), axis=-1), waterfall, (0, 1))
                    # freq_idx = tilts.shape[0] // 2
                    # waterfall(normalize(np.nanmean(slices[freq_idx, -1], -1), axis=-1))
                    fig3, _, _ = subplots((energies-np.nanmean(energies))*1e3,
                                          partial(imshow, diverging=False, vmin=-1, vmax=1, cmap='viridis', cbar=False),
                                          (0, -1,))
            else:
                fig1 = None
                fig2 = None
                fig3 = None
            # plt.show()
            return energies, tilts, slices, (fig1, fig2, fig3)


class AutoParameterTree(ParameterTree):
    """Creates a pyqtgraph.ParameterTree from a Python dictionary

    Currently valid types for the dictionary values are: dict, int, float, str, list
    """
    def __init__(self, dictionary):
        super(AutoParameterTree, self).__init__()

        self.dictionary = dictionary
        self.parameters = Parameter.create(name='params', type='group', children=self.make_parameters(dictionary))
        self.setParameters(self.parameters, showTop=False)

    def make_parameters(self, dictionary):
        pg_parameters = []
        for key, value in dictionary.items():
            if isinstance(value, dict):
                pg_parameters += [dict(name=key, type='group', children=self.make_parameters(value))]
            elif isinstance(value, int):
                pg_parameters += [dict(name=key, type='int', value=value)]
            elif isinstance(value, float):
                pg_parameters += [dict(name=key, type='float', value=value, step=value/100)]
            elif isinstance(value, str) or isinstance(value, list):
                pg_parameters += [dict(name=key, type='str', value=value)]
            else:
                LOGGER.debug('Unrecognised type: %s  %s %s' % (type(value), key, value))
        return pg_parameters

    def make_dictionary(self, parameters=None):
        if parameters is None:
            parameters = self.parameters.saveState()['children']
        dictionary = dict()
        for key, value in parameters.items():
            if value['type'] == 'group':
                if 'children' in value:
                    dictionary[key] = self.make_dictionary(value['children'])
                else:
                    dictionary[key] = None
            else:
                if value['type'] == 'str':
                    if value['value'].startswith('['):
                        dictionary[key] = eval(value['value'])
                    else:
                        dictionary[key] = value['value']
                else:
                    dictionary[key] = value['value']
        return dictionary

    def default(self):
        self.parameters = Parameter.create(name='params', type='group', children=self.make_parameters(self.dictionary))
        self.setParameters(self.parameters, showTop=False)


# TODO:
#     - Add shear to the interactive
#     - Allow one to also group bands in the interactive band listing
#     - If bands already exist, plot them
#     - If no configuration file exists, create it from the default
class InteractiveAnalysis(InteractiveBase):
    def __init__(self, images, configuration=None, variables=None):
        self.configuration = configuration
        self.analysis_class = FullDatasetModeDetection(images, configuration)

        super(InteractiveAnalysis, self).__init__(self.analysis_class.dataset, variables)

        self.all_bands = np.asarray(self._none_array(self._original_shape[:-2]), dtype=object)
        self.all_configurations = np.asarray(self._none_array(self._original_shape[:-2]), dtype=object)

    def get_qt_ui(self):
        return InteractiveAnalysisUI(self)


class InteractiveAnalysisUI(InteractiveBaseUi):
    def __init__(self, interactive_base):
        super(InteractiveAnalysisUI, self).__init__(interactive_base)
        _, _, self.e_to_px, self.k_to_px = _make_axis_functions(self.object.configuration)
        self._band_data_items = []
        try:
            self.ImageDisplay.axis_values['bottom'] = self.object.configuration['k_axis']
            self.ImageDisplay.axis_values['left'] = self.object.configuration['energy_axis']
            gui_axes = self.ImageDisplay.get_axes()
            for ax, name in zip(gui_axes, ["bottom", "left", "top", "right"]):
                if self.ImageDisplay.axis_values[name] is not None:
                    setattr(ax, 'axis_values', self.ImageDisplay.axis_values[name])
                if self.ImageDisplay.axis_units[name] is not None:
                    ax.setLabel(self.ImageDisplay.axis_units[name])
        except Exception as e:
            print(e)

        self.configuration_tree = AutoParameterTree(self.object.configuration)
        self.splitter.addWidget(self.configuration_tree)

        self.default_button = QtWidgets.QPushButton('Default')
        self.default_button.clicked.connect(self._default_tree)
        self.splitter_2.insertWidget(1, self.default_button)

        self.nonnan_button = QtWidgets.QPushButton('Go to non NaN')
        self.nonnan_button.clicked.connect(self._go_to_non_nan)
        self.splitter_2.insertWidget(1, self.nonnan_button)

    def _plot(self):
        super(InteractiveAnalysisUI, self)._plot()
        if type(self.object.all_bands[tuple(self._indxs)]) != float:
            bands = np.asarray(self.object.all_bands[tuple(self._indxs)])
            try:
                self._plot_bands(bands)
            except:
                print('Failed')

    def _default_tree(self):
        self.configuration_tree.default()

    def _go_to_non_nan(self):
        current_indices = np.array([self._indxs]).transpose()
        flat_index = np.ravel_multi_index(current_indices, self.object._original_shape[:-2])[0]

        while flat_index < np.prod(self.object._original_shape[:-2]) - 1:
            next_indices = np.unravel_index(flat_index, self.object._original_shape[:-2])
            if type(self.object.all_bands[next_indices]) != float:
                flat_index += 1
            else:
                break
        for idx, (old, new) in enumerate(zip(current_indices.flatten(), next_indices)):
            if old != new:
                self.indx_spinboxes[idx].setValue(new)

    # def new_image(self):
    #     super(InteractiveAnalysisUI, self).new_image()
        # print('Clearing %d band_data_items' % len(self._band_data_items))
        # [self.ImageDisplay.removeItem(p) for p in self._band_data_items]
        # self.configuration_tree.default()

    def analyse(self):
        img = self.object.images[tuple(self._indxs)]
        configuration = self.configuration_tree.make_dictionary()
        if configuration['image_preprocessing'] is None:
            configuration['image_preprocessing'] = dict()
        configuration = {**self.object.configuration, **configuration}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bands = self.object.analysis_class._make_bands_single_image(img, configuration=configuration)

        if len(configuration['selected_bands']) == 0:
            configuration['selected_bands'] = list(range(len(bands)))
        self._configuration = configuration

        # self._band_data_items = []
        self._analysed_bands = []
        for ii, band in enumerate(bands):
            if ii in configuration['selected_bands']:
                self._analysed_bands += [band]
        self._plot_bands(self._analysed_bands)

    def _plot_bands(self, bands):
        [self.ImageDisplay.removeItem(p) for p in self._band_data_items]

        configuration = self.configuration_tree.make_dictionary()
        if configuration['image_preprocessing'] is None:
            configuration['image_preprocessing'] = dict()
        configuration = {**self.object.configuration, **configuration}
        if len(configuration['selected_bands']) == 0:
            configuration['selected_bands'] = list(range(len(bands)))

        self._band_data_items = []
        for ii, band in enumerate(bands):
            # if ii in configuration['selected_bands']:
            pitem = pg.PlotDataItem(self.k_to_px(band[:, 0]), self.e_to_px(band[:, 1]),
                                    pen=pg.intColor(ii, len(bands)))
            self.ImageDisplay.addItem(pitem)
            # time.sleep(0.1)
            self._band_data_items += [pitem]

    def save(self):
        configuration = self._configuration
        bands = self._analysed_bands
        self.object.all_bands[tuple(self._indxs)] = bands
        self.object.all_configurations[tuple(self._indxs)] = configuration
        self.next_image()
        self.analyse()


def test():
    from nplab.utils.gui import QtGui, get_qt_app
    app = get_qt_app()
    p = dict(
            image_preprocessing=dict(),
            peak_finding=dict(peak_width=3,
                              # find_peaks=dict(height=0.007, prominence=0.00001),
                              find_peaks=dict(height=0.01, prominence=0.01)),
            clustering=dict(energy_limit=[10, 30], min_cluster_size=30, min_cluster_distance=3, noise_cluster_size=3,),
            make_bands=dict(k0=-0.1, k_acceptance=1, bandwidth=0.0005),
            analyse_bands=dict(k_range_fit=1.5),
            testing='asdf'
        )
    # p = dict(a=1, b=1.1, c='hey hey', d=dict(e='Yep', f=None))

    t = AutoParameterTree(p)
    print('Input: ', p)
    print('Output: ', t.make_dictionary())

    win = QtGui.QWidget()
    layout = QtGui.QGridLayout()
    win.setLayout(layout)
    # layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."),
    #                  0, 0, 1, 2)
    layout.addWidget(t, 1, 0, 1, 1)
    win.show()
    app.exec_()
# test()

def clusterND(points, shear=0, scale=1, scale2=1, **kwargs):
    shear_matrix = np.identity(points.shape[1])
    shear_matrix[0, 1] = shear
    scaling_matrix = np.identity(points.shape[1])
    scaling_matrix[0, 0] = scale
    scaling_matrix[-1, -1] = scale2

    sheared = np.dot(points, shear_matrix)
    scaled = np.dot(sheared, scaling_matrix)

    defaults = dict(distance_threshold=2, compute_full_tree=True, linkage='single')
    [kwargs.setdefault(key, value) for key, value in defaults.items() if key not in kwargs]
    model = AgglomerativeClustering(None, **kwargs)
    clusters = model.fit(scaled)

    if points.shape[1] == 3:
        n_images = int(np.max(points[:, 2])) + 1
        a, b = square(n_images)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, fig)
        gs1 = gridspec.GridSpecFromSubplotSpec(a, b, gs[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(a, b, gs[1])
        # fig, axs = plt.subplots(a, b, sharex=True, sharey=True)
        axs = gs1.subplots(sharex=True, sharey=True)
        for idx in range(n_images):
            ax = axs.flatten()[idx]
            indices = points[:, 2] == idx
            ax.scatter(*points[indices].transpose(), c=clusters.labels_[indices])
        axs = gs2.subplots(sharex=True, sharey=True)
        for idx in range(n_images):
            ax = axs.flatten()[idx]
            indices = points[:, 2] == idx
            ax.scatter(*scaled[indices].transpose(), c=clusters.labels_[indices])


def fit_peak(spectra, xaxis=None, find_peaks_kwargs=None, n_peaks=None, axplot=None):
    spectra = gaussian_filter(np.copy(spectra), 2)

    if xaxis is None:
        xaxis = np.arange(len(spectra))
    if find_peaks_kwargs is None:
        find_peaks_kwargs = dict()
    default_kwargs = dict(distance=10, width=1, prominence=10)
    [find_peaks_kwargs.update({key: value}) for key, value in default_kwargs.items() if key not in find_peaks_kwargs]

    peak_indices, peak_properties = find_peaks(spectra, **find_peaks_kwargs)
    if n_peaks is None:
        n_peaks = len(peak_indices)
    if len(peak_indices) > 1:
        # Sort by height and clip
        heights = spectra[peak_indices]
        sorter = np.argsort(heights)[::-1]
        peak_indices = peak_indices[sorter][:n_peaks]
        # Sort by index (so things are always ordered)
        sorter = np.argsort(peak_indices)
        peak_indices = peak_indices[sorter][::-1]

    centers = []
    for idx in range(n_peaks):
        try:
            peak = peak_indices[idx]
            _spectra = spectra[peak-3:peak+3]
            _xaxis = xaxis[peak-3:peak+3]
            fit = np.polyfit(_xaxis, _spectra, 2)
            centers += [-fit[1]/(2*fit[0])]
        except IndexError:
            centers += [np.nan]
    if axplot is not None:
        axplot.plot(xaxis, spectra)
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(peak_indices))]
        axplot.vlines(centers, -np.max(spectra)/20, np.max(spectra)/20, colours)
    return np.array(centers)


def fit_spectra(spectra, xaxis=None, find_peaks_kwargs=None, n_peaks=None, fit_in_one=False, plot=None,
                peak_model='VoigtModel'):
    spectra = gaussian_filter(np.copy(spectra), 1)
    if xaxis is None:
        xaxis = np.arange(len(spectra))
    if find_peaks_kwargs is None:
        find_peaks_kwargs = dict()
    default_kwargs = dict(distance=10, width=1, prominence=10)
    [find_peaks_kwargs.update({key: value}) for key, value in default_kwargs.items() if key not in find_peaks_kwargs]

    peak_indices, peak_properties = find_peaks(spectra, **find_peaks_kwargs)
    # print(peak_indices)
    if plot is not None:
        # print(peak_indices)
        # fig, ax = plt.subplots(1, 1)
        ax = plot
        ax.plot(xaxis, spectra)
        ax.vlines(peak_indices, 0, np.max(spectra) / 10, 'k')
    if len(peak_indices) > 1:
        if n_peaks is not None:
            # Sort by height and select limit
            heights = spectra[peak_indices]
            sorter = np.argsort(heights)[::-1]
            peak_indices = peak_indices[sorter][:n_peaks]
            new_properties = dict()
            for key, value in peak_properties.items():
                new_properties[key] = value[sorter][:n_peaks]
            peak_properties = new_properties
        sorter = np.argsort(peak_indices)
        peak_indices = peak_indices[sorter]
        new_properties = dict()
        for key, value in peak_properties.items():
            new_properties[key] = value[sorter]
        peak_properties = new_properties

    if plot is not None:
        ax.vlines(peak_indices, -np.max(spectra) / 10, 0, 'r')

    if 'prominences' in peak_properties:
        prominences = (peak_properties['prominences'], (peak_properties['left_bases'], peak_properties['right_bases']))
    else:
        prominences = peak_prominences(spectra, peak_indices)
    if 'widths' in peak_properties:
        widths = (peak_properties['widths'], peak_properties['width_heights'], (peak_properties['left_ips'], peak_properties['right_ips']))
    else:
        widths = peak_widths(spectra, peak_indices, prominence_data=prominences)
    # guess = dict(background=np.percentile(spectra, 10))
    if fit_in_one:
        guess = dict()
        for idx, peak in enumerate(peak_indices):
            guess['peak%d_center' % idx] = xaxis[peak]
            width = widths[0][idx]
            guess['peak%d_width' % idx] = width
            prominence = prominences[0][idx]
            guess['peak%d_amplitude' % idx] = prominence * width
        print(guess)
        # model = lmfit.models.ConstantModel()
        model = None
        for idx, _ in enumerate(peak_indices):
            if model is None:
                model = lmfit.models.VoigtModel(prefix='peak%d_' % idx)
            else:
                model += lmfit.models.VoigtModel(prefix='peak%d_' % idx)
        print(model.param_names)
        params_guess = model.make_params(**guess)
        fit = model.fit(spectra, params_guess, x=xaxis)
        plt.plot(xaxis, spectra)
        plt.plot(xaxis, fit.init_fit)
        plt.plot(xaxis, fit.best_fit)
    else:
        best_fit = dict()
        for idx, peak in enumerate(peak_indices):
            pk_model = getattr(lmfit.models, peak_model)
            # model = lmfit.models.PolynomialModel(1) + pk_model(prefix='peak%d_' % idx)
            model = lmfit.models.ConstantModel() + pk_model(prefix='peak%d_' % idx)
            guess = dict()
            width = widths[0][idx]
            prominence = prominences[0][idx]
            # print(len(peak_indices), idx)
            if len(peak_indices) == 1:
                roi = [np.max([0, int(peak - 2 * width)]),
                       np.min([len(xaxis), int(np.round(peak + 2 * width))])]
            else:
                if len(peak_indices) - 1 > idx > 0:
                    roi = [np.max([int(peak_indices[idx-1]+width), int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width)), int(peak_indices[idx+1]-width)])]
                elif idx > 0:
                    roi = [np.max([int(peak_indices[idx-1]+width), int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width))])]
                else:
                    roi = [np.max([0, int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width)), int(peak_indices[idx+1]-width)])]

            guess['peak%d_center' % idx] = xaxis[peak]
            guess['peak%d_sigma' % idx] = width
            guess['peak%d_amplitude' % idx] = prominence * width
            guess['c'] = np.percentile(spectra[roi[0]:roi[1]], 1)
            # guess['c0'] = np.percentile(spectra[roi[0]:roi[1]], 1)
            # guess['c1'] = 0
            # guess['c2'] = 0
            # guess['c3'] = 0
            params_guess = model.make_params(**guess)
            if plot is not None:
                # print(model.param_names)
                ax.plot(xaxis[roi[0]:roi[1]], model.eval(params_guess, x=xaxis[roi[0]:roi[1]]), 'k--')
            try:
                fit = model.fit(spectra[roi[0]:roi[1]], params_guess, x=xaxis[roi[0]:roi[1]])
                for key, value in fit.best_values.items():
                    if 'peak' in key:
                        best_fit[key] = value
            except Exception as e:
                print('Failed fit: ', idx, peak, peak_indices, guess, e)
                for key, value in guess.items():
                    if 'peak' in key:
                        best_fit[key] = np.nan
            if plot is not None:
                _best_fit = dict(best_fit)
                try:
                    _best_fit['c0'] = fit.best_fit['c']
                    # _best_fit['c0'] = fit.best_fit['c0']
                    # _best_fit['c1'] = fit.best_fit['c1']
                except:
                    _best_fit['c'] = 0
                    # _best_fit['c0'] = 0
                    # _best_fit['c1'] = 0
                _best_fit = model.make_params(**_best_fit)
                # print(model.eval(_best_fit, x=xaxis[roi[0]:roi[1]]))
                ax.plot(xaxis[roi[0]:roi[1]], model.eval(_best_fit, x=xaxis[roi[0]:roi[1]]), 'k.-')
                try:
                    ax.plot(xaxis[roi[0]:roi[1]], fit.best_fit, 'r')
                except Exception as e:
                    pass
    return best_fit


def fit_ground_state(band, xaxis=None, debug=False):
    if xaxis is None:
        xaxis = np.arange(len(band), dtype=np.float)
        xaxis -= np.mean(xaxis)

    # Remove np.nan
    indices = np.where(~np.isnan(band))
    band = band[indices]
    xaxis = xaxis[indices]

    # brillouin, _ = find_peaks(band, distance=10, width=5)
    if debug:
        # print(brillouin)
        plt.figure()
        plt.plot(xaxis, band)
        # plt.plot(xaxis[brillouin], band[brillouin], 'x')
    linear = np.polyfit(xaxis, band, 1)

    model = lmfit.models.LinearModel() + lmfit.models.SineModel()
    # print(model.param_names)
    # print(np.squeeze(np.diff(xaxis[brillouin])))
    guess = dict(slope=linear[0], intercept=linear[1], amplitude=(np.max(band) - np.min(band))/2,
                 # frequency=2*np.pi/np.squeeze(np.diff(xaxis[brillouin])),
                 frequency=2*np.pi/35,
                 shift=-np.pi/2)
    params_guess = model.make_params(**guess)
    fit = model.fit(band, params_guess, x=xaxis)
    if debug:
        plt.plot(xaxis, fit.init_fit, 'k--')
        plt.plot(xaxis, fit.best_fit)
    return fit.best_values, fit.best_fit, xaxis


# SCHRODINGER EQUATION SIMULATIONS
MASS = 3.3e-5  # 1.2
MODE = 'sinusoid'  #  'looser'  # 'tighter'  #
BAND_EDGE_FACTOR = 1  # Determines the threshold for what to consider a trapped band in the eigenvalue spectrum


def sinusoid(depth, period, periods=5, size=101, bkg_value=0, mass=1e-3):
    x = np.linspace(-periods*period, periods*period, size)
    potential = np.asarray(depth * np.cos(x * 2*np.pi / period) + bkg_value, dtype=np.complex)
    mass_array = np.ones(size) * mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def sinusoid_tighter(depth, period, periods=5, size=101, bkg_value=0, mass=1e-3):
    x = np.linspace(-periods*period, periods*period, size)
    # osc = np.cos(x * 2*np.pi / period)
    # potential = depth * np.sign(osc) * np.asarray(osc, dtype=np.complex) ** 2 + bkg_value

    osc = np.sqrt((np.cos(x * 2*np.pi / period) + 1)/2) * 2 - 1
    potential = depth * np.asarray(osc, dtype=np.complex) + bkg_value

    mass_array = np.ones(size) * mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def sinusoid_looser(depth, period, periods=5, size=101, bkg_value=0, mass=1e-3):
    x = np.linspace(-periods*period, periods*period, size)
    osc = (((np.cos(x * 2*np.pi / period) + 1) / 2)**2) * 2 - 1
    potential = depth * np.asarray(osc, dtype=np.complex) + bkg_value
    mass_array = np.ones(size) * mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def analyse_modes(eigenvalues, n_traps, potential_maximum=None, n_modes=None):
    """

    # TODO: when is a mode unbound? When any part of it's spectra is above the binding, or when it's average energy is above the binding?
    # Currently it assumes that when any part of the mode is unbound, the mode is unbound
    :param eigenvalues:
    :param n_traps:
    :param potential_maximum:
    :param n_modes:
    :return:
    """
    if potential_maximum is None:
        potential_maximum = 0
    energies = np.real(eigenvalues)

    # Selecting only bound energies
    energies = energies[energies < potential_maximum]
    unbound = np.argmin(np.abs(energies - potential_maximum))  # the index of the unbound threshold

    # Finding the band edges by finding the peaks in the energy differential
    band_edges, _props = find_peaks(np.diff(energies), width=[0.1, 2], prominence=1e-4)
    # Only selecting the bands that are where we expect them to be (each band should contain n_traps number of modes)
    mask = [not (_x+1) % n_traps for _x in band_edges]
    band_edges = band_edges[np.argwhere(mask)][:, 0]
    # Adding the unbound limit to the band edges
    band_edges = np.append(band_edges, unbound)

    # If there are no bound modes, return NaNs
    if len(band_edges) <= 1:
        if n_modes is not None:
            return np.repeat(np.nan, n_modes), np.repeat(np.nan, n_modes), np.full((n_modes, 2), np.nan)
        else:
            return np.nan, np.nan, np.nan

    # Extract energies from the indices
    _band_edges = np.append([-1], band_edges)
    bands = [energies[idx+1:idxnext+1] for idx, idxnext in zip(_band_edges, _band_edges[1:])]
    band_centers = np.array([np.mean(_band) for _band in bands])
    band_energy = np.array([(np.min(_band, -1), np.max(_band, -1)) for _band in bands])
    band_widths = band_energy[:, 1] - band_energy[:, 0]
    # band_energy = np.transpose([np.min(bands, -1), np.max(bands, -1)])

    # Reshape bands. Useful when we want all outputs to be the same shape
    if n_modes is not None:
        if len(band_centers) >= n_modes:
            band_centers = band_centers[:n_modes]
            band_widths = band_widths[:n_modes]
            band_energy = band_energy[:n_modes]
        else:
            fill_na = np.full((n_modes - len(band_centers), ), np.nan)
            fill_na2 = np.full((n_modes - len(band_centers), 2), np.nan)
            band_centers = np.append(band_centers, fill_na)
            band_widths = np.append(band_widths, fill_na)
            band_energy = np.append(band_energy, fill_na2, 0)
    return band_centers, band_widths, band_energy

# def find_bound_modes(pot, kin, *args):
#     # Should be groups of N traps
#     n_traps = len(find_peaks(-np.diag(np.real(pot))))
#     vals, vecs = solve(pot + kin)
#     band_edges, _props = find_peaks(np.diff(np.real(vals)), width=[0.1, 2])
#     unbound = np.argmin(np.abs(vals - pot.max()))
#     # print(band_edges, unbound, n_traps)
#     band_edges = band_edges[band_edges < unbound]
#     # print(band_edges)
#     assert all([not (_x+1) % n_traps for _x in band_edges])
#     band_centers = []
#     band_widths = []
#     band_gaps = []
#     old_values = None
#     for idx in range(len(band_edges)):
#         if idx == 0:
#             values = vals[0:band_edges[0]+1]
#         else:
#             values = vals[band_edges[idx-1]+1:band_edges[idx]+1]
#             band_gaps += [values.min() - old_values.max()]
#         band_centers += [np.mean(values)]
#         band_widths += [np.max(values) - np.min(values)]
#         old_values = np.copy(values)
#     return np.array(band_centers), np.array(band_widths), np.array(band_gaps)


def run_simulations(depths, periods, backgrounds=0, masses=3e-5, mode='sinusoid', size=1001, n_traps=10, n_bands=5,
                    disable_output=False):
    """

    :param depths:
    :param periods:
    :param backgrounds:
    :param masses:
    :param mode:
    :param size:
    :param n_traps:
    :param n_bands:
    :return:
    """
    try:
        len(depths)
    except:
        depths = [depths]
    try:
        len(periods)
    except:
        periods = [periods]
    try:
        len(backgrounds)
    except:
        backgrounds = [backgrounds]
    try:
        len(masses)
    except:
        masses = [masses]

    if mode == 'sinusoid':
        func = sinusoid
    elif mode == 'tighter':
        func = sinusoid_tighter
    elif mode == 'looser':
        func = sinusoid_looser
    else:
        raise ValueError()

    values = []
    analysed = []
    for depth in tqdm(depths, 'run_simulations', disable=disable_output):
        _vals = []
        # _analysed = []
        for period in periods:
            _valss = []
            # _nlsd = []
            for mass in masses:
                # print(depth, period, n_traps, size, mass)
                pot, kin, x = func(depth, period, n_traps, size, mass=mass)
                # print('# of NaN: ', np.sum(np.isnan(pot+kin)), '# of infs: ', np.sum(np.isinf(pot+kin)))
                vals, _ = solve(pot + kin)
                _values = []
                # _nlslds = []
                for background in backgrounds:
                    _eig = vals + background
                    _values += [_eig]
                    _c, _w, _e = analyse_modes(_eig, n_traps=n_traps, potential_maximum=BAND_EDGE_FACTOR * depth,
                                               n_modes=n_bands)
                    analysed += [(_c, _w, _e)]
                _valss += [_values]
                # _nlsd += [_nlslds]
            _vals += [_valss]
            # _analysed += [_nlsd]
        values += [_vals]
        # analysed += [_analysed]
    iter_shape = (len(depths), len(periods), len(masses), len(backgrounds))
    analysed_centers = np.squeeze(np.reshape([a[0] for a in analysed], iter_shape + _c.shape))
    analysed_widths = np.squeeze(np.reshape([a[1] for a in analysed], iter_shape + _w.shape))
    # print('Debug: ', _e.shape, [a[2].shape for a in analysed])
    analysed_edges = np.squeeze(np.reshape([a[2] for a in analysed], iter_shape + _e.shape))
    return np.squeeze(values), (analysed_centers, analysed_widths, analysed_edges)


def run_simulations_dataset(dataset_index, max_iterations=1, depths=None, results=None, _index=0):
    """Recursive simulations

    :param dataset_index:
    :param max_iterations:
    :param depths:
    :param results:
    :param _index:
    :param from_file:
    :return:
    """
    with h5py.File(collated_analysis, 'r') as dfile:
        laser_separations = dfile['laser_separations'][...]
    period = np.abs(2*np.pi / laser_separations[dataset_index])
    if depths is None:
        depths = np.linspace(0.1, 5.1, 11)
    eigenvalues, (centers, widths, edges) = run_simulations(depths, [period], 0, MASS, MODE)

    # centers, widths, _ = [], [], []
    # for depth, _eig in tqdm(zip(depths, eigenvalues), 'Schrodinger analysis'):
    #     _c, _w, _ = analyse_modes(_eig, n_traps=10, potential_maximum=BAND_EDGE_FACTOR * depth, n_modes=5)
    #     centers += [_c]
    #     widths += [_w]
    # centers = np.array(centers)
    # widths = np.array(widths)
    if results is None:
        results = dict(depths=depths, centers=centers, eigenvalues=eigenvalues, widths=widths)
    else:
        results = dict(depths=np.append(depths, results['depths']),
                       centers=np.append(centers, results['centers'], axis=0),
                       widths=np.append(widths, results['widths'], axis=0),
                       eigenvalues=np.append(eigenvalues, results['eigenvalues'], axis=0))

    if _index < max_iterations:
        mode_separations = np.diff(centers, axis=-1)
        # print(mode_separations)
        next_depths = []
        for _idx in range(2):
            mode_separation = mode_separations[:, _idx]
            nan_index = np.argwhere(np.isnan(mode_separation))[-1][0]
            # print(nan_index, mode_separation, np.isnan(mode_separation), np.argwhere(np.isnan(mode_separation)))
            if nan_index < len(depths) - 1:
                _next = np.linspace(depths[nan_index], depths[nan_index + 1], 11)
                next_depths += [np.linspace(_next[1], _next[-2], 9)]
        # print(next_depths)
        return run_simulations_dataset(dataset_index, max_iterations, np.array(next_depths).flatten(),
                                       results, _index + 1)
    else:
        return results


def fit_theory(dataset_index, selected_indices=None, run_sims=False):
    if dataset_index == 1:
        return None
    with h5py.File(collated_analysis, 'r') as dfile:
        laser_separations = dfile['laser_separations'][...]
    period = np.abs(2 * np.pi / laser_separations[dataset_index])

    # Experiment
    data, bands, config, analysis_results, variables = get_experimental_data(dataset_index)
    if 'data_axes_order' in config:
        data = np.transpose(data, config['data_axes_order'])
    if selected_indices is None:
        selected_indices = tuple([slice(x) for x in bands.shape])
    exper_energy_array = analysis_results[0][selected_indices]
    # print(exper_energy_array.shape)
    # n_bands = exper_energy_array.shape[-1]
    if len(exper_energy_array.shape) > 1:
        initial_shape = exper_energy_array.shape[:-1]
        # exper_energy_array = exper_energy_array
    else:
        initial_shape = (1, )
        exper_energy_array = np.array([exper_energy_array])

    # Theory
    if run_sims:
        results = run_simulations_dataset(dataset_index)
        np.save(get_data_path('2021_07_conveyorbelt/simulations/dataset%d_simulations' % dataset_index), results)
    else:
        results = np.load(get_data_path('2021_07_conveyorbelt/simulations/dataset%d_simulations.npy' % dataset_index), allow_pickle=True).take(0)

    theory_depths = results['depths']
    theory_centers = results['centers']
    all_splittings = np.diff(theory_centers, axis=-1)
    first_splitting = all_splittings[..., 0]
    theory_depth_vs_splitting = interp1d(first_splitting, theory_depths, bounds_error=False, fill_value=np.nan)

    # Fitting
    fitted_results = dict(depths=[], centers=[], widths=[], edges=[])
    n_bands = 5
    exper_split = np.diff(exper_energy_array*1e3, axis=-1)[..., 0]  # [(e[1] - e[0]) * 1e3 for e in exper_energy_array]
    fitted_depths = theory_depth_vs_splitting(exper_split)  #[theory_depth_vs_splitting(e) for e in exper_split]
    # print(fitted_depths.shape)
    for depth in tqdm(fitted_depths.flatten(), 'fit_theory'):
        if np.isnan(depth):
            centers = np.full((n_bands, ), np.nan)
            widths = np.full((n_bands, ), np.nan)
            edges = np.full((n_bands, 2), np.nan)
        else:
            _, (centers, widths, edges) = run_simulations([depth], [period], 0, MASS, MODE, n_bands=n_bands, disable_output=True)
        fitted_results['depths'] += [depth]
        fitted_results['centers'] += [centers]
        fitted_results['widths'] += [widths]
        fitted_results['edges'] += [edges]
    fitted_results['depths'] = np.reshape(fitted_results['depths'], initial_shape)
    fitted_results['centers'] = np.reshape(fitted_results['centers'], initial_shape + (n_bands, ))
    fitted_results['widths'] = np.reshape(fitted_results['widths'], initial_shape + (n_bands, ))
    fitted_results['edges'] = np.reshape(fitted_results['edges'], initial_shape + (n_bands, 2))

    return fitted_results


def plot_theory_fit(dataset_index, selected_indices, run_fit=False, run_sims=False,
                    ax=None, imshow_kw=None, plot_kw=None, fill_kw=None, plotting_kw=None):
    """

    :param dataset_index:
    :param selected_indices:
    :param run_sims:
    :param imshow_kw:
    :param plot_kw:
    :param fill_kw:
    :return:
    """
    data, bands, config, analysis_results, variables = get_experimental_data(dataset_index)
    if run_fit:
        fitted_results = fit_theory(dataset_index, selected_indices, run_sims)
        centers, edges, depths = [fitted_results[x] for x in ['centers', 'edges', 'depths']]
    else:
        fitted_results = np.load(get_data_path('2021_07_conveyorbelt/simulations/dataset%d_fits.npy' % dataset_index), allow_pickle=True).take(0)
        centers, edges, depths = [fitted_results[x][selected_indices] for x in ['centers', 'edges', 'depths']]
        if len(centers.shape) == 1:
            centers = [centers]
            edges = [edges]
            depths = [depths]
    power_axis = np.array(variables['vwp']) * 1e3
    # frequency_axis = np.array(variables['f'])

    band = bands[selected_indices]
    image = data[selected_indices]
    exper_energy_array = analysis_results[0][selected_indices]

    try:
        shp = band.shape
        if ax is not None:
            return None
        if len(shp) == 1:
            shape = square(shp[0])
        elif len(shp) == 2:
            shape = shp
        else:
            return None
        fig, _axs = plt.subplots(shape[1], shape[0])
        axs = _axs.flatten()

        flattened_bands = band.flatten()
        flattened_images = np.reshape(image, (np.prod(shp), image.shape[-2], image.shape[-1]))
        flattened_exp_energy = np.reshape(exper_energy_array, (np.prod(shp), exper_energy_array.shape[-1]))
    except Exception as e:
        shape = (1, 1)
        flattened_bands = [band]
        flattened_images = [image]
        flattened_exp_energy = np.array([exper_energy_array])
        if ax is None:
            fig, _axs = plt.subplots(*shape)
            axs = [_axs]
        else:
            fig = ax.figure
            axs = [ax]

    if imshow_kw is None: imshow_kw = dict()
    imshow_kw = {**dict(diverging=False, cbar=False, cmap='Greys', norm=LogNorm()), **imshow_kw}
    if plot_kw is None: plot_kw = dict()
    if fill_kw is None: fill_kw = dict()
    fill_kw = {**dict(facecolor='C8'), **fill_kw}
    if plotting_kw is None: plotting_kw = dict()
    plotting_kw = {**dict(show_label=True, show_bands=True, show_theory=True), **plotting_kw}

    iterable = (axs, flattened_bands, flattened_images, flattened_exp_energy,
                centers, edges, depths)
    for ax, bnd, img, exper_energy, cntrs, dgs, depth in zip(*iterable):
        if 'brillouin_plot' in config:
            k0_offset = config['brillouin_plot']['k0_offset']
        else:
            k0_offset = 0
        imshow(img, ax, xaxis=np.array(config['k_axis'])-k0_offset, yaxis=config['energy_axis'], **imshow_kw)
        if plotting_kw['show_bands']:
            # [ax.plot(*_band.transpose(), **plot_kw) for _band in bnd]
            if 'plot_max_band' in plotting_kw:
                max_band = plotting_kw['plot_max_band']
            else:
                max_band = len(bnd)
            for idx, _band in enumerate(bnd):
                if idx < max_band:
                    if 'k_filtering' in plotting_kw:
                        mask = np.abs(_band[:, 0]-k0_offset) < plotting_kw['k_filtering']
                        _band = _band[mask]
                    ax.plot(_band[:, 0]-k0_offset, _band[:, 1], **plot_kw)

        if plotting_kw['show_theory']:
            if not np.isnan(depth):
                cntrs *= 1e-3
                dgs *= 1e-3
                e_offset = exper_energy[0] - np.nanmin(cntrs)
                cntrs += e_offset
                dgs += e_offset
                for edge in dgs:
                    lw = 1e-4
                    ax.fill_between(np.array(config['k_axis'])-k0_offset, edge[0] - lw / 2, edge[1] + lw / 2, **fill_kw)

        if plotting_kw['show_label']:
            if 'label_string' in plotting_kw:
                label = ''
            else:
                try:
                    # power_label = power_axis[selected_indices[-1]]
                    power_label = variables['normalised_power_axis'][selected_indices[-1]] * 1e3
                    label = 'P=%.1fmW%sm$^{-2}$\n' % (power_label, mu)
                except:
                    label = ''
                if not np.isnan(depth):
                    label += '$V_{eff}$=%.2fmeV' % depth
            ax.text(0.5, 0.95, label, ha='center', va='top', transform=ax.transAxes)

    # if not np.isnan(depth):
    #     eigenvalues, (centers, widths, edges) = run_simulations([depth], [period], 0, MASS, MODE, n_bands=5)
    #     # centers, widths, _ = analyse_modes(eigenvalues, n_traps=10, potential_maximum=BAND_EDGE_FACTOR * depth,
    #     #                                    n_modes=5)
    #     centers *= 1e-3
    #     widths *= 1e-3
    #     # print(centers, exper_energy)
    #     centers -= np.nanmin(centers)
    #     centers += exper_energy[0]
    #
    #     # print(centers)
    #     for tf, lw in zip(centers, widths):
    #         lw = 1e-4
    #         ax.fill_between(config['k_axis'], tf - lw / 2, tf + lw / 2, color='C8', alpha=0.7)
    #     ax.set_title(r'$V_{depth}$=%.2f' % (depth, ))


# def find_two_parameters(ground_state, splitting, xaxis, yaxis, plot=False, colours=None):
#     if isinstance(plot, plt.Axes):
#         arg = plot
#     elif not plot:
#         plt.close('all')
#         arg = None
#     else:
#         fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
#         imshow(theory_groundstate, axs[0], xaxis=xaxis, yaxis=yaxis, diverging=False)
#         imshow(theory_splitting, axs[1], xaxis=xaxis, yaxis=yaxis, diverging=False)
#         X, Y = np.meshgrid(xaxis, yaxis)
#         axs[0].contour(X, Y, theory_groundstate, [ground_state], colors='w')
#         axs[1].contour(X, Y, theory_splitting, [splitting], colors='w')
#         arg = axs[2]
#     _, _, intersection_points, lines = contour_intersections([theory_groundstate, theory_splitting],
#                                                          [[ground_state], [splitting]], arg,
#                                                          [xaxis] * 2, [yaxis] * 2, colours)
#     if arg is None:
#         plt.close('all')
#     return intersection_points, lines


if __name__ == '__main__':
    TOGGLE = 'sinusoidal schrodinger'
    LOGGER.setLevel('INFO')

    if TOGGLE == 'cluster analysis':
        import h5py
        from microcavities.utils import random_choice
        collated_data_path = get_data_path('2021_07_conveyorbelt/collated_data.h5')
        collated_analysis_path = get_data_path('2021_07_conveyorbelt/collated_analysis.h5')

        example_config = dict(
            plotting=False, #['make_bands'],  #True,  # ['image_preprocessing', 'peak_finding']
            image_preprocessing=dict(
                # normalization_percentiles=[0, 100],
                # low_pass_threshold=0.4
            ),
            peak_finding=dict(peak_width=3, savgol_filter=dict(), find_peaks=dict(height=0.01, prominence=0.01)),
            clustering=dict(#shear=-0.03, #scale=0.01,
                            # AgglomerativeClustering=dict(n_clusters=15,
                            #                              # distance_threshold=None,
                            #                              # compute_distances=True
                            #                              ),
                            energy_limit=30,
                            min_cluster_size=10, min_cluster_distance=3),
            make_bands=dict(k0=-0.1,
                            k_acceptance=0.5,
                            # k_acceptance=1,
                            bandwidth=0.3),
            analyse_bands=dict(k_range_fit=1.5)
        )

        with h5py.File(collated_analysis_path, 'r') as dfile:
            laser_separations = dfile['laser_separations'][...]

        dataset_index = 3
        with h5py.File(collated_data_path, 'r') as dfile:
            dset = dfile['alignment%d/scan' % dataset_index]
            data = dset[...]
            _v = dset.attrs['variables']
            variables = eval(_v)
            eax = dset.attrs['eaxis']
            kax = dset.attrs['kaxis']
        example_config['k_axis'] = kax
        example_config['energy_axis'] = eax
        example_config['laser_angle'] = laser_separations[dataset_index]

        # for example_image in [data[-1, 5, -1], data[-1, 6, -3]]:  #data[1, 9, 8], data[2, 7, -4], data[0, 2, 5], data[1, 3, 5], data[2, 10, 9], data[1, 2, 1]]:
        #     example_config['plotting'] = ['make_bands']
        #     tst = SingleImageModeDetection(example_image, example_config)
        #     # labels, mask = tst.cluster()
        #     tst.make_bands()
        #     del example_config['clustering']['shear']

        # example_image, _indices = random_choice(data, (0, 1, 2), True)
        # print(_indices)
        # example_image = data[-1, -5, 5]
        # example_image = data[-1, 2, -2]
        # tst = SingleImageModeDetection(example_image, example_config)
        # tst.analyse_bands()

        example_config['plotting'] = True
        example_full_config = dict(# shear_slope=0.01, period=1,
                                   example_config)
        full_fit = FullDatasetModeDetection(data, example_full_config)
        # full_fit._analyse_bands_single_image(data[3, 1, 1])
        # full_fit._analyse_bands_single_image()
        full_fit.analyse_bands()
        # # print(full_fit._calculate_single_shear(data[-1, 0, 0]))
        # # shear = full_fit.calculate_shear()[0]
        # full_fit.cluster()
        # full_fit.analyse_bands(n_bands=5)
        # print(shear.shape)
        # imshow(shear)

        plt.show()
    elif TOGGLE == 'sinusoidal schrodinger':
        depth = 100
        period = 15
        mass = 3e-5
        eigenvalues = run_simulations([depth], [period], masses=mass)
        centers, widths, _ = analyse_modes(eigenvalues, 10)

        def taylor_expansion_splitting(depth, period, mass):
            return hbar * np.sqrt(2 * depth / (electron_mass * mass * period**2))
        print(centers)
        print(np.diff(centers)[0])
        print(taylor_expansion_splitting(depth, period, mass))