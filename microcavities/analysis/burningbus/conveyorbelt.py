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
from microcavities.experiment.utils import magnification
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

plt.style.use(os.path.join(os.path.dirname(get_data_path('')), 'Papers/Conveyor/python/paper_style.mplstyle'))
plt.rcParams["pgf.texsystem"] = "pdflatex"

density_cmap = 'BlueYellowRed'


LOGGER = create_logger('Fitting')
LOGGER.setLevel('WARN')

folder_name = '2022_08_conveyorbelt'
collated_data_path = get_data_path('%s/collated_data.h5' % folder_name)
collated_analysis = get_data_path('%s/collated_analysis.h5' % folder_name)
spatial_scale = magnification('rotation_pvcam', 'real_space')[0] * 1e6
momentum_scale = magnification('rotation_pvcam', 'k_space')[0] * 1e-6

hbar = 6.582119569 * 10 ** (-16) * 10 ** 3 * 10 ** 12   # in meV.ps
c = 3 * 10 ** 14 * 10 ** -12                            # Speed of Light   um/ps
me = 511 * 10 ** 6 / c ** 2                             # Free electron mass   meV/c^2

DETUNING = 1550.9923163262479-1546.0828304664699 + 3.2
RABI = 9.7 / 2
MASS = 4.6e-05 * me

with h5py.File(collated_analysis, 'r') as dfile:
    laser_separations = dfile['laser_separations'][...]
dataset_order = np.argsort(np.abs(laser_separations))
normalized_laser_separations = normalize(np.abs(laser_separations))
colormap_laser_separation = cm.get_cmap('Greens_r')((normalized_laser_separations - 0.24)/1.2)
label_colour = (216 / 255., 220 / 255., 214 / 255., 0.9)

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


"""UTILITY FUNCTIONS"""


class dummy_formatter(ScalarFormatter):
    def __init__(self, offset, *args, **kwargs):
        super(dummy_formatter, self).__init__(*args, **kwargs)
        self.set_useOffset(offset)
        self.format = '%g'

    def format_data(self, value):
        return '%g' % value


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

def power_density(power, wavevector):
    """Utility function just for this project"""
    _mom = 2*np.pi / 0.805
    angle_of_incidence = np.abs(wavevector) / _mom  # approximately true for small angles
    return power_density_angled(power, angle_of_incidence, area_of_ellipse(40, 20),)


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
    if config is None:
        config = dict()
    # ellipse_a = 20
    # ellipse_b = 40
    # _mom = 2*np.pi / 0.805
    # angle_of_incidence = np.abs(laser_separations[dataset_index]) / _mom  # approximately true for small angles
    # laser_size = np.pi * ellipse_a * ellipse_b / np.sin(np.pi/2-angle_of_incidence)
    # norm_ax = variables['vwp'] / laser_size
    norm_ax = power_density(variables['vwp'], laser_separations[dataset_index])

    variables['normalised_power_axis'] = norm_ax
    if 'data_axes_order' in config:
        data = np.transpose(data, config['data_axes_order'])
    config['k_axis'] = kax
    config['energy_axis'] = eax
    config['laser_angle'] = laser_separations[dataset_index]

    return data, variables, config


def get_experimental_data(dataset_index):
    data, variables, config = get_experimental_data_base(dataset_index)

    if configurations[dataset_index] is None:
        analyse = False
    else:
        analyse = True

    if analyse:
        cls = FullDatasetModeDetection(data, config)
        cls.configuration['plotting'] = False
        bands = np.load(get_data_path('%s/bands/bands_dataset%d.npy' % (folder_name, dataset_index)), allow_pickle=True)
        if 'data_axes_order' in config:
            bands = np.transpose(bands, config['data_axes_order'][:-2])
        analysed_bands = cls.analyse_bands(bands)
    else:
        with h5py.File(get_data_path('%s/collated_analysed_data.h5' % folder_name), 'r') as full_file:
            dset = full_file['speeds%d' % (dataset_index + 1)]
            mode_tilts = dset[...]
        if dataset_index == 1:
            mode_tilts /= 2
        bands = None
        analysed_bands = (None, mode_tilts, None, None)
        # bands = None
        # analysed_bands = None
    return data, bands, config, analysed_bands, variables


def get_selected_tilt_data(dataset_index, select=True):
    _, _, config, analysis_results, variables = get_experimental_data(dataset_index)
    mode_tilts = -remove_outliers(analysis_results[1])

    shape = mode_tilts.shape
    all_freq = np.array(variables['f'])

    full_frequency_axis = np.linspace(-9, 9, 19)
    if len(shape) == 4:
        mode_tilts = np.transpose(mode_tilts, (1, 0, 2, 3))
    all_modes = np.reshape(mode_tilts, (shape[0], np.prod(shape[1:])))

    available_frequency = np.argwhere([a in all_freq for a in full_frequency_axis]).flatten()
    left_index = np.min(available_frequency)
    right_index = np.max(available_frequency)
    left_append = np.full((left_index, all_modes.shape[1]), np.nan)
    right_append = np.full((18 - right_index, all_modes.shape[1]), np.nan)
    all_modes = np.concatenate([left_append, all_modes, right_append], 0)

    if select:
        if dataset_index == 0:
            selected_modes = mode_tilts[:, :, 2:, 1:4]
        elif dataset_index == 1:
            selected_modes = mode_tilts[:, 2:]
        elif dataset_index == 2:
            selected_modes = mode_tilts[:, 2:, 0]
        elif dataset_index == 3:
            selected_modes = mode_tilts[:, 2:]
        elif dataset_index == 4:
            selected_modes = mode_tilts[:, :, 2:, 1]
        elif dataset_index == 5:
            selected_modes = mode_tilts[:, :, 2:, :2]
        elif dataset_index == 6:
            selected_modes = mode_tilts[:, :, 2:, 1:4]
        elif dataset_index == 7:
            selected_modes = mode_tilts[:, :, 2:, 1]
        elif dataset_index == 8:
            selected_modes = mode_tilts[:, :, 2:, :2]

        _shape = selected_modes.shape
        selected_modes = np.reshape(selected_modes, (_shape[0], np.prod(_shape[1:])))
        left_append = np.full((left_index, selected_modes.shape[1]), np.nan)
        right_append = np.full((18 - right_index, selected_modes.shape[1]), np.nan)
        selected_modes = np.concatenate([left_append, selected_modes, right_append], 0)
    else:
        selected_modes = np.nan

    return selected_modes, all_modes, full_frequency_axis, config


"""EXPERIMENTAL DATA FITTING"""


def smoothened_find_peaks(x, *args, **kwargs):
    """Simple extension of find_peaks to give more than single-pixel accuracy"""
    smoothened = savgol_filter(x, 5, 3)
    sampling = interp1d(range(len(smoothened)), smoothened, 'quadratic')
    new_x = np.linspace(0, len(smoothened)-1, len(smoothened)*10)
    new_y = sampling(new_x)
    results = find_peaks(new_y, *args, **kwargs)
    return new_x[results[0]], results[1]


def _make_axis_functions(configuration):
    if 'energy_axis' not in configuration:
        e_roi = configuration['energy_roi']
        _wvls = spectrometer_calibration('rotation_acton_old', 803, '1')[e_roi[1]:e_roi[0]:-1]
        energy_axis = 1240 / _wvls
    else:
        energy_axis = configuration['energy_axis']
    energy_func = interp1d(range(len(energy_axis)), energy_axis, bounds_error=False, fill_value=np.nan)
    e_inverse = interp1d(energy_axis, range(len(energy_axis)))

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
            _wvls = spectrometer_calibration('rotation_acton_old', 803, '1')[e_roi[1]:e_roi[0]:-1]
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
                    # print(np.min(band[:, 1]), np.max(band[:, 1]), self.configuration['energy_axis'])
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


class InteractiveAnalysis(InteractiveBase):
    # TODO:
    #     - Add shear to the interactive
    #     - Allow one to also group bands in the interactive band listing
    #     - If bands already exist, plot them
    #     - If no configuration file exists, create it from the default

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
            pitem = pg.PlotDataItem(self.k_to_px(band[:, 0]), self.e_to_px(band[:, 1]),
                                    pen=pg.intColor(ii, len(bands)))
            self.ImageDisplay.addItem(pitem)
            self._band_data_items += [pitem]

    def save(self):
        configuration = self._configuration
        bands = self._analysed_bands
        self.object.all_bands[tuple(self._indxs)] = bands
        self.object.all_configurations[tuple(self._indxs)] = configuration
        self.next_image()
        self.analyse()


"""SCHRODINGER EQUATION SIMULATIONS"""


def analytical_coupling_strength(potential_depth, lattice_period, mass=MASS):
    return 4 * potential_depth * np.exp(-np.sqrt(2*mass*potential_depth)*lattice_period/hbar)


def Hamiltonian_k(k, potential, delta_k=10., mass=MASS, detuning=DETUNING, rabi=RABI, n_bands=6):
    G = delta_k

    # TODO  compare to a model where the photon and the exciton are fully separated (mass)
    space_size = 2*n_bands + 1

    # Kinetic energy
    Hk0 = np.diag([hbar ** 2 * (k - x * G) ** 2 / (2 * mass) for x in range(-n_bands, n_bands + 1)])
    Hk0 -= np.eye(space_size) * detuning / 2

    # Potential energy
    pot = [potential / 2] * (space_size - 1)
    Hv = np.diag(pot, -1) + np.diag(pot, 1)
    Hv += np.eye(space_size) * detuning / 2

    # Coupling to exciton
    H1row = np.hstack([Hk0, rabi * np.eye(space_size)])
    H2row = np.hstack([rabi * np.eye(space_size), Hv])
    return np.vstack([H1row, H2row])


def Hamiltonian_x(t, potential, delta_k, frequency, periods=6, n_points=32, mass=MASS, detuning=DETUNING, rabi=RABI):
    single_period = 2 * np.pi / np.abs(delta_k)

    if periods is None:
        x = np.linspace(-21, 20, n_points)
    else:
        x = np.linspace(-single_period * periods/2 - 0.1*single_period, single_period*periods/2, n_points)
    D2 = np.diag(-2*np.ones(n_points)) + np.diag(np.ones(n_points-1), 1) + np.diag(np.ones(n_points-1), -1)
    dx = np.diff(x)[0]
    D2 /= dx**2
    Hk0 = -D2 * hbar ** 2 / (2 * mass)
    Hk0 -= np.eye(n_points) * detuning / 2
    Hv = (potential * np.cos(delta_k * x - 2*np.pi*frequency * t) + detuning / 2) * np.eye(n_points)
    H1row = np.hstack([Hk0, rabi * np.eye(n_points)])
    H2row = np.hstack([rabi * np.eye(n_points), Hv])
    return np.vstack([H1row, H2row])


def Hamiltonian_kt(k, t, delta_k, frequency, potential, mass=MASS, detuning=DETUNING, rabi=RABI, n_bands=6):
    """ Floquet-Bloch Hamiltonian

    :param k: float. Momentum
    :param t: float. Time
    :param delta_k: float. Reciprocal vector
    :param frequency: float
    :param potential: float
    :param mass: float
    :param detuning: float
    :param rabi: float
    :param n_bands: int
    :return:
    """
    G = delta_k
    space_size = 2*n_bands + 1
    omega = 2 * np.pi * frequency

    # Kinetic energy
    Hk0 = np.diag([hbar ** 2 * (k - x * G) ** 2 / (2 * mass) for x in range(-n_bands, n_bands + 1)]) #, dtype=complex)
    Hk0 -= np.eye(space_size) * detuning / 2

    # Potential energy
    pot = [potential / 2] * (space_size - 1)
    Hv = np.diag(pot, -1) * np.exp(1j * omega * t) + \
         np.diag(pot, 1) * np.exp(-1j * omega * t)
    Hv += np.eye(space_size) * detuning / 2

    # Coupling to exciton
    H1row = np.hstack([Hk0, rabi * np.eye(space_size)])
    H2row = np.hstack([rabi * np.eye(space_size), Hv])
    return np.vstack([H1row, H2row])


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


def rk_timestep(psi, hamiltonian, t, dt, noise_level=0.2):
    K11 = -1j * np.matmul(hamiltonian(t), psi) / hbar
    K21 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + K11 * dt / 2) / hbar
    K31 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + K21 * dt / 2) / hbar
    K41 = -1j * np.matmul(hamiltonian(t + dt), psi + dt * K31) / hbar

    return psi + (K11 + 2 * K21 + 2 * K31 + K41) * dt / 6 + noise_level*np.random.rand(len(psi))


def solve_timerange(psi0, hamiltonian, timerange):
    full_psi = np.zeros((len(psi0), len(timerange)), dtype=complex)
    for idx_t, t in enumerate(timerange):
        full_psi[:, idx_t] = psi0
        psi0 = rk_timestep(psi0, hamiltonian, t, np.diff(timerange)[0])
    return full_psi


def solve_for_krange(krange, hamiltonian):
    bands = []
    modes = []
    for k in krange:
        H = hamiltonian(k)
        E, eig_vectors = np.linalg.eig(H)
        idx_sort = np.argsort(E.real)
        bands += [E[idx_sort]]
        modes += [eig_vectors[:, idx_sort]]
    return np.array(bands), np.array(modes)


def farfield(hamiltonian, starting_vectors, timerange):
    N = len(starting_vectors) // 2
    rho = np.zeros((N, len(timerange)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(vec, hamiltonian, timerange)
        psikw = np.fft.fftshift(np.fft.fft2(psi[:N, :]))
        rho += np.abs(psikw) ** 2
        if np.isnan(rho).any():
            break
    return rho


"""SIMULATIONS FOR EXPERIMENT"""


def run_simulations(depths, periods, backgrounds=0, masses=MASS, n_bands=20,
                    disable_output=False, detuning=DETUNING, rabi=RABI, k_axis=None):
    """Run simulations for a grid of depths, periods, backgrounds, and/or masses

    :param depths:
    :param periods:
    :param backgrounds:
    :param masses:
    :param n_bands:
    :param disable_output:
    :param detuning:
    :param rabi:
    :param k_axis:
    :return:
    """
    try: len(depths)
    except: depths = [depths]
    try: len(periods)
    except: periods = [periods]
    try: len(backgrounds)
    except: backgrounds = [backgrounds]
    try: len(masses)
    except: masses = [masses]

    if k_axis is None:
        k_axis = np.linspace(-3, 3, 301)

    values = []
    for depth in tqdm(depths, 'run_simulations', disable=disable_output):
        _vals = []
        for period in periods:
            _valss = []
            for mass in masses:
                bands, _ = solve_for_krange(k_axis,
                                            partial(Hamiltonian_k, n_bands=n_bands, mass=mass, detuning=detuning,
                                                    rabi=rabi, potential=depth, delta_k=2*np.pi/period))
                _values = []
                for background in backgrounds:
                    _eig = bands + background
                    _values += [_eig]
                _valss += [bands]
            _vals += [_valss]
        values += [_vals]
    return np.squeeze(values), k_axis


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
        depths = np.linspace(0.1, 10.1, 101)
    theory_bands, theory_kaxis = run_simulations(depths, [period], 0, MASS)

    if results is None:
        results = dict(depths=depths, bands=theory_bands, k_axis=theory_kaxis)
    else:
        results = dict(depths=np.append(depths, results['depths']),
                       bands=np.append(theory_bands, results['bands'], axis=0),
                       k_axis=np.append(theory_kaxis, results['k_axis'], axis=0))

    return results


def fit_theory(dataset_index, selected_indices=None, run_sims=True, adjust_tilt=False):
    if dataset_index == 1:
        return None
    with h5py.File(collated_analysis, 'r') as dfile:
        laser_separations = dfile['laser_separations'][...]
    period = np.abs(2 * np.pi / laser_separations[dataset_index])

    # Experiment
    _, bands, config, analysis_results, variables = get_experimental_data(dataset_index)
    if selected_indices is None:
        selected_indices = tuple([slice(x) for x in bands.shape])
    exper_energy_array = analysis_results[0][selected_indices]

    if len(exper_energy_array.shape) > 1:
        initial_shape = exper_energy_array.shape[:-1]
    else:  # ensures that the first axis is always cw power, even if the dataset doesn't have that variable
        initial_shape = (1,)
        exper_energy_array = np.array([exper_energy_array])
    exper_split = np.diff(exper_energy_array*1e3, axis=-1)[..., 0]

    # Adjusting the split to be perpendicular to the bands
    if adjust_tilt:
        angle = analysis_results[1][selected_indices] * np.diff(config['k_axis'])[0] / np.diff(config['energy_axis'])[0]
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

    # Theory
    if run_sims:
        results = run_simulations_dataset(dataset_index)
        np.save(get_data_path('%s/simulations/dataset%d_simulations' % (folder_name, dataset_index)), results)
    else:
        results = np.load(get_data_path('%s/simulations/dataset%d_simulations.npy' % (folder_name, dataset_index)),
                          allow_pickle=True).take(0)
    theory_depths = results['depths']
    theory_bands = results['bands']
    theory_centers = np.amin(theory_bands, 1)

    all_splittings = np.diff(theory_centers, axis=-1)
    first_splitting = all_splittings[..., 0]
    theory_depth_vs_splitting = interp1d(first_splitting, theory_depths, bounds_error=False, fill_value=np.nan)

    # Fitting
    fitted_depths = theory_depth_vs_splitting(exper_split)

    # Re-running simulations for the fitted values
    fitted_results = dict(depths=[], bands=[], k_axis=[])
    n_bands = 20
    for depth in tqdm(fitted_depths.flatten(), 'fit_theory'):
        if np.isnan(depth):
            bands = np.full((301, 82), np.nan)
            k_axis = np.full((301, ), np.nan)
        else:
            bands, k_axis = run_simulations([depth], [period], 0, MASS, n_bands=n_bands, disable_output=True)
        fitted_results['depths'] += [depth]
        fitted_results['bands'] += [bands]
        fitted_results['k_axis'] += [k_axis]
    fitted_results['depths'] = np.reshape(fitted_results['depths'], initial_shape)
    fitted_results['bands'] = np.reshape(fitted_results['bands'], initial_shape + (301, 82))
    fitted_results['k_axis'] = np.reshape(fitted_results['k_axis'], initial_shape + (301, ))

    return fitted_results


def plot_theory_fit(dataset_index, selected_indices, run_fit=False, run_sims=False, axis_offsets=(0, 0),
                    ax=None, imshow_kw=None, plot_kw=None, fill_kw=None, plotting_kw=None, label_kw=None):
    """

    :param dataset_index:
    :param selected_indices:
    :param run_sims:
    :param imshow_kw:
    :param plot_kw:
    :param fill_kw:
    :return:
    """
    if label_kw is None: label_kw = dict()

    data, bands, config, analysis_results, variables = get_experimental_data(dataset_index)
    if run_fit:
        fitted_results = fit_theory(dataset_index, selected_indices, run_sims)
        theory_bands, theory_kaxis, depths = [fitted_results[x] for x in ['bands', 'k_axis', 'depths']]
    else:
        fitted_results = np.load(get_data_path('%s/simulations/dataset%d_fits.npy' % (folder_name, dataset_index)), allow_pickle=True).take(0)
        theory_bands, theory_kaxis, depths = [fitted_results[x][selected_indices] for x in ['bands', 'k_axis', 'depths']]
        if len(theory_kaxis.shape) == 1:
            depths = [depths]

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
    if plotting_kw is None: plotting_kw = dict()
    plotting_kw = {**dict(show_label=True, show_bands=True, show_theory=True), **plotting_kw}

    iterable = (axs, flattened_bands, flattened_images, flattened_exp_energy, depths)
    for ax, bnd, img, exper_energy, depth in zip(*iterable):
        xaxis = np.array(config['k_axis'])-axis_offsets[0]
        yaxis = np.array(config['energy_axis'])
        yaxis *= 1e3
        imshow(normalize(img), ax, xaxis=xaxis, yaxis=yaxis, **imshow_kw)
        if plotting_kw['show_bands']:
            if 'plot_max_band' in plotting_kw:
                max_band = plotting_kw['plot_max_band']
            else:
                max_band = len(bnd)
            for idx, _band in enumerate(bnd):
                if idx < max_band:
                    if 'k_filtering' in plotting_kw:
                        mask = np.abs(_band[:, 0]-axis_offsets[0]) < plotting_kw['k_filtering']
                        _band = _band[mask]
                    ax.plot(_band[:, 0]-axis_offsets[0], _band[:, 1]*1e3, **plot_kw)
        if plotting_kw['show_theory']:
            if not np.isnan(depth):
                theory_bands = theory_bands.real
                theory_bands -= np.min(theory_bands)
                theory_bands += axis_offsets[1]
                min_idx, max_idx = [np.argmin(np.abs(theory_kaxis-x)) for x in [xaxis.min(), xaxis.max()]]
                if 'n_bands' in plotting_kw:
                    ax.plot(theory_kaxis[min_idx:max_idx], theory_bands[min_idx:max_idx, :plotting_kw['n_bands']], **fill_kw)
                else:
                    print(theory_kaxis.shape, theory_bands.shape)
                    ax.plot(np.squeeze(theory_kaxis)[min_idx:max_idx],
                            np.squeeze(theory_bands)[min_idx:max_idx], **fill_kw)
            else:
                print('Depth is NaN')
        if plotting_kw['show_label']:
            if 'label_string' in plotting_kw:
                label = ''
            else:
                label = '%s$k_{laser}$=%.2g%sm$^{-1}$\n' % (greek_alphabet['delta'], config['laser_angle'], greek_alphabet['mu'])
                try:
                    power_label = variables['normalised_power_axis'][selected_indices[-1]] * 1e3
                    label += '$P_s$=%.1fmW%sm$^{-2}$\n' % (power_label, greek_alphabet['mu'])
                except:
                    pass
                if not np.isnan(depth):
                    label += '$V_{eff}$=%.2fmeV' % depth
            ax.text(0.5, 0.99, label, ha='center', va='top', transform=ax.transAxes, **label_kw)

        ax.set_xlim(xaxis.min(), xaxis.max())
        _formatter = dummy_formatter(axis_offsets[1], True, None, True)
        ax.yaxis.set_major_formatter(_formatter)


def plot_theory_density(dataset_index, selected_indices, fig_ax=None, rerun=False,
                        ground_state_pixel=None, ground_state_energy=None, imshow_kwargs=None):
    if imshow_kwargs is None: imshow_kwargs = dict()

    data, variables, config = get_experimental_data_base(dataset_index)
    simulations = np.load(get_data_path('%s/simulations/dataset%d_fits.npy' % (folder_name, dataset_index)),
                          allow_pickle=True).take(0)
    depths = simulations['depths']

    if len(data.shape) < 5:
        # ensures that the first axis is always cw power, even if the dataset doesn't have that variable
        selected_indices = (0,) + selected_indices
        depths = np.array([depths])
        cw = None
    else:
        try:
            cw = np.array(variables['cw'])[selected_indices[0]]
        except:
            cw = np.array(variables['power'])[selected_indices[0]]
    fit_veff = depths[selected_indices]
    power = np.array(variables['vwp'])[selected_indices[2]]
    freq = np.array(variables['f'])[selected_indices[1]]
    h5pylabel = 'deltak={deltak}/'
    if cw is not None:
        h5pylabel += 'cw={cw}/'
    h5pylabel += 'power={power}/f={f}'
    h5pylabel = h5pylabel.format(deltak=config['laser_angle'], cw=cw, power=power, f=freq)
    print('Dataset details: ', h5pylabel, ' Veff=%g' % fit_veff)

    path = get_data_path('%s/simulations/pl_density.h5' % folder_name)

    # Getting (or simulating, if not available) all the data
    with h5py.File(path, 'a') as dfile:
        # Simulation parameters
        # n_points = 151
        # periods = 8
        # times = np.linspace(-100, 100, 4001)
        n_points = 101
        periods = 8
        # times = np.linspace(-50, 50, 2001)
        # times = np.linspace(-75, 75, 2001)
        times = np.linspace(-100, 100, 2001)
        kwargs = dict(potential=fit_veff, periods=periods, n_points=n_points, delta_k=config['laser_angle'])

        # Calculating momentum axis
        single_period = 2 * np.pi / np.abs(config['laser_angle'])
        if periods is None:
            x = np.linspace(-21, 20, n_points)
        else:
            x = np.linspace(-single_period * periods / 2 - 0.1 * single_period, single_period * periods / 2, n_points)
        dx = np.diff(x)[0]
        theory_kax = np.linspace(-np.pi/dx, np.pi/dx, n_points)

        # Calculating static eigenvalues and eigenvectors, which are used as starting points in the temporal evolution
        # of the Floquet Hamiltonian
        static_hamiltonian = Hamiltonian_x(0, frequency=0, **kwargs)
        values, vectors = np.linalg.eig(static_hamiltonian)

        if rerun or (h5pylabel not in dfile):
            floquet_hamiltonian = partial(Hamiltonian_x, frequency=freq * 1e-3, **kwargs)
            idx = np.argsort(values)
            vectors = vectors[:, idx]
            theory_density = farfield(floquet_hamiltonian, vectors, times)
            if rerun:
                try:
                    del dfile[h5pylabel]
                except:
                    pass
            dfile.create_dataset(h5pylabel, data=theory_density)
        else:
            dset = dfile[h5pylabel]
            theory_density = dset[...]
        dE = hbar * np.pi / times[-1]
        theory_eax = (np.linspace(-dE, dE, len(times)) * len(times) / 2)[::-1]

        print('n_points: ', n_points, '  theory_density.shape: ', theory_density.shape)
        print('Theory k_ax: ', theory_kax[:3], '...', theory_kax[-3:])

    # Cropping the data to the physically relevant area
    if ground_state_pixel is None:
        k0 = theory_density[theory_density.shape[0]//2]
        idxs = find_peaks(normalize(k0), 0.01)[0]
        try:
            ground_state_pixel = np.max(idxs)
            print('Ground state pixel: ', ground_state_pixel)
        except:
            ground_state_pixel = 50
            print('Failed at ground state pixel: ', ground_state_pixel)
    if ground_state_energy is None:
        ground_state_energy = np.mean(config['energy_axis']) * 1e3
    theory_eax -= theory_eax[ground_state_pixel]
    theory_eax += ground_state_energy
    bottom_lim = config['energy_axis'][-1] * 1e3
    top_lim = config['energy_axis'][0] * 1e3
    max_idx = np.argmin(np.abs(theory_eax - bottom_lim))
    min_idx = np.argmin(np.abs(theory_eax - top_lim))

    fig, ax = create_axes(fig_ax)
    _kwargs = dict(cbar=False, diverging=False, norm=LogNorm(1e-5, 1), cmap='Greys')
    imshow_kwargs = {**_kwargs, **imshow_kwargs}
    imshow(normalize(np.fliplr(theory_density.transpose()[min_idx:max_idx])), ax,
           xaxis=theory_kax, yaxis=theory_eax[min_idx:max_idx], **imshow_kwargs)
    ax.set_xlim(config['k_axis'][0], config['k_axis'][-1])
    _formatter = dummy_formatter(ground_state_energy, True, None, True)
    _formatter._offset_threshold = 2
    _formatter.format = '%.1g'
    ax.yaxis.set_major_formatter(_formatter)
    return fig, ax


def compare_to_experiment(dataset_index, selected_indices, ground_state_energy=0., ground_state_pixel=None,
                          max_bands=None, rerun=False, arbitrary_k_scale=1, show_label=False, show_theory=True):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.01))
    plot_kw = dict(color='#a25bdc', ls='--', alpha=0.3)
    fill_kw = dict(alpha=0.5, color=(142 / 255, 37 / 255, 228 / 255, 0.3))
    plot_theory_fit(dataset_index, selected_indices, ax=axs[0], plot_kw=plot_kw, fill_kw=fill_kw,
                    plotting_kw=dict(n_bands=max_bands, show_label=show_label, show_theory=show_theory), axis_offsets=(-0.12, ground_state_energy))

    plot_theory_density(dataset_index, selected_indices, axs[1], rerun=rerun,
                        ground_state_pixel=ground_state_pixel, ground_state_energy=ground_state_energy,
                        arbitrary_k_scale=arbitrary_k_scale)
    label_grid(fig._gridspecs[0], 'Energy [eV]', 'left', offset=0.05)
    label_grid(fig._gridspecs[0], 'Momentum [%sm$^{-1}$]' % greek_alphabet['mu'], 'bottom')
    return fig, axs


if __name__ == '__main__':
    def sinusoidal_k(k, potential, delta_k=10., mass=MASS, n_bands=6):
        G = delta_k

        # TODO  compare to a model where the photon and the exciton are fully separated (mass)
        space_size = 2 * n_bands + 1

        # Kinetic energy
        Hk0 = np.diag([hbar ** 2 * (k - x * G) ** 2 / (2 * mass) for x in range(-n_bands, n_bands + 1)])

        # Potential energy
        pot = [potential / 2] * (space_size - 1)
        Hv = np.diag(pot, -1) + np.diag(pot, 1)

        return Hk0 + Hv


    def Hamiltonian_delta_k(k, potential, delta_k=10., mass=MASS, detuning=DETUNING, rabi=RABI, n_bands=6):
        G = delta_k

        # TODO  compare to a model where the photon and the exciton are fully separated (mass)
        space_size = 2 * n_bands + 1

        # Kinetic energy
        Hk0 = np.diag([hbar ** 2 * (k - x * G) ** 2 / (2 * mass) for x in range(-n_bands, n_bands + 1)])
        Hk0 -= np.eye(space_size) * detuning / 2

        # Potential energy
        Hv = potential * np.ones((space_size, space_size))
        Hv += np.eye(space_size) * detuning / 2

        # Coupling to exciton
        H1row = np.hstack([Hk0, rabi * np.eye(space_size)])
        H2row = np.hstack([rabi * np.eye(space_size), Hv])
        return np.vstack([H1row, H2row])


    def test_hamiltonians():
        potential = 0
        deltak = 0.43
        vals = []

        for period in [3, 5, 7]:
            hx = Hamiltonian_x(0, potential, deltak, 0, period, n_points=101)
            val1, vec1 = np.linalg.eig(hx)
            _sort_idx = np.argsort(val1)
            val1 = val1[_sort_idx]
            vec1 = vec1[:, _sort_idx]
            vals += [val1]

        ks = np.linspace(-2, 2, 101)
        hk = partial(Hamiltonian_k, potential=potential, delta_k=deltak, n_bands=6)
        bands, modes = solve_for_krange(ks, hk)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ks, bands[:, :])
        [ax.plot(val[:100], '--') for val in vals]
        ax2 = ax.twiny()
        ax2.plot(bands[:, :])


    def testing_nonhermitian():
        gs_widths = []
        gaps = []
        ampl = 1
        k = np.linspace(-2, 2, 501)
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 51)
        for theta in thetas:
            p = ampl * np.exp(1j * theta)
            hk = partial(Hamiltonian_k, potential=p, delta_k=0.6, n_bands=6)
            bands, modes = solve_for_krange(k, hk)
            gs_widths += [np.max(bands[:, 0]) - np.min(bands[:, 0])]
            gaps += [np.min(bands[:, 1]) - np.max(bands[:, 0])]
        gs_widths = np.array(gs_widths)
        gaps = np.array(gaps)

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, fig)
        _gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0])
        ax = plt.subplot(_gs[0])
        ax.plot(thetas, gs_widths.real, '.-')
        ax2 = ax.twinx()
        ax2.plot(thetas, gs_widths.imag, '.-', color='C1')
        ax = plt.subplot(_gs[1])
        ax.plot(thetas, gaps.real)

        p = ampl * np.exp(1j * 0)
        hk = partial(Hamiltonian_k, potential=p, delta_k=0.6, n_bands=6)
        bands, modes = solve_for_krange(k, hk)
        _gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[1])
        axs = _gs.subplots()
        axs[0].plot(k, bands[:, :2].real)
        axs[1].plot(k, bands[:, :2].imag)

        p = ampl * np.exp(1j * np.pi / 2)
        hk = partial(Hamiltonian_k, potential=p, delta_k=0.6, n_bands=6)
        bands, modes = solve_for_krange(k, hk)
        _gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[2])
        axs = _gs.subplots()
        axs[0].plot(k, bands[:, :2].real)
        axs[1].plot(k, bands[:, :2].imag)

        fig, axs = plt.subplots(2, 4, sharex=True)
        k = np.linspace(-2, 2, 501)
        for p, ax in zip([np.sqrt(5) * 1j, 1 + 2j, 2 + 1j, np.sqrt(5)], axs.transpose()):
            hk = partial(Hamiltonian_k, potential=p, delta_k=0.6, n_bands=6)
            bands, modes = solve_for_krange(k, hk)
            ax[0].plot(k, bands[:, :3].real)
            _bands = bands[:, :3]
            if p.real < 1e-5:
                _b2 = _bands[:, -1]
                _b01 = _bands[:, :-1]
                mask = np.diff(_b01.real, axis=-1)[:, 0] < 1e-5
                for idx, msk in enumerate(mask):
                    if msk:
                        indxs = np.argsort(_b01[idx].imag)
                        _b01[idx] = _b01[idx][indxs]
                # indxs = np.argsort(_b01.imag, axis=-1)
                # _b01[mask] = _b01[mask][indxs[mask]]

                _bands = np.concatenate([_b01, _b2[:, np.newaxis]], 1)
            ax[1].plot(k, _bands.imag)
        for idx in range(len(axs[0]) - 1):
            axs[0, idx].sharey(axs[0, idx + 1])
            axs[1, idx].sharey(axs[1, idx + 1])

        fig, axs = plt.subplots(2, 3, sharex=True)
        k = np.linspace(-2, 2, 501)
        for p, ax in zip([2, 2 + 1j, 2 + 2j], axs.transpose()):
            hk = partial(Hamiltonian_k, potential=p, delta_k=0.6, n_bands=6)
            bands, modes = solve_for_krange(k, hk)
            ax[0].plot(k, bands[:, :3].real)
            _bands = bands[:, :3]
            if p.real < 1e-5:
                _b2 = _bands[:, -1]
                _b01 = _bands[:, :-1]
                mask = np.diff(_b01.real, axis=-1)[:, 0] < 1e-5
                for idx, msk in enumerate(mask):
                    if msk:
                        indxs = np.argsort(_b01[idx].imag)
                        _b01[idx] = _b01[idx][indxs]
                # indxs = np.argsort(_b01.imag, axis=-1)
                # _b01[mask] = _b01[mask][indxs[mask]]

                _bands = np.concatenate([_b01, _b2[:, np.newaxis]], 1)
            ax[1].plot(k, _bands.imag)
        for idx in range(len(axs[0]) - 1):
            axs[0, idx].sharey(axs[0, idx + 1])
            axs[1, idx].sharey(axs[1, idx + 1])
