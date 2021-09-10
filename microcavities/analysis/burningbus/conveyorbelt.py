# -*- coding: utf-8 -*-
import numpy as np

from microcavities.utils.plotting import *
import lmfit
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter
from microcavities.simulations.quantum_box import kinetic_matrix, normalise_potential, plot, solve
from matplotlib.colors import LogNorm
from microcavities.utils import apply_along_axes, random_choice
from microcavities.analysis.characterisation import *
from microcavities.analysis.phase_maps import low_pass
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.neighbors import KDTree
from skimage.feature import peak_local_max
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# import shapely
import pyqtgraph as pg
from nplab.utils.log import create_logger
from itertools import combinations
from copy import deepcopy
from functools import partial
LOGGER = create_logger('Fitting')
LOGGER.setLevel('WARN')


spatial_scale = magnification('rotation_pvcam', 'real_space')[0] * 1e6
momentum_scale = magnification('rotation_pvcam', 'k_space')[0] * 1e-6


# EXPERIMENTAL DATA FITTING

def smoothened_find_peaks(x, *args, **kwargs):
    """Simple extension of find_peaks to give more than single-pixel accuracy"""
    smoothened = savgol_filter(x, 5, 3)
    sampling = interp1d(range(len(smoothened)), smoothened, 'quadratic')
    new_x = np.linspace(0, len(smoothened)-1, len(smoothened)*10)
    new_y = sampling(new_x)
    results = find_peaks(new_y, *args, **kwargs)
    return new_x[results[0]], results[1]


class FitQHOModes:
    """
    TODO: sanity checks that the number of expected lobes for band are there
    """
    def __init__(self, image, configuration):
        # if image is None:
        #     idx = np.random.randint(_data.shape[0])
        #     idx2 = np.random.randint(_data.shape[1])
        #     idx3 = np.random.randint(_data.shape[2])
        #     print(idx, idx2, idx3)
        #     image = _data[idx, idx2, idx3]
        # if configuration is None:
        #     configuration = config
        self.config = configuration
        self.smoothened_image = low_pass(normalize(image), self.config['low_pass_threshold'])

        if 'energy_axis' not in self.config:
            e_roi = self.config['energy_roi']
            _wvls = spectrometer_calibration('rotation_acton', 803, '2')[e_roi[1]:e_roi[0]:-1]
            self.energy_axis = 1240 / _wvls
        else:
            self.energy_axis = self.config['energy_axis']
        self.energy_func = interp1d(range(len(self.energy_axis)), self.energy_axis, bounds_error=False, fill_value=np.nan)
        self.e_inverse = interp1d(self.energy_axis, range(len(self.energy_axis)))
        if 'k_axis' not in self.config:
            self.k_axis = np.arange(self.smoothened_image.shape[1], dtype=np.float) - self.config['k_masking']['k0']
            self.k_axis *= momentum_scale
        else:
            self.k_axis = self.config['k_axis']
        self.k_func = interp1d(range(len(self.k_axis)), self.k_axis, bounds_error=False, fill_value=np.nan)
        self.k_inverse = interp1d(self.k_axis, range(len(self.k_axis)))
        # print(self.energy_axis.shape, self.k_axis.shape, self.smoothened_image.shape, image.shape)
        # plt.figure()
        # plt.plot(self.energy_axis)
        # plt.plot(range(10), self.energy_func(range(10)))

    def _mask_by_momentum(self, peaks, k_range=None):
        # Eliminates peaks that are outside the given momentum range
        if k_range is None:
            if 'k_range' in self.config:
                k_range = self.config['k_range']
            else:
                k_range = (np.min(peaks[:, 0])-1, np.max(peaks[:, 0]) + 1)
        return np.logical_and(peaks[:, 0] > k_range[0], peaks[:, 0] < k_range[1])

    def _mask_by_intensity(self, peak_intensities, threshold=0.01):
        return peak_intensities > threshold * np.max(peak_intensities)

    def mask_untrapped(self, peaks, min_energy=None):
        config = self.config['k_masking']
        k0 = config['k0']
        slope = config['k_slope']
        if min_energy is None:
            min_energy = np.max(peaks[:, 1]) - config['e_offset']
            LOGGER.debug('mask_untrapped min_energy=%g' % min_energy)
        mask1 = (peaks[:, 0]-k0)*slope+min_energy > peaks[:, 1]
        mask2 = -(peaks[:, 0]-k0)*slope+min_energy > peaks[:, 1]
        mask = np.logical_and(mask1, mask2)
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(peaks[:, 0], peaks[:, 1], '.')
        # ax.plot(peaks[:, 0], (peaks[:, 0]-k0)*slope+min_energy)
        # ax.plot(peaks[:, 0], -(peaks[:, 0]-k0)*slope+min_energy)
        return peaks[mask], min_energy

    def find_peaks_1d(self, ax=None, **plot_kwargs):
        config = self.config['peak_finding']
        peaks = []
        for idx, x in enumerate(self.smoothened_image.transpose()):
            # pks = find_peaks(x, **self.config['find_peaks_kw'])[0]
            pks = smoothened_find_peaks(x, **config['peaks_1d'])[0]
            peaks += [(idx, pk) for pk in pks]
        peaks = np.asarray(peaks, dtype=np.float)
        if self.config['plotting'] and ax is None:
            _, ax = plt.subplots(1, 1)
        if ax is not None:
            imshow(self.smoothened_image, ax, cmap='Greys', diverging=False, norm=LogNorm(),
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            defaults = dict(ms=1, ls='None', marker='.')
            for key, value in defaults.items():
                if key not in plot_kwargs:
                    plot_kwargs[key] = value
            ax.plot(self.k_func(peaks[:, 0]), self.energy_func(peaks[:, 1]), **plot_kwargs)
        return peaks

    def find_peaks_2d(self, ax=None, **plot_kwargs):
        config = self.config['peak_finding']
        peaks = peak_local_max(self.smoothened_image, **config['peaks_2d'])
        if self.config['plotting'] and ax is None:
            _, ax = plt.subplots(1, 1)
        if ax is not None:
            defaults = dict(ms=1, ls='None', marker='.')
            for key, value in defaults.items():
                if key not in plot_kwargs:
                    plot_kwargs[key] = value
            imshow(self.smoothened_image, ax, cmap='Greys', diverging=False, norm=LogNorm(),
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            ax.plot(self.k_func(peaks[:, 1]), self.energy_func(peaks[:, 0]), **plot_kwargs)
        return peaks[:, ::-1]

    def make_bands(self, peaks1d=None, peaks2d=None, n_bands=None, ax=None, plot_kwargs=None):
        """Create bands from 1D and 2D peak fitting

        Algorithm proceeds as follows:
            Starting from 2D peaks, create chains of nearest-neighbours from the 1D peak fitting:
                - The neighbouring distance can be configured
                - Excludes points that are outside the expected trapping cone (see mask_untrapped)
                - Excludes points that are too far away from the expected mode linewidth (prevents considering chains
                that climb up the thermal line)
            Groups the chains into bands:
                - Only considers chains that are long enough (min_band_points)
                - If the average energy of the chain is similar to an existing band (min_band_distance), append the two
                chains (ensures that different lobes of the same band get grouped correctly. Would fail for heavily
                tilted bands with high contrast between lobes)
            Sanity checks:
                - The average momentum of the band needs to be near k0 (k_acceptance). Otherwise, exclude
                - Sorts the band by energy

        :param peaks1d:
        :param peaks2d:
        :param ax:
        :return:
        """
        if plot_kwargs is None:
            plot_kwargs = dict()
        if self.config['plotting'] and ax is None:
            _, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey='all', sharex='all')
        elif ax is None:
            ax0 = None
            ax1 = None
            ax2 = None
        else:
            try:
                ax0, ax1, ax2 = ax
            except Exception as e:
                ax0 = ax
                ax1 = ax
                ax2 = ax
        if peaks1d is None:
            peaks1d = self.find_peaks_1d(ax0, **plot_kwargs)
        if peaks2d is None:
            peaks2d = self.find_peaks_2d(ax1, **plot_kwargs)
        tree = KDTree(peaks1d)
        groups = tree.query_radius(peaks2d, self.config['peak_finding']['neighbourhood'])

        # First, select 2D peaks that have more than 2 nearest neighbours
        selected_points = []
        for idx, g in enumerate(groups):
            if len(g) > self.config['peak_finding']['min_neighbours']:
                selected_points += [peaks2d[idx]]
        selected_points = np.array(selected_points)
        # Exclude and estimate the parameters for excluding untrapped 2D peaks
        # TODO: when calculating the min_energy, add some sort of limits/pre-given estimates
        selected_points, _min_energy = self.mask_untrapped(selected_points)

        # Get chains of 1D peaks. Starting from the selected 2D points, extract chains of 1D peaks that are sequences of
        # points that are not too far apart
        config = self.config['chain_making']
        kd_tree_radius = config['kd_tree_radius']
        starting_groups = [peaks1d[g] for g in tree.query_radius(selected_points, kd_tree_radius)]
        LOGGER.debug('%d starting groups %s, %d selected points for chains' % (len(starting_groups), [len(x) for x in starting_groups], len(selected_points)))

        chains = []
        for g in starting_groups:
            max_idx = config['max_iterations']
            stop = False
            idx = 0
            new_g = np.copy(g)
            while not stop:
                if idx > max_idx:
                    raise RuntimeError('Maximum iterations reached')
                start_length = len(new_g)
                if start_length == 0:
                    stop = True
                else:
                    indices = tree.query_radius(new_g, kd_tree_radius)
                    new_g = [peaks1d[g] for g in indices]
                    new_g = np.concatenate(new_g)
                    new_g = np.unique(new_g, axis=0)
                    end_length = len(new_g)
                    idx += 1
                    if start_length == end_length:  # idx > max_idx or
                        stop = True
            _len1 = len(new_g)
            # Exclude untrapped polaritons
            new_g = self.mask_untrapped(new_g, _min_energy)[0]
            LOGGER.debug('Found chain of length %d (%d after masking) after %d iterations' % (_len1, len(new_g), idx))

            # Only include chains that are larger than a minimum size (in points)
            if len(new_g) > config['min_chain_size']:
                chains += [new_g]
        # Order the chains in energy, which prevents issues in which joining chains in different orders can give
        # different bands
        average_energies = [np.mean(p, 0)[1] for p in chains]
        argsort = np.argsort(self.energy_func(average_energies))
        chains = [chains[idx] for idx in argsort]

        tilt_limits = [config['expected_tilt']-config['tilt_width'], config['expected_tilt']+config['tilt_width']]

        def calculate_tilts(_chain, _chains=None):
            if _chains is None:
                _chains = chains
            tilts = []
            for chain in _chains:
                k1, e1 = [f(x) for f, x in zip((self.k_func, self.energy_func), np.median(chain, 0))]
                k0, e0 = [f(x) for f, x in zip((self.k_func, self.energy_func), np.median(_chain, 0))]
                deltak = (k1-k0)
                deltae = (e1-e0)
                tilts += [deltae/deltak]
            tilts = np.array(tilts)
            return tilts

        def join_chains_by_tilt(_chains):
            _new_chains = []
            _merged_chains = []
            for chain_index, chain in enumerate(_chains):
                if chain_index in _merged_chains:
                    continue
                LOGGER.debug('Chain_index: %d' % chain_index)
                tilts = calculate_tilts(chain, _chains)
                LOGGER.debug('Tilts: %s' % tilts)
                where = np.nonzero(np.logical_or(
                    np.logical_and(tilts >= tilt_limits[0], tilts <= tilt_limits[1]),
                    np.isnan(tilts)))[0]
                LOGGER.debug('Where: %s   tilts[where]: %s' % (where, tilts[where]))
                _merged_chains += [idx for idx in where]
                to_merge = [_chains[idx] for idx in where]
                merged_chain = np.concatenate(to_merge)
                _new_chains += [np.unique(merged_chain, axis=0)]
                LOGGER.debug('old_chains: %d   new_chains: %d' % (len(_chains), len(_new_chains)))
            if len(_chains) > len(_new_chains):
                return join_chains_by_tilt(_new_chains)
            else:
                return _new_chains

        LOGGER.debug('Joining chains by tilt')
        bands = join_chains_by_tilt(chains)

        def calculate_distances(_chain, _chains):
            distances = []
            for chain in _chains:
                distances += [np.sqrt(np.sum(np.abs(np.mean(chain, 0) - np.mean(_chain, 0))**2))]
            return np.array(distances)

        def join_chains_by_distance(_chains, threshold=config['min_chain_separation']):
            _new_chains = []
            _merged_chains = []
            for chain_index, chain in enumerate(_chains):
                if chain_index in _merged_chains:
                    continue
                LOGGER.debug('Chain_index: %d' % chain_index)
                distances = calculate_distances(chain, _chains)
                LOGGER.debug('Distances: %s' % distances)
                where = np.nonzero(distances < threshold)[0]
                LOGGER.debug('Where: %s   tilts[where]: %s' % (where, distances[where]))
                _merged_chains += [idx for idx in where]
                to_merge = [_chains[idx] for idx in where]
                merged_chain = np.concatenate(to_merge)
                _new_chains += [np.unique(merged_chain, axis=0)]
                LOGGER.debug('old_chains: %d   new_chains: %d' % (len(_chains), len(_new_chains)))
            if len(_chains) > len(_new_chains):
                return join_chains_by_distance(_new_chains)
            else:
                return _new_chains
        LOGGER.debug('Joining chains by distance')
        bands = join_chains_by_distance(bands)

        config = self.config['band_filtering']
        LOGGER.debug('Full chains: %s' % (str([(np.array(band).shape, np.mean(band, 0), np.percentile(band[:, 1], 10)) for band in bands])))
        # Clipping bands that snake up the thermal line
        masks = [band[:, 1] > np.percentile(band[:, 1], 90) - config['max_band_linewidth'] for band in bands]
        LOGGER.debug('Removing %s points in each band because they fall outside the linewidth' % str([np.sum(mask) for mask in masks]))
        bands = [g[m] for g, m in zip(bands, masks)]

        # Excluding groups that have average momentum too far from 0
        k0 = self.config['k_masking']['k0']
        ka = config['k_acceptance']
        average_energies = np.array([np.mean(p, 0)[1] for p in bands])
        masks = np.array([k0-ka < np.mean(g[:, 0]) < k0+ka for g in bands])
        _excluded_indices = np.arange(len(average_energies))[~masks]
        LOGGER.debug('Excluding bands by momentum: %s with %s energy %s momentum' % (_excluded_indices,
                                                                         self.energy_func(average_energies[~masks]),
                                                                         [self.k_func(np.mean(bands[x], 0)) for x in _excluded_indices]))
        bands = [g for g, m in zip(bands, masks) if m]

        # Keeping only bands that are larger than some minimum size
        bands = [band for band in bands if len(band) > config['min_band_size']]

        # Sorting by energy
        average_energies = [np.mean(p, 0)[1] for p in bands]
        argsort = np.argsort(self.energy_func(average_energies))
        bands = [bands[idx] for idx in argsort]

        # Transform into units
        bands = [np.array([self.k_func(band[:, 0]), self.energy_func(band[:, 1])]).transpose() for band in bands]

        # Make a fixed length tuple if required
        if n_bands is not None:
            if n_bands < len(bands):
                bands = bands[:n_bands]
            else:
                bands += [np.empty((0, 2))]*(n_bands-len(bands))

        LOGGER.debug('Final bands: %s' % str(list([len(band), np.mean(band[:, 0]), np.mean(band[:, 1])] for band in bands)))

        if ax2 is not None:
            imshow(self.smoothened_image, ax2, diverging=False, cmap='Greys', norm=LogNorm(), cbar=False,
                   xaxis=self.k_axis, yaxis=self.energy_axis)
            defaults = dict(ms=1, ls='None', marker='.')
            for key, value in defaults.items():
                if key not in plot_kwargs:
                    plot_kwargs[key] = value
            ax2.plot(self.k_func(selected_points[:, 0]), self.energy_func(selected_points[:, 1]), color='k', **plot_kwargs)
            for s in bands:
                ax2.plot(s[:, 0], s[:, 1], **plot_kwargs)
        return bands

    def _calculate_slice(self, linear_fit, band_width, image=None):
        if image is None:
            image = self.smoothened_image
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
        # Fit a linear trend to it:
        #   Return the k0 energy and the tilt
        # Mode profiles
        #   Density distribution?
        # Linewidths?
        #   Get energy modes at the peaks of the mode profiles?
        if bands is None:
            bands = self.make_bands()
        if n_bands is None:
            n_bands = len(bands)

        _fits = []
        _coords = []
        energies = []
        tilts = []
        slices = []
        for idx in range(n_bands):
            band_width = 10
            if idx > len(bands) or bands[idx].size == 0:
                energies += [np.nan]
                tilts += [np.nan]
                slices += [np.full((self.smoothened_image.shape[1], band_width), np.nan)]
            # elif:
            #     np.isempty(bands[idx]
            else: # idx < len(bands):
                band = bands[idx]
                fit = np.polyfit(band[:, 0], band[:, 1], 1)
                # func = np.poly1d(fit)
                k0_energy = np.poly1d(fit)(0)  # func(self.config['k_masking']['k0'])
                tilt = fit[0]
                inverse_fit = np.polyfit(self.k_inverse(band[:, 0]), self.e_inverse(band[:, 1]), 1)
                vectors, start_slice, end_slice = self._calculate_slice(inverse_fit, band_width)
                slice, coords = pg.affineSlice(self.smoothened_image, end_slice, start_slice, vectors, (0, 1), returnCoords=True)
                _coords += [coords]
                _fits += [fit]
                energies += [k0_energy]
                tilts += [tilt]
                slices += [slice]
            # else:
            #     energies += [np.nan]
            #     tilts += [np.nan]
            #     slices += [np.full((self.smoothened_image.shape[1], band_width), np.nan)]

        if self.config['plotting'] and gs is None:
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 2, fig)
        if gs is not None:
            ax0 = plt.subplot(gs[0])
            imshow(self.smoothened_image, ax0, diverging=False, cmap='Greys', norm=LogNorm(), xaxis=self.k_axis, yaxis=self.energy_axis)
            # ax0.imshow(self.smoothened_image, cmap='Greys', norm=LogNorm(), aspect='auto')

            gs1 = gridspec.GridSpecFromSubplotSpec(len(slices), 1, gs[1])
            axs = gs1.subplots()
            for idx, _fit, energ, tilt, slice in zip(range(len(_fits)), _fits, energies, tilts, slices):
                color = cm.get_cmap('Iris', len(_fits))(idx)
                func = np.poly1d(_fit)
                x_points = self.k_func([0, self.smoothened_image.shape[1]-1])
                ax0.plot(x_points, func(x_points))
                ax0.plot(self.k_func(_coords[idx][1].flatten()), self.energy_func(_coords[idx][0].flatten()),
                         '.', ms=0.3, color=color)
                try:
                    axs[idx].imshow(slice.transpose())
                    colour_axes(axs[idx], color)
                except TypeError:
                    axs.imshow(slice.transpose())

        return np.array(energies), np.array(tilts), np.array(slices)


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
            # cluster_heights = np.array([np.ptp(scaled[labels == l][:, 1]) for l in range(model.n_clusters_)])
            # print(cluster_heights)

            for label in range(model.n_clusters_):
                _points = scaled[labels == label]
                if len(_points) > 0:
                    width = np.percentile(_points[:, 1], 90) - np.percentile(_points[:, 1], 10)
                    if width > config['bandwidth']:
                        labels[labels == label] = -1
                        average_energy = np.mean(_points[:, 1])
                        LOGGER.debug('Excluding band %d by bandwidth: %s energy %s bandwidth' % (label, average_energy,
                                                                                             width))

            # widths = [np.ptp(scaled[labels == l][:, 1]) for l in range(model.n_clusters_)]
            # # widths = np.array([np.ptp(scaled[:, 1]) for band in labels])
            # masks = widths < config['bandwidth']
            # for idx, mask in enumerate(masks):
            #     if not mask:
            #         average_energy = np.mean(scaled[labels == idx][:, 1])
            #         LOGGER.debug('Excluding band %d by bandwidth: %s energy %s bandwidth' % (idx, average_energy, widths[idx]))
            # labels[masks] = -1
            # bands = [g for g, m in zip(bands, masks) if m]

        LOGGER.debug('Found %d clusters with labels %s' % (len(np.unique(labels[mask])), np.unique(labels[mask])))

        # masked_points = points[mask]
        # masked_scaled = scaled[mask]
        # masked_labels = labels[mask]
        masked_clusters = [points[labels == l] for l in np.unique(labels) if l >= 0]
        # masked_clusters_scaled = [masked_scaled[masked_labels == l] for l in np.unique(masked_labels) if l >= 0]

        # N_SAMPLES = 5000
        # RANDOM_STATE = 42
        # classifier = RandomForestClassifier(random_state=RANDOM_STATE)
        # classifier.fit(masked_scaled, masked_labels)
        # classifier.predict()
        # print([m.shape for m in masked_clusters])
        # print(masked_clusters[0])
        # tree = KDTree(scaled)
        # expanded_clusters = [tree.query_radius(cluster, r=2, count_only=False, return_distance=False) for cluster in masked_clusters_scaled]
        # print(len(expanded_clusters[0]), expanded_clusters[0])
        # cluster_classifiers = [KDTree(cluster) for cluster in masked_clusters]
        # print(len(cluster_classifiers[0].query_radius(points, r=2, count_only=False, return_distance=False)))
        # print(len(points))
        # expanded_clusters = [cluster[tree.query_radius(points, r=2, count_only=False, return_distance=False)]
        #                      for cluster, tree in zip(masked_clusters, cluster_classifiers)]

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
        if 'data_axes_order' in configuration:
            self.dataset = np.transpose(dataset, configuration['data_axes_order'])
        else:
            self.dataset = dataset
        return

    def _choose_random_image(self):
        image, indices = random_choice(self.dataset, tuple(range(len(self.dataset.shape) - 2)), return_indices=True)
        LOGGER.debug('Indices %s' % (indices,))
        return image

    def _cluster_single_image(self, image=None, ax=None):
        if image is None: image = self._choose_random_image()
        cls = SingleImageModeDetection(image, self.configuration)
        return cls.cluster(ax=ax)

    def _make_bands_single_image(self, image=None, ax=None):
        if image is None: image = self._choose_random_image()
        cls = SingleImageModeDetection(image, self.configuration)
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
            # plt.show()
            return energies, tilts, slices, (fig1, fig2, fig3)


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

def sinusoid(depth, period, periods=5, size=101, bkg_value=0, mass=1e-3):
    x = np.linspace(-periods*period, periods*period, size)
    potential = np.asarray(depth * np.cos(x * 2*np.pi / period) + bkg_value, dtype=np.complex)
    mass_array = np.ones(size) * mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def find_bound_modes(pot, kin, *args):
    # Should be groups of N traps
    n_traps = len(find_peaks(-np.diag(np.real(pot))))
    vals, vecs = solve(pot + kin)
    band_edges, _props = find_peaks(np.diff(np.real(vals)), width=[0.1, 2])
    unbound = np.argmin(np.abs(vals - pot.max()))
    print(band_edges, unbound, n_traps)
    band_edges = band_edges[band_edges < unbound]
    print(band_edges)
    assert all([not (_x+1) % n_traps for _x in band_edges])
    band_centers = []
    band_widths = []
    band_gaps = []
    old_values = None
    for idx in range(len(band_edges)):
        if idx == 0:
            values = vals[0:band_edges[0]+1]
        else:
            values = vals[band_edges[idx-1]+1:band_edges[idx]+1]
            band_gaps += [values.min() - old_values.max()]
        band_centers += [np.mean(values)]
        band_widths += [np.max(values) - np.min(values)]
        old_values = np.copy(values)
    return np.array(band_centers), np.array(band_widths), np.array(band_gaps)


def run_simulations(depths, periods, backgrounds=0, mass=3e-5):
    try:
        len(periods)
    except:
        periods = [periods]
    try:
        len(backgrounds)
    except:
        backgrounds = [backgrounds]

    values = []
    for depth in depths:
        _vals = []
        for period in periods:
            _values = []
            for background in backgrounds:
                pot, kin, x = sinusoid(depth, period, 10, 1001, background, mass=mass)
                vals, _ = solve(pot + kin)
                _values += [vals]
            _vals += [_values]
        values += [_vals]
    return np.squeeze(values)


def analyse_modes(eigenvalues, n_traps, potential_maximum=None, n_modes=None):
    if potential_maximum is None:
        potential_maximum = 0
    energies = np.real(eigenvalues)
    band_edges, _props = find_peaks(np.diff(energies), width=[0.1, 2], prominence=1e-4)
    unbound = np.argmin(np.abs(energies - potential_maximum))
    # print(band_edges, unbound, n_traps)
    band_edges = band_edges[band_edges < unbound]
    # print(band_edges)
    tst = [not (_x+1) % n_traps for _x in band_edges]
    # print(band_edges)
    band_edges = band_edges[np.argwhere(tst)][:, 0]
    # print(band_edges)
    if len(band_edges) == 0:
        if n_modes is not None:
            return np.repeat(np.nan, n_modes), np.repeat(np.nan, n_modes), np.repeat(np.nan, n_modes)
        else:
            return np.nan, np.nan, np.nan
    assert all([not (_x+1) % n_traps for _x in band_edges])
    band_centers = []
    band_widths = []
    band_gaps = []
    old_values = None
    for idx in range(len(band_edges)):
        if idx == 0:
            values = energies[0:band_edges[0]+1]
        else:
            values = energies[band_edges[idx-1]+1:band_edges[idx]+1]
            band_gaps += [values.min() - old_values.max()]
        band_centers += [np.mean(values)]
        band_widths += [np.max(values) - np.min(values)]
        old_values = np.copy(values)
    band_centers = np.array(band_centers)
    band_widths = np.array(band_widths)
    band_gaps = np.array(band_gaps)
    if n_modes is not None:
        if len(band_centers) >= n_modes:
            band_centers = band_centers[:n_modes]
            band_widths = band_centers[:n_modes]
            band_gaps = band_centers[:n_modes]
        else:
            # print(band_centers.shape)
            band_centers = np.append(band_centers, np.repeat(np.nan, n_modes - len(band_centers)))
            band_widths = np.append(band_widths, np.repeat(np.nan, n_modes - len(band_centers)))
            band_gaps = np.append(band_gaps, np.repeat(np.nan, n_modes - len(band_centers)))
            # print(band_centers.shape)
    return band_centers, band_widths, band_gaps


def find_two_parameters(ground_state, splitting, xaxis, yaxis, plot=False, colours=None):
    if isinstance(plot, plt.Axes):
        arg = plot
    elif not plot:
        plt.close('all')
        arg = None
    else:
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        imshow(theory_groundstate, axs[0], xaxis=xaxis, yaxis=yaxis, diverging=False)
        imshow(theory_splitting, axs[1], xaxis=xaxis, yaxis=yaxis, diverging=False)
        X, Y = np.meshgrid(xaxis, yaxis)
        axs[0].contour(X, Y, theory_groundstate, [ground_state], colors='w')
        axs[1].contour(X, Y, theory_splitting, [splitting], colors='w')
        arg = axs[2]
    _, _, intersection_points, lines = contour_intersections([theory_groundstate, theory_splitting],
                                                         [[ground_state], [splitting]], arg,
                                                         [xaxis] * 2, [yaxis] * 2, colours)
    if arg is None:
        plt.close('all')
    return intersection_points, lines


if __name__ == '__main__':
    LOGGER.setLevel('INFO')
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
