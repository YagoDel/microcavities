# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
import lmfit
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter
from microcavities.simulations.quantum_box import kinetic_matrix, normalise_potential, plot, solve
from matplotlib.colors import LogNorm
from microcavities.analysis.characterisation import *
from microcavities.analysis.phase_maps import low_pass
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.neighbors import KDTree
from skimage.feature import peak_local_max
from sklearn.decomposition import PCA
import shapely
import pyqtgraph as pg
from nplab.utils.log import create_logger
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
                func = np.poly1d(_fit)
                x_points = self.k_func([0, self.smoothened_image.shape[1]-1])
                ax0.plot(x_points, func(x_points))
                ax0.plot(self.k_func(_coords[idx][1].flatten()), self.energy_func(_coords[idx][0].flatten()), '.', ms=0.3)
                try:
                    axs[idx].imshow(slice.transpose())
                except TypeError:
                    axs.imshow(slice.transpose())

        return np.array(energies), np.array(tilts), np.array(slices)


class FitQHOManifolds:
    def __init__(self, dataset, configuration):
        self.dataset = dataset
        self.configuration = configuration

    def find_peaks_1d(self):
        def _func(image):
            cls = FitQHOModes(image, self.configuration)
            return cls.find_peaks_1d()
        return apply_along_axes(_func, (-2, -1), self.dataset)





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

