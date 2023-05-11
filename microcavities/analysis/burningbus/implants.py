# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
from microcavities.analysis.interactive import *
from microcavities.analysis.dispersion import *
from microcavities.analysis.phase_maps import low_pass
import h5py
from microcavities.analysis.utils import remove_cosmicrays
from tqdm import tqdm
from microcavities.analysis.utils import guess_peak
from microcavities.utils import interpolated_array


def cavity_energy(lp_energy, exciton_energy=1478, rabi=3):
    """Given a lower polariton energy, an exciton energy and a rabi, returns the cavity energy.
    Based on the 2-mode model pol=((X, Omega), (Omega, C))"""
    return (rabi**2/4 - lp_energy*(lp_energy-exciton_energy))/(exciton_energy-lp_energy)


"""UNIMPLANTED SAMPLE"""
old_path = get_data_path('2022_06_03/raw_data.h5')
calibration_file = 'tmd_acton.json'
old_corners = np.array([(729500, 13.89), (704500, 4.56), (-290500, 4.92), (-270500, 14.12)])
REFERENCE_CORNER = old_corners[3]
counts_to_mm = 100000


def old_get_scan(scan_name, y_positions=None, k0_index=253, k0_roi=3, k_roi=(0, -1), e_roi=(0, -1), norm=False):
    k_ax = (np.arange(400) - k0_index) * magnification('tmd_acton', 'k_space')[0] * 1e-6
    k_ax = k_ax[k_roi[0]:k_roi[-1]]
    with h5py.File(old_path, 'r') as df:
        group = df[scan_name]

        if y_positions is None:
            keys = group.keys()
            y_positions = np.array([float(k.split('=')[1]) for k in keys])
            y_positions.sort()
        else:
            try:
                len(y_positions)
            except:
                y_positions = [y_positions]
        # y_positions = y_positions[:4]

        k0 = []
        images = []
        for y in y_positions:
            group2 = group['row=%g' % y]
            x_positions = group2.attrs['x_positions']  #
            _k0 = []
            _images = []
            for x in x_positions:
                dset = group2['pos=%d/spectra' % (x)]
                img = np.array([d for d in dset[...]])
                cntrl_wvl = dset.attrs['central_wavelengths']
                wvls = np.array([spectrometer_calibration(calibration_file, cwvl, '2') for cwvl in cntrl_wvl])
                _k0 += [np.mean(img[:, k0_index - k0_roi:k0_index + k0_roi, e_roi[0]:e_roi[1]], 1)]
                _images += [img[..., k_roi[0]:k_roi[1], e_roi[0]:e_roi[1]]]
            if norm:
                _k0 = np.array([[normalize(k) for k in _img] for _img in _k0])
            else:
                _k0 = np.array([[k for k in _img] for _img in _k0])
            k0 += [_k0]
            images += [_images]
        images = np.array(images)
        k0 = np.array(k0)
    return np.squeeze(k0), (x_positions - REFERENCE_CORNER[0]) / counts_to_mm, y_positions - REFERENCE_CORNER[
        1], np.squeeze(1.24e6 / wvls[:, e_roi[0]:e_roi[1]]), k_ax, np.squeeze(images)


"""IMPLANTED SAMPLES CORNER MATCHING"""
path = get_data_path('2023_03_09/raw_data.h5')
EDGE_NAMES = [u'bottom', u'left', u'right', u'top']
CORNER_NAMES = ['bottom_left', 'bottom_right', 'top_right', 'top_left']
SAMPLE_NAMES = ['sample0', 'sample1', 'sample2', 'sample4', 'sampleA', 'sampleB', 'sampleC', 'sampleD', 'sampleE',
                'sampleF']
CORNER_MATCHING = [('origin', ('sample0', 'bottom', 'bottom_left')),
                   (('sample0', 'bottom', 'bottom_left'), ('sample1', 'top', 'top_left')),
                   (('sample1', 'bottom', 'bottom_right'), ('sample2', 'bottom', 'bottom_left')),
                   (('sample1', 'right', 'top_right'), ('sampleA', 'right', 'bottom_right')),
                   (('sample0', 'bottom', 'bottom_right'), ('sampleB', 'bottom', 'bottom_left')),
                   (('sampleB', 'top', 'top_left'), ('sampleD', 'bottom', 'bottom_left')),
                   (('sampleD', 'top', 'top_right'), ('sampleF', 'bottom', 'bottom_right')),
                   (('sampleF', 'right', 'bottom_right'), ('sampleE', 'left', 'bottom_left')),
                   (('sampleF', 'top', 'top_left'), ('sample4', 'bottom', 'bottom_left')),
                   (('sampleD', 'right', 'bottom_right'), ('sampleC', 'left', 'bottom_left')), ]
EXTRA_ANGLES = dict(sample0=np.pi, sample1=np.pi, sampleA=np.pi/30,
                    sampleC=-np.pi/60, sampleE=-np.pi/100, sample4=-np.pi/200)

IMPLANT_ENERGIES = dict(sample1=250, sampleA=100, sampleB=100, sample2=250, sampleC=250,
                        sampleD=240, sample3=100, sampleE=100, sampleF=100)
IMPLANT_DOSES = dict(sample1=65, sampleA=50, sampleB=25, sample2=40, sampleC=20,
                     sampleD=10, sample3=75, sampleE=35, sampleF=15)
SCAN_NAMES = []
for name in SAMPLE_NAMES:
    with h5py.File(path, 'r') as df:
        keys = df[name].keys()
        for key in keys:
            if 'scan' in key:
                _scan_name = name + '/' + key
                SCAN_NAMES += [_scan_name]


def get_positions(sample_name):
    """Gets the raw (x,y,z) positions of the all the edges of a sample
    :param sample_name: str
    :return: (4, ) list (in the order of EDGE_NAMES) of (N, 3) arrays
    """
    with h5py.File(path, 'r') as df:
        all_positions = []
        for edge in EDGE_NAMES:
            if edge in df['%s/position_calibration/edges' % sample_name].keys():
                group = df['%s/position_calibration/edges/%s' % (sample_name, edge)]
                positions = []
                for key in group.keys():
                    positions += [group[key].attrs['position']]

                if 'corner' in df['%s/position_calibration' % sample_name].keys():
                    group = df['%s/position_calibration/corner' % sample_name]
                    for key in group.keys():
                        if edge in key:
                            positions += [group[key].attrs['position']]
                positions = np.array(positions)
                positions[:, 0] /= counts_to_mm
                idx_sort = np.argsort(positions[:, 0])
                all_positions += [np.array(positions)[idx_sort][:, (0, -1)]]
    return all_positions


def find_linear_fits(sample_edges):
    """Given a list of sample edge positions, returns a dictionary of linear interpolations for each edge
    :param sample_edges: str
    :return: dict
    """
    interpolations = dict()
    for name, edge in zip(EDGE_NAMES, sample_edges):
        edge = np.array(edge, dtype=float)
        interp = np.polyfit(edge[:, 0], edge[:, 1], 1)
        interpolations[name] = interp
    return interpolations


def find_corners(sample_edges):
    """Given a list of sample edge positions, interpolates between them and finds their intersections to determine the
    sample corners
    :param sample_edges: str
    :return: dict of (x,y) points for the sample corners
    """
    interpolations = find_linear_fits(sample_edges)
    corners = dict()
    for name in CORNER_NAMES:
        splt = name.split('_')
        edge1 = interpolations[splt[0]]
        edge2 = interpolations[splt[1]]
        corners[name] = np.array([-edge1[1] + edge2[1], -edge1[1] * edge2[0] + edge2[1] * edge1[0]]) / (
                    edge1[0] - edge2[0])
    return corners


def rotate_points(points, angle):
    return np.array(
        [(np.cos(angle) * x - np.sin(angle) * y, np.sin(angle) * x + np.cos(angle) * y) for (x, y) in points])


def puzzled_together(corner_matching, extra_angles=None):
    """Rotates and offsets the edge positions of all samples to piece them back together

    :param corner_matching: (N, 2, 3) list of edge matching. Each element has two components for the two edges being
    matched. Each should contain the name of the sample, the name of the edge that will be used to determine the angle
    offset, and the name of the corner that will be used to determine the spatial offset.
    :param extra_angles: dictionary where the keys are sample names, and the values are additional angles to rotate the
    samples by
    :return:
    """
    if extra_angles is None: extra_angles = dict()
    # Creating a dictionary of sample positions that we will then rotate and piece together
    sample_names = set([c[1][0] for c in corner_matching])
    all_positions = dict()
    for name in sample_names:
        all_positions[name] = get_positions(name)

    offsets = dict()
    extra_offsets = dict()
    angles = dict()
    for pair in corner_matching:
        if pair[0] == 'origin':
            # Simply setting the first sample to be horizontal
            (name, edge_name, corner_name) = pair[1]
            interpolations = find_linear_fits(all_positions[name])
            angle = np.arctan(interpolations[edge_name][0])
            rotated = np.array([rotate_points(t, -angle) for t in all_positions[name]], dtype=object)
            if name in extra_angles:
                corners = find_corners(rotated)
                sample_center = np.mean(list(corners.values()), 0)
                extra_offsets[name] = sample_center
                _centered = [(r - sample_center) for r in rotated]
                _rotated = np.array([rotate_points(c, -extra_angles[name]) for c in _centered], dtype=object)
                rotated = [(r + sample_center) for r in _rotated]
            all_positions[name] = rotated

            offsets[name] = np.array([0, 0])
            angles[name] = angle
        else:
            (name1, edge_name1, corner_name1), (name2, edge_name2, corner_name2) = pair

            # Getting the angle of the relevant edge and the position of the chosen corner for the reference piece
            pos1 = all_positions[name1]
            interpolations = find_linear_fits(pos1)
            angle1 = np.arctan(interpolations[edge_name1][0])
            corner1 = find_corners(pos1)[corner_name1]

            # Getting the angle of the relevant edge for the piece to be moved
            pos2 = all_positions[pair[1][0]]
            interpolations = find_linear_fits(pos2)
            angle2 = np.arctan(interpolations[edge_name2][0])
            if np.max(np.abs([angle1, angle2])) > np.pi / 4:  # HACK
                if np.sign(angle1) != np.sign(angle2):
                    angle2 += np.pi
            # Rotating the piece
            rotated2 = np.array([rotate_points(t, -(angle2 - angle1)) for t in pos2], dtype=object)
            if name2 in extra_angles:
                corners = find_corners(rotated2)
                sample_center = np.mean(list(corners.values()), 0)
                extra_offsets[name2] = sample_center
                _centered = [(r - sample_center) for r in rotated2]
                _rotated = np.array([rotate_points(c, -extra_angles[name2]) for c in _centered], dtype=object)
                rotated2 = [(r + sample_center) for r in _rotated]
            corner2 = find_corners(rotated2)[corner_name2]
            offset = corner1 - corner2
            offsetted = np.array([(r + offset) for r in rotated2], dtype=object)
            all_positions[name2] = offsetted

            offsets[name2] = offset
            angles[name2] = angle2 - angle1

    return all_positions, offsets, angles, extra_offsets


"""IMPLANTED SAMPLES ENERGY FINDING"""


def get_scan(scan_name, k_roi=None, k_ax=None):
    """Extracts the raw data for a particular scan
    :param scan_name:
    :param k_roi:
    :param k_ax:
    :return:
    """
    if k_roi is None: k_roi = [150, 330]
    if k_ax is None:
        k0 = 150 + 85
        k_ax = (np.arange(400) - k0) * magnification('tmd_acton', 'k_space')[0] * 1e-6

    with h5py.File(path, 'r') as df:
        keys = df[scan_name].keys()
        images = []
        wavelengths = []
        positions = []
        for idx, key in enumerate(keys):
            dset = df['%s/%s' % (scan_name, key)]
            images += [dset[k_roi[0]:k_roi[1]]]
            wavelengths += [dset.attrs['wavelengths']]
            positions += [dset.attrs['position']]
        positions = np.array(positions, dtype=float)
        positions[:, 0] /= counts_to_mm
    return np.array(images), np.array(wavelengths), np.asarray(k_ax[k_roi[0]:k_roi[1]]), np.array(positions)


def scan_find_k0_peaks(scan_name, energy_roi=None, bkg=70, plotting=False, peak_kwargs=None):
    """Extracts the raw data for a scan, selects the k~0 momenta, and finds the energy peaks at that momentum
    :param scan_name:
    :param energy_roi:
    :param bkg:
    :param plotting:
    :param peak_kwargs:
    :return:
    """
    if energy_roi is None: energy_roi = (500, 1250)
    if peak_kwargs is None: peak_kwargs = dict()
    defaults = dict(height=5e-2, distance=40)
    peak_kwargs = {**defaults, **peak_kwargs}

    images, wavelengths, kax, pos = get_scan(scan_name)
    k0 = 85
    roi = 3
    smoothened = np.array([low_pass(img, 0.2) for img in images])
    k0_spectra = np.sum(smoothened[:, k0 - roi:k0 + roi, energy_roi[0]:energy_roi[1]] - bkg, 1)
    normalised = np.array([normalize(k) for k in k0_spectra])
    peak_indices = []
    for n in normalised:
        peaks = find_peaks(n, **peak_kwargs)[0]

        idx_sort = np.argsort(n[peaks])
        peaks = peaks[idx_sort][::-1]
        maximum_intensity_peaks = peaks[:2]
        peak_indices += [np.sort(maximum_intensity_peaks)]
    peak_indices = np.array(peak_indices)

    eaxis = 1.24e6 / wavelengths[0][energy_roi[0]:energy_roi[1]]
    peak_energies = [eaxis[pk] for pk in peak_indices]

    if plotting:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, fig)

        subplots(np.transpose(images[:, :, energy_roi[0]:energy_roi[1]], (0, 2, 1)),
                 partial(imshow, diverging=False, norm=LogNorm(), cbar=False, yaxis=eaxis),
                 gridspec_loc=gs[1])

        ax = plt.subplot(gs[0])
        waterfall(np.log(normalised + 1e-2), ax, xaxis=eaxis, peak_positions=peak_energies)

    return peak_energies, peak_indices, wavelengths, pos


def get_energies():
    """Finds the k~0 energy peaks for each scan to determine the cavity energy and uses the CORNER_MATCHING to rotate
    and offset the scan positions.
    :return:
    """
    edge_positions, offsets, angles, extra_offsets = puzzled_together(CORNER_MATCHING, EXTRA_ANGLES)

    plotting = False

    peak_kwargs = dict()
    for x in [1, 2, 3, 4, 5, 6]:
        peak_kwargs['sample4/scan%d' % x] = dict(height=2e-1, distance=50)
    peak_kwargs['sampleC/scan1'] = dict(height=1e-2, distance=15)
    for x in [1, 2, 3, 4, 5, 6]:
        peak_kwargs['sampleF/scan%d' % x] = dict(height=1e-2, distance=70)

    exclusion_indices = dict()
    exclusion_indices['sample0/scan3'] = [8]
    exclusion_indices['sample0/scan4'] = [7, 8]
    exclusion_indices['sample1/scan1'] = [3]
    exclusion_indices['sample4/scan2'] = [2, 8]
    exclusion_indices['sample4/scan3'] = [2]
    exclusion_indices['sample4/scan6'] = [2]
    exclusion_indices['sample4/scan7'] = [2, 3, 4, 5, 6, 7]
    exclusion_indices['sample4/scan8'] = [2, 3, 4, 5, 6, 7]
    exclusion_indices['sampleC/scan4'] = [0, 1, 2]
    exclusion_indices['sampleE/scan1'] = [6]
    exclusion_indices['sampleE/scan2'] = [6]
    exclusion_indices['sampleE/scan3'] = [6]

    EXCITON_LINE = 1485
    RABI = 2

    scan_energies = dict()
    scan_positions = dict()
    for sample_name in SAMPLE_NAMES:
        energies = []
        positions = []
        for scan_name in SCAN_NAMES:
            if sample_name in scan_name:
                sample_name = scan_name.split('/')[0]

                if scan_name in peak_kwargs:
                    kw = peak_kwargs[scan_name]
                else:
                    kw = dict(height=1e-2, distance=30)
                energ, _, _, pos = scan_find_k0_peaks(scan_name, peak_kwargs=kw, plotting=plotting)
                if scan_name in exclusion_indices:
                    energ = np.delete(energ, exclusion_indices[scan_name], axis=0)
                    pos = np.delete(pos, exclusion_indices[scan_name], axis=0)

                # TODO: use the cavity_energy function to make this more accurate
                cav_energy = []
                for e in energ:
                    if len(e) > 1:
                        mask = np.abs(e - EXCITON_LINE) > RABI
                        if np.all(mask):
                            print(scan_name, mask, e, energ)
                        elif np.any(mask):
                            cav_energy += [e[mask][0]]
                        else:
                            cav_energy += [np.mean(e)]
                    else:
                        cav_energy += [e[0]]

                rotated = rotate_points(pos[:, (0, -1)], -angles[sample_name])
                if sample_name in EXTRA_ANGLES:
                    sample_center = extra_offsets[sample_name]
                    _centered = [(r - sample_center) for r in rotated]
                    _rotated = rotate_points(_centered, -EXTRA_ANGLES[sample_name])
                    rotated = [(r + sample_center) for r in _rotated]
                pos = np.array([(r + offsets[sample_name]) for r in rotated])

                energies += [cav_energy]
                positions += [pos]
        scan_energies[sample_name] = energies
        scan_positions[sample_name] = positions

    return scan_energies, scan_positions, edge_positions


def implanted_polygonal_image(sample_name, parameters=None, ax=None, text_kwargs=None, **imshow_kwargs):
    """Wrapper for the microcavities.utils.plotting.polygonal_image

    Extracts the measured cavity energy for all the scans of a particular sample, and uses them to create a
    polygonal_image showing the cavity energy as a function of space

    :param sample_name:
    :param parameters:
    :param ax:
    :param text_kwargs:
    :param imshow_kwargs:
    :return:
    """
    if parameters is None:
        scan_energies, scan_positions, edge_positions = get_energies()
    else:
        scan_energies, scan_positions, edge_positions = parameters

    edges = edge_positions[sample_name]
    min_x, min_y = np.min([np.min(e, 0) for e in edges], 0)
    max_x, max_y = np.max([np.max(e, 0) for e in edges], 0)

    e = np.squeeze(np.concatenate(scan_energies[sample_name]))
    p = np.concatenate(scan_positions[sample_name])

    if 'vmin' not in imshow_kwargs: imshow_kwargs['vmin'] = 1472.

    try:
        rtrn = polygonal_image(np.round(p, 2)[:, ::-1], e, ax=ax, margins=(0, 0),
                               position_limits=(min_y, max_y, min_x, max_x), plot_kwargs=dict(color=(0.5, 0.5, 0.5)),
                               diverging=False, **imshow_kwargs)
    except Exception as exc:
        rtrn = None
        print('Failed at %s because %s' % (sample_name, exc))

    defaults = dict(color='white')
    if text_kwargs is None: text_kwargs = dict()
    text_kwargs = {**defaults, **text_kwargs}

    if sample_name in IMPLANT_DOSES:
        ax.text(np.mean([min_x, max_x]), np.mean([min_y, max_y]),
                '%s %g %g' % (sample_name, IMPLANT_DOSES[sample_name], IMPLANT_ENERGIES[sample_name]), ha='center',
                va='top', **text_kwargs)
    else:
        ax.text(np.mean([min_x, max_x]), np.mean([min_y, max_y]),
                '%s' % sample_name, ha='center', va='top', **text_kwargs)

    return rtrn


"""SRIM simulations"""


def import_SRIM_3D_datafile(file_name):
    """ Imports an SRIM 3D dataset
    The dimensions are depth, position, and number of vacancies/ions/damage events/etc.

    :param file_name: str
    :return:
    """
    with open(file_name, 'r') as dfile:
        header_break = '----------- -------------------------------------------------------------------\n'
        header = []
        while True:
            # Get next line from file
            line = dfile.readline()

            if not line:  # if we reach the end of the file without finding the header end
                raise ValueError('Header end not found')
            elif line == header_break:  # if we reach the end of the header
                header += [line]
                break
            else:
                header += [line]

    array = np.loadtxt(file_name, skiprows=len(header))  # load all the data after the header
    depths = array[:, 0]  # first column is depths
    x_axis = depths - depths[-2] // 2
    data = array[:, 1:]
    return data, depths, x_axis, header


def analyse_SRIM_3D_data(filename=None, *args, straggle_averaging_depth=200):
    if filename is None:
        img, depth, _x, h = args
    else:
        img, depth, _x, h = import_SRIM_3D_datafile(filename)

    total_vacancies = np.sum(img, 1)

    maximum_idx = np.argmax(np.sum(img, 1))
    maximum_depth = depth[maximum_idx]

    min_idx = np.argmin(np.abs(depth - (maximum_depth - straggle_averaging_depth)))
    max_idx = np.argmin(np.abs(depth - (maximum_depth + straggle_averaging_depth)))
    straggle = np.sum(img[min_idx:max_idx], 0)
    peak_params = guess_peak(straggle, _x, background_percentile=0)

    return total_vacancies, maximum_depth, straggle, peak_params


def plot_SRIM_3D_data(filename=None, *args):
    """ Default plotting for characterising 3D SRIM data

    Plots the 2D (x, depth) distribution of vacancies/ions/damage/etc., plots the total dose vs depth, and the straggle
    at the depth of maximum total dose

    :param filename:
    :param args:
    :return:
    """
    if filename is None:
        img, depth, _x, h = args
        total_vacancies, maximum_depth, straggle, peak_params = analyse_SRIM_3D_data(None, img, depth, h)
    else:
        img, depth, _x, h = import_SRIM_3D_datafile(filename)
        total_vacancies, maximum_depth, straggle, peak_params = analyse_SRIM_3D_data(filename)
    fig = figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.3)
    axs = gs.subplots()
    imshow(img, axs[0], yaxis=depth, xaxis=_x, diverging=False)
    axs[1].plot(depth, total_vacancies, color='xkcd:navy')
    axs[1].axvline(maximum_depth, color='xkcd:brick red')
    axs[1].text(maximum_depth, np.max(total_vacancies), 'd$_{\mathrm{max}}$ ~ %dnm' % (maximum_depth / 10),
                transform=axs[1].transData)
    axs[2].plot(_x, straggle, color='xkcd:brick red')
    axs[2].hlines(np.max(straggle) / 2, -peak_params['sigma'], peak_params['sigma'], color='xkcd:leaf green')
    axs[2].text(peak_params['sigma'], np.max(straggle) / 2, 'Straggle ~ %dnm' % (peak_params['sigma'] / 10),
                transform=axs[2].transData)

    label_axes(axs[0], 'x [\u212B]', 'y [\u212B]', 'Spatial distribution')
    label_axes(axs[1], 'x [\u212B]', 'Vacancies [?]', 'Depth distribution')
    label_axes(axs[2], 'x [\u212B]', 'Vacancies [?]', 'Straggle at depth maxima')

    fig.suptitle(h[0].rstrip('\n').strip('=').strip(' '))

    return fig, axs


def implant_mask(vacancy_file, width_in_angs=100, n_points=(201, 151)):
    """Returns the array that would result from implanting an area of a given width with the SRIM parameters in the
    given file

    :param vacancy_file: str. Path
    :param width_in_angs: float
    :param n_points: 2-tuple
    :return:
    """
    img, depth, x_axis, h = import_SRIM_3D_datafile(vacancy_file)
    interpolated_vacancies = interpolated_array(img, [depth, x_axis], fill_value=0)

    # Making a new depth array
    new_depth = np.linspace(np.min(depth), np.max(depth), n_points[0])

    # Making a new x array
    min_x = np.min([np.min(x_axis), -width_in_angs])
    max_x = np.max([np.max(x_axis), width_in_angs])
    new_xaxis = np.linspace(min_x, max_x, n_points[1])

    # Making a boolean mask
    mask = np.abs(new_xaxis) < width_in_angs / 2

    # Masked implants
    final_implants = np.zeros((len(new_depth), len(new_xaxis)))
    for _x, _bool in zip(new_xaxis, mask):
        if _bool:
            final_implants += np.array([interpolated_vacancies(_d, new_xaxis - _x) for _d in new_depth])
    return final_implants, new_depth, new_xaxis