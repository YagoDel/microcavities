# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from scipy.signal import convolve, fftconvolve
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from nplab.instrument.electronics.SLM import zernike_polynomial


def zernike_decomposition(image, order=10, beam_size=1, unit_circle=True):
    """Decompose phase map onto a set of Zernike polynomials

    :param image: NxM array. Phase map to be decomposed
    :param order: int. Maximum order of Zernike polynomial
    :param beam_size: float. Size of the beam to be passed to zernike_polynomial
    :param unit_circle: bool. Whether to limit the overlap integral to the unit circle
    :return: (order * (order + 1) / 2, 3) array. First two columns are the Zernike indices, last column is the overlap coefficient
    """

    coefficients = []
    for idx in range(order):
        for idx2 in np.arange(-idx, idx+1, 2):
            zer = zernike_polynomial(image.shape, idx, idx2, beam_size=beam_size, unit_circle=unit_circle)
            coefficients += [(idx, idx2, np.sum(image*zer))]

    return np.array(coefficients)


def zernike_reconstruction(array_size, coefficients, order=None, excluded_indices=None, beam_size=1, unit_circle=True):
    """Reconstruct phase map

    :param array_size: int or 2-tuple (N, M)
    :param coefficients: (N, 3) array.First two columns are the Zernike indices, last column is the overlap coefficient
    :param order: int or 2-tuple. (min, max) orders to be used
    :param excluded_indices: list of 2-tuples. Indices of Zernike polynomials to be excluded from reconstructed
    :param beam_size: float. Size of the beam to be passed to zernike_polynomial
    :param unit_circle: bool. Whether to limit the reconstruction to the unit circle
    :return: NxM array
    """
    if order is None:
        order = (0, np.max(coefficients[:, 0]))
    elif type(order) == int:
        order = (0, order)
    if excluded_indices is None:
        excluded_indices = []

    reconstructed = np.zeros(array_size)
    for idx, idx2, coef in coefficients:
        if (idx, idx2) not in excluded_indices:
            if order[1] >= idx >= order[0]:
                zer = zernike_polynomial(array_size, int(idx), int(idx2), beam_size=beam_size, unit_circle=unit_circle)
                reconstructed += zer * coef

    return reconstructed


def find_vortex(phase_map):
    """Finds vortex positions (to pixel accuracy)

    Doing line integrals around 2x2 plaquettes to find the vorticity

    :param phase_map:
    :return:
    """
    # It is necessary to ensure the phase differences are in the [-pi, pi) range
    difpx = (np.roll(phase_map, -1, 0) - phase_map + np.pi) % (2 * np.pi) - np.pi
    difmx = (np.roll(phase_map, 1, 0) - phase_map + np.pi) % (2 * np.pi) - np.pi
    difpy = (np.roll(phase_map, -1, 1) - phase_map + np.pi) % (2 * np.pi) - np.pi
    difmy = (np.roll(phase_map, 1, 1) - phase_map + np.pi) % (2 * np.pi) - np.pi

    # Adding over plaquettes
    grad = convolve(difmy, [[1, 0], [0, 0]]) + convolve(difpy, [[0, 0], [0, 1]]) + \
           convolve(difmx, [[0, 1], [0, 0]]) + convolve(difpx, [[0, 0], [1, 0]])

    vortex_pos = np.argwhere(np.round(grad / (2*np.pi)) > 0)
    antivortex_pos = np.argwhere(np.round(grad / (2*np.pi)) < 0)
    return vortex_pos, antivortex_pos


def remove_vortex(phase_map, approximate_vortex_position=None):
    """Removes a vortex/antivortex from a phase map

    Finds the vortex/antivortex nearest to the approximate position given
    Substracts a vortex/antivortex centered at that measured position
    Unwraps the phase

    :param phase_map: NxM array
    :param approximate_vortex_position: if None, it assumes vortex is at the center
    :return:
    """
    _x = np.arange(phase_map.shape[1])
    _y = np.arange(phase_map.shape[0])
    _X, _Y = np.meshgrid(_x, _y)

    if approximate_vortex_position is None:
        # By default, removes central vortex
        approximate_vortex_position = np.array(phase_map.shape)/2
    vortex_positions, anti_vortex_positions = find_vortex(phase_map)
    vx_distance_to_center = np.sum(np.abs(vortex_positions - approximate_vortex_position)**2, -1)
    avx_distance_to_center = np.sum(np.abs(anti_vortex_positions - approximate_vortex_position)**2, -1)
    if vx_distance_to_center.min() > avx_distance_to_center.min():
        vortex = anti_vortex_positions[np.argmin(avx_distance_to_center)]
        _theta = -np.arctan2(_Y - _y[vortex[0] - 1], _X - _x[vortex[1] - 1])
    elif vx_distance_to_center.min() < avx_distance_to_center.min():
        vortex = vortex_positions[np.argmin(vx_distance_to_center)]
        _theta = np.arctan2(_Y - _y[vortex[0] - 1], _X - _x[vortex[1] - 1])
    else:
        raise ValueError('There is a vortex and an antivortex equidistant from the approximate position')

    return unwrap_phase(phase_map - _theta), _theta


def low_pass(image, cutoff=0.02, filter_order=4):
    """Low pass radial Butterworth filter

    https://en.wikipedia.org/wiki/Butterworth_filter

    :param image:
    :param cutoff:
    :param filter_order:
    :return:
    """
    _x = np.linspace(-1, 1, image.shape[0])
    _y = np.linspace(-1, 1, image.shape[1])
    _x, _y = np.meshgrid(_y, _x)
    _r = np.sqrt(_x ** 2 + _y ** 2)
    filter_kernel = 1 / np.sqrt(1 + (_r / cutoff) ** (2 * filter_order))

    frequency_space = np.fft.fftshift(np.fft.fft2(image))
    filtered = filter_kernel * frequency_space
    real_space = np.fft.ifft2(filtered)
    passed = np.abs(real_space) * np.sign(image)

    return passed


def remove_fringes(image, center_offset, sigma):
    """

    TODO: rather than completely remove the frequency peak, erode it onto the background
    :param image:
    :param center_offset:
    :param sigma:
    :return:
    """
    center_offset = np.array(center_offset)
    sigma = np.array(sigma)
    _xy = [np.arange(image.shape[idx]) for idx in range(2)[::-1]]
    _x, _y = np.meshgrid(*_xy)
    image_center = np.array([x/2 for x in image.shape[::-1]])
    fltr = np.ones(image.shape)
    fltr[(((_x-image_center[0])/sigma[0])**2 + ((_y-image_center[1])/sigma[1])**2) > 1] = 0
    # for sign in [1]:
    #     center = image_center + sign * center_offset
    #     fltr -= np.exp(-((_x - center[0])**2)/(2*sigma[0]**2)-((_y - center[1])**2)/(2*sigma[1]**2))

    frequency_space = np.fft.fftshift(np.fft.fft2(image))
    rolled = np.roll(frequency_space, center_offset, (1, 0))
    imshow(np.abs(frequency_space))
    colorful_imshow([np.abs(rolled), fltr])

    filtered = fltr * frequency_space  # np.roll(fltr * frequency_space, image_center + center_offset, (0, 1))
    real_space = np.fft.ifft2(filtered)
    colorful_imshow([np.abs(frequency_space), fltr])
    return real_space #np.abs(real_space) * np.sign(image)


def analyse_fringes(image, offset=None, mask_radius=None, plot=False, peak_kwargs=None):
    """Extracts phase and visibility from an interference pattern

    :param image: 2D array
    :param offset: 2-tuple. Pixels to shift the Fourier plane. Corresponds to spatial frequency of fringes in image
    :param mask_radius: float
    :param plot: bool
    :param peak_kwargs: dict
    :return:
    """
    center = np.asarray(image.shape, int) // 2

    # Fourier plane
    fourier = np.fft.fft2(image)
    shifted_fourier = np.fft.fftshift(fourier)

    # Centering the frequency peak
    if offset is None:
        if peak_kwargs is None: peak_kwargs = dict()
        default_peak_kwargs = dict(min_distance=5, threshold_abs=0.01)
        peak_kwargs = {**default_peak_kwargs, **peak_kwargs}
        peaks = peak_local_max(normalize(np.abs(shifted_fourier)), **peak_kwargs)
        peak_intensities = [np.abs(shifted_fourier)[peak[0], peak[1]] for peak in peaks]
        # Sorting peaks in decreasing order of intensity
        idxs = np.argsort(peak_intensities)[::-1]
        sorted_peaks = peaks[idxs]

        # If the first peak is near the center, the second and third peaks are likely a frequency peaks. Otherwise, the
        # first and second peak are likely the frequency peaks. This will fail if data is very noisy.
        if np.sum(np.abs(sorted_peaks[0] - np.array(shifted_fourier.shape)/2)) < 2:
            frequency_peaks = sorted_peaks[1:3]
        else:
            frequency_peaks = sorted_peaks[:2]

        # Sorting frequency peaks in decreasing order of x (y) if the fringes are horizontal (vertical)
        _idx = np.argmax(np.abs(frequency_peaks[0]))  # index of dimension with larger offset
        idxs = np.argsort(frequency_peaks[:, _idx])  # sorting along that dimension
        frequency_peaks = frequency_peaks[idxs]

        # The offset required to center the frequency peak
        offset = (frequency_peaks[1] - frequency_peaks[0]) // 2
    rolled = np.roll(shifted_fourier, offset, (0, 1))

    # Defining a centered circular hard mask
    axes = [np.arange(x) for x in image.shape]
    grid = np.array(np.meshgrid(*axes, indexing='ij'))
    _grid = np.array([g - o for g, o in zip(grid, center)])
    r = np.sqrt(np.sum(_grid**2, 0))
    if mask_radius is None:
        mask_radius = np.sqrt(np.sum(offset**2))/2
    mask = np.asarray(r**2 < mask_radius**2, int)

    # Separating the interference term from the envelope term
    mask1 = np.fft.fftshift(mask) * fourier  # selecting the mask area around the 0th frequency
    mask2 = np.fft.fftshift(mask * rolled)  # selecting the mask area around the offset frequency
    zeroth_modulation = np.fft.ifft2(mask1)
    offset_modulation = np.fft.ifft2(mask2)

    # Visibility, phase and cw
    # factor of 2 from mask taking out half the frequency plane. 1e-6 ensures we don't divide by zero
    visibility = 2 * np.abs(offset_modulation) / (np.abs(zeroth_modulation) + 1e-6)
    angle = np.angle(offset_modulation)
    cw = np.abs(zeroth_modulation)

    if plot:
        # Plot results
        fig, axs = plt.subplots(2, 3)
        imshow(image, axs[0, 0], diverging=False, cbar=False, cmap='Greys')
        imshow_transparency(visibility, axs[0, 1], cw, cmap="Greys")
        imshow_transparency(angle, axs[0, 2], cw, cmap='coolwarm')
        colorful_imshow([np.log(np.abs(fourier) + 1e2), mask], axs[1, 0], from_black=False)
        axs[1, 0].text(0.5, 1, offset, ha='center', va='top')

        idx = image.shape[0]//2
        axs[1, 1].plot(image[idx])
        axs[1, 1].plot(cw[idx])
        ip = cw + np.abs(offset_modulation)
        im = cw - np.abs(offset_modulation)
        vs = (ip - im) / (ip + im)
        axs[1, 1].plot(ip[idx])
        axs[1, 1].plot(im[idx])
        axs[1, 2].plot(vs[idx])
        axs[1, 2].set_ylim(0, 1)
        return visibility, angle, cw, fig, axs
    return visibility, angle, cw


def test_analyse_fringes(interference_type='flat', options=None):
    """

    :param interference_type: str. One of 'flat' 'retroreflected'
    :param options: list. Can contain 'noise', 'incoherent'
    :return:
    """
    if options is None: options = []
    from microcavities.utils.functools import lg_mode

    axes = [np.linspace(-5, 5, 1001), np.linspace(-5, 5, 501)]
    grid = np.array(np.meshgrid(*axes))
    r = np.sqrt(np.sum(grid ** 2, 0))
    angle_offset = 40

    field1 = lg_mode(0, 1, axes, (0.5, 0))
    field1 /= np.max(np.abs(field1))
    incoherent = 0
    if 'noise' in options:
        field1 += 0.1 * np.random.random(r.shape) * np.exp(1j * 0.1 * 2 * np.pi * np.random.random(r.shape))
    if 'incoherent' in options:
        incoherent = 0.1 * np.abs(field1) ** 2

    if interference_type == 'flat':
        field2 = np.exp(1j * angle_offset * grid[0])
    elif interference_type == 'retroreflected':
        field2 = np.fliplr(field1) * np.exp(1j * angle_offset * grid[0])
    else:
        raise ValueError

    interference = field1 + field2
    measurement = incoherent + np.abs(interference) ** 2

    v, a, c, _, _ = analyse_fringes(measurement, plot=True);
