# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import convolve
from skimage.restoration import unwrap_phase
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
