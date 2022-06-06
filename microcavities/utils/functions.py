# -*- coding: utf-8 -*-
import numpy as np
from microcavities.utils import depth
from math import factorial
from scipy.special import hermite, genlaguerre


def gaussian(x, x0, a, gam):
    return a * np.exp(-(x-x0)**2 / (2*gam**2))


def gaussianNd(x, x0, a, gam):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x])
    X = np.meshgrid(*x)
    try:
        iter(gam)
    except:
        gam = np.array([gam] * len(X), dtype=np.float)
    gam /= 2*np.sqrt(2*np.log(2))
    try:
        iter(x0)
    except:
        x0 = np.array([x0] * len(X), dtype=np.float)
    exponent = np.zeros(X[0].shape)
    for _x, _x0, _gam in zip(X, x0, gam):
        exponent -= (_x - _x0)**2 / (2*_gam**2)
    return a * np.exp(exponent)


def lorentzian(x, x0, a, gam):
    return a * gam ** 2 / (gam ** 2 + (x - x0) ** 2)


def lorentzianNd(x, x0, a, gam):
    try:
        len(x[0])
    except:
        x = [x]
    X = np.meshgrid(*x)
    try:
        iter(gam)
    except:
        gam = np.array([gam] * len(X), dtype=np.float)
    try:
        iter(x0)
    except:
        x0 = np.array([x0] * len(X), dtype=np.float)
    quotient = np.zeros(X[0].shape)
    for _x, _x0, _gam in zip(X, x0, gam):
        quotient += ((_x - _x0) / (_gam / 2)) ** 2
    return a / (quotient + 1)


def lg_mode(radial_number, azimuthal_number, axes=None, center_offset=(0, 0)):
    """2D Laguerre Gauss modes

    >>> lg = lg_mode(0, 1)  # vortex beam of charge 1
    >>> imshow(lg)

    :param radial_number:       int
    :param azimuthal_number:    int
    :param axes:                2-tuple of 1D arrays
    :param center_offset:       2-tuple of floats
    :return: 2D array. Intensity distribution of the LG mode
    """
    if axes is None:
        axes = [np.linspace(-10, 10, 101) for _ in range(2)]

    axes = [ax - c for ax, c in zip(axes, center_offset)]

    grid = np.array(np.meshgrid(*axes))
    r = np.sqrt(np.sum(grid**2, 0))
    phi = np.arctan2(*grid[::-1])
    norm_coeff = np.sqrt((2*factorial(radial_number)) / (np.pi * factorial(radial_number + np.abs(azimuthal_number))))
    w_0 = 1

    radial = (((r*np.sqrt(2)) / w_0) ** np.abs(azimuthal_number)) * np.exp(-(r/w_0)**2) * genlaguerre(radial_number, np.abs(azimuthal_number))(2 * (r/w_0)**2)
    azimuthal = np.exp(-1j * azimuthal_number * phi)
    return norm_coeff * radial * azimuthal


def hg_mode(horizontal_number, vertical_number, axes=None, center_offset=(0, 0)):
    """2D Hermite Gauss modes

    >>> hg = hg_mode(1, 0)  # two-lobed mode along x
    >>> imshow(hg)

    :param horizontal_number:   int
    :param vertical_number:     int
    :param axes:                2-tuple of 1D arrays
    :param center_offset:       2-tuple of floats
    :return: 2D array. Intensity distribution of the HG mode
    """
    if axes is None:
        axes = [np.linspace(-10, 10, 101) for idx in range(2)]

    axes = [ax - c for ax, c in zip(axes, center_offset)]

    grid = np.array(np.meshgrid(*axes))
    r = np.sqrt(np.sum(grid**2, 0))
    w_0 = 1
    poly = [hermite(number)(np.sqrt(2) * grid[idx] / w_0) for idx, number in enumerate([horizontal_number, vertical_number])]

    return np.prod(poly, 0) * np.exp(-(r/w_0)**2)

