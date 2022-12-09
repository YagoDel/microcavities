# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from scipy.sparse.linalg import eigsh
from scipy import sparse
import logging
from functools import partial
from tqdm import tqdm

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


def make_axes(region=30, n_points=101):
    """
    :param region: float or tuple. Size (in um) of the area. If tuple, you can give (size_x, size_y) independently.
    :param n_points: integer
    :return:
    """
    if type(region) not in [tuple, list]:  # if region is a number, make it a tuple with the same number twice
        region = (region, region)

    _x = np.linspace(-region[0]/2, region[0]/2, n_points, dtype=float)
    _y = np.linspace(-region[1]/2, region[1]/2, n_points, dtype=float)
    x, y = np.meshgrid(_x, _y)
    return x, y, _x, _y


def make_k_axes(x_axes):
    dx = np.diff(x_axes[2])[0]
    _kx = np.linspace(-np.pi / dx, np.pi / dx, len(x_axes[2]))
    dy = np.diff(x_axes[3])[0]
    _ky = np.linspace(-np.pi / dy, np.pi / dy, len(x_axes[3]))
    kx, ky = np.meshgrid(_kx, _ky)

    return kx, ky, _kx, _ky
