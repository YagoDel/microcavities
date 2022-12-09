# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from nplab.utils.log import create_logger
from functools import partial
from tqdm import tqdm

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


def make_ax(region=30, n_points=101):
    """
    :param region: float or tuple. Size (in um) of the area. If tuple, you can give (size_x, size_y) independently.
    :param n_points: integer
    :return:
    """
    if type(region) not in [tuple, list]:  # if region is a number, make it a tuple with the same number twice
        region = (region, region)

    return np.linspace(-region[0]/2, region[0]/2, n_points, dtype=float)


def make_k_ax(x_ax):
    dx = np.diff(x_ax)[0]
    return np.linspace(-np.pi / dx, np.pi / dx, len(x_ax))


def solve(hamiltonian):
    E, eig_vectors = np.linalg.eig(hamiltonian)
    # linalg.eig does not necessarily return sorted values, so we sort it after:
    idx_sort = np.argsort(E.real)
    return E[idx_sort], np.asarray(eig_vectors[:, idx_sort])
