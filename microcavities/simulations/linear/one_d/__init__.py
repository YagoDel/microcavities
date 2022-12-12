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


def rk_timestep(psi, hamiltonian, t, dt):
    """Runge-Kutta 4th order time step
    :param psi: vector
    :param hamiltonian: function that returns a matrix
    :param t: float
    :param dt: float
    :return:
    """
    k11 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t), psi))) / hbar
    k21 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2), psi + k11 * dt / 2))) / hbar
    k31 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2), psi + k21 * dt / 2))) / hbar
    k41 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt), psi + dt * k31))) / hbar
    return psi + (k11 + 2 * k21 + 2 * k31 + k41) * dt / 6


def solve_timerange(hamiltonian, time_range, starting_wavefunction=None):
    """Time steps an initial wavefunction using the Runge Kutta stepper
    :param starting_wavefunction: vector
    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :return: 2d array
    """
    if starting_wavefunction is None:
        n_points = hamiltonian(0).shape[0]
        starting_wavefunction = np.random.uniform(-1, 1, (n_points,)) + 1.j * np.random.uniform(-1, 1, (n_points,))

    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in enumerate(time_range):
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
    return full_psi
