# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from nplab.utils.log import create_logger
from functools import partial
from tqdm import tqdm
from microcavities.simulations.linear.one_d import make_ax, make_k_ax, solve
from microcavities.simulations.linear.one_d import hbar, electron_mass


def rk_timestep(psi, hamiltonian, t, dt):
    """Runge-Kutta 4th order time step
    :param psi: vector
    :param hamiltonian: function that returns a matrix
    :param t: float
    :param dt: float
    :return:
    """
    k11 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t, psi), psi))) / hbar
    k21 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2, psi + k11 * dt / 2),
                                                psi + k11 * dt / 2))) / hbar
    k31 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2, psi + k21 * dt / 2),
                                                psi + k21 * dt / 2))) / hbar
    k41 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt, psi + dt * k31), psi + dt * k31))) / hbar
    return psi + (k11 + 2 * k21 + 2 * k31 + k41) * dt / 6


def solve_timerange(hamiltonian, time_range, starting_wavefunction=None):
    """Time steps an initial wavefunction using the Runge Kutta stepper
    :param starting_wavefunction: vector
    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :return: 2d array
    """
    if starting_wavefunction is None:
        n_points = hamiltonian(0, 0).shape[0]
        starting_wavefunction = 1e-3 * np.random.uniform(-1, 1, (n_points,)) + 1.j * np.random.uniform(-1, 1, (n_points,))

    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in enumerate(time_range):
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
    return full_psi
