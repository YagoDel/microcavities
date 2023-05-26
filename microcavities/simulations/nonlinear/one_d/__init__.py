# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from nplab.utils.log import create_logger
from functools import partial
from tqdm import tqdm
from microcavities.simulations.linear.one_d import make_ax, make_k_ax, solve
from microcavities.simulations.linear.one_d import hbar, electron_mass


def dissipative_edges(loss, x_ax, mode='butter', boundary_params=None):
    if boundary_params is None: boundary_params = dict()
    if mode == 'butter':
        defaults = dict(edge=6, order=4)  # default dissipative edge of 8um
        boundary_params = {**defaults, **boundary_params}

        return loss * np.sqrt(1 + (x_ax / (x_ax.max() - boundary_params['edge'])) ** (2 * boundary_params['order']))
    elif mode == 'exp':
        defaults = dict(edge=2, growth=2)  # default dissipative edge of 2um
        boundary_params = {**defaults, **boundary_params}

        edge_right = x_ax.max() - boundary_params['edge']
        edge_left = x_ax.min() + boundary_params['edge']
        factor_right = 1 + np.exp(boundary_params['growth'] * (x_ax - edge_right))
        factor_left = 1 + np.exp(-boundary_params['growth'] * (x_ax - edge_left))
        return loss * factor_right * factor_left
    else:
        raise ValueError('Mode must be one of: %s' % ['butter', 'exp'])


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


def solve_timerange(hamiltonian, time_range, starting_wavefunction=None, noise_amplitude=0):
    """Time steps an initial wavefunction using the Runge Kutta stepper
    :param starting_wavefunction: vector
    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :return: 2d array
    """
    if starting_wavefunction is None:
        n_points = hamiltonian(0, 0).shape[0]
        starting_wavefunction = 1e-3 * np.random.uniform(-1, 1, (n_points,)) + 1.j * np.random.uniform(-1, 1, (n_points,))
    noise = noise_amplitude * (np.random.random(len(time_range)) + 1j * np.random.random(len(time_range)))

    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in enumerate(time_range):
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
        starting_wavefunction += noise[idx_t] * np.abs(starting_wavefunction)
    return full_psi
