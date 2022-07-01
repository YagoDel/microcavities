# -*- coding: utf-8 -*-

from microcavities.analysis.condensation import *
from nplab.utils.log import create_logger
from functools import partial
from tqdm import tqdm


LOGGER = create_logger('Bloch simulations')
LOGGER.setLevel('WARN')

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


# Time step solver
def rk_timestep(psi, hamiltonian, t, dt, noise_level=0.2):
    """Runge-Kutta 4th order time step
    :param psi: vector
    :param hamiltonian: function that returns a matrix
    :param t: float
    :param dt: float
    :param noise_level: float
    :return:
    """
    K11 = -1j * np.matmul(hamiltonian(t), psi) / hbar
    K21 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + K11 * dt / 2) / hbar
    K31 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + K21 * dt / 2) / hbar
    K41 = -1j * np.matmul(hamiltonian(t + dt), psi + dt * K31) / hbar

    return psi + (K11 + 2 * K21 + 2 * K31 + K41) * dt / 6 + noise_level*np.random.rand(len(psi))


def solve_timerange(starting_wavefunction, hamiltonian, time_range):
    """
    :param starting_wavefunction: vector
    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :return: 2d array
    """
    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in enumerate(time_range):
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
    return full_psi


def farfield(hamiltonian, starting_vectors, time_range):
    """Calculates the farfield emission pattern of a Hamiltonian when the eigenvectors are randomly occupied

    :param hamiltonian: function. Takes one argument (time), returns an array
    :param starting_vectors:
    :param time_range:
    :return:
    """
    N = len(starting_vectors) // 2
    rho = np.zeros((N, len(time_range)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(vec, hamiltonian, time_range)
        psi_kw = np.fft.fftshift(np.fft.fft2(psi[:N, :]))
        rho += np.abs(psi_kw) ** 2
        if np.isnan(rho).any():
            break
    return rho


# Diagonalisation solver
def solve_for_krange(momentum_range, hamiltonian):
    """Diagonalises a Hamiltonian at different points in momentum space

    :param momentum_range:
    :param hamiltonian: function. Takes one argument (momentum) and returns a matrix
    :return:
    """
    bands = []
    modes = []
    for k in momentum_range:
        H = hamiltonian(k)
        E, eig_vectors = np.linalg.eig(H)
        # linalg.eig does not necessarily return sorted values, so we sort it after:
        idx_sort = np.argsort(E.real)
        bands += [E[idx_sort]]
        modes += [eig_vectors[:, idx_sort]]
    return np.array(bands), np.array(modes)


# Example Hamiltonians
def hamiltonian_k(k, potential_depth=1, period=10., mass=electron_mass*1e-5, detuning=3, rabi=6, n_bands=6):
    """Momentum space Hamiltonian for a sinusoidal potential

    :param k:
    :param potential_depth:
    :param period:
    :param mass:
    :param detuning:
    :param rabi:
    :param n_bands:
    :return:
    """
    G = 2*np.pi / period  # reciprocal vector
    space_size = 2 * n_bands + 1

    # Kinetic energy
    Hk0 = np.diag([hbar ** 2 * (k - x * G) ** 2 / (2 * mass) for x in range(-n_bands, n_bands + 1)])
    Hk0 -= np.eye(space_size) * detuning / 2

    # Potential energy
    pot = [potential_depth / 2] * (space_size - 1)
    Hv = np.diag(pot, -1) + np.diag(pot, 1)
    Hv += np.eye(space_size) * detuning / 2

    # Coupling to exciton
    H1row = np.hstack([Hk0, rabi * np.eye(space_size) / 2])
    H2row = np.hstack([rabi * np.eye(space_size) / 2, Hv])
    return np.vstack([H1row, H2row])


def hamiltonian_x(t, potential, delta_k, frequency, periods=6, n_points=32, mass=electron_mass*1e-5, detuning=3, rabi=6):
    single_period = 2 * np.pi / np.abs(delta_k)

    if periods is None:
        x = np.linspace(-21, 20, n_points)
    else:
        x = np.linspace(-single_period * periods / 2 - 0.1 * single_period, single_period * periods / 2, n_points)
        # print(x.min(), x.max())
    D2 = np.diag(-2 * np.ones(n_points)) + np.diag(np.ones(n_points - 1), 1) + np.diag(np.ones(n_points - 1), -1)
    dx = np.diff(x)[0]  # 1.25
    D2 /= dx ** 2
    Hk0 = -D2 * hbar ** 2 / (2 * mass)
    Hk0 -= np.eye(n_points) * detuning / 2
    Hv = (potential * np.cos(delta_k * x - 2 * np.pi * frequency * t) + detuning / 2) * np.eye(n_points)
    H1row = np.hstack([Hk0, rabi * np.eye(n_points)])
    H2row = np.hstack([rabi * np.eye(n_points), Hv])
    return np.vstack([H1row, H2row])


def test_hamiltonians():
    potential = 0
    period = 2*np.pi / 0.43
    vals = []

    for n_periods in [3, 5, 7]:
        hx = hamiltonian_x(0, potential, period, 0, n_periods, n_points=101)
        val1, vec1 = np.linalg.eig(hx)
        _sort_idx = np.argsort(val1)
        val1 = val1[_sort_idx]
        vec1 = vec1[:, _sort_idx]
        vals += [val1]

    hk = partial(hamiltonian_k, potential_depth=potential, period=period, n_bands=6)
    bands, modes = solve_for_krange(np.linspace(-2, 2, 101), hk)

    fig, axs = plt.subplots(1, 2)
    [axs[0].plot(val[:100], '--') for val in vals]
    axs[1].plot(bands[:, :])
