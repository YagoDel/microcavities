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


"""TIME EVOLUTION"""


def rk_timestep(psi, hamiltonian, t, dt, noise_level=0):
    """Single time step using Runge Kutta 4th order

    :param psi: vector
    :param hamiltonian: function. Should take a single input (time) and return a scipy.sparse matrix
    :param t: float
    :param dt: float
    :param noise_level: float
    :return:
    """
    K11 = -1j * (hamiltonian(t)@psi) / hbar
    K21 = -1j * (hamiltonian(t + dt / 2)@(psi + K11 * dt / 2)) / hbar
    K31 = -1j * (hamiltonian(t + dt / 2)@(psi + K21 * dt / 2)) / hbar
    K41 = -1j * (hamiltonian(t + dt)@(psi + dt * K31)) / hbar

    return psi + (K11 + 2 * K21 + 2 * K31 + K41) * dt / 6 + noise_level * np.random.rand(len(psi))


def solve_timerange(hamiltonian, time_range, starting_wavefunction=None):
    """Time steps an initial wavefunction using the Runge Kutta stepper

    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range: list of floats
    :param starting_wavefunction: vector or None
    :return:
    """
    if starting_wavefunction is None:
        n_points = hamiltonian(0).shape[0]
        starting_wavefunction = np.random.uniform(-1, 1, (n_points,)) + 1.j * np.random.uniform(-1, 1, (n_points,))

    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in tqdm(enumerate(time_range), 'solve_timerange'):
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
    return full_psi


"""DIRECT DIAGONALISATION"""


def diagonalise_hamiltonian(hamiltonian, n_eigenvalues=250):
    if sparse.issparse(hamiltonian):
        size = hamiltonian.shape[0]
        if n_eigenvalues > size: n_eigenvalues = size-1  # sparse matrices cannot be fully factorised
        vals, vecs = eigsh(hamiltonian, k=n_eigenvalues, which='SA')
        return vals, vecs
    else:
        E, eig_vectors = np.linalg.eig(hamiltonian)
        # linalg.eig does not return sorted values, so we sort them:
        idx_sort = np.argsort(E.real)
        return E[idx_sort], np.asarray(eig_vectors[:, idx_sort])


def get_eigenvector(n, vecs, size=None):
    """
    :param n: int. Index selection of eigenvector
    :param vecs: array. List of eigenvectors
    :param size: int. To determine how to reshape the eigenvector from a list to a matrix
    :return:
    """
    if size is None: size = int(np.sqrt(vecs.shape[0]/2))
    return vecs.T[n].reshape((2, size, size))


def plot_eigenvectors(vecs, vals, axes=None):
    """Plots eigenvalues and eigenvectors for a 2D polariton Hamiltonian

    Photonic and excitonic components are plotted separately

    :param vecs:
    :param vals:
    :param axes:
    :return:
    """

    if axes is None:
        x, y = make_axes()[2:]
    else:
        x, y = axes[2:]

    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(1, 2, fig, width_ratios=[1, 4])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, gs[1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(vals)
    axs = gs2.subplots()

    for idx, _axs in enumerate(axs.transpose()):
        imshow(get_eigenvector(idx, vecs)[0], _axs[0], xaxis=x, yaxis=y)
        imshow(get_eigenvector(idx, vecs)[1], _axs[1], xaxis=x, yaxis=y)
    return fig, [ax0, axs]
