# -*- coding: utf-8 -*-

# TODO: generalise to arbitrary potentials

from microcavities.simulations.linear import *


def Hamiltonian_k(k, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35):
    """Free space polariton Hamiltonian"""
    photon = hbar ** 2 * k ** 2 / (2 * mass_photon * electron_mass)
    exciton = hbar ** 2 * k ** 2 / (2 * mass_exciton * electron_mass)
    return np.array([[photon + detuning / 2, rabi / 2], [rabi / 2, exciton - detuning / 2]])


def solve_for_krange(krange, hamiltonian):
    bands = []
    modes = []
    for k in krange:
        H = hamiltonian(k)
        E, eig_vectors = np.linalg.eig(H)
        idx_sort = np.argsort(E.real)
        bands += [E[idx_sort]]
        modes += [eig_vectors[:, idx_sort]]
    return np.array(bands), np.array(modes)


def test_hamiltonian_k():
    """Testing that the Hamiltonian here agrees with the equations we use to fit experimental data"""
    from microcavities.analysis.dispersion import exciton_photon_dispersions

    DETUNING = -5
    RABI = 3

    n_points = 201

    axes = make_axes(100, n_points)  # Using a region large enough to get accurate free space propagation
    k_axes = make_k_axes(axes)

    two_modes, _ = solve_for_krange(k_axes[2], partial(Hamiltonian_k, detuning=DETUNING, rabi=RABI))
    l, u, x, p = exciton_photon_dispersions(k_axes[2], DETUNING/2, RABI, 1e-5, -DETUNING/2, 0.35, for_fit=False)

    fig, ax = plt.subplots(1, 1)
    ax.plot(k_axes[2], two_modes)
    ax.plot(k_axes[2], l, color='k', ls='--')
    ax.plot(k_axes[2], u, color='k', ls='--')
    ax.plot(k_axes[2], x, color=(0.5, 0.5, 0.5, 0.5), ls='--')
    ax.plot(k_axes[2], p, color=(0.5, 0.5, 0.5, 0.5), ls='--')