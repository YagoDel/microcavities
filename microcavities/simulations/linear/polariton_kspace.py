# -*- coding: utf-8 -*-

# TODO: generalise to arbitrary potentials

from microcavities.simulations.linear import *


def hamiltonian_k_free_space(k, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35):
    """Free space polariton Hamiltonian in the momentum basis

    :param k: float. Magnitude of the momentum vector in 2D
    :param detuning:
    :param rabi:
    :param mass_photon:
    :param mass_exciton:
    :return:
    """
    try:
        if len(k.shape) == 1:
            k = sparse.diags(k, 0)
        elif len(k.shape) == 0:
            raise TypeError
    except (TypeError, AttributeError):
        k = sparse.diags([[k]], [0])
    photon = hbar ** 2 * k ** 2 / (2 * mass_photon * electron_mass)
    exciton = hbar ** 2 * k ** 2 / (2 * mass_exciton * electron_mass)
    _rabi = rabi * sparse.eye(*photon.shape)

    photon.data += detuning/2
    exciton.data -= detuning / 2
    return sparse.bmat([[photon, _rabi / 2], [_rabi / 2, exciton]])


def solve_for_krange(krange, hamiltonian):
    bands = []
    modes = []
    for k in krange:
        energy, eig_vectors = diagonalise_hamiltonian(hamiltonian(k).toarray())
        bands += [energy]
        modes += [eig_vectors]
    return np.array(bands), np.array(modes)


def test_hamiltonian_k():
    """Testing that the Hamiltonian here agrees with the equations we use to fit experimental data"""
    from microcavities.analysis.dispersion import exciton_photon_dispersions

    DETUNING = -5
    RABI = 3

    n_points = 201

    axes = make_axes(100, n_points)  # Using a region large enough to get accurate free space propagation
    k_axes = make_k_axes(axes)

    two_modes, _ = solve_for_krange(k_axes[2], partial(hamiltonian_k_free_space, detuning=DETUNING, rabi=RABI))
    l, u, x, p = exciton_photon_dispersions(k_axes[2], DETUNING/2, RABI, 1e-5, -DETUNING/2, 0.35, for_fit=False)

    k = np.linspace(-3, 3, 41)
    ham = lambda t: hamiltonian_k_free_space(k, DETUNING, RABI)
    timerange = np.linspace(-10, 10, 2001)
    psi = solve_timerange(ham, timerange)
    ff = np.fft.fftshift(np.fft.fft(psi), -1)
    dE = hbar * np.pi / timerange[-1]
    theory_eax = (np.linspace(-dE, dE, len(timerange)) * len(timerange) / 2)[::-1]

    fig, ax = plt.subplots(1, 1)
    ax.plot(k_axes[2], two_modes, label='Numerical diagonalisation')
    ax.plot(k_axes[2], l, color='k', ls='--', label='Exact diagonalisation')
    ax.plot(k_axes[2], u, color='k', ls='--')
    ax.plot(k_axes[2], x, color=(0.5, 0.5, 0.5, 0.5), ls='--', label='Exciton/Photon')
    ax.plot(k_axes[2], p, color=(0.5, 0.5, 0.5, 0.5), ls='--')
    _, _, cbarax = imshow(np.abs(ff[:41].transpose())**2, ax, xaxis=k, yaxis=theory_eax, diverging=False, norm=LogNorm())
    _, _, cbarax = imshow(np.abs(ff[41:].transpose())**2, xaxis=k, yaxis=theory_eax, diverging=False, norm=LogNorm())

    ax.set_ylim(-8, 30)
    cbarax.set_title(r'Runge Kutta FFT($|\psi^2|$)')
    unique_legend(ax)
