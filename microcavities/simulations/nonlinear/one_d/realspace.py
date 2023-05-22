# -*- coding: utf-8 -*-

"""1D diagonalising and numerically solving of exciton-photon coupling Hamiltonians with arbitrary spatial potentials,
gain and nonlinearity

See test_ functions for example usage
"""

from microcavities.simulations.nonlinear.one_d import *
from microcavities.simulations.linear.one_d.realspace import kinetic_matrix, potential_matrix


def farfield(hamiltonian, time_range, starting_vectors=None):
    """Calculates the farfield emission pattern of a one dimensional Hamiltonian

    Given a list of starting wavefunctions, evolves them in time using a Runge Kutta stepper and extracts the farfield
    emission of each wavefunction by Fourier transforming them in space and time

    :param hamiltonian: function. Takes one argument (time), returns a 2D matrix
    :param time_range: list of times to evaluate the Hamiltonian at
    :param starting_vectors: list of vectors, integer or None. If None, If an integer,
    :return:
    """
    n_points = hamiltonian(0, 0).shape[0]
    size = int(n_points / 2)
    if starting_vectors is None:
        starting_vectors = [None]
    elif type(starting_vectors) == int:
        starting_vectors = [None] * starting_vectors

    rho = np.zeros((size, len(time_range)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(hamiltonian, time_range, vec)
        psi_reshaped = np.reshape(psi, (2, size, len(time_range)))
        photon_field = psi_reshaped[0]

        psi_kw = np.fft.fftshift(np.fft.fft2(photon_field))
        rho += np.abs(psi_kw) ** 2
        if np.isnan(rho).any():
            break

    dE = hbar * np.pi / time_range[-1]
    theory_eax = (np.linspace(-dE, dE, len(time_range)) * len(time_range) / 2)[::-1]

    return rho, theory_eax


def test_uniform_condensation():
    def hamiltonian_gain_loss(t, psi, gain, loss, saturation, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35,
                              xax=None):
        if xax is None:
            xax = make_ax()
        try:
            len(psi)
        except TypeError:
            psi = np.full(2 * len(xax), psi)
        npoints = len(xax)

        T = kinetic_matrix(len(xax), rabi, mass_photon, mass_exciton, np.diff(xax)[0])
        pot_phot = 1j * ((gain / (saturation + np.abs(psi[:npoints]) ** 2)) - loss) / 2
        pot_x = 1j * ((gain / (saturation + np.abs(psi[npoints:]) ** 2)) - loss) / 2

        U = potential_matrix(pot_phot, pot_x, detuning)
        return T + U, xax

    times = np.linspace(0, 80, 10001)
    xax = make_ax()
    kax = make_k_ax(xax)
    loss = 0.2
    saturation = 100
    threshold = loss * saturation
    gains = np.append(np.insert(np.linspace(0.5, 3, 6), 0, 0.1), 6) * threshold
    wavefunctions = []
    densities = []
    for gain in tqdm(gains):
        _ham = partial(hamiltonian_gain_loss, xax=xax, gain=gain, loss=0.2, saturation=saturation, detuning=0, rabi=6)
        ham = lambda t, psi: _ham(t, psi)[0]
        wavefunctions += [solve_timerange(ham, times)]

        density, e_ax = farfield(ham, times)
        densities += [density]
    wavefunctions = np.array(wavefunctions)
    densities = np.array(densities)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(wspace=0.3))
    colors = ['C%d' % x for x in range(6)]
    [axs[0].semilogy(times, np.sum(np.abs(t) ** 2, 0), c=c, label='photonic') for t, c in zip(wavefunctions[:, :len(xax)], colors)]
    [axs[0].semilogy(times, np.sum(np.abs(t) ** 2, 0), ls='-.', label='excitonic') for t in wavefunctions[:, len(xax):]]
    unique_legend(axs[0], multi_artist=True)
    label_axes(axs[0], 'Time [ps]', '%s$_x|%s(t)|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    axs[1].loglog(gains/threshold, np.sum(np.abs(wavefunctions[:, :101, -1]) ** 2, 1), 'k', marker='x', label='photonic')
    axs[1].loglog(gains/threshold, np.sum(np.abs(wavefunctions[:, 101:, -1]) ** 2, 1), 'k', marker='x', ls='-.', label='excitonic')
    unique_legend(axs[1], multi_artist=True)
    label_axes(axs[1], '$P/P_{th}$', '%s$_x|%s(t_{final})|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    fig = figure(figsize=(12, 8))
    b, a = square(len(gains))
    gs = gridspec.GridSpec(a, b, fig, hspace=0.02, wspace=0.02)
    axs = gs.subplots(sharex=True, sharey=True)
    for ax, d, g in zip(axs.flatten(), densities, gains/threshold):
        imshow(d.transpose(), ax, diverging=False, norm=LogNorm(), cbar=False, xaxis=kax, yaxis=e_ax)
        ax.set_ylim(-3, 3)
        ax.text(0.5, 0.99, '%.1f $P/P_{th}$' % g, transform=ax.transAxes, ha='center', va='top')
    label_grid(gs, '$k_x$ [%sm$^{-1}$]' % greek_alphabet['mu'], 'bottom', 0.04)
    label_grid(gs, 'E [meV]', 'left', 0.04)
