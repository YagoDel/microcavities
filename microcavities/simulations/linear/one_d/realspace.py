# -*- coding: utf-8 -*-

"""1D diagonalising and numerically solving of arbitrary Hamiltonians

See test_ functions for example usage
"""

from microcavities.simulations.linear.one_d import *


"""SETTING UP MATRICES"""


def _kinetic_matrix(mass, size, x_spacing=1):
    """Matrix representation of the kinetic energy operator of a quantum particle
    :param mass:
    :param size:
    :param x_spacing:
    :return:
    """
    diag = -2 * np.diag(np.ones((size,))) + np.diag(np.ones((size-1,)), 1) + np.diag(np.ones((size-1,)), -1)
    return (-1/2) * diag * (hbar**2) / (mass*electron_mass*x_spacing**2)


def kinetic_matrix(size, rabi, mass_photon=1e-5, mass_exciton=0.35, x_spacing=1):
    """Matrix representation of the kinetic energy operator of a polariton
    :param size: int
    :param rabi: float
    :param mass_photon: float
    :param mass_exciton: float
    :param x_spacing: float
    :return:
    """
    # Photon kinetic matrix
    t_photon = _kinetic_matrix(mass_photon, size, x_spacing)

    # Exciton kinetic matrix
    t_exciton = _kinetic_matrix(mass_exciton, size, x_spacing)

    # Coupling to exciton
    try:
        assert len(rabi) == size
        r = np.diag(rabi)
    except TypeError:
        r = rabi * np.eye(size)
    return np.bmat([[t_photon, r/2], [r/2, t_exciton]])


def potential_matrix(potential_photon, potential_exciton, detuning):
    """
    :param potential_photon:
    :param potential_exciton:
    :param detuning:
    :return:
    """
    assert potential_exciton.shape == potential_photon.shape

    potential_photon = np.diag(potential_photon + detuning / 2)
    potential_exciton = np.diag(potential_exciton - detuning / 2)

    off_diag = np.zeros(potential_photon.shape)
    return np.bmat([[potential_photon, off_diag], [off_diag, potential_exciton]])


"""FARFIELD EMISSION"""


def farfield(hamiltonian, time_range, starting_vectors=None):
    """Calculates the farfield emission pattern of a one dimensional Hamiltonian

    Given a list of starting wavefunctions, evolves them in time using a Runge Kutta stepper and extracts the farfield
    emission of each wavefunction by Fourier transforming them in space and time

    :param hamiltonian: function. Takes one argument (time), returns a 2D matrix
    :param time_range: list of times to evaluate the Hamiltonian at
    :param starting_vectors: list of vectors, integer or None. If None, If an integer,
    :return:
    """
    n_points = hamiltonian(0).shape[0]
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


"""EXAMPLE HAMILTONIANS"""


def harmonic_potential(omega=1, mass_photon=1e-5, mass_exciton=0.35, detuning=3, rabi=6, xax=None):
    if xax is None:
        xax = make_ax()

    T = kinetic_matrix(len(xax), rabi, mass_photon, mass_exciton, np.diff(xax)[0])
    pot = (0.5 * mass_photon * (omega * xax) ** 2)
    U = potential_matrix(pot, 0*xax, detuning)
    return T + U, xax


def hamiltonian_conveyor_x(t, period, frequency, potential_depth, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35,
                           xax=None, background=0):
    if xax is None:
        xax = make_ax()

    T = kinetic_matrix(len(xax), rabi, mass_photon, mass_exciton, np.diff(xax)[0])
    pot = potential_depth * np.cos((2 * np.pi / period) * xax - (2 * np.pi * frequency) * t) + background
    U = potential_matrix(0*xax, pot, detuning)
    return T + U, xax


def test_farfield_harmonic_potential():
    omega = 200
    DETUNING = -5
    RABI = 3
    times = np.linspace(-100, 100, 16001)

    ham, xax = harmonic_potential(omega=omega, detuning=DETUNING, rabi=RABI)
    energies, modes = solve(ham)

    density, e_ax = farfield(lambda t: ham, times, 3)
    kax = make_k_ax(xax)

    fig, ax = plt.subplots(1, 1)
    imshow(np.abs(density).transpose(), ax, norm=LogNorm(), diverging=False, xaxis=kax,
           yaxis=e_ax, interpolation='none', cbar=False)
    ax.set_ylim(-3.5, 5)
    diff_e = np.diff(energies)[0]
    for e, m in zip(energies[:4], modes[:, :4].transpose()):
        normed = normalize(m[:101]) - 0.5
        normed -= normed[0]
        ax.plot(kax, 0.5 * diff_e * normed + e, color=(0.5, 0.5, 0.5, 0.7), ls='--')
    label_axes(ax, '$k$ [um]', 'Energy [meV]')


def test_farfield_conveyor_belt():
    n_points = 101
    period = 15
    xax = make_ax(period * 8, n_points)
    depth = 1
    detuning = -4
    rabi = 8
    times = np.linspace(-100, 100, 2001)
    kax = make_k_ax(xax)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for ax, frequency in zip(axs, [0, 1e-2, 1e-1]):
        _ham = partial(hamiltonian_conveyor_x, period=period, frequency=frequency, potential_depth=depth,
                       detuning=detuning,
                       rabi=rabi, xax=xax)
        ham = lambda t: _ham(t)[0]
        density, e_ax = farfield(ham, times, 3)

        imshow(np.abs(density).transpose(), ax, norm=LogNorm(), diverging=False, xaxis=kax,
               yaxis=e_ax, interpolation='none', cbar=False)
        ax.set_ylim(-6, 6)
        ax.set_xlim(-1.5, 1.5)
        label_axes(ax, '$k$ [um]', 'Energy [meV]', 'f=%dGHz' % (frequency * 1e3))


def test_farfield_free_space():
    from microcavities.analysis.dispersion import exciton_photon_dispersions
    mass_photon = 5e-5
    mass_exciton = 0.35
    detuning = -5
    rabi = 5
    n_points = 501
    xax = np.linspace(-80, 80, n_points)
    kax = make_k_ax(xax)

    T = kinetic_matrix(len(xax), rabi, mass_photon, mass_exciton, np.diff(xax)[0])
    U = potential_matrix(0*xax, 0*xax, detuning)
    H = T + U

    ham = lambda t: H
    times = np.linspace(-100, 100, 5001)

    ff = farfield(ham, times, 1)

    fig, ax = plt.subplots(1, 1)
    imshow(ff[0].transpose(), ax, diverging=False, norm=LogNorm(), xaxis=kax, yaxis=ff[1], interpolation='none')
    lower, upper, exciton, photon = exciton_photon_dispersions(kax, detuning / 2, rabi, mass_photon, -detuning / 2,
                                                               mass_exciton, for_fit=False)
    [ax.plot(kax, y, color=c, alpha=0.3, lw=3) for y, c in zip([lower, upper], ['darkviolet', 'darkorange'])]
    [ax.plot(kax, y, color='k', alpha=0.3, lw=3, ls='--') for y in [exciton, photon]]
    ax.set_ylim(-10, 10)
