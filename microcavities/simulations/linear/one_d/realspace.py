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
    return (-1/2) * diag * (hbar**2) / (mass*electron_mass*x_spacing)


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


"""EIGENSYSTEM SOLVING"""


def rk_timestep(psi, hamiltonian, t, dt):
    """Runge-Kutta 4th order time step
    :param psi: vector
    :param hamiltonian: function that returns a matrix
    :param t: float
    :param dt: float
    :return:
    """
    # print(psi.shape)
    # print(hamiltonian(t).shape)
    k11 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t), psi))) / hbar
    # print(type(k11))
    # print(k11.transpose()[:, 0].shape)
    # print((psi + k11 * dt / 2).shape)
    # print(hamiltonian(t + dt / 2).shape)
    # print(dt)
    # print((psi + k11 * dt / 2).shape)
    # print(psi.shape)
    # print(np.matmul(hamiltonian(t + dt / 2), psi + k11 * dt / 2))

    k21 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2), psi + k11 * dt / 2))) / hbar
    k31 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt / 2), psi + k21 * dt / 2))) / hbar
    k41 = -1j * np.squeeze(np.asarray(np.matmul(hamiltonian(t + dt), psi + dt * k31))) / hbar
    # print(k11.shape, (psi + (k11 + 2 * k21 + 2 * k31 + k41) * dt / 6).shape)
    # print(k21.shape)
    # print(k31.shape)
    # print(k41.shape)
    return psi + (k11 + 2 * k21 + 2 * k31 + k41) * dt / 6


def solve_timerange(starting_wavefunction, hamiltonian, time_range):
    """
    :param starting_wavefunction: vector
    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :return: 2d array
    """
    full_psi = np.zeros((len(starting_wavefunction), len(time_range)), dtype=complex)
    for idx_t, t in enumerate(time_range):
        # print(1, starting_wavefunction.shape)
        full_psi[:, idx_t] = starting_wavefunction
        starting_wavefunction = rk_timestep(starting_wavefunction, hamiltonian, t, np.diff(time_range)[0])
        # print(2, starting_wavefunction.shape)
    return full_psi


def farfield(hamiltonian, time_range, starting_vectors=None):
    """Calculates the farfield emission pattern of a Hamiltonian when the eigenvectors are randomly occupied

    :param hamiltonian: function. Takes one argument (time), returns an array
    :param time_range:
    :param starting_vectors:
    :return:
    """
    n_points = hamiltonian(0).shape[0]
    size = int(n_points / 2)
    if starting_vectors is None:
        starting_vectors = np.array([np.random.uniform(-1, 1, (n_points, ))
                                     + 1.j * np.random.uniform(-1, 1, (n_points, ))])
    elif type(starting_vectors) == int:
        starting_vectors = np.array([np.random.uniform(-1, 1, (n_points, ))
                                     + 1.j * np.random.uniform(-1, 1, (n_points, )) for x in range(starting_vectors)])

    rho = np.zeros((size, len(time_range)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(vec, hamiltonian, time_range)
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
    U = potential_matrix(pot, pot, detuning)
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
           yaxis=e_ax, interpolation='none')
    ax.set_ylim(-5, 10)
    diff_e = np.diff(energies)[0]
    for e, m in zip(energies[:4], modes[:, :4].transpose()):
        normed = normalize(m[:101]) - 0.5
        normed -= normed[0]
        ax.plot(kax, 0.5 * diff_e * normed + e, color=(0.5, 0.5, 0.5, 0.7), ls='--')

