# -*- coding: utf-8 -*-

"""1D diagonalising and numerically solving of arbitrary Hamiltonians

See test_ functions for example usage
"""

from microcavities.simulations.linear.one_d import *

LOGGER = create_logger('Bloch simulations')
LOGGER.setLevel('WARN')

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


# Time step solver
def rk_timestep(psi, hamiltonian, t, dt):
    """Runge-Kutta 4th order time step
    :param psi: vector
    :param hamiltonian: function that returns a matrix
    :param t: float
    :param dt: float
    :return:
    """
    k11 = -1j * np.matmul(hamiltonian(t), psi) / hbar
    k21 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + k11 * dt / 2) / hbar
    k31 = -1j * np.matmul(hamiltonian(t + dt / 2), psi + k21 * dt / 2) / hbar
    k41 = -1j * np.matmul(hamiltonian(t + dt), psi + dt * k31) / hbar

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
    N = len(starting_vectors[0]) // 2
    rho = np.zeros((N, len(time_range)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(vec, hamiltonian, time_range)
        psi_kw = np.fft.fftshift(np.fft.fft2(psi[:N, :]))
        rho += np.abs(psi_kw) ** 2
        if np.isnan(rho).any():
            break
    return rho


# Diagonalisation solver
def solve(hamiltonian):
    E, eig_vectors = np.linalg.eig(hamiltonian)
    # linalg.eig does not necessarily return sorted values, so we sort it after:
    idx_sort = np.argsort(E.real)
    return E[idx_sort], eig_vectors[:, idx_sort]


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
        idx_sort = np.argsort(E.real)
        bands += [E[idx_sort]]
        modes += [eig_vectors[:, idx_sort]]
    return np.array(bands), np.array(modes)


def calculate_chern_number(hamiltonian, momentum_range, time_period, n_points=100, band_number=3, hamiltonian_kw=None):
    """Calculates the Chern number of an (x, t) Hamiltonian

    Numerically evaluates band differentials in momentum and time to extract the Berry curvature and then sums over the
    whole 2D Brillouin zone to get the Chern number.

    :param hamiltonian: function
    :param momentum_range: float
    :param time_period: float
    :param n_points: int
    :param band_number: int
    :param hamiltonian_kw: dict or None
    :return:
    """
    # Choosing a differential size that is smaller than the step size along the k and t dimensions
    delta_k = momentum_range / (100*n_points)
    delta_t = time_period / (100*n_points)

    # Defining the Hamiltonian function to that it has only two parameters: k, t
    f = np.abs(1/time_period)
    if hamiltonian_kw is None: hamiltonian_kw = dict()
    _hamiltonian = partial(hamiltonian, period=2*np.pi / momentum_range, frequency=f, **hamiltonian_kw)

    # Looping over the Brillouin zones and one full time period
    k_range = np.arange(-momentum_range / 2, momentum_range / 2, momentum_range / n_points)
    t_range = np.arange(0, time_period, time_period / n_points)

    berry_curvature = []
    for kx in tqdm(k_range, 'Brillouin zone sum'):
        _berry_curvature = []
        for t in t_range:
            # Band wavefunction evaluated at four points (k, t), (k+dk, t), (k, t+dt), (k+dk, k+dt)
            vectors = []
            for _t in [t, t + delta_t]:
                for _k in [kx, kx + delta_k]:
                    h = _hamiltonian(_k, _t)
                    eigenvalue, eigenvector = np.linalg.eig(h)
                    vector = eigenvector[:, np.argsort(np.real(eigenvalue))[band_number]]  #
                    vectors += [vector]

            # Fixing the gauge of the wavefunctions by making it real in the same BZ
            index = np.argmax(np.abs(vectors[0]))  # BZ index
            vectors = [v * np.exp(- 1j * np.angle(v[index])) for v in vectors]

            # Berry connections as partial differentials wrt k and t
            a_k = np.dot(vectors[0].transpose().conj(), (vectors[1] - vectors[0]) / delta_k)
            a_t = np.dot(vectors[0].transpose().conj(), (vectors[2] - vectors[0]) / delta_t)
            a_k_dt = np.dot(vectors[2].transpose().conj(), (vectors[3] - vectors[2]) / delta_k)
            a_t_dk = np.dot(vectors[1].transpose().conj(), (vectors[3] - vectors[1]) / delta_t)

            # Berry curvature
            _berry_curvature += [(a_t_dk - a_t) / delta_k - (a_k_dt - a_k) / delta_t]
        berry_curvature += [_berry_curvature]
    chern_number = np.sum(berry_curvature) * (momentum_range / n_points) * (time_period / n_points) / (2 * np.pi * 1j)
    return chern_number, np.array(berry_curvature)


# Example Hamiltonians
def hamiltonian_conveyor(k, t, period, frequency, potential_depth, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35, n_bands=6):
    """1D Time-dependent Bloch Hamiltonian for a conveyor belt potential on the exciton component
    :param k:
    :param t:
    :param period:
    :param frequency:
    :param potential_depth:
    :param detuning:
    :param rabi:
    :param mass_photon:
    :param mass_exciton:
    :param n_bands:
    :return:
    """
    G = 2 * np.pi / period
    space_size = 2 * n_bands + 1
    omega = 2 * np.pi * frequency

    # Kinetic energy
    photon = np.diag(
        [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_photon * electron_mass) for x in range(-n_bands, n_bands + 1)])
    photon -= np.eye(space_size) * detuning / 2
    photon = np.asarray(photon, dtype=complex)

    # Potential energy
    exciton = np.diag(
        [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_exciton * electron_mass) for x in range(-n_bands, n_bands + 1)])
    exciton = np.asarray(exciton, dtype=complex)

    pot = [potential_depth / 2] * (space_size - 1)
    exciton += np.diag(pot, -1) * np.exp(1j * omega * t) + np.diag(pot, 1) * np.exp(-1j * omega * t)
    exciton += np.eye(space_size) * detuning / 2

    # Coupling to exciton
    _rabi = np.eye(space_size) * rabi/2
    return np.bmat([[photon, _rabi], [_rabi, exciton]])


def qho(t, w_0=1, mass=electron_mass*1e-5, detuning=3, rabi=6, n_points=101):
    x = np.linspace(-21, 20, n_points)
    D2 = np.diag(-2 * np.ones(n_points)) + np.diag(np.ones(n_points - 1), 1) + np.diag(np.ones(n_points - 1), -1)
    dx = np.diff(x)[0]  # 1.25
    D2 /= dx ** 2
    Hk0 = -D2 * hbar ** 2 / (2 * mass)
    Hk0 -= np.eye(n_points) * detuning / 2
    Hk0 += (0.5 * mass * (w_0 * x) ** 2) * np.eye(n_points)
    Hv = (0.5 * mass * (w_0 * x) ** 2) * np.eye(n_points)
    H1row = np.hstack([Hk0, rabi * np.eye(n_points)])
    H2row = np.hstack([rabi * np.eye(n_points), Hv])
    return np.vstack([H1row, H2row])


def test_hamiltonians():
    from microcavities.utils.plotting import create_axes, unique_legend, label_axes

    # Quantum Harmonic Oscillator
    trap = 1

    energies, modes = solve(qho(0, trap))
    _norm = 0.25 * np.diff(energies).max()
    x = np.linspace(-21, 20, 101)

    fig, ax = create_axes()
    idx = 0
    for e, m in zip(energies[:3], modes.transpose()[:3]):
        mode = normalize(m) * _norm + e
        ax.plot(x, mode[:101], color='C%d' % idx, label='exciton')
        ax.plot(x, mode[101:],  '--', color='C%d' % idx, label='photon')
        idx += 1
    unique_legend(ax, multi_artist=True)
    label_axes(ax, 'x [um]', 'Energy', 'QHO solved in space')


def test_conveyor_chern():
    """Takes ~1h to run"""
    n_points = 11
    period = 15
    depth = 1
    detuning = -4
    rabi = 8
    ham = partial(hamiltonian_conveyor, potential_depth=depth, detuning=detuning, rabi=rabi, n_bands=20)

    chern_numbers = np.zeros([3, 10], dtype=complex)
    frequencies = np.append(np.linspace(-9e-3, -1e-4, 5), np.linspace(1e-4, 9e-3, 5))
    for bn in [0, 1, 2]:
        for idx, f in enumerate(frequencies):
            cn, bc = calculate_chern_number(ham, 2 * np.pi / period, 1/f, band_number=bn, n_points=n_points)
            print('Chern number %gGHz= ' % (f*1e3), cn)
            chern_numbers[bn, idx] = cn

    fig, axs = plt.subplots(1, 2)
    kx = np.linspace(-2*np.pi/period, 2*np.pi/period, 101)
    bands, _ = solve_for_krange(kx, partial(ham, t=0, frequency=0, period=period))
    axs[0].plot(kx, bands[:, :10].real)
    label_axes(axs[0], '$k_x$ [um]', 'E [meV]')
    [axs[1].plot(frequencies*1e3, c, label='band %d' % x) for x, c in enumerate(chern_numbers)]
    label_axes(axs[1], 'f [GHz]', 'Chern number')
    axs[1].legend()
