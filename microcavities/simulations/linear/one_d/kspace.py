# -*- coding: utf-8 -*-

"""1D diagonalising and numerically solving of arbitrary Hamiltonians

See test_ functions for example usage
"""

from microcavities.simulations.linear.one_d import *

LOGGER = create_logger('microcavities.simulations.linear.one_d.kspace')
LOGGER.setLevel('WARN')


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

    # Defining the Hamiltonian function so that it has only two parameters: k, t
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
def hamiltonian_free_space_k(k, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35):
    photon = hbar ** 2 * k ** 2 / (2 * mass_photon * electron_mass)
    exciton = hbar ** 2 * k ** 2 / (2 * mass_exciton * electron_mass)

    photon += detuning/2
    exciton -= detuning / 2
    return np.array([[photon, rabi / 2], [rabi / 2, exciton]])


def hamiltonian_conveyor_k(k, t, period, frequency, potential_depth, detuning, rabi, mass_photon=1e-5,
                           mass_exciton=0.35, n_bands=6, background=0):
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
    :param background:
    :return:
    """
    G = 2 * np.pi / period
    space_size = 2 * n_bands + 1
    omega = 2 * np.pi * frequency

    # Kinetic energy
    photon = np.diag(
        [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_photon * electron_mass) for x in range(-n_bands, n_bands + 1)])
    photon += np.eye(space_size) * detuning / 2
    photon = np.asarray(photon, dtype=complex)

    # Potential energy
    exciton = np.diag(
        [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_exciton * electron_mass) for x in range(-n_bands, n_bands + 1)])
    exciton = np.asarray(exciton, dtype=complex)

    pot = [potential_depth / 2] * (space_size - 1)
    exciton += np.diag(pot, -1) * np.exp(1j * omega * t) + np.diag(pot, 1) * np.exp(-1j * omega * t)
    exciton -= np.eye(space_size) * (detuning / 2 + background)

    # Coupling to exciton
    _rabi = np.eye(space_size) * rabi/2
    return np.bmat([[photon, _rabi], [_rabi, exciton]])


def hamiltonian_floquet_conveyor(k, t, period, frequency, floquet_depth, potential_depth, detuning, rabi, mass_photon=1e-5,
                                 mass_exciton=0.35, n_bands=6, background=0, n_periods=10):
    h0 = hamiltonian_conveyor_k(k, 0, period, frequency, potential_depth, detuning, rabi, mass_photon,
                                mass_exciton, n_bands, background)
    floquet_coupling = np.diag([floquet_depth]*h0.shape[0])
    blocks = []
    for idx1 in range(n_periods):
        floquet_offset = (idx1 - n_periods//2) * hbar * 2 * np.pi * frequency
        _row = []
        for idx2 in range(n_periods):
            if idx2 == idx1:
                _row += [h0+floquet_offset]
            elif np.abs(idx2 - idx1) == 1:
                _row += [floquet_coupling]
            else:
                _row += [np.zeros(floquet_coupling.shape)]
        blocks += [_row]
    return np.block(blocks)


def test_conveyor_chern():
    """Takes ~5min to run"""
    n_points = 11
    period = 15
    depth = 1
    detuning = -4
    rabi = 8
    ham = partial(hamiltonian_conveyor_k, potential_depth=depth, detuning=detuning, rabi=rabi, n_bands=20)

    chern_numbers = np.zeros([3, 10], dtype=complex)
    frequencies = np.append(np.linspace(-9e-3, -1e-4, 5), np.linspace(1e-4, 9e-3, 5))
    for bn in [0, 1, 2]:
        for idx, f in enumerate(frequencies):
            cn, bc = calculate_chern_number(ham, 2 * np.pi / period, 1/f, band_number=bn, n_points=n_points)
            print('Chern number %gGHz= ' % (f*1e3), cn)
            chern_numbers[bn, idx] = cn

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    kx = np.linspace(-2*np.pi/period, 2*np.pi/period, 101)
    bands, _ = solve_for_krange(kx, partial(ham, t=0, frequency=1e-3, period=period))
    axs[0].plot(kx, bands[:, :10].real)
    label_axes(axs[0], '$k_x$ [um]', 'E [meV]')

    [axs[1].plot(frequencies*1e3, c, label='band %d' % x) for x, c in enumerate(chern_numbers)]
    label_axes(axs[1], 'f [GHz]', 'Chern number')
    axs[1].legend()
    axs[1].set_ylim(-2, 2)


def test_farfield_free_space():
    from microcavities.analysis.dispersion import exciton_photon_dispersions

    n_bands = 0
    space_size = 2 * n_bands + 1
    mass_photon = 5e-5
    mass_exciton = 0.35
    detuning = -5
    rabi = 5
    G = 2 * np.pi / 15

    def free_space(k):
        # Kinetic energy
        photon = np.diag(
            [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_photon * electron_mass) for x in range(-n_bands, n_bands + 1)])
        photon += np.eye(space_size) * detuning / 2
        photon = np.asarray(photon, dtype=complex)

        # Potential energy
        exciton = np.diag(
            [hbar ** 2 * (k - x * G) ** 2 / (2 * mass_exciton * electron_mass) for x in range(-n_bands, n_bands + 1)])
        exciton = np.asarray(exciton, dtype=complex)
        exciton -= np.eye(space_size) * detuning / 2

        # Coupling to exciton
        _rabi = np.eye(space_size) * rabi / 2
        return np.bmat([[photon, _rabi], [_rabi, exciton]])

    k_ax = np.linspace(-10, 10, 101)

    theory_bands, _ = solve_for_krange(k_ax, free_space)

    lower, upper, exciton, photon = exciton_photon_dispersions(k_ax, (detuning)/2, rabi, mass_photon, -detuning/2,
                                                               mass_exciton, for_fit=False)

    fig, ax = plt.subplots(1, 1)
    ax.plot(k_ax, theory_bands, 'r')
    [ax.plot(k_ax, y, color=c, alpha=0.3, lw=3) for y, c in zip([lower, upper], ['darkviolet', 'darkorange'])]
    [ax.plot(k_ax, y, color='k', alpha=0.3, lw=3, ls='--') for y in [exciton, photon]]


