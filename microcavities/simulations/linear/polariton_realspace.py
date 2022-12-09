# -*- coding: utf-8 -*-

""" Numerical solution of arbitrary 2D polariton Hamiltonians

Constructs and solves exciton-photon coupled Hamiltonians with arbitrary potentials in for the photon and exciton
components. To be used for physical systems with finite boundaries (i.e. not for periodic systems and bands)
"""

from microcavities.simulations.linear import *
from dxf.lattices import *


"""SETTING UP MATRICES"""


def _kinetic_matrix(mass, size, x_spacing=1):
    """Matrix representation of the kinetic energy operator of a quantum particle
    :param mass:
    :param size:
    :param x_spacing:
    :return:
    """
    diag = np.ones([size])
    diags = np.array([diag, -2*diag, diag])
    # D is a sparse matrix which has -2 on the main diagonal and 1 on the two neighbouring diagonals
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), size, size)
    T = -1/2 * sparse.kronsum(D, D)
    return T * (hbar**2) / (mass*electron_mass*x_spacing**2)


def kinetic_matrix(rabi, mass_photon=1e-5, mass_exciton=0.35, axes=None):
    """Matrix representation of the kinetic energy operator of a polariton
    :param rabi: float
    :param mass_photon: float
    :param mass_exciton: float
    :param axes: tuple
    :return:
    """
    if axes is None:
        axes = make_axes()
    size = axes[0].shape[0]
    x_spacing = np.diff(axes[3])[0]

    # Photon kinetic matrix
    t_photon = _kinetic_matrix(mass_photon, size, x_spacing)

    # Exciton kinetic matrix
    t_exciton = _kinetic_matrix(mass_exciton, size, x_spacing)

    # Coupling to exciton
    r = rabi * sparse.eye(size**2)
    return sparse.bmat([[t_photon, r/2], [r/2, t_exciton]]), axes


def potential_matrix(potential_photon, potential_exciton, detuning):
    """
    :param potential_photon:
    :param potential_exciton:
    :param detuning:
    :return:
    """
    assert potential_exciton.shape == potential_photon.shape
    assert potential_exciton.shape[0] == potential_exciton.shape[1]

    size = potential_photon.shape[0]
    potential_photon = sparse.diags(potential_photon.reshape(size**2) + detuning / 2, 0)
    potential_exciton = sparse.diags(potential_exciton.reshape(size**2) - detuning / 2, 0)

    return sparse.bmat([[potential_photon, None], [None, potential_exciton]])


def make_hamiltonian(potential_photon, potential_exciton, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35, axes=None):
    kinetic, axes = kinetic_matrix(rabi, mass_photon, mass_exciton, axes=axes)

    if type(potential_photon) in [int, float, complex]:
        potential_photon *= np.ones(axes[0].shape)
    if type(potential_exciton) in [int, float, complex]:
        potential_exciton *= np.ones(axes[0].shape)
    potential = potential_matrix(potential_photon, potential_exciton, detuning)
    return kinetic + potential, axes


"""EIGENSYSTEM SOLVING"""


def solve_polariton_static(hamiltonian, n_eigenvalues=250):

    #
    # if axes is None:
    #     axes = make_axes()  # 30 um square area
    size = hamiltonian.shape[0]
    # x_spacing = np.diff(axes[3])[0]
    if n_eigenvalues > 2 * (size**2): n_eigenvalues = 2*(size**2)-1
    #
    # potential = potential_matrix(potential_photon, potential_exciton, detuning)
    # kinetic = kinetic_matrix(size, rabi, mass_photon, mass_exciton, x_spacing)

    # Compute eigenvalues and eigenvectors
    vals, vecs = eigsh(hamiltonian, k=n_eigenvalues, which='SA')
    return vals, vecs


def get_eigenvector(n, vecs, size=None):
    """
    :param n: int. Index selection of eigenvector
    :param vecs: array. List of eigenvectors
    :param size: int. To determine how to reshape the eigenvector from a list to a matrix
    :return:
    """
    if size is None: size = int(np.sqrt(vecs.shape[0]/2))
    return vecs.T[n].reshape((2, size, size))


def plot_eigenvectors(x, y, vecs, vals):
    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(1, 2, fig, width_ratios=[1, 4])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, gs[1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(vals)
    axs = gs2.subplots()
    # fig, axs = plt.subplots(4, 2, figsize=(9, 9))
    for idx, _axs in enumerate(axs.transpose()):
        imshow(get_eigenvector(idx, vecs)[0], _axs[0], xaxis=x, yaxis=y)
        imshow(get_eigenvector(idx, vecs)[1], _axs[1], xaxis=x, yaxis=y)
    return fig, [ax0, axs]


"""FARFIELD CODE"""


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


def solve_timerange(psi0, hamiltonian, timerange):
    """Time steps an initial wavefunction using the Runge Kutta stepper
    :param psi0: vector
    :param hamiltonian: function. Should take a single input (time) and return a scipy.sparse matrix
    :param timerange: list
    :return:
    """
    full_psi = np.zeros((len(psi0), len(timerange)), dtype=complex)
    for idx_t, t in tqdm(enumerate(timerange), 'solve_timerange'):
        full_psi[:, idx_t] = psi0
        psi0 = rk_timestep(psi0, hamiltonian, t, np.diff(timerange)[0])
    return full_psi


def farfield(hamiltonian, timerange, starting_vectors=None):
    """
    :param hamiltonian:
    :param timerange:
    :param starting_vectors:
    :return:
    """
    n_points = hamiltonian(0).shape[0]
    size = int(np.sqrt(n_points / 2))
    if starting_vectors is None:
        starting_vectors = np.array([np.random.uniform(-1, 1, (n_points, ))
                                     + 1.j * np.random.uniform(-1, 1, (n_points, ))])

    # N = starting_vectors.shape[1] // 2
    rho = np.zeros((size, size, len(timerange)))
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(vec, hamiltonian, timerange)
        psi_reshaped = np.reshape(psi, (2, size, size, len(timerange)))

        photon_field = psi_reshaped[0]
        psikw = np.fft.fftshift(np.fft.fft2(photon_field, axes=(0, 1, 2)))
        rho += np.abs(psikw) ** 2
        if np.isnan(rho).any():
            break

    dE = hbar * np.pi / timerange[-1]
    theory_eax = (np.linspace(-dE, dE, len(timerange)) * len(timerange) / 2)[::-1]

    return rho, theory_eax


"""EXAMPLE HAMILTONIANS"""


# def hamiltonian_x_qho(omega, rabi, detuning, mass_photon=1e-5, mass_exciton=0.35, axes=None):
#     if axes is None:
#         axes = make_axes()  # 30 um square area
#     # size = axes[0].shape[0]
#     # x_spacing = np.diff(axes[3])[0]
#     _potential = 0.5 * mass_photon * (omega**2) * (axes[0]**2 + axes[1]**2)
#     potential = potential_matrix(_potential, np.zeros(axes[0].shape), detuning)
#     # kinetic = kinetic_matrix(size, rabi, mass_photon, mass_exciton, x_spacing=x_spacing)
#     return add_kinetic(potential, rabi,  mass_photon, mass_exciton, axes)

    # return potential + kinetic, axes
    # nx,ny = np.arange(0,4), np.arange(0,4)
    # ana_vals = hbar**2*(nx+ny+1)*omega**2/(2*mass)
    # return V


def single_circle(radius, depth_photon, depth_exciton, center=(0, 0), background=0, axes=None,
                  potential=None):
    """Creates a potential with a single, circular well

    :param radius:
    :param depth_photon:
    :param depth_exciton:
    :param center:
    :param background:
    :param axes:
    :param potential:
    :return:
    """
    if axes is None:
        axes = make_axes()  # 30 um square area
    potential_photon = np.full(axes[0].shape, background)
    potential_exciton = np.full(axes[0].shape, background)

    mask = ((axes[0]-center[0])**2 + (axes[1]-center[1])**2) < radius**2

    potential_photon[mask] = depth_photon + background
    potential_exciton[mask] = depth_exciton + background
    return potential_photon, potential_exciton, axes


"""TEST FUNCTIONS"""


def test_potential_single_circles():
    """Simply tests that the potentials look like how we expect them to look"""
    fig, axs = plt.subplots(1, 2)
    vphot, vex, axes = single_circle(5, -10, 5)
    imshow(vphot, axs[0], xaxis=axes[2], yaxis=axes[3], aspect='equal')
    imshow(vex, axs[1], xaxis=axes[2], yaxis=axes[3], aspect='equal')
    label_axes(axs[0], 'x [um]', 'y [um]', 'Photon (-10meV)')
    label_axes(axs[1], 'x [um]', 'y [um]', 'Exciton (+5meV)')


def test_hamiltonian_freespace_x():
    """Compares the far-field emission arising from the potential_matrix/kinetic_matrix in this file to that expected
    from solving the Hamiltonian in k-space"""
    from microcavities.simulations.linear.polariton_kspace import solve_for_krange, hamiltonian_k_free_space

    DETUNING = -5
    RABI = 3

    n_points = 201
    times = np.linspace(-25, 25, 8001)

    axes = make_axes(100, n_points)  # Using a region large enough to get accurate free space propagation
    k_axes = make_k_axes(axes)
    # size = axes[0].shape[0]
    # x_spacing = np.diff(axes[3])[0]

    # potential = potential_matrix(np.zeros(axes[0].shape), np.zeros(axes[0].shape), DETUNING)
    # kinetic = kinetic_matrix(size, RABI, x_spacing=x_spacing)
    # _ham = (potential + kinetic)
    # _ham = hamiltonian_x_free_space(RABI, DETUNING, axes=axes)
    # kinetic, axes = kinetic_matrix(RABI, axes=axes)
    # potential = potential_matrix(0, 0, DETUNING)
    # _ham = (potential + kinetic)
    _ham, axes = make_hamiltonian(0, 0, DETUNING, RABI, axes=axes)

    ham = lambda t: _ham

    density, e_ax = farfield(ham, times)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for idx, index in enumerate([80, 90, 100]):
        imshow(np.abs(density[index]).transpose(), axs[idx], norm=LogNorm(), diverging=False, xaxis=k_axes[2],
               yaxis=e_ax, interpolation='none', cbar=False)
        axs[idx].set_ylim(-5, 15)
        axs[idx].set_xlim(-3, 3)

        kax2 = np.sqrt(k_axes[2] ** 2 + k_axes[2][index] ** 2)
        two_modes, _ = solve_for_krange(kax2, partial(hamiltonian_k_free_space, detuning=DETUNING, rabi=RABI))
        axs[idx].plot(k_axes[2], two_modes)
        axs[idx].set_title('$k_y$=%.1f' % k_axes[2][index])
    label_grid(fig, '$k_x$ [um]', 'bottom')
    label_grid(fig, 'E [meV]', 'left')


def test_hamiltonian_qho():
    """Simulates the eigenvalues and eigenvectors for a photonic quadratic potential"""
    omega = 200
    DETUNING = -5
    RABI = 3
    MASS_PHOTON= 1e-5

    axes = make_axes()
    _potential = 0.5 * MASS_PHOTON * (omega**2) * (axes[0]**2 + axes[1]**2)
    _ham, axes = make_hamiltonian(_potential, 0, DETUNING, RABI, MASS_PHOTON, axes=axes)

    energies, vectors = solve_polariton_static(_ham, 20)

    fig = figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, fig, wspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0], hspace=0.3)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[1], hspace=0.02, wspace=0.03)
    ax0 = plt.subplot(gs1[0])
    ax0.plot(axes[2], _potential[int(_potential.shape[0]//2)] + DETUNING/2)
    ax0.hlines(energies, axes[2].min(), axes[2].max(), ls='--', color=(0.5, 0.5, 0.5))
    ax0.set_ylim(-3.5, 5)
    ax0.set_xlim(-8, 8)
    label_axes(ax0, 'x [um]', 'E [meV]', 'Potential')
    ax1 = plt.subplot(gs1[1])
    ax1.plot(energies)
    label_axes(ax1, 'Eigennumber', 'E [meV]', 'Eigenvalues')

    axs = gs2.subplots(sharex=True, sharey=True)
    for idx, ax in enumerate(axs.flatten()):
        vec = get_eigenvector(idx, vectors)
        imshow(vec[0], ax, xaxis=axes[2], yaxis=axes[3], cbar=False)
    label_grid(gs2, '$x$ [um]', 'bottom', offset=0.03)
    label_grid(gs2, '$y$ [um]', 'left')
    label_grid(gs2, 'Wavefunctions', 'top', offset=0.02)
    fig.suptitle(r'Polariton ($\Omega_{Rabi}=%d$, $\Delta=%d$)' % (RABI, DETUNING))


def test_infinite_potential_well():

    def infinite_potential_well(mass, width):
        analytical_eigenvalues = np.array([[(nx**2+ny**2) * (np.pi * hbar)**2 / (2*mass*(width**2))
                                            for nx in np.arange(1, 5)]
                                           for ny in np.arange(1, 5)]).flatten()
        return np.sort(analytical_eigenvalues)

    DETUNING = 0
    RABI = 0
    MASS_PHOTON= 4e-5
    DEPTH = -1000
    WIDTH = 10

    axes = make_axes(20, 41)

    vphot = np.full(axes[0].shape, 0)
    mask = np.logical_and(np.abs(axes[0]) < WIDTH/2, np.abs(axes[1]) < WIDTH/2)
    vphot[mask] = DEPTH
    _ham, axes = make_hamiltonian(vphot, 0, DETUNING, RABI, MASS_PHOTON, axes=axes)

    energies, vectors = solve_polariton_static(_ham, 20)
    analytical = infinite_potential_well(MASS_PHOTON*electron_mass, WIDTH) + DEPTH + DETUNING/2

    fig = figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, fig, wspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0], hspace=0.3)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[1], hspace=0.02, wspace=0.03)
    ax0 = plt.subplot(gs1[0])
    ax0.plot(axes[2], vphot[int(vphot.shape[0]//2)] + DETUNING/2)
    ax0.hlines(energies, axes[2].min(), axes[2].max(), ls='--', color=(0.5, 0.5, 0.5))
    ax0.hlines(analytical, axes[2].min(), axes[2].max(), ls='-.', color='r')
    ax0.set_ylim(DEPTH-0.5, DEPTH+3.5)

    label_axes(ax0, 'x [um]', 'E [meV]', 'Potential')
    ax1 = plt.subplot(gs1[1])
    ax1.plot(energies, color=(0.5, 0.5, 0.5), label='numerical')
    ax1.plot(analytical, color='r', label='analytical')
    ax1.legend()
    label_axes(ax1, 'Eigennumber', 'E [meV]', 'Eigenvalues')

    axs = gs2.subplots(sharex=True, sharey=True)
    for idx, ax in enumerate(axs.flatten()):
        vec = get_eigenvector(idx, vectors)
        imshow(vec[0], ax, xaxis=axes[2], yaxis=axes[3], cbar=False, interpolation='none', aspect='equal')
    label_grid(gs2, '$x$ [um]', 'bottom', offset=0.03)
    label_grid(gs2, '$y$ [um]', 'left')
    label_grid(gs2, 'Wavefunctions', 'top', offset=0.02)
    fig.suptitle(r'Free photon ($\Omega_{Rabi}=0$)')
