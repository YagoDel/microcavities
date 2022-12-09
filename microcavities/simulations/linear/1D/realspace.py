# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools

"""
Make a dimer
- Compare lifetimes of symmetric and antisymmetric states
- Extract tight-binding parameters
- How do the tight-binding parameters vary as a function of separation, width, background gain/loss, gain/loss in the barrier
"""

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


def polariton_mass(detuning, rabi_splitting, photon_mass=1e-5, exciton_mass=0.35):
    hopfield = 0.5 * (1 + detuning/np.sqrt(detuning**2 + rabi_splitting**2))
    return 1 / (hopfield / exciton_mass + (1-hopfield) / photon_mass)


def kinetic_matrix(size=101, mass=1e-3, x_spacing=1.0):
    """

    :param size: int. Size of the square array to be returned
    :param mass: ratio to free electron mass
    :param x_spacing: in microns
    :return:
    """
    try:
        assert len(mass) == size
    except TypeError:
        mass = np.ones(size) * mass
    mass = np.pad(mass, (1, 1), 'edge')
    mass_forward = np.mean([np.roll(mass, -1), mass], 0)
    mass_backward = np.mean([np.roll(mass, 1), mass], 0)
    diag = -(1/mass_forward + 1/mass_backward)[1:-1]  #np.mean([mass_forward, mass_backward], 0)[1:-1]  # - 1/(np.roll(mass, 1) + mass) - 1/(mass + np.roll(mass, -1))
    up = 1/mass_forward[1:-2]  # 1 / (mass + np.roll(mass, 1))
    down = 1/mass_backward[2:-1]  # 1 / (mass + np.roll(mass, -1))
    # print(mass, mass_forward, mass_backward, np.mean([mass_forward, mass_backward], 0))
    matrix = np.diag(diag) + np.diag(up, 1) + np.diag(down, -1)
    diff = matrix / (x_spacing**2)
    return - hbar**2 * diff / (2 * electron_mass)

    #
    # # try:
    # #     assert len(mass) == size
    # #     # print(mass[:10])
    # #     mass = np.array(mass)
    # #     mass_diag = 1 / mass
    # #     mass_offset = 1 / np.mean([mass[1:], mass[:-1]], 0)
    # #     diff = (-2 * np.diag(mass_diag) + np.diag(mass_offset, 1) + np.diag(mass_offset, -1)) / (x_spacing**2)
    # #     # diff = (-2 * np.diag(np.ones(size)/mass) + np.diag(np.ones(size-1)/mass[:-1], 1) + np.diag(np.ones(size-1)/mass[:-1], -1)) / (x_spacing**2)
    # #     # print("Yep")
    # # except Exception as e:
    # #     # print(e)
    # #     diff = (-2 * np.diag(np.ones(size)) + np.diag(np.ones(size-1), 1) + np.diag(np.ones(size-1), -1)) / (x_spacing**2)
    # #     diff = diff / mass
    # # # print(np.diag(diff)[:10])
    # diff = (-2 * np.diag(np.ones(size)) + np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)) / (
    #             x_spacing ** 2)
    # diff = diff / mass
    # return - hbar**2 * diff / (2 * electron_mass)


def single_trap(depth, width, size=101, bkg_value=0, mass=1e-3, mass_landscape=False):
    x = np.linspace(-5*width, 5*width, size)
    potential = np.zeros(size, np.complex) + bkg_value
    potential[np.abs(x) < width / 2] += depth
    if mass_landscape:
        mass_array = np.ones(size) * mass[0]
        mass_array[np.abs(x) < width / 2] = mass[1]
    else:
        try:
            mass_array = np.ones(size) * mass[0]
        except TypeError:
            mass_array = np.ones(size) * mass

    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def double_trap(depth, barrier, width, separation=None, size=101, bkg_value=0.0, mass=1e-3, mass_landscape=False):
    if separation is None:
        separation = 3 * width
    region = np.max([5*separation, 20*width])  # np.max([3*separation, 10*width])
    x = np.linspace(-region, region, size)
    potential = np.zeros(size, np.complex) + bkg_value
    potential[np.abs(x) < separation/2 + width] = depth + bkg_value
    potential[np.abs(x) < separation/2] = barrier + bkg_value
    if mass_landscape:
        mass_array = np.zeros(size) + mass[0]
        mass_array[np.abs(x) < separation / 2 + width] = mass[1]
        if len(mass) == 2:
            mass_array[np.abs(x) < separation / 2] = mass[0]
        if len(mass) == 3:
            mass_array[np.abs(x) < separation/2] = mass[2]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x, mass_array)
    else:
        try:
            mass_array = mass[0]
        except TypeError:
            mass_array = mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x


def single_polariton_trap(trap_depth, width, implant='photonic', photon_mass=1e-5, exciton_mass=0.35,
                          photon_lifetime=10, exciton_lifetime=1e5, detuning=0, rabi_splitting=10, size=101,
                          mass_landscape=True):
    """

    :param trap_depth:
    :param width:
    :param separation:
    :param implant:
    :param photon_mass:
    :param exciton_mass:
    :param cavity_lifetime:
    :param exciton_lifetime:
    :param detuning:
    :param rabi_splitting:
    :return:
    """
    exciton_energy = 0
    photon_energy = float(detuning)
    hopfield_in_trap = 0.5 * (1 + detuning/np.sqrt(detuning**2 + rabi_splitting**2))
    mass_in_trap = 1 / (hopfield_in_trap / exciton_mass + (1-hopfield_in_trap) / photon_mass)
    decay_in_trap = hopfield_in_trap / exciton_lifetime + (1-hopfield_in_trap) / photon_lifetime
    energy_in_trap = 0.5 * (exciton_energy + photon_energy - np.sqrt(rabi_splitting**2 + detuning**2))  # arbitrarily setting the exciton energy to 0

    # print('Ex: %g, Ph: %g, X: %g, m: %g, decay: %g, E: %g' % (exciton_energy, photon_energy, hopfield_in_trap, mass_in_trap, decay_in_trap, energy_in_trap))

    if implant == 'photonic':
        photon_energy = detuning + trap_depth
    elif implant == 'excitonic':
        exciton_energy = trap_depth
    else:
        raise ValueError
    detuning = photon_energy - exciton_energy

    hopfield_out_trap = 0.5 * (1 + detuning/np.sqrt(detuning**2 + rabi_splitting**2))
    mass_out_trap = 1 / (hopfield_out_trap / exciton_mass + (1-hopfield_out_trap) / photon_mass)
    decay_out_trap = hopfield_out_trap / exciton_lifetime + (1-hopfield_out_trap) / photon_lifetime
    energy_out_trap = 0.5 * (exciton_energy + photon_energy - np.sqrt(rabi_splitting**2 + detuning**2))

    # print('Ex: %g, Ph: %g, X: %g, m: %g, decay: %g, E: %g' % (exciton_energy, photon_energy, hopfield_out_trap, mass_out_trap, decay_out_trap, energy_out_trap))
    # print(mass_in_trap, mass_out_trap)

    pot, kin, x = single_trap(energy_in_trap - energy_out_trap - 1j * (decay_in_trap - decay_out_trap), width, size,
                              energy_out_trap - 1j * decay_out_trap, [mass_out_trap, mass_in_trap], mass_landscape)
    pot = normalise_potential(pot)
    return pot, kin, x


def double_polariton_trap(trap_depth, width, separation=None, implant='photonic', photon_mass=1e-5, exciton_mass=0.35,
                          photon_lifetime=10, exciton_lifetime=1e5, detuning=0, rabi_splitting=10, size=101,
                          mass_landscape=True):
    """

    :param trap_depth:
    :param width:
    :param separation:
    :param implant:
    :param photon_mass:
    :param exciton_mass:
    :param cavity_lifetime:
    :param exciton_lifetime:
    :param detuning:
    :param rabi_splitting:
    :return:
    """
    exciton_energy = 0
    photon_energy = float(detuning)
    hopfield_in_trap = 0.5 * (1 + detuning/np.sqrt(detuning**2 + rabi_splitting**2))
    mass_in_trap = 1 / (hopfield_in_trap / exciton_mass + (1-hopfield_in_trap) / photon_mass)
    decay_in_trap = hopfield_in_trap / exciton_lifetime + (1-hopfield_in_trap) / photon_lifetime
    energy_in_trap = 0.5 * (exciton_energy + photon_energy - np.sqrt(rabi_splitting**2 + detuning**2))  # arbitrarily setting the exciton energy to 0

    # print('Ex: %g, Ph: %g, X: %g, m: %g, decay: %g, E: %g' % (exciton_energy, photon_energy, hopfield_in_trap, mass_in_trap, decay_in_trap, energy_in_trap))

    if implant == 'photonic':
        photon_energy = detuning + trap_depth
    elif implant == 'excitonic':
        exciton_energy = trap_depth
    else:
        raise ValueError
    detuning = photon_energy - exciton_energy

    hopfield_out_trap = 0.5 * (1 + detuning/np.sqrt(detuning**2 + rabi_splitting**2))
    mass_out_trap = 1 / (hopfield_out_trap / exciton_mass + (1-hopfield_out_trap) / photon_mass)
    decay_out_trap = hopfield_out_trap / exciton_lifetime + (1-hopfield_out_trap) / photon_lifetime
    energy_out_trap = 0.5 * (exciton_energy + photon_energy - np.sqrt(rabi_splitting**2 + detuning**2))

    # print('Ex: %g, Ph: %g, X: %g, m: %g, decay: %g, E: %g' % (exciton_energy, photon_energy, hopfield_out_trap, mass_out_trap, decay_out_trap, energy_out_trap))
    # print(mass_in_trap, mass_out_trap)

    pot, kin, x = double_trap(energy_in_trap - energy_out_trap - 1j * (decay_in_trap - decay_out_trap), 0, width,
                              separation, size, energy_out_trap - 1j * decay_out_trap, [mass_out_trap, mass_in_trap],
                              mass_landscape)
    pot = normalise_potential(pot)
    return pot, kin, x


def normalise_potential(matrix):
    if len(matrix.shape) > 1:
        pot = np.diag(matrix)
    else:
        pot = np.asarray(matrix)
    assert pot[0] == pot[-1]
    matrix_return = pot - pot[0].real  # Making the edges zero energy
    if len(matrix.shape) > 1:
        matrix_return = np.diag(matrix_return)
    return matrix_return


def solve(matrix, sort=True, lifetime=False):
    size = matrix.shape[0]
    vals, vecs = np.linalg.eig(matrix)
    vecs = vecs.T
    if sort:
        idxs = np.argsort(vals.real)
        vals = vals[idxs][:int(size/2)]
        vecs = vecs[idxs][:int(size/2)]
        if lifetime:
            if not any(vals.real < 0):
                print('No bound states. Sorting by lifetime')
                idxs = np.argsort(vals.imag)[::-1]
                vals = vals[idxs]
                vecs = vecs[idxs]
            # return np.full(vals.shape, np.nan), np.full(vecs.shape, np.nan)
    return vals, vecs #np.sign(np.real(vals)) * np.sqrt(np.abs(vals)), vecs


def transform_fourier(matrix, scaling=1, direction='forward'):
    assert direction in ['forward', 'backward']
    N = matrix.shape[0]
    forward = np.array([[np.exp(-1j * 2 * np.pi * scaling * n * k / N) for n in range(N)] for k in range(N)])
    backward = np.linalg.inv(forward)
    if direction == 'forward':
        return np.matmul(np.matmul(backward, matrix), forward)
    elif direction == 'backward':
        return np.matmul(np.matmul(forward, matrix), backward)


def plot(pot, kin, x, bound_modes=True, couplings=True):
    vals, vecs = solve(pot+kin)
    if bound_modes:
        n_bound_modes = int(np.sum(vals.real < 0))
        print("Bound modes: %d" % n_bound_modes)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)
        _gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0])
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, _gs[0], hspace=0.01)
        gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, _gs[1], hspace=0.01)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[1], hspace=0.02)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[2], hspace=0.01, wspace=0.01)
        _poss = gs.get_grid_positions(fig)
        fig.text(np.mean([_poss[2][-1], _poss[3][-1]]), _poss[1], 'Mode profiles', ha='center')
    else:
        n_bound_modes = int(np.sum(vals.real < 0))
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 2)
        _gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0])
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, _gs[0], hspace=0.01)
        gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, _gs[1], hspace=0.01)
        # gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[1], hspace=0.02)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[1], hspace=0.01, wspace=0.01)

    ax0 = plt.subplot(gs00[0])
    ax0.plot(vals.real, '.-')
    ax0.set_xticklabels([])
    ax0.set_ylabel('Eigen energy')
    ax1 = plt.subplot(gs00[1], sharex=ax0)
    ax1.plot(vals.imag, '.-')
    ax1.set_xlabel('Eigen index')
    ax1.set_ylabel('Eigen lifetime')

    ax2 = plt.subplot(gs01[0])
    ax2.plot(x, np.real(np.diag(pot)), 'b')
    ax2.set_xticklabels([])
    ax2.set_ylabel('Energy')
    ax3 = plt.subplot(gs01[1], sharex=ax2)
    ax3.plot(x, np.imag(np.diag(pot)), 'r')
    ax3.set_xlabel('Position / um')
    ax3.set_ylabel('Lifetime')

    if bound_modes:
        ax01 = plt.subplot(gs1[0])
        ax01.plot(vals.real[:n_bound_modes+2], '.-')
        ax01.set_xticklabels([])
        ax01.set_ylabel('Eigen energy')
        ax01.set_title("%d bound modes" % n_bound_modes)
        ax11 = plt.subplot(gs1[1])
        ax11.plot(vals.imag[:n_bound_modes+2], '.-')
        ax11.set_xlabel('Eigen index')
        ax11.set_ylabel('Eigen energy')
    for idx in range(4):
        ax = plt.subplot(gs2[idx])
        ax.plot(x, vecs[idx])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if idx % 2 == 0:
            ax.set_ylabel('Field')
        if idx >= 2:
            ax.set_xlabel('Position')
    if couplings:
        coupling = vals[1] - vals[0]
        fig.suptitle("Coupling: %g + 1j * %g meV (%g%%)" % (coupling.real, coupling.imag, 100*coupling.imag/coupling.real))
        # on_site = vals[1] + vals[0]
    return fig


def test_free_space():
    N = 1353
    MASS = 3e-5  # electron_mass
    SPATIAL_SCALE = 0.03
    # x = np.arange(N) * SPATIAL_SCALE
    k_range = np.linspace(0, 1, N//2) * np.pi / (2*SPATIAL_SCALE)

    kin = kinetic_matrix(N, MASS, SPATIAL_SCALE)
    vals, vecs = solve(kin)
    analytical = (hbar*k_range)**2 / (2*MASS*electron_mass)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(vals, label='numerical')
    axs[0].plot(analytical, '-.', label='analytical')
    axs[0].legend()
    axs[1].plot(analytical, vals)


def test_infinite_potential_well():
    N = 501
    MASS = 3e-5
    WIDTH = 20
    DEPTH = -20

    def infinite_potential_well(mass, width):
        return np.array([n**2 * np.pi**2 * hbar**2 / (2*mass*width**2) for n in range(1, 11)])

    pot, kin, x = single_trap(DEPTH, WIDTH, N, 0, MASS)
    vals, vecs = solve(pot + kin)
    vals2, _ = solve(transform_fourier(pot + kin))
    analytical = infinite_potential_well(MASS*electron_mass, WIDTH)+DEPTH

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(x, np.diag(pot).real)
    axs[0].hlines(analytical, WIDTH/2, 3*WIDTH/2, colors='k', label='analytical')
    axs[0].hlines(vals[:10], -3*WIDTH/2, -WIDTH/2, colors='r', label='numerical')
    axs[0].hlines(vals2[:10], -5*WIDTH/2, -3*WIDTH/2, colors='b', label='numerical')
    axs[0].legend(loc='upper right')
    axs[1].plot(analytical, vals[:10], '.')
    axs[1].plot(analytical, analytical, '--', alpha=0.5)


def test_harmonic_potential():
    SIZE = 501
    MASS = 3e-5
    OMEGA0 = 4
    XRANGE = 5e1

    def harmonic_potential_energies(omega0):
        return np.array([(2*n + 1) * hbar*omega0/2 for n in range(10)])

    def harmonic_potential(omega0, mass, size=1001, x_range=10):
        x = np.linspace(-x_range/2, x_range/2, size)
        return np.diag(mass * electron_mass * (omega0 * x)**2 / 2), x

    pot, x = harmonic_potential(OMEGA0, MASS, SIZE, XRANGE)
    kin = kinetic_matrix(SIZE, MASS, np.diff(x)[0])
    vals, vecs = solve(pot + kin)
    vals2, vecs2 = solve(transform_fourier(pot + kin))

    analytical = harmonic_potential_energies(OMEGA0)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(x, np.diag(pot).real)
    ax2 = axs[0].twinx()
    ax2.hlines(analytical, -XRANGE/4, 0, colors='r', label='analytical')
    ax2.hlines(vals[:10], 0, XRANGE/4, colors='k', label='numerical')
    ax2.hlines(vals2[:10], XRANGE/4, XRANGE/2, colors='b', label='numerical_FFT')
    ax2.legend()
    axs[1].plot(analytical, vals[:10], '.')
    axs[1].plot(analytical, analytical, '--', alpha=0.5)


def test_momentum_space():
    x_scaling = 0.28
    mass = 0.035
    N = 101
    index = np.linspace(0, 1, N//2) * (hbar / (2*mass*electron_mass))
    k = np.arange(N)*np.pi/(x_scaling*N)
    kinetic_kmatrix = hbar * np.diag((1*k)**2) / (2*mass*electron_mass)
    kinetic_xmatrix = kinetic_matrix(N, mass, x_scaling)

    vals, vecs = solve(kinetic_xmatrix)
    vals2, vecs2 = solve(kinetic_kmatrix)
    vals3, vecs3 = solve(transform_fourier(kinetic_xmatrix))
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(vals, 'k')
    axs[0].plot(vals2, '--')
    axs[0].plot(vals3, '-.')
    axs[1].plot(np.sqrt(vals), 'k')
    axs[1].plot(np.sqrt(vals2), '--')
    axs[1].plot(np.sqrt(vals3), '-.')
    # axs[1].plot(index, 'k')
    axs[2].plot(vals/vals2)
    axs[2].plot(vals/vals3)

    # Test:
    #     Solve momentum space on the square potential and the quadratic potential
    #     Build the two mode polariton matrix and plot the dispersions for free space
    # Sinusoidal potential
    #     Can we see smaller splittings?


def polaritons():
    def coupling_hamiltonian(k, mass_x=1, mass_p=1e-5, detuning=0, rabi=1):
        return np.array([[(hbar*k)**2 / (2*mass_x)+detuning, rabi], [rabi, (hbar*k)**2 / (2*mass_p)]])

    krange = np.linspace(-1e-2, 1e-2, 101)
    energies = np.array([np.sort(np.linalg.eig(coupling_hamiltonian(k))[0]) for k in krange])
    [plt.plot(e) for e in energies.transpose()]

    SIZE = 501
    exciton_mass = 10
    photon_mass = 1
    rabi = 10
    exciton_hamiltonian = transform_fourier(kinetic_matrix(SIZE, exciton_mass))
    photon_hamiltonian = transform_fourier(kinetic_matrix(SIZE, photon_mass))
    rabi_hamiltonian = np.full((SIZE, SIZE), rabi)
    polariton_hamiltonian = np.block([[exciton_hamiltonian, rabi_hamiltonian],
                                      [rabi_hamiltonian, photon_hamiltonian]])
    vals, _ = np.linalg.eig(polariton_hamiltonian)
    plt.semilogy(np.sort(np.abs(vals)))


if __name__ == '__main__':
    path = '/Users/Yago/Desktop/Meetings/'

    aspect = 21*3/(17*7)
    figsize=8
    fig, axs = plt.subplots(3, 7, True, True, gridspec_kw=dict(wspace=0.01, hspace=0.01), figsize=(figsize, figsize*aspect))
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(trapped[idx, :, :, 0, 1].real)

    fig = plt.figure(figsize=(9.5, 7))
    gs = gridspec.GridSpec(3, 3, wspace=0.1)
    axes = [('Well width', 'Well depth'), ('Well separation', 'Well depth'), ('Well separation', 'Well width')]
    axes2 = ['Well separation\n0.5-3.5um', 'Well width\n1-3um', 'Well depth\n1-20meV']
    axes3 = ['Real', 'Imag', 'Imag / Real\n+-16%%']
    for indx2 in range(3):
        for indx in range(3):
            gs2 = gridspec.GridSpecFromSubplotSpec(4, 5, gs[indx, indx2], hspace=0.01, wspace=0.01)
            for idx in range(20):
                ax = plt.subplot(gs2[idx])
                if indx == 0:
                    _mat = couplings[:, :, idx]
                elif indx == 1:
                    _mat = couplings[:, idx, :]
                elif indx == 2:
                    _mat = couplings[idx, :, :]
                if indx2 == 0:
                    matrix = _mat.real
                    val=None
                elif indx2 == 1:
                    matrix = _mat.imag
                    val=None
                elif indx2 == 2:
                    matrix = _mat.imag / _mat.real
                    val = 0.16
                if val is None:
                    val = np.max(np.abs([np.nanmax(matrix), np.nanmin(matrix)]))
                ax.imshow(matrix, vmin=-val, vmax=val, cmap='RdBu')
                ax.contour(_mat.real, [0.1, 0.2], linewidths=0.3, cmap='Greys_r', vmin=0.2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            _poss = gs2.get_grid_positions(fig)
            fig.text(np.mean(_poss[2:]), np.min(_poss[:2]), axes[indx][0], ha='center', va='top')
            fig.text(np.min(_poss[2:]), np.mean(_poss[:2]), axes[indx][1], ha='right', va='center', rotation=90)
        _poss = gs.get_grid_positions(fig)
        fig.text(np.min(_poss[2:])-0.06, np.mean([_poss[0][indx2], _poss[1][indx2]]), axes2[indx2], rotation=90, va='center', weight='bold')
        fig.text(np.mean([_poss[2][indx2], _poss[3][indx2]]), np.max(_poss[:2])+0.04, axes3[indx2], ha='center', va='top', weight='bold')




    indxs = np.argwhere(couplings.real>0.1)

    plot(*single_trap(-10, 10, size=501, bkg_value=-hbar * 0.1j))
    plot(*single_trap(-10, 1, size=501, bkg_value=-hbar * 0.1j))

    plot(*double_polariton_trap(10, implant='excitonic', width=2, separation=1, size=501))
    plot(*double_polariton_trap(10, implant='photonic', width=2, separation=1, size=501))

    plot(*double_trap(-10+hbar*0.1j, 0, 1, separation=1, size=501, mass=1e-4, bkg_value=-hbar*0.1j))
    plot(*double_trap(-10, -hbar*0.1j, 1, separation=1, size=501, mass=1e-4, bkg_value=hbar*1j))
    plot(*double_trap(-10, -hbar*0.01j, 1, separation=1, size=501, mass=1e-4, bkg_value=hbar*0.001j))
    plot(*double_trap(-1, -hbar*0.1j, 1, separation=1, size=501, mass=1e-4, bkg_value=hbar*0.001j))
    plot(*double_trap(-1, -hbar*0.1j, 2, separation=1, size=501, mass=1e-4, bkg_value=hbar*0.001j))
    plot(*double_trap(-1, -hbar*0.1j, 2, separation=1.5, size=501, mass=1e-4, bkg_value=hbar*0.001j))
    plot(*double_trap(-1, -hbar*0.1j, 2, separation=2, size=501, mass=1e-4, bkg_value=hbar*0.001j))


    def double_trap_tight_binding(depth, barrier, width, separation=None, size=101, v_0=0, mass=1):
        pot, kin = double_trap(depth, barrier, width, separation, size, v_0, mass)
        vals, vecs = solve(pot + kin)

        coupling = vals[1] - vals[0]
        on_site = vals[1] + vals[0]
        return coupling, on_site
    print(double_trap_tight_binding(-10, 0, 1, size=501, v_0=1j))
    print(double_trap_tight_binding(-10, 0-1j, 1, size=501, v_0=1j))

    barriers = np.linspace(-9, 9, 101)
    couplings = []
    on_sites = []
    for barrier in barriers:
        coupling, on_site = double_trap_tight_binding(-10, barrier, 1)
        couplings += [coupling]
        on_sites += [on_site]
    fig, axs = plt.subplots(1, 2, True)
    axs[0].semilogy(barriers, couplings, '.-')
    axs[0].set_ylabel('Coupling')
    axs[1].plot(barriers, on_sites)
    axs[1].set_ylabel('On site')
    [ax.set_xlabel('Barrier') for ax in axs]
    fig.savefig(path + 'tigh_binding_barrier.png', dpi=1200)

    separations = np.linspace(0.1, 10, 101)
    couplings = []
    on_sites = []
    for separation in separations:
        coupling, on_site = double_trap_tight_binding(-10, 0, 1, separation, size=1001)
        couplings += [coupling]
        on_sites += [on_site]
    fig, axs = plt.subplots(1, 2, True)
    axs[0].semilogy(separations, couplings, '.-')
    axs[0].set_ylabel('Coupling')
    axs[1].plot(separations, on_sites)
    axs[1].set_ylabel('On site')
    [ax.set_xlabel('Separation') for ax in axs]
    fig.savefig(path + 'tigh_binding_separation.png', dpi=1200)

    pot, kin = single_trap(1, 0.1, 10, 501)
    pot, kin = double_trap(-10, 0, 1, size=501)
    vals, vecs = solve(pot + kin)
    print(vals[1] - vals[0])
    pot, kin = double_trap(-10, -9, 1, size=501)
    vals, vecs = solve(pot + kin)
    print(vals[1] - vals[0])
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 2)
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, gs[0])
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[1])
    ax0 = plt.subplot(gs0[0])
    ax0.plot(vals.real, '.-')
    ax1 = plt.subplot(gs0[1])
    ax1.plot(vals.imag, '.-')
    ax2 = plt.subplot(gs0[2])
    ax2.plot(np.real(np.diag(pot)))
    ax2.plot(np.imag(np.diag(pot)))
    for idx in range(4):
        ax = plt.subplot(gs1[idx])
        ax.plot(vecs[idx])
