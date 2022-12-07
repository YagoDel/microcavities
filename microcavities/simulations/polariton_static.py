#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *
from scipy.sparse.linalg import eigsh
from scipy import sparse
import logging
from dxf.lattices import *

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


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
    r = rabi * sparse.eye(size**2)
    return sparse.bmat([[t_photon, r], [r, t_exciton]])


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


def solve_polariton_static(potential_photon, potential_exciton, detuning, rabi, mass_photon=1e-5, mass_exciton=0.35,
                           axes=None, n_eigenvalues=250):
    if axes is None:
        axes = make_axes()  # 30 um square area
    size = potential_photon.shape[0]
    x_spacing = np.diff(axes[3])[0]
    if n_eigenvalues > 2 * (size**2): n_eigenvalues = 2*(size**2)-1

    potential = potential_matrix(potential_photon, potential_exciton, detuning)
    kinetic = kinetic_matrix(size, rabi, mass_photon, mass_exciton, x_spacing)

    # Compute eigenvalues and eigenvectors
    vals, vecs = eigsh(potential + kinetic, k=n_eigenvalues, which='SA')
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


"""POTENTIAL FUNCTIONS"""


def harmonic_potential(omega, mass, axes=None):
    if axes is None:
        axes = make_axes()  # 30 um square area
    potential = 0.5 * mass * (omega**2) * (axes[0]**2 + axes[1]**2)
    return potential, axes
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


def test_solver():
    """
    Need to compare the numerical solution to analytics:
        - Get the farfield pattern from realspace and compare it to the exact k-space solution
        - Compare the results for a small potential that is the same in both the exciton and the photon

    Currently the comparison with the square well does not give agreement, indicating there's a problem with the
    Hamiltonian and it feels like a missing factor of 2
    """

    def infinite_potential_well(mass, width):
        analytical_eigenvalues = np.array([[(nx**2+ny**2) * (np.pi * hbar)**2 / (2*mass*width**2)
                                            for nx in np.arange(0, 4)]
                                           for ny in np.arange(0, 4)]).flatten()
        return np.sort(analytical_eigenvalues)

    def polariton_mass(detuning, rabi_splitting, photon_mass=1e-5, exciton_mass=0.35):
        hopfield = 0.5 * (1 + detuning / np.sqrt(detuning ** 2 + rabi_splitting ** 2))
        return 1 / (hopfield / exciton_mass + (1 - hopfield) / photon_mass)

    axes = make_axes()  # 30 um square area
    vphot = np.full(axes[0].shape, 0)
    vex = np.full(axes[0].shape, 0)
    WIDTH = 10
    DETUNING = -20
    RABI = 0
    DEPTH = -2

    mask = np.logical_and(np.abs(axes[0]) < WIDTH/2, np.abs(axes[1]) < WIDTH/2)
    vphot[mask] = DEPTH
    vex[mask] = 0

    vals, vecs = solve_polariton_static(vphot, vex, DETUNING, RABI, mass_photon=1e-5, mass_exciton=0.35,
                                        axes=axes, n_eigenvalues=100)
    fig, axs = plot_eigenvectors(axes[2], axes[3], vecs, vals)
    axs[0].plot(infinite_potential_well(polariton_mass(DETUNING, RABI)*electron_mass, WIDTH)+np.min(vals))
    raise NotImplementedError
