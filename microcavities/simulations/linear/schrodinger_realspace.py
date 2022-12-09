#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:11:30 2022

@author: Simoneunit
"""

from microcavities.utils.plotting import *
from scipy.sparse.linalg import eigsh
from scipy import sparse
import logging
from dxf.lattices import *

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2


def make_axes(region, n_points):
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


def kinetic_matrix(mass, size, x_spacing=1):
    diag = np.ones([size])
    diags = np.array([diag, -2*diag, diag])
    # D is a sparse matrix which has -2 on the main diagonal and 1 on the two neighbouring diagonals
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), size, size)
    T = -1/2 * sparse.kronsum(D, D)
    return T * (hbar**2) / (mass*electron_mass*x_spacing**2)


def solve(V, mass, axes, n_eigenvalues=250):
    size = V.shape[0]
    x_spacing = np.diff(axes[3])[0]
    if n_eigenvalues > size**2: n_eigenvalues = (size**2)-1

    # Create a matrix for Kinetic energy
    T = kinetic_matrix(mass, size, x_spacing)
    # V is a 2D grid reshaped into N**2 vector to be stretched along main diagonal
    U = sparse.diags(V.reshape(size**2), 0)
    # Create the full Hamiltonian
    H = T + U   # N**2 x N**2 dimension

    # Compute eigenvalues and eigenvectors
    vals, vecs = eigsh(H, k=n_eigenvalues, which='SA')  # SM means we are looking for only the smallest eigenvalues
    return vals, vecs


def get_eigenvector(n, vecs, size=None):
    """
    :param n: int. Index selection of eigenvector
    :param vecs: array. List of eigenvectors
    :param size: int. To determine how to reshape the eigenvector from a list to a matrix
    :return:
    """
    if size is None: size = int(np.sqrt(vecs.shape[0]))
    return vecs.T[n].reshape((size, size))


def plot_eigenvectors(x, y, vecs):
   for n in range(0, 4):
       fig = plt.figure(figsize=(6,6))
       plt.contourf(x, y, get_eigenvector(n, vecs), 20)
       plt.colorbar(label="psi")#plot PDF with 20 countour levels by e()**2 with 20 countor levels
       return fig


"""POTENTIAL FUNCTIONS"""


def harmonic_potential(omega, mass, axes=None):
    if axes is None:
        axes = make_axes(30, 101)
    potential = 0.5 * mass * (omega**2) * (axes[0]**2 + axes[1]**2)
    return potential, axes
    # nx,ny = np.arange(0,4), np.arange(0,4)
    # ana_vals = hbar**2*(nx+ny+1)*omega**2/(2*mass)
    # return V


def single_circle(radius, depth, center=(0, 0), background=0, axes=None, potential=None):
    """Creates a potential with a single, circular well

    :param radius: float, in um. Radius of the well
    :param depth: float, in meV. Depth of the well
    :param center: tuple of floats, in um. Center point of the well
    :param background: float, in meV. Background potential value.
    :param axes: 4-tuple. See make_axes
    :param potential: array. Potential onto which add this single circle. If not given, it's assumed to be uniform with
    background value
    :return:
    """
    if axes is None:
        axes = make_axes(30, 101)
    if potential is None:
        potential = np.full(axes[0].shape, background)
    mask = ((axes[0]-center[0])**2 + (axes[1]-center[1])**2) < radius**2
    potential[mask] = depth + background
    return potential, axes


def double_circle(radius, depth, distance, center=(0, 0), background=0, axes=None):
    """Creates a double well

    :param radius:
    :param depth:
    :param distance:
    :param center:
    :param background:
    :param axes:
    :return:
    """
    if axes is None:
        axes = make_axes(30, 101)
    potential = np.full(axes[0].shape, background)
    center1 = (center[0] - distance/2, center[1])
    center2 = (center[0] + distance/2, center[1])

    potential = single_circle(radius, depth, center1, background, axes, potential)[0]
    potential = single_circle(radius, depth, center2, background, axes, potential)[0]
    return potential, axes


def potential_lattice(centers, radius, depth=-10, background=0, axes=None):
    if axes is None:
        axes = make_axes(30, 101)

    potential = np.full(axes[0].shape, background)
    for center in centers:
        potential = single_circle(radius, depth, center, background, axes, potential)[0]
    return potential, axes


"""TEST FUNCTIONS"""


def test_potential_circles():
    """Simply tests that the potentials look like how we expect them to look"""
    fig, axs = plt.subplots(1, 3)
    V, axes = single_circle(5, -10)
    imshow(V, axs[0], xaxis=axes[2], yaxis=axes[3], aspect='equal')

    centers = np.reshape([[(x, y) for x in np.linspace(-10, 10, 3)] for y in np.linspace(-10, 10, 3)], (9, 2))
    V, axes = potential_lattice(centers, 3, -10)
    imshow(V, axs[1], xaxis=axes[2], yaxis=axes[3], aspect='equal')

    V, axes = double_circle(3, -10, 4)
    imshow(V, axs[2], xaxis=axes[2], yaxis=axes[3], aspect='equal')


def test_solver():
    """Using the analytical formula for a QHO, makes sure the numerical solver is giving correct results"""
    axes = make_axes(30, 51)

    omega = 1e-3
    mass = 1e-2
    potential, axes = harmonic_potential(omega, mass * electron_mass, axes)

    analytical_eigenvalues = np.array([[hbar*omega*(nx+ny+1)
                                        for nx in np.arange(0, 4)]
                                       for ny in np.arange(0, 4)]).flatten()
    analytical_eigenvalues = np.sort(analytical_eigenvalues)

    numerical_eigenvalues, eigenvectors = solve(potential, mass, axes)

    # fig, axs = plt.subplots(2, 3)
    fig = figure(2)
    _gs = gridspec.GridSpec(2, 1, fig)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, _gs[0], width_ratios=[1, 2])
    axs0 = gs0.subplots()
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, _gs[1])
    axs1 = gs1.subplots()
    imshow(potential, axs0[0], xaxis=axes[2], yaxis=axes[3])
    axs0[1].plot(analytical_eigenvalues, 'k--', label='analytical')
    axs0[1].plot(numerical_eigenvalues[:len(analytical_eigenvalues)], 'r.-', label='numerical')
    axs0[1].legend()

    for idx in range(3):
        imshow(get_eigenvector(idx, eigenvectors), axs1[idx], xaxis=axes[2], yaxis=axes[3])


if __name__ == "__main__":
    
    V = potential_lattice(x, y, honeycomb_flaked, 0.8, depth=-50, bkg_value=10);
    imshow(V, xaxis=_x, yaxis=_y)
    vals,vecs = solve(size,mass,electron_mass,x_spacing,V)
    
    
    V = potential_circle(x, y, -2,0, 2, 0, 0.5)  
    plt.figure(); plt.imshow(V)   #plot 2D potential


    fig, axs = plt.subplots(1, 3, figsize=(9,9))
    for n in range(0,4):
        ax = axs[n-1]
        psi = get_e(n)
        pdf = get_e(n)**2
        ax.imshow(x,y,psi)
    
        
        
    for n in range(0,4):
        plt.figure(figsize=(6,6))
        plt.contourf(x, y, get_e(n), 20) 
        plt.colorbar(label = "psi")#plot PDF with 20 countour levels by e()**2 with 20 countor levels
    
    
    coupling=vals[1]-vals[0]
    
    plt.plot(vals,'-o')
    #plt.plot(ana_vals, '-v')
    plt.ylabel("$E/\hbar \omega$")
    
    
    
    # get the absolute path of the module.
    import os
    import sys
    curent_path = os.getcwd()
    module_path = curent_path[:-9]
    sys.path.append(module_path)
    from tbee.lattice import *
    from tbee.save import *
    # %matplotlib inline
    from math import cos, sin, sqrt
    from math import pi as PI
    
    unit_cell = [{'tag': b'a', 'r0': (0, 0)}]
    prim_vec = [(1., 0.), (0., 1.)]
    
    lat = lattice(unit_cell=unit_cell, prim_vec=prim_vec)
    sav = save(dir_name='square')
    
    n1, n2 = 8, 8
    lat.get_lattice(n1=n1, n2=n2)
    fig_lat = lat.plot(ms=15)
    sav.fig_lat(fig_lat, 'lattice')



