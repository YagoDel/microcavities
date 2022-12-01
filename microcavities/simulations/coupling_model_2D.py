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
mass = 1e-5
size = 250
region = 35
x_spacing = 2*region/(size-1)

_x = np.linspace(-region, region, size, dtype = float)
_y = np.linspace(-region,region, size, dtype = float)
x, y = np.meshgrid(_x, _y)


def harmonic_potential(x,y):
    omega = 4
    V = 0.5*omega**2*(x**2+y**2)
    nx,ny = np.arange(0,4), np.arange(0,4)
    ana_vals = hbar**2*(nx+ny+1)*omega**2/(2*mass)
    return V


def potential_circle(x,y,x1,y1,x2,y2,r):
    #Create a x-y space
    V = np.zeros(x.shape)
    depth=-10
    bkg_value=0
    mask1 = (x-x1)**2+(y-y1)**2<r**2
    mask2 = (x-x2)**2+(y-y2)**2<r**2
    print(mask1)
    V[mask1] = depth + bkg_value
    V[mask2] = depth + bkg_value
    return V


def potential_lattice(x, y, centers, r, depth=-10, bkg_value=0):
    V = np.zeros(x.shape) + bkg_value
    for cx, cy in centers:
        mask = (x-cx)**2+(y-cy)**2<r**2
        V[mask] = depth + bkg_value  
    return V


def potential_square_latt(x,y,xc,yc,r):
    
    # for _x, _y in zip(xc, yc):
    
    V = np.zeros(x.shape)
    depth=-10
    bkg_value=0
    xc = square_flaked[:,0]
    yc = square_flaked[:,1]
    mask = (x- xc)**2+(y-xy)**2<r**2
    print(mask)
    V[mask] = depth + bkg_value
    return V


def solve(size,mass,electron_mass,x_spacing,V):
    diag = np.ones([size])
    diags = np.array([diag, -2*diag, diag])
    #D is a sparse matrix which has -2 on the main diagonal and 1 on the two neighbouring diagonals
    D = sparse.spdiags(diags, np.array([-1,0,1]), size, size)  
    T = -1/2 * sparse.kronsum(D,D)
    T_dim = T*hbar**2/ (mass*electron_mass*x_spacing**2)
    #V is a 2D grid reshaped into N**2 vector to be stretched along main diagonal
    U = sparse.diags(V.reshape(size**2), (0))
    H = T_dim+U   #N**2 x N**2 dimension   
    #compute eigenvalues and eigenvectors
    vals, vecs = eigsh(H, k=250, which='SA')  #SM means we are looking for only the smallest eigenvalues
    return vals,vecs


def get_e(n): 
    return vecs.T[n].reshape((size,size))


def plot(x,y):
   for n in range(0,4):
       fig = plt.figure(figsize=(6,6))
       plt.contourf(x, y, get_e(n), 20) 
       plt.colorbar(label = "psi")#plot PDF with 20 countour levels by e()**2 with 20 countor levels
       return fig


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



