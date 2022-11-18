#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:11:30 2022

@author: Simone
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from scipy import sparse

hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2
mass = 1e-5
size = 150
region = 10

#Create a x-y space
x,y = np.meshgrid(np.linspace(-region, region, size, dtype = float),
                  np.linspace(-region,region, size, dtype = float))
x_spacing=2*region/(size-1)

def harmonic_potential(x,y):
    omega = 4
    V = 0.5*omega**2*(x**2+y**2)
    nx,ny = np.arange(0,4), np.arange(0,4)
    ana_vals = hbar**2*(nx+ny+1)*omega**2/(2*mass)
    return V

def potential_circle(x,y,x1,y1,x2,y2,r):
    V = np.zeros(x.shape)
    depth=-10
    bkg_value=0
    mask1 = (x-x1)**2+(y-y1)**2<r**2
    mask2 = (x-x2)**2+(y-y2)**2<r**2
    print(mask1)
    V[mask1] = depth + bkg_value
    V[mask2] = depth + bkg_value
    return V

V = potential_circle(x, y, 0,0,3, 0, 1)  
V = harmonic_potential(x,y)

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
    vals, vecs = eigsh(H, k=10, which='SM')  #SM means we are looking for only the smallest eigenvalues
    return vals,vecs

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
vals, vecs = eigsh(H, k=10, which='SM')  #SM means we are looking for only the smallest eigenvalues


def get_e(n): 
    return vecs.T[n].reshape((size,size))

plt.figure(); plt.imshow(V)         #plot 2D potential

plt.figure(figsize=(9,9))
plt.contourf(x, y, get_e(0), 20)   #plot PDF with 20 countour levels by e()**2 with 20 countor levels
plt.colorbar(label = "psi")

coupling=vals[1]-vals[0]

plt.plot(vals,'-o')
#plt.plot(ana_vals, '-v')
plt.ylabel("$E/\hbar \omega$")



from types import SimpleNamespace
def make_2d_box(N):
    L = 1
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    # The second order derivative D2
    # (for simplicity we have dx=dy and so D2 is the same for x and y)
    dx = np.diff(x)[0]
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2
    One = sparse.eye(N)

    D2_x_2d = sparse.kron(D2, One)
    D2_y_2d = sparse.kron(One, D2)

    hbar = 0.658
    mass = 1e-5
    T = -(hbar**2 / (2 * mass)) * (D2_x_2d + D2_y_2d)
    V = 0
    H = T + V
    return SimpleNamespace(**locals())
def calc_box_2d(box):
    # Takes about 1 minute on a mac book pro 2019, 2.6 GHz 6-Core, for N=300
    eigs, eigv = sparse.linalg.eigsh(box.H, k=10, which='SM')
    return SimpleNamespace(**locals())

box_2d = make_2d_box(N=150)
res_box_2d = calc_box_2d(box_2d)
nx, ny = np.meshgrid(np.arange(1, 5), np.arange(1, 5))
#Compute analytical eigenvalues
ana_eigs = np.sort(((box_2d.hbar**2 / (2*box_2d.mass)) * (np.pi**2/box_2d.L**2) * (nx**2 + ny**2)).flatten())[0:10]

fig, axs = plt.subplots(1, 4, figsize=(16,6))
for n in range(1, 5):
    ax = axs[n-1]
    psi = res_box_2d.eigv.T[n-1]  
    psi_n = np.reshape(psi, (box_2d.N, box_2d.N))
    ax.set_title("n = %d" % (n-1))
    ax.imshow(np.abs(psi_n), origin='lower')
    ax.set_xticks([0, len(box_2d.x)//4, len(box_2d.x)//2, (3*len(box_2d.x))//4, len(box_2d.x)])
    ax.set_xticklabels(['0', '', '', '', 'L'])
    ax.set_yticks([0, len(box_2d.y)//4, len(box_2d.y)//2, (3*len(box_2d.y))//4, len(box_2d.y)])
    ax.set_yticklabels(['', '', '', '', 'L'])
    
plt.figure(figsize=(7,5))
plt.plot(res_box_2d.eigs, 'o-', color = 'r',  label='numerical')
plt.plot(ana_eigs, 'o', label='analytical')
plt.ylabel("$E$")
plt.legend();






