#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:11:30 2022

@author: Simoneunit
"""
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path
from scipy import sparse
import logging
import sys
import itertools
from dxf.lattices import *

LOGGER = logging.getLogger('Lattices')
LOGGER.propagate = False
LOGGER.setLevel('DEBUG')
fh = logging.StreamHandler(sys.stdout)
f = logging.Formatter('[%(name)s] - %(levelname)s: %(message)s - %(asctime)s ', '%H:%M')
fh.setFormatter(f)
LOGGER.addHandler(fh)

# PRECISION = 6
hbar = 0.658  # in meV ps
electron_mass = 5.68e3  # in meV ps2 um-2
mass = 1e-5
size = 250
region = 35
x_spacing=2*region/(size-1)

# x,y = np.meshgrid(np.linspace(-10, region, size, dtype = float),
                  # np.linspace(-10,region, size, dtype = float))

_x = np.linspace(-region, region, size, dtype = float)
_y = np.linspace(-region,region, size, dtype = float)
x, y = np.meshgrid(_x, _y)


# def dxf_circle(center, radius, layer=1):
#     """Makes a circle string
#     :param center:
#     :param radius:
#     :param layer:
#     :return:
#     """
#     circle = ("0\nCIRCLE\n8\n%d\n10\n%." + str(PRECISION) + "f\n20\n%." + str(PRECISION) + "f\n40\n%." + str(PRECISION) + "f\n") % (layer, center[0], center[1], radius)
#     return circle

# def points_to_dxf(centers, trap_size, layer=1):
#     """
#     :param centers:
#     :param trap_size:
#     :param layer:
#     :return:
#     """
#     string = ''
#     for center in centers:
#         string += dxf_circle(center, trap_size, layer)
#     return string

# def general_lattice(unit_vectors, n_sites=(10, 10), site_distance=1., offset=(0, 0)):
#     """Returns a tuple of (x, y) lattice positions
#     Can be used for making photonic crystal lattices, but also for making lattices for parameter variations
#     :param unit_vectors: either one or two unit vectors depending on whether you want a row or a lattice
#     :param n_sites: either an integer or a two-tuple of integers
#     :param site_distance:
#     :param offset:
#     :return:
#     """
#     LOGGER.debug('Running general_lattice: %s, %s, %s, %s' % (unit_vectors, n_sites, site_distance, offset))
#     assert len(offset) == 2
#     unit_vectors = np.array(unit_vectors, np.float)
#     if len(unit_vectors.shape) == 1:
#         unit_vectors = np.array([unit_vectors, [0, 0]])
#         assert type(n_sites) == int
#         n_sites = (n_sites, 1)
#     elif type(n_sites) == int:
#         n_sites = (n_sites, n_sites)
#     unit_vectors /= np.linalg.norm(unit_vectors, None, -1)  # normalising the unit vectors

#     centers = ()
#     for nx in range(n_sites[0]):
#         for ny in range(n_sites[1]):
#             center_x = site_distance * (nx * unit_vectors[0][0] + ny * unit_vectors[1][0]) - offset[0]
#             center_y = site_distance * (nx * unit_vectors[0][1] + ny * unit_vectors[1][1]) - offset[1]
#             centers += ((center_x, center_y), )
#     return np.array(centers, np.float)

# def unit_cell_lattice(unit_cell, unit_vectors, n_sites=(10, 10), site_distance=1.):
#     """Returns a set of lattice positions
#     :param unit_cell:
#     :param unit_vectors:
#     :param n_sites:
#     :param site_distance:
#     :return:
#     """
#     LOGGER.debug('Running unit_cell_lattice: %s, %s, %s, %s' % (unit_cell, unit_vectors, n_sites, site_distance))
#     unit_centers = general_lattice(unit_vectors, n_sites, site_distance)
#     unit_cell = np.array(unit_cell) * site_distance
#     centers = ()
#     for center in unit_centers:
#         for uc in unit_cell:
#             cx = center[0] + uc[0]
#             cy = center[1] + uc[1]
#             centers += ([cx, cy], )
#     return np.array(centers, np.float)

# def lattice(unit_cell, unit_vectors, site_distance=1., trap_size=0.2, cropping_vertices=None, offset=(0, 0),
#             center=(0, 0), output='string'):
#     """Utility function to creates a cropped lattice
#     :param unit_cell: positions within the unit cell
#     :param unit_vectors: two vectors defining the lattice axis
#     :param site_distance: separation between adjacent unit cells
#     :param trap_size: size of lattice sites
#     :param cropping_vertices: corners of the polygon to contain the lattice
#     :param offset: vector defining an offset between the lattice and the cropping polygon. Useful for creating specific
#                    lattice edges
#     :param center: starting position of the lattice
#     :param output: if 'string' the return is a DXF string. Otherwise, it is a list of lattice sites
#     :return:
#     """
#     LOGGER.debug('Running lattice: %s, %s, %s, %s, %s, %s, %s, %s' % (unit_cell, unit_vectors, site_distance, trap_size, cropping_vertices, offset, center, output))
#     unit_vectors = np.array(unit_vectors, np.float)
#     if cropping_vertices is None:
#         cropping_vertices = ([0, 0],
#                              10 * site_distance * unit_vectors[0],
#                              10 * site_distance * (unit_vectors[0] + unit_vectors[1]),
#                              10 * site_distance * unit_vectors[1])
#     cropping_vertices = np.array(cropping_vertices, np.float)
#     cropping_centroid = np.mean(cropping_vertices, 0)
#     cropping_radius = np.sqrt(np.max(np.sum(np.abs(cropping_vertices - cropping_centroid)**2, 1)))

#     unit_cell_polygon = Polygon([[0, 0], unit_vectors[0] * site_distance,
#                                  (unit_vectors[0] + unit_vectors[1]) * site_distance, unit_vectors[1] * site_distance])
#     unit_cell_radius = unit_cell_polygon.boundary.distance(unit_cell_polygon.centroid)

#     n = int(np.ceil(cropping_radius / unit_cell_radius))

#     offset = (n * np.array(unit_cell_polygon.centroid) - cropping_centroid) + np.array(offset)
#     all_centers = unit_cell_lattice(unit_cell, unit_vectors, n, site_distance)
#     all_centers_offset = all_centers - offset
#     cropped_centers = cropped_positions(all_centers_offset, cropping_vertices)
#     cropped_centers_offset = cropped_centers + np.array(center)

#     if output == 'string':
#         return points_to_dxf(cropped_centers_offset, trap_size)
#     else:
#         return cropped_centers_offset

# def cropped_positions(old_centers, cropping_vertices, accuracy=None):
#     """Crops positions to those contained within a polygon
#     :param old_centers: original lattice positions
#     :param cropping_vertices: vertices of the cropping polygon
#     :param accuracy: the radius of acceptance to determine whether a point is in the polygon or not
#     :return: old_centers contained within the cropping_vertices
#     """
#     old_centers = np.array(old_centers, np.float)
#     if accuracy is None:
#         # by default, set the acceptance as half of the minimum distance between any two points
#         pairs = np.array(list(itertools.combinations(old_centers, 2)))
#         distances = np.sum(np.diff(pairs, axis=1) ** 2, -1)
#         accuracy = np.min([np.min(distances) / 10., 1e-5])
#     polygon = matplotlib.path.Path(cropping_vertices)
#     centers = old_centers[polygon.contains_points(old_centers, radius=accuracy)]
#     return centers


# def square_lattice(site_distance=1., trap_size=0.2, cropping_vertices=None, offset=(0, 0), center=(0, 0), output='string'):
#     return lattice(([0, 0], ), ([1, 0], [0, 1]), site_distance, trap_size, cropping_vertices, offset, center, output)

# def kagome_lattice(site_distance=1., trap_size=0.2, cropping_vertices=None, offset=(0, 0), center=(0, 0), asymmetry=0, output='string'):
#     return lattice((1-asymmetry) * np.array(([0, 0], [0.5, 0], [np.cos(np.pi / 3) / 2, np.sin(np.pi / 3) / 2])),
#                    ([1, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)]),
#                    site_distance, trap_size, cropping_vertices, offset, center, output)

# def honeycomb_lattice(site_distance=1., trap_size=0.2, cropping_vertices=None, offset=(0, 0), center=(0, 0), output='string'):
#     return lattice(([0, 0], [0, 1/np.sqrt(3)]), ([1, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)]),
#                    site_distance, trap_size, cropping_vertices, offset, center, output)

# def apply_strain(strain_function, strain_strength, positions, center=None):
#     if center is None:
#         center = np.mean(positions, 0)
#     centered = positions - center
#     strained = strain_function(centered, strain_strength)
#     return strained + center

# def radial_strain(cartesian_positions, strain_strength):
#     """Following the Rechstmann paper
#     :param cartesian_positions:
#     :param strain_strength:
#     :return:
#     """
#     radial_positions = [np.sqrt(cartesian_positions[:, 0]**2 + cartesian_positions[:, 1]**2),
#                         np.arctan2(cartesian_positions[:, 1], cartesian_positions[:, 0])]
#     strain = strain_strength * radial_positions[0]**2 * np.array([np.sin(3 * radial_positions[1]),
#                                                                   np.cos(3 * radial_positions[1])])
#     strained_positions = [radial_positions[0] + strain[0], radial_positions[1] + strain[1]]
#     return np.swapaxes([strained_positions[0] * np.cos(strained_positions[1]),
#                         strained_positions[0] * np.sin(strained_positions[1])], 0, 1)

# def strained_graphene(strain_strength, *args, **kwargs):
#     kwargs['output'] = 'points'
#     positions = honeycomb_lattice(*args, **kwargs)
#     strained = apply_strain(radial_strain, strain_strength, positions)
#     return strained

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



