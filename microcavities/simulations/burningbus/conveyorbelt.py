# -*- coding: utf-8 -*-

from microcavities.simulations.quantum_box import kinetic_matrix, normalise_potential, plot, solve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import find_peaks
import h5py


def sinusoid(depth, period, periods=5, size=101, bkg_value=0, mass=1e-3):
    x = np.linspace(-periods*period, periods*period, size)
    potential = np.asarray(depth * np.cos(x * 2*np.pi / period) + bkg_value, dtype=np.complex)
    mass_array = np.ones(size) * mass
    return np.diag(potential), kinetic_matrix(size, mass_array, np.diff(x)[0]), x

plot(*sinusoid(2, 1, 10, 101), bound_modes=False)
plot(*sinusoid(2, 1, 10, 101), bound_modes=True)


pot, kin, x = sinusoid(2, 1, 10, 101)


def find_bound_modes(pot, kin, *args):
    # Should be groups of N traps
    n_traps = len(find_peaks(-np.diag(np.real(pot))))
    vals, vecs = solve(pot + kin)
    band_edges, _props = find_peaks(np.diff(np.real(vals)), width=[0.1, 2])
    unbound = np.argmin(np.abs(vals - pot.max()))
    print(band_edges, unbound, n_traps)
    band_edges = band_edges[band_edges < unbound]
    print(band_edges)
    assert all([not (_x+1) % n_traps for _x in band_edges])
    band_centers = []
    band_widths = []
    band_gaps = []
    old_values = None
    for idx in range(len(band_edges)):
        if idx == 0:
            values = vals[0:band_edges[0]+1]
        else:
            values = vals[band_edges[idx-1]+1:band_edges[idx]+1]
            band_gaps += [values.min() - old_values.max()]
        band_centers += [np.mean(values)]
        band_widths += [np.max(values) - np.min(values)]
        old_values = np.copy(values)
    return np.array(band_centers), np.array(band_widths), np.array(band_gaps)


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


print('a: ', find_bound_modes(*sinusoid(2, 1, 10, 101)))
print('b: ', find_bound_modes(*sinusoid(2, 1, 10, 1001)))
print('c: ', find_bound_modes(*sinusoid(2, 1, 10, 2001)))


# The data analysis says that the polaritons have a mass of 3e-5 (2021_03_29)

depths = np.linspace(0.1, 10.1, 51)
periods = np.linspace(0.3, 20.3, 41)
values = []
for depth in depths:
    _values = []
    for period in periods:
        pot, kin, x = sinusoid(depth, period, 10, 1001, mass=3e-5)
        vals, _ = solve(pot + kin)
        _values += [vals]
    values += [_values]
values = np.array(values)
with h5py.File('/Users/yago/Documents/github/microcavities/microcavities/simulations/burningbus/simulations_2021_04_30.h5', 'a') as dfile:
    dset = dfile.create_dataset('eigen_energies', data=values)
    dset.attrs.create('depth', depths)
    dset.attrs.create('period', periods)
with h5py.File('/Users/yago/Documents/github/microcavities/microcavities/simulations/burningbus/simulations_2021_04_30.h5', 'r') as dfile:
    dset = dfile['eigen_energies']
    values = dset[...]
    depths = dset.attrs['depth']
    periods = dset.attrs['period']

gs = np.real(np.mean(values[..., :20], -1))
gs[gs > 0] = np.nan
split = np.real(np.mean(values[..., 20:40], -1)) - gs

fig, axs = plt.subplots(1, 2, True, True)
axs[0].imshow(gs)
axs[1].imshow(split)
contour1 = axs[0].contour(gs, [-2])
contour2 = axs[1].contour(split, [1])

x, y = [np.arange(-_x, _x) for _x in [25, 40]]
X, Y = np.meshgrid(x, y)
img1 = X**2 - Y**2
img2 = X**2 + Y**2
contour1 = plt.contour(img1, [-2, 20], colors='r')
contour2 = plt.contour(img2, [50, 100, 200], colors='k')
line1 = LineString(contour1.collections[0].get_paths()[0].interpolated(1).vertices)
line2 = LineString(contour2.collections[0].get_paths()[0].interpolated(1).vertices)

line1 = MultiLineString([path.interpolated(1).vertices for linecol in contour1.collections for path in linecol.get_paths()])
line2 = MultiLineString([path.interpolated(1).vertices for linecol in contour2.collections for path in linecol.get_paths()])
for ln in line1.geoms:
    plt.plot(*ln.xy)
for ln in line2.geoms:
    plt.plot(*ln.xy)
for pnt in line1.intersection(line2).geoms:
    plt.plot(*pnt.xy, 'kx')
plt.plot(*line1.xy)
plt.plot(*line2.xy)
inter = line1.intersection(line2)
try:
    for pnt in inter.geoms:
        plt.plot(*pnt.xy, 'x')
except:
    plt.plot(*inter.xy, 'x')

import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier


def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points
def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])
def contour_points(contour, steps=1):
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])
def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def find_depth_periodicity(ground_state, splitting):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    imshow(gs, axs[0], xaxis=periods, yaxis=depths, diverging=False)
    imshow(split, axs[1], xaxis=periods, yaxis=depths, diverging=False)
    X, Y = np.meshgrid(periods, depths)
    # axs[0].contour(gs, [ground_state], colors='w')
    # axs[1].contour(split, [splitting], colors='w')
    axs[0].contour(X, Y, gs, [ground_state], colors='w')
    axs[1].contour(X, Y, split, [splitting], colors='w')
    contour1 = axs[2].contour(X, Y, gs, [ground_state])
    contour2 = axs[2].contour(X, Y, split, [splitting])

    cluster_size = 100
    eps = 0.1
    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    intersection_points = intersection(points1, points2, eps)
    if len(intersection_points) > 0:
        intersection_points = cluster(intersection_points, cluster_size)
        axs[2].scatter(intersection_points[:, 0], intersection_points[:, 1], s=20)
        return intersection_points
    else:
        return intersection_points

find_depth_periodicity(-2, 1)
