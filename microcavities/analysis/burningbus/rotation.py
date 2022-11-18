# -*- coding: utf-8 -*-
from microcavities.analysis import *
from microcavities.utils.plotting import *
from nplab.utils.log import create_logger
import h5py
from scipy.ndimage.measurements import center_of_mass
from microcavities.analysis.phase_maps import low_pass
from microcavities.utils import apply_along_axes
from functools import partial
from scipy.signal import fftconvolve
from microcavities.experiment.utils import magnification_function, spectrometer_calibration
from microcavities.experiment.utils import magnification

LOGGER = create_logger('Rotation')

n_cmap = mycmap
delta_n_cmap = 'RdBu_r'
lz_cmap = 'PuOr'

rotation_path = '/Users/yago/Desktop/DDrive/Papers/Rotation/'

"""Defining axis"""


def make_xaxis(xlims, ylims, xscale, centers=(0, 0)):
    """Creates spatial axis

    Each value pair (x[indx1], y[indx2]) corresponds to the spatial position of the pinhole at indices (indx1, indx2)

    :param xlims: 3-tuple of (min, max, n_values) to be passed to np.linspace. min, max are in values of motor steps
    :param ylims: 3-tuple of (min, max, n_values) to be passed to np.linspace. min, max are in values of motor steps
    :param xscale: float, scaling factor in units of micron per motor steps
    :param centers: 2-tuple, in microns
    :return: x, y   1D arrays of micron positions of the pinhole
    """
    x = np.linspace(*xlims)
    y = np.linspace(*ylims)
    x -= np.mean(x)
    x *= xscale
    x -= centers[0]

    y -= np.mean(y)
    y *= xscale
    y = y[::-1]  # pinholing starts at positive y, so need to reverse the axis
    y -= centers[1]
    return x, y


def create_kaxis(dataset, kscale, k0):
    """Creates momenta axis

    :param dataset: N-D array of camera images. Last two axis of the array are kx, ky
    :param kscale: float, scaling factor in units of inverse micron per pixel
    :param k0: float, in pixel units
    :return:
        kx_func, ky_func    interpolation functions
        kx, ky              1D arrays of momenta in the camera images
    """
    _ky = ((np.arange(dataset.shape[-2]) - k0[0]) * kscale)[::-1]  # CCD images need the y axis reversed
    _kx = ((np.arange(dataset.shape[-1]) - k0[1]) * kscale)
    _kx_func = partial(np.interp, xp=np.arange(dataset.shape[-1]), fp=_kx)
    _ky_func = partial(np.interp, xp=np.arange(dataset.shape[-2]), fp=_ky)

    return _kx_func, _ky_func, _kx, _ky


"""Geometric overlap functions"""


def overlap_between_circles(radius, distance):
    """
    Following https://mathworld.wolfram.com/Circle-CircleIntersection.html#:~:text=Two%20circles%20may%20intersect%20in,known%20as%20the%20radical%20line.

    :param radius:
    :param distance:
    :return:
    """
    return 2 * (radius ** 2) * np.arccos(distance/(2*radius)) - distance * np.sqrt(4 * radius**2 - distance**2) / 2


def missing_overlap(radius, distance):
    total_square = distance ** 2
    total_circle = np.pi * radius ** 2
    return total_square - total_circle + 2 * overlap_between_circles(radius, distance)


def corrected_sum(array, radius, distance):
    """
    :param array: 2D np.ndarray. Array of pinholed values to sum over
    :param radius: float. In micron
    :param distance: float. In micron
    :return:
    """
    pinhole_area = np.pi * radius ** 2
    overlap_factor = overlap_between_circles(radius, distance) / pinhole_area
    missing_factor = missing_overlap(radius, distance) / pinhole_area

    overlap1 = (array[1:] + array[:-1]) / 2   # nearest neighbour average over the 1st axis
    overlap2 = (array[:, 1:] + array[:, :-1]) / 2   # nearest neighbour average over the 2nd axis
    missing = fftconvolve(array, [[1, 1], [1, 1]], 'valid') / 4  # nearest neighbour average over the first 2 axes
    correction = np.sum(missing * missing_factor) - np.sum(overlap_factor * overlap1) - np.sum(overlap_factor * overlap2)

    return np.sum(array) + correction


"""Utility functions"""


def angular_momentum_array(vector_field, axes=None):
    """Angular momentum map

    Cross product of r x k

    :param vector_field: 2-tuple of 2D arrays of the kx and ky momenta fields
    :param axes: 2-tuple of 2D arrays of the x and y spatial fields
    :return:
    """
    if axes is None:
        axes = np.array([(np.arange(x)-x/2) for x in vector_field[0].shape])
        X, Y = np.meshgrid(*axes)
    else:
        X, Y = axes
    return vector_field[0] * Y - vector_field[1] * X


def angular_momentum_average(vector_field, density, axes=None):
    """Total angular momentum

    Given the vector field and density of a wavefunction, we calculate the angular momentum as that of a solid body (the
    sum over the whole body of the mass times the cross product of velocity and position).
    Since our measurement is momenta, not velocity, we can simply do a weighted average over the density in space rather
    than a sum over the mass.

    :param vector_field: 2-tuple of 2D arrays of the kx and ky momenta fields
    :param density: 2D array of the wavefunction density
    :param axes: 2-tuple of 2D arrays of the x and y spatial fields
    :return:
    """
    ll = angular_momentum_array(vector_field, axes)
    return hbar * np.average(ll, (0, 1), density)


def measure_momenta(image, k_funcs, k_axes, plott=True, ax=None):
    """Measurement of the momenta of a PL cloud

    Assumes the given CCD image is a cloud with a single centroid. Would not work well if there's multiple momenta
    components in the same image.

    :param image: 2D array. CCD image
    :param k_funcs: 2-tuple of functions. Interpolation functions that transform pixels into inverse microns
    :param k_axes: 2-tuple of 1D arrays.
    :param plott: bool
    :param ax:
    :return:
    """
    image = normalize(image)
    mask = image > 0.5
    CoM = np.squeeze(np.array(center_of_mass(image, mask, 1)))
    if plott:
        fig, ax = create_axes(ax)
        imshow(image, ax, cbar=False, xaxis=k_axes[0], yaxis=k_axes[1])
        ax.plot([0, k_funcs[0](CoM[1])], [0, k_funcs[1](CoM[0])], 'k')
    return k_funcs[0](CoM[1]), k_funcs[1](CoM[0])


def get_field(data, k_funcs, k_axes):
    """Measurement of momenta for a whole pinholed condensate

    :param data: 4D array. First two axis correspond to the pinhole x and y positions. Last two axis are the CCD image
                kx,ky axis
    :param k_funcs: 2-tuple of functions. Interpolation functions that transform pixels into inverse microns
    :param k_axes: 2-tuple of 1D arrays.
    :return:
    """
    spatial_image = np.sum(data, (-2, -1))
    momenta = np.array(
        apply_along_axes(partial(measure_momenta, k_funcs=k_funcs, k_axes=k_axes, plott=False), (2, 3), data, 2))
    return spatial_image, momenta


def plot_quiver(data, x_axes, k_funcs, k_axes, ax=None, **kwargs):
    fig, ax = create_axes(ax)
    spatial_image, momenta = get_field(data, k_funcs, k_axes)
    imshow(spatial_image, ax, cbar=False, xaxis=x_axes[0], yaxis=x_axes[1], aspect='equal')
    ax.quiver(x_axes[0], x_axes[1], momenta[0], momenta[1], **kwargs)
    ax.quiver(x_axes[0], x_axes[1], momenta[0], momenta[1], **kwargs)
    return fig, ax, momenta


"""Vector field functions"""


def jacobian(vector_field, axes=None):
    """Numerical Jacobian of an ND vector field

    :param vector_field:
    :param axes:
    :return:
    """
    if axes is None:
        axes = [np.arange(x) for x in vector_field[0].shape]
    _jacobian = []
    for component in vector_field:
        _jcbn = []
        for idx_ax in range(len(vector_field)):
            _jcbn += [np.gradient(component, axis=idx_ax) / (np.gradient(axes[idx_ax])[0])]
        _jacobian += [_jcbn]
    return np.array(_jacobian)


def divergence(vector_field, axes=None):
    jcb = jacobian(vector_field, axes=axes)
    return np.sum([jcb[(idx, )*len(axes)] for idx in range(len(axes))], 0)


def curl(vector_field, axes=None):
    jcb = jacobian(vector_field, axes=axes)
    return jcb[1, 0] - jcb[0, 1]  # TODO: this only works for 2D


def test_curl_divergence():
    x = np.linspace(-25, 25, 101) + 1e-6
    y = np.linspace(-50, 50, 201) + 1e-6
    extent = [y[0], y[-1], x[-1], x[0]]
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    lines = (X / X, 0*X)
    vortex = (X / R, Y / R)
    vortex2 = (-X / R, -Y / R)
    source = (-Y / R, X / R)
    source2 = (Y / R, -X / R)

    fields = [lines, source, source2, vortex, vortex2]

    fig, axs = plt.subplots(3, len(fields), sharex=True, sharey=True)
    for idx, field in enumerate(fields):
        div = divergence(field, (x, y))
        crl = curl(field, (x, y))
        axs[0, idx].quiver(Y[::10, ::10], X[::10, ::10], field[0][::10, ::10], field[1][::10, ::10])
        imshow(div, axs[1, idx], xaxis=y, yaxis=x)
        imshow(crl, axs[2, idx], xaxis=y, yaxis=x)
    for ax in axs.flatten():
        ax.set_aspect('equal')


def vortex_field(x, y, scale=1):
    Y, X = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    return -scale * np.cos(theta), scale * np.sin(theta)


def source_field(x, y, scale=1):
    Y, X = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return scale * np.sin(theta), scale * np.cos(theta)


def saddle_field(x, y, scale=1):
    Y, X = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return scale*r * np.cos(theta), scale * r * np.sin(theta)


def momentum_quantum(wavefunction, axes=None):
    if axes is None:
        axes = np.array([(np.arange(x) - x / 2) for x in wavefunction.shape[::-1]])
    vector_field = np.array(np.gradient(wavefunction, *axes))
    return np.imag(np.conj(wavefunction) * vector_field)[::-1]
