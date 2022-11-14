# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
from microcavities.utils.plotting import create_axes
from nplab.utils.log import create_logger
import h5py
from scipy.ndimage.measurements import center_of_mass
from microcavities.analysis.phase_maps import low_pass
from microcavities.utils import apply_along_axes
from functools import partial
from scipy.signal import fftconvolve
from microcavities.experiment.utils import magnification_function, spectrometer_calibration
from microcavities.experiment.utils import magnification
plt.rcParams['pdf.fonttype'] = 'truetype'
plt.rcParams['svg.fonttype'] = 'none'

LOGGER = create_logger('Rotation')

cdict = {'red': [(0.0, 0.0, 1.0),
                 (0.25, 0.0, 0.0),
                 (0.5, 1.0, 1.0),
                 (0.75, 1.0, 1.0),
                 (1.0, 0.0, 0.0)],
         'green': [(0.0, 0.0, 1.0),
                   (0.25, 0.4, 0.4),
                   (0.5, 1.0, 1.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)],
         'blue': [(0.0, 1.0, 1.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]}

viscm = colors.LinearSegmentedColormap('viscm', cdict, 256)

n_cmap = viscm #'viscm'
delta_n_cmap = 'RdBu_r'
lz_cmap = 'PuOr'

# pixels_per_step = 2 * 150 / 300000
# pixels_per_step = (134-26) / 70000

# data_path = get_data_path('2022_01_26/raw_data.h5')
# data_path = get_data_path('2022_10_25/scan1.h5')

# figure_data_path = get_data_path('2022_01_26/figure_data.h5')
rotation_path = '/Users/yago/Desktop/DDrive/Papers/Rotation/'
mu = '\u03BC'
Delta = '\u0394'
hbar_u = '\u0127'
# momentum_scale = 0.019634954084936207
# spatial_scale = 0.4
# momentum_scale = 0.01325359400733194  # magnification('rotation_acton', 'k_space')[0] * 1e-6
# spatial_scale = 0.27  # magnification('rotation_acton', 'real_space')[0] * 1e6
hbar = 6.582119569 * 10 ** (-16) * 10 ** 3 * 10 ** 12   # in meV.ps
c = 3 * 10 ** 14 * 10 ** -12                            # Speed of Light   um/ps


"""# Defining spatial axis"""
def make_xaxis(xlims, ylims, xscale=None, centers=(0, 0)):
    if xscale is None:
        xscale = spatial_scale * pixels_per_step
    x = np.linspace(*xlims)
    y = np.linspace(*ylims)
    x -= np.mean(x)
    # x *= pixels_per_step
    x *= xscale
    x -= centers[0]

    y -= np.mean(y)
    # y *= pixels_per_step
    y *= xscale
    y = y[::-1]
    y -= centers[1]
    return x, y

# x_axis, y_axis = make_xaxis((2020000.0, 2060000.0, 11), (6155000.0, 6195000.0, 11))
# x_axis, y_axis = make_xaxis((2037000.0, 2107000.0, 22), (6463000.0, 6533000.0, 21))
# x_axis, y_axis = make_xaxis((2054500.0, 2096500.0, 13), (6470000.0, 6515500.0, 14))
# x_axis, y_axis = make_xaxis((2137000.0, 2179000.0, 13), (6480000.0, 6525500.0, 14), (2, 1))
# x_axis, y_axis = make_xaxis((2137000.0, 2186000.0, 15), (6473000.0, 6525500.0, 16), centers=(2, 1))

# # x_axis = np.linspace(2020000.0, 2060000.0, 11)
# # y_axis = np.linspace(6155000.0, 6195000.0, 11)
# # x_axis = np.linspace(2037000.0, 2107000.0, 22)
# # y_axis = np.linspace(6463000.0, 6533000.0, 21)
# # x_axis = np.linspace(2054500.0, 2096500.0, 13)
# # y_axis = np.linspace(6470000.0, 6515500.0, 14)
# x_axis = np.linspace(2137000.0, 2179000.0, 13)
# y_axis = np.linspace(6480000.0, 6525500.0, 14)
# x_axis -= np.mean(x_axis)
# x_axis *= pixels_per_step
# x_axis *= spatial_scale
# y_axis -= np.mean(y_axis)
# y_axis *= pixels_per_step
# y_axis *= spatial_scale
# y_axis = y_axis[::-1]
# frequencies = [-7.5, -5, -3, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 3, 5, 7.5]
# frequencies = [-7, -5, -3, -1, -0.1, 0.1, 1, 3, 5, 7]
# x_axis -= 0.8
# y_axis += 2.5

"""Definining k axis"""
# Finding k=0
# with h5py.File(data_path, 'r') as dfile:
#     below_threshold = np.array([(dfile['full_scan1/below_threshold/%s' % name][...]-110) for name in ['matisse', 'toptica']])
# lp = np.array([low_pass(x, 0.05) for x in below_threshold])
# k0_CoM = np.array([center_of_mass(x) for x in lp])
# k0_argmax = np.array([np.argmax(np.mean(lp, ax), axis=1) for ax in [-1, -2]])
# # # k0_eye = [155, 169]
# # # k0_eye = [241, 281]  # [281, 241]
# # # k0_eye = [240, 330]  # [281, 241]
# # # k0_eye = [242, 285]  # [281, 241]
# # # k0_eye = [250, 295]  # [281, 241]
# # #
# # # # k0_eye = [145, 179]
# # # # k0_eye = [165, 159]
# # #
# # # with h5py.File(data_path, 'r') as dfile:
# # #     bkg = dfile['full_scan1/just_toptica_bkg'][...]
# # #     static_condensate = np.asarray(dfile['full_scan1/just_toptica'][...], float) - bkg
# # #     static_condensate = static_condensate[..., ::-1, ::-1]  # inverting the k-plane (from the k-lens)
# # #
# # # ky = ((np.arange(static_condensate.shape[-2]) - k0_eye[0]) * momentum_scale)[::-1]
# # # kx = ((np.arange(static_condensate.shape[-1]) - k0_eye[1]) * momentum_scale)
# # # kx_func = partial(np.interp, xp=np.arange(static_condensate.shape[-1]), fp=kx)
# # # ky_func = partial(np.interp, xp=np.arange(static_condensate.shape[-2]), fp=ky)


def create_kaxis(dataset, kscale, k0=None, method='CoM'):
    if k0 is None:
        if method == 'CoM':
            k0 = np.array([center_of_mass(x) for x in lp])
        elif method == 'max':
            k0 = np.array([np.argmax(np.mean(lp, ax), axis=1) for ax in [-1, -2]])
        else:
            raise ValueError('k0=None and unrecognised method %s' % method)
    _ky = ((np.arange(dataset.shape[-2]) - k0[0]) * kscale)[::-1]
    _kx = ((np.arange(dataset.shape[-1]) - k0[1]) * kscale)
    _kx_func = partial(np.interp, xp=np.arange(dataset.shape[-1]), fp=_kx)
    _ky_func = partial(np.interp, xp=np.arange(dataset.shape[-2]), fp=_ky)

    return _kx_func, _ky_func, _kx, _ky


# with h5py.File(data_path, 'r') as dfile:
#     tst = dfile['kspace/f=1GHz'][...]
#
#     # bkg = dfile['full_scan1/just_toptica_bkg'][...]
#     # static_condensate = np.asarray(dfile['full_scan1/just_toptica'][...], float) - bkg
#     # static_condensate = static_condensate[..., ::-1, ::-1]  # inverting the k-plane (from the k-lens)
# # kx_func, ky_func, kx, ky = create_kaxis(static_condensate, [155, 169])
# # kx_func, ky_func, kx, ky = create_kaxis(tst, [250, 258])
# # kx_func, ky_func, kx, ky = create_kaxis(tst, [245, 260])
# kx_func, ky_func, kx, ky = create_kaxis(tst, [260, 290])


"""Utility functions"""


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


def photon_density(camera_count, nd_filter=1, exposure_time=1e-4, lifetime=20e-12, hopfield=0.23, alpha=0.25):
    optical_losses = 0.97  # at the objective
    optical_losses *= 0.97  # at the dichroic
    optical_losses *= 0.99  # at the bandpass filter
    optical_losses *= (0.97**2)  # estimated reflection from 2 mirrors
    # optical_losses *= (0.95**4)  # estimated transmission through 4 lenses
    # optical_losses *= 0.1  # estimated spectrometer grating
    # optical_losses *= 0.026  # directly measured
    optical_losses *= 0.3125  # directly measured

    # nd_filter = 1.5e-6  # we used ND6
    # nd_filter = 1  # we used ND0
    quantum_efficiency = 0.4  # 0.75  # camera QE
    efficiency = nd_filter * quantum_efficiency * optical_losses / 2  # factor of two from both directions in the mcav

    # exposure_time = 2  # in seconds
    # exposure_time = 1e-4  # in seconds
    # alpha = 1/4  #0.25  # CCD
    photon_flux = camera_count * alpha / (efficiency * exposure_time)

    # lifetime = 23e-12  #14e-12  #2.1e-12  # 2.5e-12
    # hopfield = 0.23  #0.09  # 0.025
    polariton = photon_flux * lifetime / hopfield

    # Spatial filtering with a 50um pinhole
    # radius = 50 / magnification_function([0.01, 0.25], wavelength=800e-9)[0]
    # area = (np.pi * radius**2)

    return polariton   # , polariton / area


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

    :param array:
    :param radius:
    :param distance:
    :return:
    """
    pinhole_area = np.pi * radius ** 2
    overlap_factor = overlap_between_circles(radius, distance) / pinhole_area
    missing_factor = missing_overlap(radius, distance) / pinhole_area

    overlap1 = (array[1:] + array[:-1]) / 2
    overlap2 = (array[:, 1:] + array[:, :-1]) / 2
    missing = fftconvolve(array, [[1, 1], [1, 1]], 'valid') / 4
    correction = np.sum(missing * missing_factor) - np.sum(overlap_factor * overlap1) - np.sum(overlap_factor * overlap2)

    return np.sum(array) + correction


def angular_momentum_array(vector_field, axes=None):
    if axes is None:
        axes = np.array([(np.arange(x)-x/2) for x in vector_field[0].shape])
        X, Y = np.meshgrid(*axes)
    else:
        X, Y = axes
    return vector_field[0] * Y - vector_field[1] * X  # hbar *


def angular_momentum_average(vector_field, density, axes=None):
    ll = angular_momentum_array(vector_field, axes)
    return hbar * np.average(ll, (0, 1), density)


def angular_momentum_quantum(wavefunction, axes=None):
    if axes is None:
        axes = np.array([(np.arange(x)-x/2) for x in wavefunction.shape[::-1]])
    Axes = np.array(np.meshgrid(*axes))
    vector_field = np.array(np.gradient(wavefunction, *axes))
    momenta = np.imag(np.conj(wavefunction) * vector_field)[::-1]
    cross = np.cross(Axes, vector_field[::-1], axis=0)
    ang_moment = np.imag(np.conj(wavefunction) * cross)
    ang_moment_classical = angular_momentum_classical(momenta, axes)

    fig, axs = plt.subplots(2, 3)
    imshow(np.abs(wavefunction), axs[0,0])
    imshow(np.angle(wavefunction), axs[0,1])
    axs[1, 0].quiver(axes[0][::10], axes[1][::10], *momenta[:, ::10, ::10])
    imshow(ang_moment[1:-1, 1:-1], axs[1, 1])
    imshow(ang_moment_classical[1:-1, 1:-1], axs[1, 2])

    expectation = np.sum(ang_moment[1:-1, 1:-1])  # removes edge effects from np.gradient
    expectation_c = np.sum(ang_moment_classical[1:-1, 1:-1])  # removes edge effects from np.gradient
    density = np.sum(np.abs(wavefunction[1:-1, 1:-1])**2)
    return expectation / density, expectation_c / density


def momentum_quantum(wavefunction, axes=None):
    if axes is None:
        axes = np.array([(np.arange(x) - x / 2) for x in wavefunction.shape[::-1]])
    vector_field = np.array(np.gradient(wavefunction, *axes))
    return np.imag(np.conj(wavefunction) * vector_field)[::-1]


def measure_momenta(image, k_funcs=None, k_axes=None, plott=True, ax=None):
    # if k_functions is None:
    #     k_functions = (kx_func, ky_func)
    # if k_axes is None:
    #     k_axes = (kx, ky)
    image = normalize(image)
    mask = image > 0.5
    CoM = np.squeeze(np.array(center_of_mass(image, mask, 1)))
    if plott:
        fig, ax = create_axes(ax)
        imshow(image, ax, cbar=False, xaxis=k_axes[0], yaxis=k_axes[1])
        ax.plot([0, k_funcs[0](CoM[1])], [0, k_funcs[1](CoM[0])], 'k')
    return k_funcs[0](CoM[1]), k_funcs[1](CoM[0])


def get_field(data, k_funcs, k_axes):
    spatial_image = np.sum(data, (-2, -1))
    momenta = np.array(apply_along_axes(partial(measure_momenta, k_funcs=k_funcs, k_axes=k_axes, plott=False), (2, 3), data, 2))
    return spatial_image, momenta


def plot_quiver(data, x_axes, k_funcs, k_axes, ax=None, **kwargs):
    # if axes is None:
    #     axes = (x_axis, y_axis)
    fig, ax = create_axes(ax)
    spatial_image, momenta = get_field(data, k_funcs, k_axes)
    imshow(spatial_image, ax, cbar=False, xaxis=x_axes[0], yaxis=x_axes[1], aspect='equal')
    ax.quiver(x_axes[0], x_axes[1], momenta[0], momenta[1], **kwargs)
    ax.quiver(x_axes[0], x_axes[1], momenta[0], momenta[1], **kwargs)
    return fig, ax, momenta


def plot_differential(frequency, axes=None, **kwargs):
    if axes is None:
        axes = (x_axis, y_axis)
    img, field = get_differential_field(frequency)
    _, ax = imshow(img, xaxis=axes[0], yaxis=axes[1], cbar=False)
    ax.quiver(axes[0], axes[1], field[0], field[1], **kwargs)
