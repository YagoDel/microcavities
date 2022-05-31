# -*- coding: utf-8 -*-
import numpy as np

from microcavities.utils.plotting import *


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


def angular_momentum_classical(vector_field, axes=None):
    if axes is None:
        axes = np.array([(np.arange(x)-x/2) for x in vector_field[0].shape[::-1]])
    Axes = np.array(np.meshgrid(*axes))
    return np.cross(Axes, vector_field, axis=0)


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
