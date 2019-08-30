# -*- coding: utf-8 -*-

import numpy as np


def magnification(focus_array, wavelength=780e-9):
    """Returns the scaling, either in real-space or k-space, at the focal plane
    of a series of lenses.
    Can be used to calibrate the scale of a detector, like a CCD. If the pixel
    size is p (in the same units as wavelength), the size in scaled units is
    p/magnification(focus_array)


    :param list focus_array: list of focusing distances of lenses between the sample and the desired plane.
    :param float wavelength: wavelength in whatever units the focus_array is in. Defaults to 780e-9 (in meters).
    :return: magnification at the final plane, and a list of magnification at all of the intermediate planes.
    :rtype: type

    """

    assert len(focus_array) > 1

    if len(focus_array) % 2:
        # For an odd number of lenses, you are measuring k-space, so we use the wavelength to get the wavenumber
        kp = 2 * np.pi / wavelength  # wavenumber in m-1
        m = focus_array[0] / kp
        m_array = [m]
        m2, m_array2 = magnification(focus_array[1:])
        m_array += list([m * x for x in m_array2])
        m *= m2
    else:
        # For an even number of lenses, you are measuring real space
        m = 1
        m_array = [m]
        for idx in range(len(focus_array)/2):
            m *= focus_array[2*idx + 1] / focus_array[2*idx]
            m_array += [m]
    return m, m_array


def spectrometer_calibration(pixel=None, wavelength=800):
    if pixel is None:
        pixel = np.arange(1340)
    return (-7.991E-06 * wavelength + 2.454E-02) * pixel + (-2.131E-04 * wavelength + 1.937E-01) + wavelength

