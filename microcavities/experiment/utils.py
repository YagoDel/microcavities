# -*- coding: utf-8 -*-

from microcavities.utils import yaml_loader
import numpy as np
import yaml
import os
import json


def magnification(calibration_path, space, wavelength=800e-9):
    if not os.path.isabs(calibration_path):
        settings = yaml_loader(os.path.join(os.path.dirname(__file__), '..', 'settings.yaml'))
        calibration_path = os.path.join(settings['calibrations_path'], calibration_path)
    if os.path.splitext(calibration_path)[1] == '':
        calibration_path += '.json'
    with open(calibration_path) as dfile:
        calibration = json.load(dfile)
    mag = magnification_function(calibration[space], wavelength)
    pixel_size = calibration['pixel_size']
    return pixel_size / mag[0], mag[1]


def magnification_old(focus_array=None, wavelength=780e-9, camera=None, settings_path='settings.yaml'):
    """Returns the appropriate scaling

    If a camera name and measurement space is given, it looks in the settings file to find the appropriate lens arrays
    and pixel sizes for the scaling. Otherwise calls magnification_function directly

    :param focus_array:
    :param wavelength:
    :param camera: 2-tuple of str. Camera name and measurement plane (real_space or k_space)
    :param settings_path:
    :return:
    """
    if camera is None:
        return magnification_function(focus_array, wavelength)
    else:
        camera_name, space = camera
        if not os.path.isabs(settings_path):
            settings_path = os.path.join(os.path.dirname(__file__), settings_path)
        with open(settings_path, 'r') as settings_file:
            full_settings = yaml.full_load(settings_file)['calibrations']
        if camera_name not in full_settings:
            raise AttributeError('%s is not in the given yaml' % camera_name)
        camera_settings = full_settings[camera_name]
        pixel_size = camera_settings['pixel_size']
        if focus_array is None:
            focus_array = camera_settings['calibrations'][space]['y']['lenses']
        m, m_array = magnification_function(focus_array, wavelength)
        if space == 'real_space':
            return pixel_size / m, m_array
        elif space == 'k_space':
            # pixel size assumed to be in microns and conversion factor needs to be inverse micron
            return pixel_size*1e-6 / (m * 1e6), m_array


def magnification_function(focus_array, wavelength=780e-9):
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
        m2, m_array2 = magnification_function(focus_array[1:])
        m_array += list([m * x for x in m_array2])
        m *= m2
    else:
        # For an even number of lenses, you are measuring real space
        m = 1
        m_array = [m]
        for idx in range(int(len(focus_array)/2)):
            m *= focus_array[2*idx + 1] / focus_array[2*idx]
            m_array += [m]
    return m, m_array


def spectrometer_calibration_old(pixel=None, wavelength=800, grating='1200'):
    if pixel is None:
        pixel = np.arange(-670, 670)
        # pixel = np.arange(1, 1341)
        # pixel = np.arange(-1340, 0)
    # return (-7.991E-06 * wavelength + 2.454E-02) * pixel + (-2.131E-04 * wavelength + 1.937E-01) + wavelength
    if grating == '1200':
        return (-9.04865e-06 * wavelength + 2.53741e-02) * pixel + 0.18 + wavelength
    elif grating == '1800':
        return (-1.38343e-05 * wavelength + 2.0021e-02) * pixel + wavelength


def spectrometer_calibration(calibration_file, wavelength, grating=None):
    """
    Reads from a calibration file that contains the detector size being used, and the dispersion, and returns the
    wavelength range shown in a detector

    Example JSONs:
        {
          "detector_size": 100,
          "dispersion": 0.01
        }
        {
          "detector_size": 100,
          "dispersion": [0.0001, 0.02]
        }
        {
          "detector_size": 2048,
          "dispersion": {"1": 0.014, "2": [0.0001, 0.02]},
          "offset": {"1": [0.00001, 1]}
        }
    :param calibration_file: str. path to a calibration JSON
    :param wavelength: float. Central wavelength at which to evaluate the dispersion
    :param grating: str. Index of the grating in the JSON file
    :return:
    """
    if not os.path.isabs(calibration_file):
        settings = yaml_loader(os.path.join(os.path.dirname(__file__), '..', 'settings.yaml'))
        calibration_file = os.path.join(settings['calibrations_path'], calibration_file)
    if os.path.splitext(calibration_file)[1] == '':
        calibration_file += '.json'
    with open(calibration_file, 'r') as dfile:
        calibration = json.load(dfile)
    detector_size = calibration['detector_size']

    dispersion = calibration['dispersion']
    if isinstance(dispersion, dict):
        dispersion = dispersion[grating]
    poly = np.poly1d(dispersion)  # poly1d handles it whether you give it a number on an iterable
    dispersion_value = poly(wavelength)

    offset_value = 0
    if 'offset' in calibration:
        offset = calibration['offset']
        if isinstance(offset, dict):
            if grating in offset:
                offset = offset[grating]
            else:
                offset = 0

        poly = np.poly1d(offset)
        offset_value = poly(wavelength)

    pixels = np.arange(detector_size, dtype=np.float)
    pixels -= np.mean(pixels)
    delta_wvl = pixels * dispersion_value

    return wavelength + delta_wvl + offset_value


def quick_scan(setter, values, measurement):
    data = []
    for value in values:
        setter(value)
        data += [measurement()]
    return np.array(data)
