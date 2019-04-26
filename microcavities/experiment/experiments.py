# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import curve_fit


def auto_focus(camera, stage, position_range, start_pos=None, method='size'):
    """Optimises the stage position by looking at the PL on the camera

    :param camera: camera instance with a raw_image function
    :param stage: stage instance with a move function
    :param position_range: range of positions to maximise emission over
    :param start_pos: middle position for the scan
    :param method: either 'size' or 'percentile'
    :return:
    """
    if start_pos is None:
        start_pos = stage.position
    positions = np.linspace(start_pos - position_range / 2.,
                            start_pos + position_range / 2., 21)

    images = []
    for pos in positions:
        stage.move(pos, 'y')
        images += [camera.raw_image(update_latest_frame=True)]

    if method == 'size':
        # Estimates the size of the spot by finding the half-maximum, and counting how many pixels are above it
        analysed = []
        for img in images:
            maxval = np.percentile(img, 99.9)
            minval = np.percentile(img, 20)
            fwhm = np.copy(img - minval)
            fwhm[fwhm < (maxval - minval) / 2.] = 0
            fwhm[fwhm > 0] = 1
            analysed += [fwhm]
        analysed *= -1  # so that the focus is at the maxima, not the minima
    elif method == 'percentile':
        analysed = np.percentile(images, 99.9)
    elif method == 'mask':
        # Creates a mask that only looks at the region where the average intensity is >80% of the max
        mask = np.mean(images, 0)
        threshold = np.percentile(mask, 20) + 0.8 * (np.percentile(mask, 99.9) - np.percentile(mask, 20))
        mask[mask < threshold] = 0
        mask[mask > 0] = 1
        mask = np.array(mask, np.bool)
        # Extract the emission inside the mask region and use it as the target function
        analysed = []
        for img in images:
            analysed += [np.sum(img[mask])]
    else:
        raise TypeError('Unrecognised method: %s' % method)

    max_idx = np.argmax(analysed)
    max_pos = positions[max_idx]
    stage.move(max_pos, 'y')

    return max_pos, images, analysed, positions


def auto_wavelength(camera, laser, wavelength_range=3):
    # TODO: test on actual data
    """Optimises the laser wavelength by looking at the PL on the camera

    :param camera: camera instance with a raw_image function
    :param laser: laser instance with a wavelength property (in nm) with setter and getter
    :param wavelength_range: range of wavelengths (in nm) to maximise over
    :return:
    """
    current_wvl = laser.wavelength
    wavelengths = np.linspace(current_wvl - wavelength_range / 2., current_wvl + wavelength_range / 2., 21)

    emission = []
    for wvl in wavelengths:
        laser.wavelength = wvl
        time.sleep(1)
        img = camera.raw_image()
        emission += [np.percentile(img, 99)]

    max_idx = np.argmax(emission)
    max_wvl = wavelengths[max_idx]
    laser.wavelength = max_wvl

    return max_wvl, emission, wavelengths


def auto_exposure(camera, saturation_threshold):
    """Scales the camera exposure so that the maximum signal is the saturation threshold

    :param camera: camera instance with a raw_image function
    :param saturation_threshold: float
    :return:
    """
    image = camera.raw_image(update_latest_frame=True)

    # Using percentile instead of np.max means hot pixels are ignored. However, if the exposed region is less than 1% of
    # the image, this will fail and will over-estimate the appropriate exposure
    image_max = np.percentile(image, 99.9)

    new_exposure = camera.exposure * saturation_threshold / image_max

    camera.exposure = new_exposure

    return new_exposure


def threshold(camera, power_control, method='power'):
    start_power = power_control.power
    powers = np.linspace(start_power / 5., 3 * start_power, 31)

    images = []
    for power in powers:
        power_control.power = power
        time.sleep(1)
        images += [camera.raw_image(update_latest_frame=True)]

    if method == 'power':
        line = np.percentile(images, 99.9, (1, 2))

        def piecewise_linear(x, x0, b, k1, k2):
            condlist = [x < x0, (x >= x0)]
            funclist = [lambda _x: k1 * _x + b, lambda _x: k1 * _x + b + k2 * (_x - x0)]
            return np.piecewise(x, condlist, funclist)

        params, e = curve_fit(piecewise_linear, powers, line)
        power_control.threshold = params[0]

        return power_control.threshold, powers, line, params

    elif method == 'energy':
        # TODO: make analysis code for finding the condensation threshold
        raise NotImplementedError




