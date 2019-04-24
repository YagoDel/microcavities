# -*- coding: utf-8 -*-

import numpy as np
import time


def auto_focus_emission(camera, stage, position_range, start_pos=None):
    """Optimises the stage position by looking at the PL on the camera

    :param camera: camera instance with a raw_image function
    :param stage: stage instance with a move function
    :param position_range: range of positions to maximise emission over
    :param start_pos: middle position for the scan
    :return:
    """
    if start_pos is None:
        start_pos = stage.position
    positions = np.linspace(start_pos - position_range / 2.,
                            start_pos + position_range / 2., 21)

    emission = []
    for pos in positions:
        stage.move(pos, 'y')
        img = camera.raw_image()
        emission += [np.percentile(img, 99)]

    max_idx = np.argmax(emission)
    max_pos = positions[max_idx]
    stage.move(max_pos, 'y')

    return max_pos, emission, positions


def auto_wavelength(camera, laser, wavelength_range=3):
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
    image = camera.raw_image()

    # Using percentile instead of np.max means hot pixels are ignored. However, if the exposed region is less than 1% of
    # the image, this will fail and will over-estimate the appropriate exposure
    image_max = np.percentile(image, 99)

    new_exposure = camera.exposure * saturation_threshold / image_max

    camera.exposure = new_exposure

    return new_exposure
