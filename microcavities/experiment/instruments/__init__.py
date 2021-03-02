# -*- coding: utf-8 -*-

from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.Andor import Andor
import os
import json
import numpy as np


class AndorActon(Andor):
    def __init__(self, acton_address, acton_calibration_file=None, andor_settings_filepath=None):
        super(AndorActon, self).__init__(andor_settings_filepath)
        self.spectrometer = SP2750(acton_address)
        self.calibration_file = acton_calibration_file
        self.wavelength = self.wavelength  # ensures the widget displays things

    @property
    def wavelength(self):
        return self.spectrometer.wavelength

    @wavelength.setter
    def wavelength(self, wvl):
        self.spectrometer.wavelength = wvl
        if wvl > 10:
            self.x_axis = self.wavelengths
        else:
            self.x_axis = None
        self.update_widgets()

    @property
    def wavelengths(self):
        return self.get_wavelengths()

    # CALIBRATED MEASUREMENT
    @property
    def calibration_file(self):
        """Path to the calibration file"""
        if self._calibration_file is None:
            self._calibration_file = os.path.join(os.path.dirname(__file__), 'default_calibration.json')
        return self._calibration_file

    @calibration_file.setter
    def calibration_file(self, path):
        """Ensures the path is absolute and points to a .json file"""
        if not os.path.isabs(path):
            default_directory = os.path.dirname(__file__)
            path, ext = os.path.splitext(path)
            if ext != 'json':
                if ext != '':
                    self._logger.warn('Changing file type to JSON')
                ext = 'json'
                path = os.path.join(default_directory, path + '.' + ext)
        self._calibration_file = path

    def get_wavelengths(self):
        """Returns the current wavelength range being shown on a detector attached to the SP2750"""
        return self._get_wavelengths(self.calibration_file, self.wavelength, self.spectrometer.get_grating())

    @staticmethod
    def _get_wavelengths(calibration_file, wavelength, grating=None):
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
                offset = offset[grating]

            poly = np.poly1d(offset)
            offset_value = poly(wavelength)

        pixels = np.arange(detector_size, dtype=np.float)
        pixels -= np.mean(pixels)
        delta_wvl = pixels * dispersion_value

        return wavelength + delta_wvl + offset_value