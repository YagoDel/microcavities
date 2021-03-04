# -*- coding: utf-8 -*-

from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.Andor import Andor
from microcavities.experiment.utils import spectrometer_calibration
import os
import json
import numpy as np


class AndorActon(Andor):
    def __init__(self, acton_address, acton_calibration_file=None, *args, **kwargs):
        super(AndorActon, self).__init__(*args, **kwargs)
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
        return spectrometer_calibration(self.calibration_file, self.wavelength, self.spectrometer.get_grating())
