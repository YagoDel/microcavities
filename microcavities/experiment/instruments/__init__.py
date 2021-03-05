# -*- coding: utf-8 -*-

from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.Andor import Andor
from microcavities.experiment.instruments.tcp_connection import PvcamClient
from microcavities.experiment.utils import spectrometer_calibration, magnification
import os
import json
import numpy as np


def unitful_camera_factory(camera_class):
    class UnitfulCamera(camera_class):
        def __init__(self, camera_calibration_file, *args, **kwargs):
            super(UnitfulCamera, self).__init__(*args, **kwargs)
            self.camera_calibration_file = camera_calibration_file
            self.space = 'real_space'

        @property
        def space(self):
            return self._space

        @space.setter
        def space(self, value):
            self._space = value
            magn, _ = magnification(self.camera_calibration_file, value)
            with open(self.camera_calibration_file) as dfile:
                calibration = json.load(dfile)
            detector_shape = calibration['detector_shape']
            pixel_size = calibration['pixel_size']
            xaxis = np.arange(detector_shape[0], dtype=np.float)
            xaxis -= np.mean(xaxis)
            xaxis *= pixel_size/magn
            yaxis = np.arange(detector_shape[1], dtype=np.float)
            yaxis -= np.mean(yaxis)
            yaxis *= pixel_size/magn
            if value == 'real_space':  # default units for display is microns
                xaxis *= 1e6
                yaxis *= 1e6
                self.axis_units['bottom'] = u"\u03BCm"
                self.axis_units['left'] = u"\u03BCm"
            if value == 'k_space':  # default units for display is inverse microns
                xaxis *= 1e-6
                yaxis *= 1e-6
                self.axis_units['bottom'] = u"\u03BCm\u207B\u00b1"
                self.axis_units['left'] = u"\u03BCm\u207B\u00b1"
            self.x_axis = xaxis
            self.y_axis = yaxis
            self.update_widgets()

        @property
        def camera_calibration_file(self):
            """Path to the calibration file"""
            if self._camera_calibration_file is None:
                self._camera_calibration_file = os.path.join(os.path.dirname(__file__),
                                                      'calibrations',
                                                      'default_camera_calibration.json')
            return self._camera_calibration_file

        @camera_calibration_file.setter
        def camera_calibration_file(self, path):
            """Ensures the path is absolute and points to a .json file"""
            if not os.path.isabs(path):
                default_directory = os.path.join(os.path.dirname(__file__), 'calibrations')
                path, ext = os.path.splitext(path)
                if ext != 'json':
                    if ext != '':
                        self._logger.warn('Changing file type to JSON')
                    ext = 'json'
                    path = os.path.join(default_directory, path + '.' + ext)
            self._camera_calibration_file = path
    return UnitfulCamera


def camera_spectrometer_factory(camera_class, spectrometer_class):
    class Combined(camera_class):
        def __init__(self, calibration_file, spectrometer_args, *args, **kwargs):
            super(Combined, self).__init__(*args, **kwargs)
            self.spectrometer = spectrometer_class(*spectrometer_args)
            self.calibration_file = calibration_file
            self.wavelength = self.wavelength  # ensures the widget displays things

        @property
        def wavelength(self):
            return self.spectrometer.wavelength

        @wavelength.setter
        def wavelength(self, wvl):
            self.spectrometer.wavelength = wvl
            if wvl > 10:
                self.x_axis = self.wavelengths
                self.axis_units['bottom'] = "nm"
            elif hasattr(self, 'space'):
                self.space = self.space
            else:
                self.x_axis = None
                self.axis_units['bottom'] = None
            self.update_widgets()

        @property
        def wavelengths(self):
            return self.get_wavelengths()

        # CALIBRATED MEASUREMENT
        @property
        def calibration_file(self):
            """Path to the calibration file"""
            if self._calibration_file is None:
                self._calibration_file = os.path.join(os.path.dirname(__file__),
                                                      'calibrations',
                                                      'default_spectrometer_calibration.json')
            return self._calibration_file

        @calibration_file.setter
        def calibration_file(self, path):
            """Ensures the path is absolute and points to a .json file"""
            if not os.path.isabs(path):
                default_directory = os.path.join(os.path.dirname(__file__), 'calibrations')
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

    return Combined


AndorActon_base = camera_spectrometer_factory(unitful_camera_factory(Andor), SP2750)
PrincetonActon = camera_spectrometer_factory(PvcamClient, SP2750)


class AndorActon(AndorActon_base):
    def __init__(self, *args, **kwargs):
        super(AndorActon, self).__init__(*args, **kwargs)
        self.dark_image = None
        self.reference_image = None
        self.to_filter = False

    def dark_capture(self):
        current_shutter = self.Shutter
        self.Shutter = [current_shutter[0], 2] + current_shutter[2:]
        img = self.raw_image(False, True)
        self.Shutter = current_shutter
        self.dark_image = img
        return img

    def filter_function(self, img):
        if self.to_filter:
            if self.dark_image is not None:
                img -= self.dark_image
            if self.background is not None:
                img -= self.background
            if self.reference_image is not None:
                img /= self.reference
        else:
            if self.backgrounded:
                return img - self.background
            else:
                return img
        return img

# class AndorActon(Andor):
#     def __init__(self, acton_address, acton_calibration_file=None, *args, **kwargs):
#         super(AndorActon, self).__init__(*args, **kwargs)
#         self.spectrometer = SP2750(acton_address)
#         self.calibration_file = acton_calibration_file
#         self.wavelength = self.wavelength  # ensures the widget displays things
#
#     @property
#     def wavelength(self):
#         return self.spectrometer.wavelength
#
#     @wavelength.setter
#     def wavelength(self, wvl):
#         self.spectrometer.wavelength = wvl
#         if wvl > 10:
#             self.x_axis = self.wavelengths
#         else:
#             self.x_axis = None
#         self.update_widgets()
#
#     @property
#     def wavelengths(self):
#         return self.get_wavelengths()
#
#     # CALIBRATED MEASUREMENT
#     @property
#     def calibration_file(self):
#         """Path to the calibration file"""
#         if self._calibration_file is None:
#             self._calibration_file = os.path.join(os.path.dirname(__file__), 'default_calibration.json')
#         return self._calibration_file
#
#     @calibration_file.setter
#     def calibration_file(self, path):
#         """Ensures the path is absolute and points to a .json file"""
#         if not os.path.isabs(path):
#             default_directory = os.path.dirname(__file__)
#             path, ext = os.path.splitext(path)
#             if ext != 'json':
#                 if ext != '':
#                     self._logger.warn('Changing file type to JSON')
#                 ext = 'json'
#                 path = os.path.join(default_directory, path + '.' + ext)
#         self._calibration_file = path
#
#     def get_wavelengths(self):
#         """Returns the current wavelength range being shown on a detector attached to the SP2750"""
#         return spectrometer_calibration(self.calibration_file, self.wavelength, self.spectrometer.get_grating())
