# -*- coding: utf-8 -*-

from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.Andor import Andor
# from microcavities.experiment.instruments.power_control import AcoustoOpticModulator
from nplab.utils.notified_property import NotifiedProperty
from nplab.instrument.camera.ST133.pvcam import Pvcam
import nplab.instrument.camera.ST133.pvcam_h as pv
from microcavities.experiment.instruments.tcp_connection import PvcamClient
from microcavities.experiment.utils import spectrometer_calibration, magnification
import os
import json
import numpy as np
import matplotlib.pyplot as plt


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
            xaxis = np.arange(detector_shape[0], dtype=np.float)
            xaxis -= np.mean(xaxis)
            xaxis *= magn
            yaxis = np.arange(detector_shape[1], dtype=np.float)
            yaxis -= np.mean(yaxis)
            yaxis *= magn
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
PrincetonActon_remote = camera_spectrometer_factory(PvcamClient, SP2750)


class Princeton(Pvcam):
    def __init__(self, *args, **kwargs):
        Pvcam.__init__(self, *args, **kwargs)
        # super(Pvcam, self).__init__(*args, **kwargs)
        self.dark_image = None
        self.reference_image = None
        self.to_filter = False

    def auto_exposure(self, min_value=1e4, max_value=6e4, max_attempts=10, precision=None, max_exposure=30):
        # if max_value is None:
        #     max_value = 0.8 * (2 ** self.BitDepth)
        # if min_value is None:
        #     min_value = 0.1 * (2 ** self.BitDepth)
        image = self.raw_image(False, True)
        current_exposure = self.exposure
        # Takes the 10th brightest pixel
        ignore_n_pixels = 10
        percentile = 100 * (1 - ignore_n_pixels / np.prod(image.shape))
        brightest_pixel = np.percentile(image, percentile)
        okay = True
        attempt = 0

        while (brightest_pixel > max_value or brightest_pixel < min_value) and okay:
            attempt += 1

            # adjust the exposure time
            if brightest_pixel > max_value:
                print("REDUCE exposure time...\n")
                new_exposure = current_exposure / 2
            elif brightest_pixel < min_value:
                print("INCREASE exposure time...\n")
                new_exposure = current_exposure / brightest_pixel * max_value * 0.99
            else:
                raise ValueError('This should not happen')
            if new_exposure > max_exposure:
                new_exposure = max_exposure
                okay = False
            # try the new exposure time
            previous_exposure = current_exposure
            current_exposure = new_exposure
            self.exposure = new_exposure

            image = self.raw_image(False, True)
            brightest_pixel = np.percentile(image, percentile)

            if precision is not None:
                # don't keep on trying the same exposure
                if np.abs(previous_exposure - current_exposure) < precision: okay = False
            # don't keep on trying forever
            if attempt > max_attempts: okay = False

        return image

    def dark_capture(self):
        current_shutter = self.get_param(pv.PARAM_SHTR_OPEN_MODE)
        self.set_param(pv.PARAM_SHTR_OPEN_MODE, pv.OPEN_NEVER)
        img = self.raw_image(False, True)
        self.set_param(pv.PARAM_SHTR_OPEN_MODE, current_shutter)
        # pvcam.set_param(pv.PARAM_SHTR_OPEN_MODE, pv.OPEN_NEVER)
        # current_shutter = self.Shutter
        # self.Shutter = [current_shutter[0], 2] + current_shutter[2:]
        # img = self.raw_image(False, True)
        # self.Shutter = current_shutter
        self.dark_image = np.asarray(img, float)
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

    def image_counts(self, auto=False, *args, **kwargs):
        if auto:
            raw_img = self.auto_exposure(*args, **kwargs)
        else:
            raw_img = self.raw_image(False, True)
        img = raw_img - self.dark_image
        return img / self._exposure_time

    def image_backgrounded(self, *args, **kwargs):
        img = self.image_counts(*args, **kwargs)
        return img - self.background_image

    def image_referenced(self, *args, **kwargs):
        return self.image_backgrounded(*args, **kwargs) / self.reference_image


PrincetonActon_base = camera_spectrometer_factory(Princeton, SP2750)


class PrincetonActon(PrincetonActon_base):
    def __init__(self, *args, **kwargs):
        super(PrincetonActon, self).__init__(*args, **kwargs)

    def take_spectra(self, wvl_min, wvl_max, overlap=0.1, auto_expose=True, backgrounded=False, referenced=False, app=None, *args, **kwargs):
        wvls1 = spectrometer_calibration(self.calibration_file, wvl_min, self.spectrometer.get_grating())
        wvls2 = spectrometer_calibration(self.calibration_file, wvl_max, self.spectrometer.get_grating())
        rng1 = wvls1[-1] - wvls1[0]
        rng2 = wvls2[-1] - wvls2[0]
        rng = (1-overlap) * np.min([rng1, rng2])
        steps = int(np.ceil((wvl_max - wvl_min + rng) / rng))
        center_wavelengths = np.linspace(wvl_min, wvl_max, steps)
        if app is not None:
            print(center_wavelengths)
            app.processEvents()
        images = []
        start_exp = self.exposure
        for wvl in center_wavelengths:
            self.wavelength = wvl
            if referenced:
                images += [self.image_referenced(auto=auto_expose, *args, **kwargs)]
            elif backgrounded:
                images += [self.image_backgrounded(auto=auto_expose, *args, **kwargs)]
            else:
                images += [self.image_counts(auto=auto_expose, *args, **kwargs)]

            if auto_expose:
                self.exposure = start_exp
            if app is not None:
                print(wvl)
                app.processEvents()
        return np.array(images), center_wavelengths

    def plot_spectra(self, images, wavelengths, roi=(200, 220), grating=None, reference=None, background=None,
                     binning=1, *args, **kwargs):
        if grating is None:
            grating = self.spectrometer.get_grating()
        images = np.copy(images)
        if background is not None:
            images -= background
        if reference is not None:
            images /= reference
        spectra = np.mean(images[:, roi[0]:roi[1]], 1)
        wvl_axis = [spectrometer_calibration(self.calibration_file, wvl, grating)[::binning] for wvl in wavelengths]
        [plt.plot(x, y, *args, **kwargs) for x, y in zip(wvl_axis, spectra)]


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

    def auto_exposure(self, min_value=None, max_value=None, max_attempts=10, precision=None, max_exposure=30):
        if max_value is None:
            max_value = 0.8 * (2 ** self.BitDepth)
        if min_value is None:
            min_value = 0.1 * (2 ** self.BitDepth)
        image = self.raw_image(False, True)
        current_exposure = self.Exposure
        # Takes the 10th brightest pixel
        ignore_n_pixels = 10
        percentile = 100 * (1 - ignore_n_pixels / np.prod(image.shape))
        brightest_pixel = np.percentile(image, percentile)
        okay = True
        attempt = 0

        while (brightest_pixel > max_value or brightest_pixel < min_value) and okay:
            attempt += 1

            # adjust the exposure time
            if brightest_pixel > max_value:
                print("REDUCE exposure time...\n")
                new_exposure = current_exposure / 2
            elif brightest_pixel < min_value:
                print("INCREASE exposure time...\n")
                new_exposure = current_exposure / brightest_pixel * max_value * 0.99
            else:
                raise ValueError('This should not happen')
            if new_exposure > max_exposure:
                new_exposure = 30
                okay = False
            # try the new exposure time
            previous_exposure = current_exposure
            current_exposure = new_exposure
            self.Exposure = new_exposure

            image = self.raw_image(False, True)
            brightest_pixel = np.percentile(image, percentile)

            if precision is not None:
                # don't keep on trying the same exposure
                if np.abs(previous_exposure - current_exposure) < precision: okay = False
            # don't keep on trying forever
            if attempt > max_attempts: okay = False

        return image


class AndorActonAom(AndorActon):
    def __init__(self, aom_args, *args, **kwargs):
        super(AndorActonAom, self).__init__(*args, **kwargs)
        self.aom1 = AcoustoOpticModulator(1, *aom_args)
        self.aom2 = AcoustoOpticModulator(2, *aom_args)

    @NotifiedProperty
    def Exposure(self):
        return self._exposure_aom_time

    @Exposure.setter
    def Exposure(self, value):
        self.aom1.exposure = value
        self.aom2.exposure = value
        self.set_andor_parameter('Exposure', value)
        self._exposure_aom_time = value


class AndorAom(unitful_camera_factory(Andor)):
    def __init__(self, aom_args, *args, **kwargs):
        super(AndorAom, self).__init__(*args, **kwargs)
        self.aom1 = AcoustoOpticModulator(1, *aom_args)
        self.aom2 = AcoustoOpticModulator(2, *aom_args)

    @NotifiedProperty
    def Exposure(self):
        return self._exposure_aom_time

    @Exposure.setter
    def Exposure(self, value):
        self.aom1.exposure = value
        self.aom2.exposure = value
        self.set_andor_parameter('Exposure', value)
        self._exposure_aom_time = value
