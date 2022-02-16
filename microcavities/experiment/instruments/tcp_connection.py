# -*- coding: utf-8 -*-

from nplab.instrument.server_instrument import create_client_class, create_server_class
from nplab.instrument.camera.ST133.pvcam import Pvcam, PvcamSdk
from nplab.instrument.camera.Andor import Andor
from nplab.instrument.camera.camera_scaled_roi import DisplayWidgetRoiScale
from weakref import WeakSet
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class MyPvcam(Pvcam):
    def __init__(self, *args, **kwargs):
        super(MyPvcam, self).__init__(*args, **kwargs)

    def get_preview_widget(self):
        self._logger.debug('Getting preview widget')
        if self._preview_widgets is None:
            self._preview_widgets = WeakSet()
        new_widget = MyDisplayWidgetRoiScale()
        self._preview_widgets.add(new_widget)
        return new_widget


class MyDisplayWidgetRoiScale(DisplayWidgetRoiScale):
    def __init__(self, *args, **kwargs):
        super(MyDisplayWidgetRoiScale, self).__init__(*args, **kwargs)
        self.local_normalize = True
        self.local_normalize_size = 10
        self.local_normalize_factor = None

    # Local normalizing
    def update_image(self, newimage):
        newimage = np.asarray(newimage, np.float)
        if self.local_normalize:
            if self.local_normalize_factor is None:
                self.local_normalize_factor = gaussian_filter(newimage, sigma=self.local_normalize_size)
            newimage /= self.local_normalize_factor
        return super(MyDisplayWidgetRoiScale, self).update_image(newimage)



PvcamServer = create_server_class(MyPvcam)
PvcamClient = create_client_class(MyPvcam,
                                  list(PvcamSdk.__dict__.keys()) + ["get_camera_parameter", "set_camera_parameter", 'raw_image'],
                                  ('get_qt_ui', "get_control_widget", "get_preview_widget", 'get_wavelength', 'set_wavelength'),
                                  ('latest_frame', 'latest_raw_frame', 'exposure'),
                                  ('_preview_widgets', 'wavelength', 'exposure'))

AndorServer = create_server_class(Andor)
AndorClient = create_client_class(Andor,
                                  list(Andor.__dict__.keys()) + ['get_camera_parameter', 'get_camera_parameter', 'raw_image'],
                                  ('get_qt_ui',  "get_control_widget", "get_preview_widget"),
                                  ('parameters', 'latest_frame', 'latest_raw_frame'),
                                  ('_preview_widgets', 'ui', '_ShowGUIMixin__gui_instance'))


class PvactonClient(PvcamClient):
    metadata_property_names = PvcamClient.metadata_property_names + ('wavelength', )

    def __init__(self, camera_address, spectrometer_address,  **kwargs):
        # super(Pvacton, self).__init__(camera_device, **kwargs)
        # SP2750.__init__(self, spectrometer_address)
        PvcamClient.__init__(self, camera_address)
        self.spectrometer = SP2750(spectrometer_address)
        self.x_axis = spectrometer_calibration(None, self.get_wavelength())

    def get_wavelength(self):
        wvl = self.spectrometer.get_wavelength()
        return wvl

    def set_wavelength(self, wvl):
        self.spectrometer.set_wavelength_fast(wvl)
        self.x_axis = spectrometer_calibration(None, self.get_wavelength())

    wavelength = property(get_wavelength, set_wavelength)
