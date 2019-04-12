# -*- coding: utf-8 -*-
from nplab.instrument import Instrument
from nplab.instrument.server_instrument import create_client_class
from nplab.instrument.stage.SigmaKoki import HIT, SHOT
from nplab.instrument.electronics.Meadowlark import VariableRetarder
from nplab.instrument.Flipper.thorlabs_MFF002 import ThorlabsMFF
from nplab.instrument.electronics.NewportPowermeter import NewportPowermeter
from nplab.instrument.stage.wheel_of_power import PowerWheelMixin
from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.ST133.pvcam import Pvcam, PvcamSdk
from nplab.instrument.camera.Andor.andornplab import Andor

import nidaqmx
import numpy as np
import re
import time


class RetarderPower(VariableRetarder, PowerWheelMixin):
    def __init__(self, port):
        super(RetarderPower, self).__init__(port)
        self._raw_min = 2
        self._raw_max = 8
        # self.channel = 2
        self.multiple_calibrations = [None, None]

    def prepare_calibration(self, calibration):
        calibration = np.copy(calibration)
        powers = calibration[1]
        calibration = calibration[:, powers < 1]

        maxidx = np.argmax(powers)
        minidx = np.argmin(powers)

        return calibration[:, maxidx:minidx]

    # def _calibration_functions(self, calibration=None):
    #     if calibration is None:
    #         calibration = self.calibration
    #     self.cal_to_raw = interp1d(calibration[1], calibration[0])
    #     self.raw_to_cal = interp1d(calibration[0], calibration[1])
    #     if self.channel == 2:
    #         self.calibration2 = np.copy(calibration)
    #         self.cal_to_raw2 = interp1d(calibration[1], calibration[0])
    #         self.raw_to_cal2 = interp1d(calibration[0], calibration[1])
    #     elif self.channel == 3:
    #         self.calibration3 = np.copy(calibration)
    #         self.cal_to_raw3 = interp1d(calibration[1], calibration[0])
    #         self.raw_to_cal3 = interp1d(calibration[0], calibration[1])
    #     else:
    #         print 'Unrecognised channel'
    #
    # @property
    # def power2(self):
    #     self.channel = 2
    #     return self.raw_to_cal2(self.raw_power)
    #
    # @power2.setter
    # def power2(self, value):
    #     self.channel = 2
    #     self.raw_power = self.cal_to_raw2(value)
    #
    # @property
    # def power3(self):
    #     self.channel = 3
    #     return self.raw_to_cal3(self.raw_power)
    #
    # @power3.setter
    # def power3(self, value):
    #     self.channel = 3
    #     self.raw_power = self.cal_to_raw3(value)

    @property
    def raw_power(self):
        return self.voltage

    @raw_power.setter
    def raw_power(self, value):
        self.voltage = value


class Flipper(Instrument):

    def __init__(self, device="Dev1", channel="PFI0"):
        self.id = "%s/%s" % (device, channel)
        self.state = None

    def on(self):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan('Dev1/PFI0')
            task.write(True)
        self.state = True

    def off(self):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan('Dev1/PFI0')
            task.write(False)
        self.state = False

    def toggle(self):
        if self.state:
            self.off()
        else:
            self.on()


PvcamClient = create_client_class(Pvcam,
                                  PvcamSdk.__dict__.keys() + ["get_camera_parameter", "set_camera_parameter"],
                                  ('get_qt_ui', "get_control_widget", "get_preview_widget", 'get_wavelength', 'set_wavelength'),
                                  ('latest_frame', 'latest_raw_frame'),
                                  ('_preview_widgets', 'wavelength'))

AndorClient = create_client_class(Andor,
                                  Andor.__dict__.keys() + ['get_camera_parameter', 'get_camera_parameter', 'raw_image'],
                                  ('get_qt_ui',  "get_control_widget", "get_preview_widget"),
                                  ('parameters', 'latest_frame', 'latest_raw_frame'),
                                  ('_preview_widgets', ))


# def calibrate(self, wvl, to_device=True):
#     if to_device:
#         calibrated = (wvl - 1.937E-01) / (1-2.131E-04)
#     else:
#         calibrated = (-2.131E-04 * wvl + 1.937E-01) + wvl
#     return calibrated
# setattr(SP2750, 'calibrate', calibrate)


class Spectrometer(SP2750):
    def __init__(self, *args, **kwargs):
        super(Spectrometer, self).__init__(*args, **kwargs)

    @property
    def wavelength(self):
        return self.get_wavelength()
    @wavelength.setter
    def wavelength(self, value):
        self.set_wavelength_fast(value)


class PvactonClient(PvcamClient):
    def __init__(self, camera_address, spectrometer_address,  **kwargs):
        # super(Pvacton, self).__init__(camera_device, **kwargs)
        # SP2750.__init__(self, spectrometer_address)
        PvcamClient.__init__(self, camera_address)
        self.spectrometer = SP2750(spectrometer_address)

    def set_offset_scaling(self):
        string = self.spectrometer.query("?NM")
        raw_wvl = float(re.findall(" ([0-9]+\.[0-9]+) ", string)[0])

        def pos_to_unit(pos, axis):
            if axis == 'bottom':
                offset = (-7.991E-06 * raw_wvl + 2.454E-02) * (-self.resolution[0] / 2) + (
                    -2.131E-04 * raw_wvl + 1.937E-01) + raw_wvl
                scale = (-7.991E-06 * raw_wvl + 2.454E-02)
                return map(lambda x: scale * x + offset, pos)
            elif axis == 'left':
                return map(lambda x: x, pos)
            else:
                raise ValueError
        setattr(self, 'pos_to_unit', pos_to_unit)

        # self.unit_offset[0] = (-7.991E-06 * raw_wvl + 2.454E-02) * (-self.resolution[0] / 2) + (
        #             -2.131E-04 * raw_wvl + 1.937E-01) + raw_wvl
        # self.scaling[0] = (-7.991E-06 * raw_wvl + 2.454E-02)

    def get_wavelength(self):
        wvl = self.spectrometer.get_wavelength()
        self.set_offset_scaling()
        return wvl

    def set_wavelength(self, wvl):
        self.spectrometer.set_wavelength_fast(wvl)
        # self.unit_offset[0] = self.spectrometer.calibrate(wvl) - self.unit_scale[0] * self.resolution[0] / 2
        self.set_offset_scaling()

    wavelength = property(get_wavelength, set_wavelength)


class NdWheel(SHOT, PowerWheelMixin):
    def __init__(self, address):
        super(NdWheel, self).__init__(address)
        self._count_deg = 200.
        self._raw_max = 360

    def move(self, angle, axis=1, relative=False, wait=True):
        if not relative:
            curr_angle = self.get_position(axis)
            angle_diff = angle - curr_angle

            if angle_diff < -10:
                angle_diff += 360

            counts = angle_diff * self._count_deg
        else:
            counts = angle * self._count_deg

        super(NdWheel, self).move(counts, axis, relative=True, wait=wait)

    def get_position(self, axis=1):
        counts = super(NdWheel, self).get_position(axis)
        if hasattr(counts, '__iter__'):
            counts = np.array(counts)
        return (counts/self._count_deg) % 360

    @property
    def raw_power(self):
        return self.position

    @raw_power.setter
    def raw_power(self, value):
        self.move(value)

    def prepare_calibration(self, calibration):
        angles = calibration[0]
        powers = calibration[1]

        minidx = np.argmin(powers)
        self.move(angles[minidx])
        self.set_origin()

        powers = np.roll(powers, -minidx)
        angles = np.roll(angles, -minidx) % 360

        maxidx = np.argmin(np.abs(angles - 320))
        powers = powers[:maxidx]
        angles = angles[:maxidx]

        return np.array([angles, powers])


class PowerMeterFlipper(NewportPowermeter):
    def __init__(self, motor_address, *args, **kwargs):
        super(PowerMeterFlipper, self).__init__(*args, **kwargs)
        self.flipper = ThorlabsMFF(motor_address)

    @property
    def power(self):
        """
        Reads the instantaneous power
        """
        flip = False
        if self.flipper.state == 0:
            self.set_state(1)
            time.sleep(4)
            flip = True
        power = self.query('PM:Power?')
        if flip:
            self.flipper.set_state(0)
            time.sleep(1)
        return float(power)

    def set_state(self, value):
        if value:
            self.flipper.write(0x046A, param1=0x01, param2=0x01)
        else:
            self.flipper.write(0x046A, param1=0x01, param2=0x02)


# sample_surface = 4221500  % 4220500 # 4220000
class SampleMovement(HIT):
    axis_names = ['x', 'y']
    axis_LUT = dict(zip(['x', 'y'], [2, 1]))
    # axis_lim = dict(zip(['x', 'y'], [None, (3420000, 4420000)]))
    axis_lim = dict(zip(['x', 'y'], [None, (-789000, 789000)]))
    # axis_lim = dict(zip(['x', 'y'], [None, None]))

    def __init__(self, address):
        super(SampleMovement, self).__init__(address)
        for ax in self.axis_names:
            self.set_speed(ax, 1, 100000, 1000)

    def move(self, counts, axes=None, relative=False, wait=True, safe=True):
        if axes is None:
            axes = self.axis_names
        elif not hasattr(axes, '__iter__'):
            axes = (axes, )
        if not hasattr(counts, '__iter__'):
            counts = [counts] * len(axes)
        if safe:
            for count, ax in zip(counts, axes):
                if self.axis_lim[ax] is not None:
                    lim = self.axis_lim[ax]
                    if relative:
                        pos = self.get_position((self.axis_LUT[ax], ))[0]
                        assert lim[0] < pos + count < lim[1]
                    else:
                        assert lim[0] < count < lim[1]
        super(SampleMovement, self).move(counts, axes, relative, wait)

    def mechanical_home(self, axes):
        assert 'y' not in axes
        super(SampleMovement, self).mechanical_home(axes)

    def define_line(self, xz_points):
        xz_points = np.array(xz_points, dtype=np.float)
        point = xz_points[0]
        vec = xz_points[0]-xz_points[1]
        slope = vec[1] / vec[0]
        offset = point[1] - point[0] * slope
        def line(x):
            return slope * x + offset
        self.line = line

    def define_plane(self, xyz_points):
        xyz_points = np.array(xyz_points, dtype=np.float)
        point = xyz_points[0]
        vec1 = xyz_points[0]-xyz_points[1]
        vec2 = xyz_points[0]-xyz_points[2]
        normal = np.cross(vec1, vec2)
        print point, vec1, vec2, normal, normal[:2]
        def z_func(x, y):
            return (np.dot(point, normal) - np.dot([x,y], normal[:2])) / normal[2]
        self.plane = z_func

    def move_correct(self, x, relative=False):
        if hasattr(x, "__iter__"):
            HIT.move(self, x[0], 2, relative)
            HIT.move(self, x[1], 1, relative)
            HIT.move(self, self.plane(*x), 1, relative)
        else:
            HIT.move(self, x, 2, relative)
            HIT.move(self, self.line(x), 1, relative)


class Stages(HIT):

    def __init__(self, address):
        super(Stages, self).__init__(address)

        self.axis_names = ("spectrometer_lens", "k_lens", "filter_x", "filter_y", "stokes", "interferometer_delay")
        self.axis_LUT = dict(zip(self.axis_names, (3, 0, 4, 5, 1, 2)))
        for axis in range(6):
            self.set_speed(axis, 1, 500000, 1000)

        self.axis_toggle = dict(k_lens=dict(on=2330000, off=7000000),
                                filter_y=dict(off=8604180, small=338640, medium=3446640, big=6511640),
                                filter_x=dict(on=2091000, off=6000000))
        # 20um pinhole x,y = [2204000, 338640]
        # 50um pinhole x, y = [2127000, 3417640]
        # 100um pinhole x, y = [2117000, 6482000]

    def toggle(self, axis, state):
        if axis in self.axis_toggle:
            dictionary = self.axis_toggle[axis]
            if state in dictionary:
                self.move(dictionary[state], axis)
            else:
                self._logger.warn('Unrecognised state %s. Needs to be one of %s' %(state, dictionary.keys()))
        else:
            self._logger.warn('Axis %s does not have toggle values' % axis)

            # states = np.array(self.axis_toggle[axis])
            # pos = self.get_position(axis)[0]
            # current_state = np.argmin(np.abs(states - pos))
            # self.move(states[(current_state + 1) % len(states)])


"""OLD and UNUSED"""
class Pvacton(Pvcam):
    def __init__(self, camera_device, spectrometer_address,  **kwargs):
        # super(Pvacton, self).__init__(camera_device, **kwargs)
        # SP2750.__init__(self, spectrometer_address)
        Pvcam.__init__(self, camera_device, **kwargs)
        self.spectrometer = SP2750(spectrometer_address)

    def get_wavelength(self):
        wvl = self.spectrometer.get_wavelength()
        self.unit_offset[0] = wvl
        return wvl

    def set_wavelength(self, wvl):
        self.spectrometer.set_wavelength_fast(wvl)
        self.unit_offset[0] = self.spectrometer.calibrate(wvl) - self.unit_scale[0] * self.resolution[0] / 2

    wavelength = property(get_wavelength, set_wavelength)