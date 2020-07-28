# -*- coding: utf-8 -*-
from nplab.instrument import Instrument
from nplab.instrument.visa_instrument import VisaInstrument
from nplab.instrument.server_instrument import create_client_class, create_server_class
from nplab.instrument.stage.SigmaKoki import HIT, SHOT
from nplab.instrument.electronics.Meadowlark import VariableRetarder
from nplab.instrument.Flipper.thorlabs_MFF002 import ThorlabsMFF
from nplab.instrument.electronics.NewportPowermeter import NewportPowermeter
from nplab.instrument.stage.wheel_of_power import PowerWheelMixin
from nplab.instrument.spectrometer.Acton import SP2750
from nplab.instrument.camera.ST133.pvcam import Pvcam, PvcamSdk
from nplab.instrument.camera.Andor import Andor
from microcavities.experiment.utils import spectrometer_calibration

import nidaqmx
import nidaqmx.stream_writers
import numpy as np
import re
import time


class Matisse(VisaInstrument):
    """See http://www.moi-lab.org/uploads/Matisse%20Programmes%20Guide.pdf"""
    def __init__(self, address):
        super(Matisse, self).__init__(address)

    def wrapped_command(self, command):

        if command.endswith('?'):
            return self.query_value(command)
        else:
            full_reply = self.query(command)
            assert "OK" in full_reply

    def query_value(self, command):
        full_reply = self.query(command)

        # Match at least one number, followed by anything except whitespaces, and ending in another number
        number_string = re.findall('\d+\S*\d', full_reply)[0]

        try:
            number = int(number_string)
        except Exception as e:
            number = float(number_string)

        return number

    @property
    def wavelength(self):
        return self.query_value("MOTBI:WL?")

    @wavelength.setter
    def wavelength(self, value):
        self.wrapped_command("MOTBI:WL %g" % value)


class RetarderPower(VariableRetarder, PowerWheelMixin):
    def __init__(self, port, channel=2, direction='up'):
        super(RetarderPower, self).__init__(port, channel)
        self._raw_min = 2
        self._raw_max = 8
        self.direction = direction
        self.multiple_calibrations = [None, None]

    def prepare_calibration(self, calibration):
        calibration = np.copy(calibration)
        powers = calibration[1]
        calibration = calibration[:, powers < 1]

        if self.direction == 'up':
            maxidx = np.argmax(powers)
            minidx = np.argmin(powers)
        elif self.direction == 'down':
            maxidx = np.argmin(powers)
            minidx = np.argmax(powers)
        else:
            raise ValueError('Direction unrecognised: %s' % self.direction)

        return calibration[:, minidx:maxidx]

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


class AOM(Instrument):
    def __init__(self, device="Dev1", channel="ao0"):
        self.id = "%s/%s" % (device, channel)
        self.task = None

    def close_task(self):
        self.task.stop()
        self.task.close()
        self.task = None

    def sawtooth(self, frequency=100, duty_cycle=10, amplitude=1):
        """

        :param frequency: in Hz
        :param duty_cycle: out of 100
        :param amplitude: in V
        :return:
        """
        assert duty_cycle / (frequency * 100.) > 1e-5  # If you make too small a pulse, it isn't really square

        if self.task is not None:
            self.close_task()

        waveform = np.append(amplitude * np.ones(duty_cycle), np.zeros(100 - duty_cycle))

        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan(self.id)
        self.task.timing.cfg_samp_clk_timing(rate=frequency * 100, samps_per_chan=100,
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(self.task.out_stream, auto_start=True)
        writer.write_many_sample(waveform)

    def cw(self, amplitude):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.id)
            writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(task.out_stream, auto_start=True)
            writer.write_one_sample(amplitude)


PvcamServer = create_server_class(Pvcam)
PvcamClient = create_client_class(Pvcam,
                                  list(PvcamSdk.__dict__.keys()) + ["get_camera_parameter", "set_camera_parameter", 'raw_image'],
                                  ('get_qt_ui', "get_control_widget", "get_preview_widget", 'get_wavelength', 'set_wavelength'),
                                  ('latest_frame', 'latest_raw_frame'),
                                  ('_preview_widgets', 'wavelength'))

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


class NdWheel(SHOT, PowerWheelMixin):
    axis_names = (1, )

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
    on_state = 0

    def __init__(self, motor_address, *args, **kwargs):
        super(PowerMeterFlipper, self).__init__(*args, **kwargs)
        self.flipper = ThorlabsMFF(motor_address)

    @property
    def power(self):
        """
        Reads the instantaneous power
        """
        flip = False
        if self.flipper.state != self.on_state:
            self.set_state(self.on_state)
            time.sleep(4)
            flip = True
        power = self.query('PM:Power?')
        if flip:
            self.flipper.set_state((self.on_state + 1) % 2)
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
    axis_LUT = dict(list(zip(['x', 'y'], [2, 1])))
    # axis_lim = dict(zip(['x', 'y'], [None, (3420000, 4420000)]))
    axis_lim = dict(list(zip(['x', 'y'], [None, None]))) #(-789000, 789000)]))
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
        print(point, vec1, vec2, normal, normal[:2])
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
        self.axis_LUT = dict(list(zip(self.axis_names, (3, 0, 4, 5, 1, 2))))
        for axis in range(6):
            self.set_speed(axis, 1, 500000, 1000)

        self.axis_toggle = dict(k_lens=dict(on=2330000, off=7000000),
                                filter_y=dict(off=8604180, small=338640, medium=3394640, big=6475000),
                                filter_x=dict(off=6000000, small=2204000, medium=1983000, big=2030000))
        # 20um pinhole x,y = [2204000, 338640]
        # 50um pinhole x, y = [1983000, 3394640]
        # 100um pinhole x, y = [2030000, 6475000] [1955000, 6460000]
        self._tomography_limits = [6586000, 7246000]

    def toggle(self, axis, state):
        if axis in self.axis_toggle:
            dictionary = self.axis_toggle[axis]
            if state in dictionary:
                self.move(dictionary[state], axis)
            else:
                self._logger.warn('Unrecognised state %s. Needs to be one of %s' % (state, list(dictionary.keys())))
        else:
            self._logger.warn('Axis %s does not have toggle values' % axis)

    def tomography(self, kvalue):
        """

        :param kvalue: float. In inverse micron
        :return:
        """
        counts = np.interp(kvalue, np.linspace(-4, 4), np.linspace(self._tomography_limits[0],
                                                                   self._tomography_limits[1]))
        self.move(counts, 'spectrometer_lens')


if __name__ == "__main__":
    print('Opening')
    andor_server = AndorServer(("localhost", 9999))
    andor_server._logger.setLevel('DEBUG')
    pvcam = PvcamClient(("172.27.25.39", 9999))
    print('Opened')
    # pvcam.capture()
    pvcam.show_gui()
