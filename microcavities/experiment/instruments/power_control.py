# -*- coding: utf-8 -*-
# from nplab.instrument import Instrument
# from nplab.instrument.stage.SigmaKoki import SHOT
from nplab.instrument.electronics.Meadowlark import VariableRetarder
from nplab.instrument.Flipper.thorlabs_MFF002 import ThorlabsMFF
from nplab.instrument.electronics.NewportPowermeter import NewportPowermeter
from nplab.instrument.stage.wheel_of_power import PowerWheelMixin
# import nidaqmx
# import nidaqmx.stream_writers
import time
import numpy as np


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

    @property
    def raw_power(self):
        return self.voltage

    @raw_power.setter
    def raw_power(self, value):
        self.voltage = value


# class Flipper(Instrument):
#
#     def __init__(self, device="Dev1", channel="PFI0"):
#         super(Flipper, self).__init__()
#         self.id = "%s/%s" % (device, channel)
#         self.state = None
#
#     def on(self):
#         with nidaqmx.Task() as task:
#             task.do_channels.add_do_chan('Dev1/PFI0')
#             task.write(True)
#         self.state = True
#
#     def off(self):
#         with nidaqmx.Task() as task:
#             task.do_channels.add_do_chan('Dev1/PFI0')
#             task.write(False)
#         self.state = False
#
#     def toggle(self):
#         if self.state:
#             self.off()
#         else:
#             self.on()


# class NdWheel(SHOT, PowerWheelMixin):
#     axis_names = (1, )
#
#     def __init__(self, address):
#         super(NdWheel, self).__init__(address)
#         self._count_deg = 200.
#         self._raw_max = 360
#
#     def move(self, angle, axis=1, relative=False, wait=True):
#         if not relative:
#             curr_angle = self.get_position(axis)
#             angle_diff = angle - curr_angle
#
#             if angle_diff < -10:
#                 angle_diff += 360
#
#             counts = angle_diff * self._count_deg
#         else:
#             counts = angle * self._count_deg
#
#         super(NdWheel, self).move(counts, axis, relative=True, wait=wait)
#
#     def get_position(self, axis=1):
#         counts = super(NdWheel, self).get_position(axis)
#         if hasattr(counts, '__iter__'):
#             counts = np.array(counts)
#         return (counts/self._count_deg) % 360
#
#     @property
#     def raw_power(self):
#         return self.position
#
#     @raw_power.setter
#     def raw_power(self, value):
#         self.move(value)
#
#     def prepare_calibration(self, calibration):
#         angles = calibration[0]
#         powers = calibration[1]
#
#         minidx = np.argmin(powers)
#         self.move(angles[minidx])
#         self.set_origin()
#
#         powers = np.roll(powers, -minidx)
#         angles = np.roll(angles, -minidx) % 360
#
#         maxidx = np.argmin(np.abs(angles - 320))
#         powers = powers[:maxidx]
#         angles = angles[:maxidx]
#
#         return np.array([angles, powers])


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