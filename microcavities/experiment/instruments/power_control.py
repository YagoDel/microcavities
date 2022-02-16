# -*- coding: utf-8 -*-
from nplab.instrument.message_bus_instrument import queried_property
from nplab.instrument.stage.SigmaKoki import SHOT
from nplab.instrument.electronics.Meadowlark import VariableRetarder
from nplab.instrument.Flipper.thorlabs_MFF002 import ThorlabsMFF
from nplab.instrument.electronics.NewportPowermeter import NewportPowermeter
from nplab.instrument.stage.wheel_of_power import PowerWheelMixin
from nplab.instrument.electronics.rigol import RigolDG1022Lite, rigol_property
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

    def plot_calibration(self, ax=None):
        if ax is None:
            from microcavities.utils.plotting import plt
            fig, ax = plt.subplots(1, 1)
        ax.plot(*self.calibration)


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
        angles = (np.roll(angles, -minidx) - angles[minidx]) % 360

        maxidx = np.argmin(np.abs(angles - 320))
        powers = powers[:maxidx]
        angles = angles[:maxidx]

        return np.array([angles, powers])

    def plot_calibration(self, ax=None):
        if ax is None:
            from microcavities.utils.plotting import plt
            fig, ax = plt.subplots(1, 1)
        ax.semilogy(*self.calibration)


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


class AcoustoOpticModulator(RigolDG1022Lite, PowerWheelMixin):
    DELAY = 0.01

    def __init__(self, default_channel=1, *args, **kwargs):
        super(AcoustoOpticModulator, self).__init__(*args, **kwargs)
        self.DEFAULT_CHANNEL = default_channel
        self.output = 0
        self.offset = 0.1
        self.output_polarity = 'normal'
        self.trigger_source = 'external'
        self.output_mode = 'normal'
        self.mode('pulse')
        self.frequency = 500
        # self.voltage_high = 0.1
        self.voltage_low = 0
        self.pulse_width = 1e-3
        self.pulse_period = self.pulse_width + 1e-3
        self.phase = 0
        self.output = 1

    def mode(self, mode):
        assert mode in ['dc', 'pulse', 'chopper']
        self._mode = mode

        if mode == 'dc':
            self.waveform = 'dc'
            self.output_polarity = 'normal'
            self.offset = self.voltage_high
        elif mode == 'pulse':
            self.waveform = 'pulse'
            self.voltage_high = self.offset
            self.output_polarity = 'normal'
            self.burst = 1
            self.burst_mode = 'triggered'
            self.burst_ncycles = 1
            self.burst_idle = 'bottom'
        elif mode == 'chopper':
            self.waveform = 'pulse'
            self.voltage_high = self.offset
            self.output_polarity = 'inverted'
            self.burst = 0

    @property
    def amplitude(self):
        if self._mode == 'dc':
            return self.offset
        elif self._mode == 'pulse':
            return self.voltage_high
        else:
            raise ValueError()
    @amplitude.setter
    def amplitude(self, value):
        assert 0 <= value <= 1
        if self._mode == 'dc':
            self.offset = value
        elif self._mode == 'pulse':
            self.voltage_high = value
        else:
            raise ValueError()

    @property
    def exposure(self):
        return self.pulse_width

    @exposure.setter
    def exposure(self, value):
        # self.frequency = 1 / 1.1*value
        self.pulse_period = 1.1*value  # does not matter, as long as it's larger than the pulse width
        self.pulse_width = value

    @property
    def raw_power(self):
        return self.amplitude

    @raw_power.setter
    def raw_power(self, value):
        self.amplitude = value

    def calibrate(self, *args, **kwargs):
        self.mode('dc')
        super(AcoustoOpticModulator, self).calibrate(*args, **kwargs)
        # TODO: automatically reset the mode after the background thread ends

    def plot_calibration(self, ax=None):
        if ax is None:
            from microcavities.utils.plotting import plt
            fig, ax = plt.subplots(1, 1)
        ax.plot(*self.calibration)
