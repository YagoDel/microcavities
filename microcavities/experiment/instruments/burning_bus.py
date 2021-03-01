# -*- coding: utf-8 -*-
from nplab.instrument import Instrument
from nplab.instrument.visa_instrument import VisaInstrument
from nplab.instrument.temperatureControl import TemperatureControlMixin
import re
import nidaqmx
import nidaqmx.stream_writers
import numpy as np


class Matisse(VisaInstrument, Instrument):
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


class OxfordItc(VisaInstrument, TemperatureControlMixin, Instrument):
    def __init__(self, address):
        connection_settings = dict(baud_rate=115200, timeout=5000, read_termination='\n', write_termination='\n')
        super(OxfordItc, self).__init__(address, settings=connection_settings)

    def get_temperature(self):
        reply = self.query('READ:DEV:MB1.T1:TEMP:SIG:TEMP')
        temp = float(reply.split(':')[-1][:-1])
        return temp
    temperature = property(fget=get_temperature)


class AOM(Instrument):
    def __init__(self, device="Dev1", channel="ao0"):
        super(AOM, self).__init__()
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

