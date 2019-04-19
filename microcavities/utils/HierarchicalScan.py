# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app  #, QtCore, uic
from nplab.utils.log import create_logger
import nplab.utils.send_mail as email
import nplab.datafile as df
from microcavities.analysis.utils import SortingKey
import h5py
import yaml
import pymsgbox
import numpy as np
import time, datetime
import os
from collections import OrderedDict


DRY_RUN = False


class HierarchicalScan(object):
    """
    Base class for implementing general hierarchical scans, for both experiments and analysis.

    By hierarchical scans I mean grid scans of many different parameters simultaneously. Each level in the hierarchy
    corresponds to a different parameter, e.g.
        power = 1
            angle = 0
            angle = 1
        power = 2
            angle = 0
            angle = 1

    It is designed thinking that datasets could be HDF5 datasets, or they could be folder directories, or anything else.
    We have abstracted those details by keeping a 'level' object and leaving the user to reimplement the necessary
    methods to address these objects.
    The user must reimplement self.variable_name_value, self.is_lowest_level, and self.next_levels

    TODO: implement analysis at different depths
    TODO: variable creation from a yaml. Both the experiment and the analysis should have access to this file.
    """
    def __init__(self, settings_yaml, **kwargs):
        super(HierarchicalScan, self).__init__()

        self.variables = OrderedDict()

        full_yaml = yaml.load(open(settings_yaml, 'r'))
        self.settings_yaml = full_yaml

        logger_name = "HierarchicalScan"
        if "logger_name" in full_yaml:
            logger_name = full_yaml["logger_name"]
        elif "logger_name" in kwargs:
            logger_name = kwargs["logger_name"]
        self._logger = create_logger(logger_name)

        if "logger_level" in full_yaml:
            self._logger.setLevel(full_yaml["logger_level"])
        elif "logger_level" in kwargs:
            self._logger.setLevel(kwargs["logger_level"])

        self.series_name = 'DefaultSeriesName'
        if 'series_name' in full_yaml:
            self.series_name = full_yaml['series_name']

        self.email_when_done = False
        if 'email_when_done' in full_yaml:
            self.email_when_done = True
            if 'email' in full_yaml:
                self.email_address = full_yaml['email']
            else:
                self.email_address = pymsgbox.prompt('Message box!', 'Title')
                if self.email_address is None:
                    self.email_when_done = False

        for variable in full_yaml['variables']:
            if 'name' in variable:
                name = variable['name']
            else:
                name = self.get_default_name(variable)
                if name in self.variables:
                    self._logger.warn("Repeated variable name. Are you sure you know what you're doing?")

            first_value = variable['values'][0]
            if type(first_value) == str:
                if first_value == 'linear':
                    values = np.linspace(*variable['values'][1:])
                elif first_value == 'random':
                    values = np.random.uniform(*variable['values'][1:])
                else:
                    raise ValueError("Unrecognised variable value type: %s" % name)
            else:
                values = variable['values']

            self.variables[name] = values

    @staticmethod
    def get_default_name(dictionary):
        if 'name' in dictionary:
            name = dictionary['name']
        else:
            name = ''
            if 'instrument' in dictionary:
                name += dictionary['instrument']
            if 'property' in dictionary:
                name += '_' + dictionary['property']
            elif 'function' in dictionary:
                name += '_' + dictionary['function']
        if len(name) == 0:
            raise ValueError("Variable name was not given or couldn't make automatically make one: %s" % dictionary)
        name = name.lstrip('_')
        return name

    @staticmethod
    def save_array(name, array, save_type):
        if DRY_RUN:
            print 'Save array: ', name, array, save_type
            return
        if save_type == 'npy':
            np.save(name, array)
        else:
            np.savetxt(name + '.' + save_type, array)

    @staticmethod
    def load_array(name, save_type):
        if DRY_RUN:
            print 'Loading array: ', name, save_type
            return
        if save_type == 'npy':
            array = np.load(name)
        else:
            array = np.loadtxt(name + '.' + save_type)
        return array

    def iteration_function(self, level, name, value):
        """Function that runs at each level of the scan. Needs overwriting

        :param int level:
        :param str name:
        :param int/float value:
        :return:
        """
        self._logger.debug("Called iteration_function: %s %s %s. Folder_name: %s" %
                           (level, name, value, self.folder_name))

    def final_function(self):
        """Function that runs at the hierarchical bottom of the scan, once all parameters are set. Needs overwriting

        :return:
        """
        self._logger.debug("Called final_function. Folder_name: %s" % self.folder_name)

    def run(self, level=0):
        self.abort = False
        t0 = time.time()
        self._iterate(level)
        total_time = time.time() - t0
        self._logger.info("Scan finished after %s" % datetime.timedelta(seconds=total_time))
        if self.email_when_done:
            email.send_email(self.email_address, "HierarchicalScan %s finished at %s, after %s" %
                             (self.series_name,
                              datetime.datetime.now().strftime('%H %M %S'),
                              datetime.timedelta(seconds=total_time)))

    def abort_scan(self):
        return False

    def _iterate(self, level):
        """
        Recursive function to iterate over a hierarchical dataset. Given an input level, it will either perform analysis
        on it, or perform recursion over it's sublevels

        :param level: level over which you want to act
        :return:
        """
        if level == 0:
            self.folder_name = [self.series_name]

        if level < len(self.variables):
            name, values = self.variables.items()[level]
            self._logger.debug('Iterating %s' % name)
            level += 1
            for value in values:
                if self.abort_scan():
                    break
                self._logger.debug('%s = %g' % (name, value))

                try:
                    self.folder_name[level] = '%s=%g' % (name, value)
                except IndexError:
                    self.folder_name += ['%s=%g' % (name, value)]

                self.iteration_function(level, name, value)
                self._iterate(level)
        else:
            self.final_function()


class ExperimentScan(HierarchicalScan):
    """
    tester.yaml
        base_path: "C:/Users/Rotation/Desktop/TestExperiment"
        series_name: power_scan
        save_type: HDF5

        variables:
          - {instrument: power_wheel,
             function: move,
             values: [linear, 0, 100, 2],
             metadata:[{instrument: temp_gauge, property: temperature}]}

        measurements:
          - {instrument: andor2, function: raw_image, kwargs: {update_latest_frame: True}, save: True}
    """
    def __init__(self, settings_yaml, instrument_dictionary, gui, **kwargs):
        super(ExperimentScan, self).__init__(settings_yaml, logger_name="ExperimentScan", **kwargs)
        self.instr_dict = instrument_dictionary
        self.pyqt_app = get_qt_app()

        if not DRY_RUN:
            self.gui = gui
            self.gui.abortScan = False

        self.measurements = self.settings_yaml["measurements"]
        if isinstance(self.measurements, list):
            self.measurements = {'depth%d' % len(self.variables): self.measurements}

        self.save_type = 'HDF5'
        if "save_type" in self.settings_yaml:
            self.save_type = self.settings_yaml["save_type"]

    def iteration_function(self, level, name, value):
        """At each level of the iteration, we get an instrument and set a property or call a function

        :param int level:
        :param str name:
        :param int/float value:
        :return:
        """
        super(ExperimentScan, self).iteration_function(level, name, value)  # Purely for logging

        dictionary = self.settings_yaml['variables'][level-1]
        instr = self.instr_dict[dictionary['instrument']]
        dictionary['value'] = value
        call_dictionary(instr, dictionary)
        self.pyqt_app.processEvents()

        attributes = self.get_attributes(dictionary)
        folder_name = '/'.join(self.folder_name)
        self.save_attributes(attributes, folder_name)

        self.measure(level)

    def measure(self, level):
        if 'depth%d' % level in self.measurements:
            self._logger.debug('Measuring at level=%d: %s' % (level, self.measurements['depth%d' % level]))

            for measurement in self.measurements['depth%d' % level]:
                if 'wait' in measurement:
                    time.sleep(measurement['wait'])

                if len(measurement.keys()) > 1:
                    measure_name = self.get_default_name(measurement)
                    current_folder = list(self.folder_name[:level+1])
                    file_name = current_folder + [measure_name]
                    file_name = '/'.join(file_name)
                    measurement = yaml.load(str(measurement).replace("scan_file_name", file_name))

                    instr = self.instr_dict[measurement['instrument']]
                    data = call_dictionary(instr, measurement)
                    self.pyqt_app.processEvents()

                    if 'save' in measurement and measurement['save']:
                        self.save(data, file_name, measurement)

    def save(self, data, file_name, dictionary):
        """

        :param data:
        :param file_name:
        :param dictionary:
        :return:
        """
        attributes = self.get_attributes(dictionary)

        self.pyqt_app.processEvents()

        self._logger.debug("Saving %s: %s %s %s" % (file_name, dictionary, data, attributes))
        self._logger.info('Saving: %s' % file_name)
        if self.save_type == 'HDF5':
            self._logger.debug('Creating group: %s %s' % (file_name, attributes))
            if not DRY_RUN:
                self.instr_dict['HDF5'].create_dataset(file_name, data=data, attrs=attributes)
        else:
            lst = file_name.split('/')
            folder_name = '/'.join(lst[:-1])
            if not os.path.exists(folder_name):
                self._logger.debug('Making dir: %s' % folder_name)
                if not DRY_RUN:
                    os.makedirs(folder_name)
            self.save_array(file_name, attributes, self.save_type)
            self.save_attributes(attributes, file_name)
        self.pyqt_app.processEvents()

    def save_attributes(self, attributes, file_name):
        if attributes is not None:
            if self.save_type == 'HDF5':
                self._logger.debug('Creating group: %s %s' % (file_name, attributes))
                if not DRY_RUN:
                    self.instr_dict['HDF5'].create_group(file_name, attributes)
            else:
                if not os.path.exists(file_name):
                    self._logger.debug('Making dir: %s' % file_name)
                    if not DRY_RUN:
                        os.makedirs(file_name)
                self.save_array(file_name + '/attrs', attributes, self.save_type)
            self.pyqt_app.processEvents()

    def abort_scan(self):
        """
        A scan can be paused or aborted at each iteration. This relies on the GUI having the attributes _abort_scan and
        _pause_scan and the function play_scan

        :return:
        """
        if DRY_RUN:
            return False
        if self.gui._abort_scan:
            toggle = pymsgbox.confirm('Do you want to abort?', 'Confirm abort', ["Yes", 'No'])
            if toggle == "Yes":
                return True
            else:
                self.gui._abort_scan = False
                self.gui.play_scan()
                return False
        elif self.gui._pause_scan:
            while self.gui._pause_scan:
                time.sleep(0.05)
                self.pyqt_app.processEvents()
            return False
        else:
            return False

    def run(self, level=0):
        # Ensures that the scan pause and abort options are False
        if not DRY_RUN:
            for button, value in zip(['Abort', 'Play'], [False, True]):
                getattr(self.gui, 'actionScan' + button).setChecked(not value)
                getattr(self.gui, 'actionScan' + button).trigger()

            if 'base_path' in self.settings_yaml:
                filename = self.settings_yaml['base_path']
                self.instr_dict['HDF5'].close()
                self.instr_dict['HDF5'] = df.DataFile(filename)
                self.instr_dict['HDF5'].make_current()
            if self.instr_dict["HDF5"] is None:
                self.gui.menuNewExperiment()

        super(ExperimentScan, self).run(level)

        if not DRY_RUN:
            self.instr_dict['HDF5'].close()

    def get_attributes(self, base_dictionary):
        attributes = None
        if 'metadata' in base_dictionary:
            attributes = {}
            for dictionary in base_dictionary['metadata']:
                instr = self.instr_dict[dictionary['instrument']]
                value = call_dictionary(instr, dictionary)
                name = self.get_default_name(dictionary)
                attributes[name] = value
        self._logger.debug('Getting attributes: %s' % attributes)
        return attributes


class AnalysisScan(HierarchicalScan):
    # TODO: add support for analysing hierarchical files for which we do not have the yaml
    """
    tester.yaml
        experiment_yaml_path: 'C:/Users/Rotation/polariton.riken/Python_lab/Experiments/scripts/test_exper.yaml'
        save_path: "C:/Users/Rotation/Desktop/TestAnalysis"
        raw_data_file:

        analysis_functions:
          depth1:
          - {data_name: andor2_raw_image, function_name: roi_intensity}
          - {data_name: andor2_raw_image, function_name: fit_gaussian}
    """
    def __init__(self, settings_yaml, **kwargs):
        full_yaml = yaml.load(open(settings_yaml, 'r'))
        self.analysis_yaml = full_yaml
        super(AnalysisScan, self).__init__(self.analysis_yaml['experiment_yaml_path'],
                                           logger_name="AnalysisScan", **kwargs)
        self.analysed_data = OrderedDict()
        self.analysis_functions = self.analysis_yaml["analysis_functions"]
        if isinstance(self.analysis_functions, list):
            self.analysis_functions = {'depth%d' % len(self.variables): self.analysis_functions}

        self.save_type = 'HDF5'
        if "save_type" in self.settings_yaml:
            self.save_type = self.settings_yaml["save_type"]

        if self.save_type == 'HDF5' and not DRY_RUN:
            if 'base_path' in self.settings_yaml:
                self.HDF5 = h5py.File(self.settings_yaml['base_path'], 'r')
            elif 'raw_data_file' in self.analysis_yaml:
                self.HDF5 = h5py.File(self.analysis_yaml['raw_data_file'], 'r')

    def get_data(self, file_name):
        if DRY_RUN:
            self._logger.debug('Getting data: %s' % file_name)
            self.load_array(file_name, self.save_type)
            return 1, 0

        if self.save_type == 'HDF5':
            self._logger.debug('Opening group: %s' % file_name)
            data = self.HDF5[file_name][...]
            attrs = dict(self.HDF5[file_name].attrs)
        else:
            data = self.load_array(file_name, self.save_type)
            attrs = {}
        return data, attrs

    def run(self, level=0):
        self.analysed_data = OrderedDict()
        super(AnalysisScan, self).run(level)
        self._reshape_results()

        if 'save_path' in self.analysis_yaml:
            print self.analysis_yaml['save_path']
            h5file = df.DataFile(self.analysis_yaml['save_path'])
            for name, data in self.analysed_data.items():
                self._logger.debug('Creating dataset %s %s %s' % (name, data.shape, data.dtype))
                h5file.create_dataset(name, data=data)
            h5file.flush()
            h5file.close()

    def iteration_function(self, level, name, value):
        """At each level of the iteration, we get an instrument and set a property or call a function

        :param int level:
        :param str name:
        :param int/float value:
        :return:
        """
        super(AnalysisScan, self).iteration_function(level, name, value)  # Purely for logging

        if 'depth%d' % level in self.analysis_functions:
            for analysis_dict in self.analysis_functions['depth%d' % level]:
                data_name = analysis_dict["data_name"]

                current_folder = list(self.folder_name[:level + 1])
                file_name = current_folder + [data_name]
                file_name = '/'.join(file_name)
                raw_data, attributes = self.get_data(file_name)
                exec("from microcavities.analysis.analysis_functions import %s" % analysis_dict['function_name'])
                args = []
                if 'args' in analysis_dict:
                    args = analysis_dict['args']
                kwargs = {}
                if 'kwargs' in analysis_dict:
                    kwargs = analysis_dict['kwargs']
                data, args, kwargs = eval(analysis_dict['function_name'])(raw_data, *args, **kwargs)
                analysis_dict['args'] = args
                analysis_dict['kwargs'] = kwargs

                if 'analysed_name' in analysis_dict:
                    analysed_name = analysis_dict['analysed_name']
                else:
                    analysed_name = analysis_dict['function_name']

                if analysed_name in self.analysed_data:
                    self.analysed_data[analysed_name] += [data]
                else:
                    self.analysed_data[analysed_name] = [data]

    def _reshape_results(self):
        """
        Since self.analyse stores results as flat-arrays, and that's not terribly useful, this reshapes them into numpy
        arrays where each dimension corresponds to one of the parameters in the hierarchy

        :return:
        """
        self._logger.debug('Reshaping results')

        new_shape = tuple([len(var) for var in self.variables.values()])
        self._logger.debug('New shape: %s' % new_shape)

        for name, result in self.analysed_data.items():
            if not isinstance(result, np.ndarray):
                self._logger.debug('Making into array')
                result = np.array(result)

            current_shape = result.shape
            self._logger.debug('Current %s shape: %s' % (name, current_shape))

            self.analysed_data[name] = result.reshape(new_shape + current_shape[1:])

    def _extract_hierarchy(self, levels):
        """
        Iterates over levels, checking that only one parameter is being changed, and storing the parameter values (if any)

        :param levels:
        :return:
        """
        self._logger.debug('Extracting variables from: %s' % levels)

        var_name = None
        values = []
        for level in levels:
            name, value = self.variable_name_value(level)

            values += [value]
            if var_name is None:
                var_name = name
            elif var_name != name:
                raise ValueError('There are differently named variables in the same hierarchical level: %s %s' %
                                 (var_name, name))

        if var_name is not None:
            self.variables[var_name] = values

    def get_random_group(self, level):
        """
        Iterates through the hdf5 file by randomly selecting a group lower down the hierarchy (which matches the form
        'varname=varvalue') until it finds a group called groupName.

        Say your file has the following hierarchy:
            FullDataset:
                Focus=1:
                    Power=2:
                        Spin:
                            Image1
                            Image2
                    Power=3:
                        Spin:
                            Image1
                            Image2
                Focus=2:
                    Power=2:
                        Spin:
                            Image1
                            Image2
                    Power=3:
                        Spin:
                            Image1
                            Image2
        This function allows you to randomly select one of the 'Spin' datasets, returning one of the following strings:
            FullDataset/Focus=1/Power=2/Spin
            FullDataset/Focus=1/Power=3/Spin
            FullDataset/Focus=2/Power=2/Spin
            FullDataset/Focus=2/Power=3/Spin
        :param level: starting level
        :return:
        """

        if self.is_lowest_level(level):
            return level
        else:
            levels = self.next_levels(level)
            indices = range(len(levels))
            return self.get_random_group(levels[np.random.choice(indices)])


def call_dictionary(obj, dictionary):
    print "call_dictionary: %s %s" % (obj, dictionary)

    if DRY_RUN:
        print "call_dictionary: %s %s" % (obj, dictionary)
        return 1

    if 'function' in dictionary and 'property' in dictionary:
        raise ValueError("Provided a function and a property. Please provide only one.")
    elif 'function' in dictionary:
        if 'value' in dictionary:
            args = [dictionary['value']]
        else:
            args = []
        kwargs = dict()
        if 'args' in dictionary:
            args = dictionary['args']
        if 'kwargs' in dictionary:
            kwargs = dictionary['kwargs']
        return getattr(obj, dictionary['function'])(*args, **kwargs)
    elif 'property' in dictionary:
        if 'value' in dictionary:
            return setattr(obj, dictionary['property'], dictionary['value'])
        else:
            return getattr(obj, dictionary['property'])
    else:
        raise ValueError


if __name__ == "__main__":
    # tst = HierarchicalScan(
    #     r'C:\Users\Rotation\Documents\ExperimentControl\microcavities\microcavities\experiment\experiment_scan.yaml')
    # tst._logger.setLevel("DEBUG")
    # tst.run()

    # instruments = dict(andor='Yay', streak='Yay2', power_wheel='Yay3', temperature_control='Yay4')
    # tst = ExperimentScan(
    #     r'C:\Users\Rotation\Documents\ExperimentControl\microcavities\microcavities\experiment\experiment_scan.yaml',
    #     instruments, None)
    # tst._logger.setLevel("DEBUG")
    # tst.run()

    tst = AnalysisScan(
        r'C:\Users\Rotation\Documents\ExperimentControl\microcavities\microcavities\experiment\analysis_scan.yaml')
    tst._logger.setLevel("DEBUG")
    tst.run()