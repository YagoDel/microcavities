# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app, QtWidgets, QtCore, uic
from nplab.utils.log import create_logger
import nplab.utils.send_mail as email
import nplab.datafile as df
from nplab.experiment.gui import ExperimentWithProgressBar, ExperimentStopped
from microcavities.utils.HierarchicalDatastructure import SortingKey
from microcavities.analysis.streak import open_image
from microcavities.utils import string_to_number, yaml_loader, get_data_path, get_data_directory
import h5py
import yaml
import pymsgbox
import numpy as np
import time
import datetime
import os
import re
import sys
from collections import OrderedDict
import threading


DRY_RUN = False


class HierarchicalScan(ExperimentWithProgressBar):
    """
    Base class for implementing general hierarchical scans, for both experiments and analysis.

    Hierarchical scan means a grid scans of many different parameters simultaneously. Each level in the hierarchy
    corresponds to a different parameter, e.g.
        power = 1
            angle = 0
            angle = 1
        power = 2
            angle = 0
            angle = 1

    It is designed without specifying the type of data structure that could be used, but has been mostly tested on HDF5
    We have abstracted those details by keeping a 'level' object and leaving the user to reimplement the necessary
    methods to address these objects.
    The user may re-implement self.iteration_function, self.final_function, and self.abort_scan
    """
    def __init__(self, settings_yaml, **kwargs):
        """

        :param settings_yaml: yaml with at least the following keys
            - logger_name (opt)
            - logger_level (opt)
            - series_name (opt)
            - email_when_done (opt)
            - variables (required)
        :param kwargs:
        """
        super(HierarchicalScan, self).__init__()

        self.variables = OrderedDict()
        self.progress = 1
        self.progress_maximum = None
        full_yaml = yaml_loader(settings_yaml)

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
                self.email_address = pymsgbox.prompt('What email do you want to notify?', 'Email')
                if self.email_address is None:
                    self.email_when_done = False

    def update_progress(self, progress):
        """Update the progress bar (NB should only be called from within run()"""
        if not self.running:
            # if run was called directly, fail gracefully
            self._logger.debug("Progress: {}".format(progress))
            return
        try:
            self._progress_bar.setValueLater(progress)
        except AttributeError:
            self._logger.warn("Error setting progress bar to {} (are you running via run_modally()?)".format(progress))
        if self._stop_event.is_set():
            raise ExperimentStopped()

    def make_variables(self, yaml_file):
        for variable in yaml_file['variables']:
            if 'name' in variable:
                name = variable['name']
            else:
                name = self.get_default_name(variable)
                if name in self.variables:
                    self._logger.warn("Repeated variable name. Are you sure you know what you're doing?")

            if name == 'repeat':
                values = list(range(variable['values']))
            else:
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
        self.progress_maximum = np.prod([len(x) for x in list(self.variables.values())])

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
            print('Save array: ', name, array, save_type)
            return
        if save_type == 'npy':
            np.save(name, array)
        elif save_type == 'HDF5':
            with h5py.File(name[0]) as data_file:
                data_file.create_dataset(name[1], array.shape, array.dtype, array)
        # elif save_type == 'local':
        #     name = file_name.split('/')[-1]
        #     if name in self.results:
        #         self.results[name] += [array]
        #     else:
        #         self.results[name] = [array]
        else:
            np.savetxt(name + '.' + save_type, array)

    @staticmethod
    def load_array(name, save_type):
        if DRY_RUN:
            print('Loading array: ', name, save_type)
            return
        if save_type == 'npy':
            return np.load(name)
        elif save_type == 'HDF5':
            with h5py.File(name[0]) as data_file:
                array = data_file[name[1]][...]
            return array
        elif save_type == 'streak':
            return open_image(name + '.tif')[0]
        else:
            return np.loadtxt(name + '.' + save_type)

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

    def run_modally(self, *args, **kwargs):
        self.progress = 1
        self.progress_maximum = np.prod([len(x) for x in list(self.variables.values())])
        super(HierarchicalScan, self).run_modally(*args, **kwargs)

    def prepare_run(self):
        self.make_variables(self.settings_yaml)

    def run(self):
        """Starts the recursive iteration, timing it and emailing when done

        :return:
        """
        t0 = time.time()
        self.prepare_run()
        self._iterate(level=0)
        total_time = time.time() - t0
        self._logger.info("Scan finished after %s" % datetime.timedelta(seconds=total_time))
        if self.email_when_done:
            email.send_email(self.email_address, "HierarchicalScan %s finished at %s, after %s" %
                             (self.series_name,
                              datetime.datetime.now().strftime('%H %M %S'),
                              datetime.timedelta(seconds=total_time)))

    def abort_scan(self):
        """Utility function to abort HierarchicalScan

        :return: bool. If True, the scan will be aborted at the next iteration.
        """
        return False

    def _iterate(self, level):
        """Utility function to recursively iterate over a hierarchy of variables

        Given an input level, if there are sublevels, it will call self.iteration_function on each of the sublevels and
        then call self._iterate on those sublevels. If there are no sublevels, it will call self.final_function

        :param level: int. level over which to iterate
        :return:
        """
        if level == 0:
            self.folder_name = [self.series_name]

        if level < len(self.variables):
            name, values = list(self.variables.items())[level]
            self._logger.debug('Iterating %s' % name)
            level += 1
            for value in values:
                self.update_progress(self.progress)
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
            self.progress += 1
            self.final_function()

    def _reshape_results(self, results):
        """
        Since sometimes results are stored as as flat-arrays, and that's not terribly useful, this reshapes them into
        numpy arrays where each dimension corresponds to one of the parameters in the hierarchy

        :return:
        """
        self._logger.debug('Reshaping results')

        new_shape = tuple([len(var) for var in list(self.variables.values())])
        self._logger.debug('New shape: %s' % (new_shape,))

        new_results = dict()
        for name, result in list(results.items()):
            if not isinstance(result, np.ndarray):
                self._logger.debug('Making into array')
                result = np.array(result)

            current_shape = result.shape
            self._logger.debug('Current %s shape: %s' % (name, current_shape))

            new_results[name] = result.reshape(new_shape + current_shape[1:])
        return new_results


class ExperimentScan(HierarchicalScan):
    """
    tester.yaml
        file_name: "C:/Users/Rotation/Desktop/TestExperiment"
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
    def __init__(self, instrument_dictionary, gui, settings_yaml=None, **kwargs):
        self.instr_dict = instrument_dictionary
        self.pyqt_app = get_qt_app()

        if not DRY_RUN:
            self.gui = gui
            self.gui._abort_scan = False
        if settings_yaml is None:
            settings_yaml = self.settings_popup()

        super(ExperimentScan, self).__init__(settings_yaml, logger_name="ExperimentScan", **kwargs)

        self.measurements = self.settings_yaml["measurements"]

        self.save_type = 'HDF5'
        if "save_type" in self.settings_yaml:
            self.save_type = self.settings_yaml["save_type"]
        if self.save_type == 'local':
            self.results = []

        if self.save_type == 'HDF5':
            try:
                self.instr_dict['HDF5'].close()
            except:
                pass  # ignore if already closed

    def iteration_function(self, level, name, value):
        """At each level of the iteration, we get an instrument and set a property or call a function

        :param int level:
        :param str name:
        :param int/float value:
        :return:
        """
        super(ExperimentScan, self).iteration_function(level, name, value)  # Purely for logging

        dictionary = self.settings_yaml['variables'][level-1]
        if 'instrument' in dictionary:
            instr = self.instr_dict[dictionary['instrument']]
        else:
            instr = None  # for some iterations (notably repeated measurements), an instrument is not needed
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

                if len(list(measurement.keys())) > 1:
                    measure_name = self.get_default_name(measurement)
                    current_folder = list(self.folder_name[:level+1])
                    file_name = current_folder + [measure_name]
                    file_name = '/'.join(file_name)
                    # The next line allows one to access the file_name from the original yaml file
                    measurement = yaml.full_load(str(measurement).replace("scan_file_name", file_name))
                    instr = self.instr_dict[measurement['instrument']]

                    data = call_dictionary(instr, measurement)
                    self.pyqt_app.processEvents()
                    if 'save' in measurement and measurement['save']:
                        self.save(data, file_name, measurement)
                else:
                    self._logger.warn('Measurement is an empty dictionary: %s' % measurement)

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
        elif self.save_type == 'local':
            name = file_name.split('/')[-1]
            if name in self.results:
                self.results[name] += [data]
            else:
                self.results[name] = [data]
        elif self.save_type != 'None':
            lst = file_name.split('/')
            folder_name = '/'.join(lst[:-1])
            if not os.path.exists(folder_name):
                self._logger.debug('Making dir: %s' % folder_name)
                if not DRY_RUN:
                    os.makedirs(folder_name)
            self.save_array(file_name, attributes, self.save_type)
            self.save_attributes(attributes, file_name)
        else:
            self._logger.debug('Saving is disabled')

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

    def prepare_run(self):
        super(ExperimentScan, self).prepare_run()
        if isinstance(self.measurements, list):
            self.measurements = {'depth%d' % len(self.variables): self.measurements}

    def run(self):
        # Ensures that the scan pause and abort options are False
        if not DRY_RUN:
            for button, value in zip(['Abort', 'Play'], [False, True]):
                getattr(self.gui, 'actionScan' + button).setChecked(not value)
                getattr(self.gui, 'actionScan' + button).trigger()

            if self.save_type == 'local':
                self.results = dict()
            elif 'raw_data_file' in self.settings_yaml:
                path = self.settings_yaml['raw_data_file']
                if not os.path.isabs(path):
                    path = get_data_path(path)
                # self.base_path = os.path.dirname(path)
                self.instr_dict['HDF5'] = df.DataFile(path)
            elif 'base_path' in self.settings_yaml:
                # self.base_path = self.settings_yaml['base_path']
                path = self.settings_yaml['base_path']
                self.instr_dict['HDF5'] = df.DataFile(path)
            elif self.instr_dict["HDF5"] is None:
                self.gui.menuNewExperiment()
            else:
                raise RuntimeError('Do not know how to handle instr_dict.HDF5: %s' % self.instr_dict['HDF5'])
            self.instr_dict['HDF5'].make_current()

            # if 'file_name' in self.settings_yaml:
            #     filename = self.settings_yaml['file_name']
            #     if isinstance(self.instr_dict['HDF5'], h5py.File):
            #         self.instr_dict['HDF5'].close()
            #     self.instr_dict['HDF5'] = df.DataFile(filename)
            #     self.instr_dict['HDF5'].make_current()
            # if self.instr_dict["HDF5"] is None:
            #     self.gui.menuNewExperiment()

        super(ExperimentScan, self).run()

        if not DRY_RUN:
            if self.save_type == 'local':
                self.results = self._reshape_results(self.results)
            else:
                self.instr_dict['HDF5'][self.series_name].attrs['yaml'] = yaml.dump(self.settings_yaml)
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

    def settings_popup(self):
        ui = ExperimentYamlSetup(self.instr_dict, self.gui)
        ui.exec_()
        yaml_path = ui.yaml_path
        return yaml_path


class AnalysisScan(HierarchicalScan):
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
        full_yaml = yaml_loader(settings_yaml)

        # Sometimes you want to let the analysis figure out on its own what the variables are, instead of reading them
        # from the yaml (if you've lost the yaml for example)
        passing_yaml = full_yaml
        if 'variables' not in passing_yaml:
            extract_hierarchy = True
            passing_yaml['variables'] = []
        else:
            extract_hierarchy = False

        super(AnalysisScan, self).__init__(passing_yaml, logger_name="AnalysisScan", **kwargs)

        self.save_type = 'HDF5'
        if "save_type" in self.settings_yaml:
            self.save_type = self.settings_yaml["save_type"]

        if self.save_type == 'HDF5' and not DRY_RUN:
            file_name = None
            if 'file_name' in self.settings_yaml:
                file_name = self.settings_yaml['file_name']
            elif 'raw_data_file' in self.settings_yaml:
                file_name = self.settings_yaml['raw_data_file']
            if file_name is not None:
                if os.path.isabs(file_name):
                    self.HDF5 = h5py.File(file_name, 'r')
                elif 'directory' in self.settings_yaml:
                    self.HDF5 = h5py.File(os.path.join(self.settings_yaml['directory'], file_name), 'r')
                else:
                    self.HDF5 = h5py.File(get_data_path(file_name), 'r')

        self.analysed_data = None
        if "analysis_functions" in self.settings_yaml:
            self.analysis_functions = self.settings_yaml["analysis_functions"]
        elif "measurements" in self.settings_yaml:
            self.analysis_functions = []
            for idx, measurement in enumerate(self.settings_yaml['measurements']):
                dataname = self.get_default_name(measurement)
                self.analysis_functions += [dict(function_name='raw_%s' % dataname, data_name=dataname)]
        else:
            raise RuntimeError("Neither analysis_functions or measurements were provided in the yaml")

        if extract_hierarchy:
            self.extract_hierarchy()

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

    def run(self):
        self.analysed_data = OrderedDict()
        super(AnalysisScan, self).run()
        self.analysed_data = self._reshape_results(self.analysed_data)

        if 'save_path' in self.settings_yaml:
            self._logger.debug('Save path: %s' % self.settings_yaml['save_path'])
            h5file = df.DataFile(self.settings_yaml['save_path'])
            for name, data in list(self.analysed_data.items()):
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

                args = []
                if 'args' in analysis_dict:
                    args = analysis_dict['args']
                kwargs = {}
                if 'kwargs' in analysis_dict:
                    kwargs = analysis_dict['kwargs']

                if analysis_dict['function_name'].startswith('raw'):
                    data = np.copy(raw_data)
                else:
                    exec("from microcavities.analysis.analysis_functions import %s" % analysis_dict['function_name'])
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

    def make_variables(self, yaml_file):
        if 'variables' in yaml_file:
            super(AnalysisScan, self).make_variables(yaml_file)
        else:
            self.extract_hierarchy()
        if isinstance(self.analysis_functions, list):
            self.analysis_functions = {'depth%d' % len(self.variables): self.analysis_functions}

    def extract_hierarchy(self):
        """Base call for the recursive iterator to get a file's hierarchy

        To be used in situations where we do not have access to the yaml used to run the experiment

        :return:
        """
        if self.series_name == 'DefaultSeriesName':
            reply = pymsgbox.confirm('The series name is DefaultSeriesName. Is that what you want?')
            if reply == 'Cancel':
                if 'series_name' in self.settings_yaml:
                    self.series_name = self.settings_yaml['series_name']
                elif self.save_type == 'HDF5' and len(list(self.HDF5.keys())) == 1:
                    self.series_name = list(self.HDF5.keys())[0]
                else:
                    self._logger.warn('series_name could not be determined. Please provide an experiment yaml, a data '
                                      'file with just one key at the top level, or a series_name in the')
                    return
        self.variables = OrderedDict()
        self._extract_hierarchy(self.series_name)

    def _extract_hierarchy(self, level_name):
        """Recursive function for extracting a hierarchy

        Makes use of the self.next_levels and self.is_lowest_level methods so that different types of file structures
        can be analysed

        :param level_name: string
        :return:
        """
        self._logger.debug('Extracting variables from: %s' % level_name)

        var_name = None
        next_levels = self.next_levels(level_name)
        for level in next_levels:
            if not self.is_lowest_level(level):
                # last_level = level.split('/')[-1]
                last_level = os.path.split(level)[-1]
                name, value = re.findall(r'(.+)=(\d+(?:\.\d*)?(?:e[+-]\d+)?)', last_level)[0]
                value = string_to_number(value)
                self._logger.debug('Extracted variable and value: %s = %g' % (name, value))

                if var_name is None:
                    var_name = name
                elif var_name != name:
                    raise ValueError('There are differently named variables in the same hierarchical level: %s %s' %
                                     (var_name, name))

                if var_name not in self.variables:
                    self.variables[var_name] = [value]
                else:
                    if value not in self.variables[var_name]:
                        self.variables[var_name] += [value]

                self._extract_hierarchy(level)

    def next_levels(self, level_name):
        """Currently only works for HDF5
        Returns the sorted keys under the current level

        :param level_name:
        :return:
        """
        if self.save_type == 'HDF5':
            keys = list(self.HDF5[level_name].keys())
            keys.sort(key=SortingKey)
            return [level_name + '/' + key for key in keys]
        elif self.save_type == 'streak':
            directory = os.listdir(level_name)
            directory.sort(key=SortingKey)
            return [os.path.join(level_name, x) for x in directory]
        else:
            raise ValueError('Unrecognised save_type: %s' % self.save_type)

    def is_lowest_level(self, level_name):
        """Returns whether a level is a Dataset (True) or a Group (False)"""
        if self.save_type == 'HDF5':
            return isinstance(self.HDF5[level_name], h5py.Dataset)
        elif self.save_type == 'streak':
            return os.path.isfile(level_name)
        else:
            raise ValueError('Unrecognised save_type: %s' % self.save_type)

    def get_random_group(self, level):
        """
        Iterates through the file structure by randomly selecting a group lower down the hierarchy until it gets to the
        lowest level. It requires the self.next_levels and self.is_lowest_level methods

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
            indices = list(range(len(levels)))
            return self.get_random_group(levels[np.random.choice(indices)])


def call_dictionary(obj, dictionary):
    """Utility function to get/set properties or call functions

    :param obj: object containing the properties/methods to be called
    :param dictionary: dict. Keys should have one of either 'function' or 'property'. If 'function', can also provide
    list (args) and named (kwargs) arguments to be passed when calling the method. If 'property' and 'value' given, the
    property is set; otherwise it returns the value of the property
    :return:
    """
    if DRY_RUN:
        print("call_dictionary: %s %s" % (obj, dictionary))
        return 1

    # if 'name' in dictionary and dictionary['name'] == 'repeat':
    #     return None

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
        raise ValueError("Neither function nor property provided. Please provide only one.")


class ExperimentYamlSetup(QtWidgets.QDialog):
    """
    Functionality
        GUI for adding measurements
            Add instrument name, function/method name
        Data file
        Series name
        GUI for adding variables
    """
    def __init__(self, instrument_dictionary, gui):
        super(ExperimentYamlSetup, self).__init__()
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        self._n_variables = 0
        self._n_measurements = 0
        self.variables = []
        self.measurements = []
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ExperimentYamlSetup.ui'), self)

        # Set default data path
        data_path = get_data_path()
        data_directory = get_data_directory()
        self.lineEdit_filename.setText(data_path.lstrip(data_directory))
        self.lineEdit_seriesname.setText('DefaultScan')

        self._connect_gui()

    def _connect_gui(self):
        self.spinBox_variables.valueChanged.connect(self.update_variables)
        self.spinBox_measurements.valueChanged.connect(self.update_measurements)
        self.pushButton_yaml.pressed.connect(self.make_yaml)
        # self.pushButton_run.pressed.connect(self.run)
        # self.pushButton_analyse.pressed.connect(self.analyse)

    @property
    def filename(self):
        path = os.path.normpath(self.lineEdit_filename.text())
        if not os.path.isabs(path):
            data_directory = get_data_directory()
            path = os.path.join(data_directory, path)
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        return path

    def update_variables(self):
        # instrument property/function (args, kwargs) value_min value_max steps linear/random
        # defaults = [''] * 8
        n_var = int(self.spinBox_variables.value())
        if n_var > self._n_variables:
            for row in range(self._n_variables, n_var):
                widgets = [QtWidgets.QLineEdit('') for x in range(4)]
                widgets += [QtWidgets.QPlainTextEdit('')]
                widgets += [QtWidgets.QLineEdit('') for x in range(3)]
                qbox = QtWidgets.QComboBox()
                qbox.addItem('linear')
                qbox.addItem('random')
                widgets += [qbox]
                self.variables += [widgets]
                [self.gridLayout_variables.addWidget(wdgt, row+3, col) for wdgt, col in
                 zip(widgets, range(len(widgets)))]
        else:
            for row in range(n_var, self._n_variables):
                [wdgt.deleteLater() for wdgt in self.variables[row]]
            self.variables = self.variables[:n_var]
        self._n_variables = int(n_var)
        if n_var > 3:
            self.adjustSize()
        return

    def update_measurements(self):
        # name instrument save property/function (args, kwargs)
        n_mes = int(self.spinBox_measurements.value())
        if n_mes > self._n_measurements:
            for row in range(self._n_measurements, n_mes):
                widgets = [QtWidgets.QLineEdit('') for _ in range(4)]
                widgets += [QtWidgets.QPlainTextEdit('')]
                widgets += [QtWidgets.QLineEdit('')]
                chk_bx = QtWidgets.QCheckBox()
                chk_bx.setChecked(True)
                widgets += [chk_bx]

                self.measurements += [widgets]
                [self.gridLayout_measurements.addWidget(wdgt, row+3, col) for wdgt, col in
                 zip(widgets, range(len(widgets)))]
        else:
            for row in range(n_mes, self._n_measurements):
                [wdgt.deleteLater() for wdgt in self.measurements[row]]
            self.measurements = self.measurements[:n_mes]
        self._n_measurements = int(n_mes)
        if n_mes > 3:
            self.adjustSize()
        return

    def make_yaml(self):
        variables = []
        for idx, var_wdgts in enumerate(self.variables):
            instrument, prop, func = [x.text() for x in var_wdgts[:3]]
            variable = dict(instrument=instrument)
            print(variable)
            if prop != '' and func != '':
                raise ValueError('Both a function and a property variable given at %d' % idx)
            elif func != '':
                variable['function'] = func
                args = str(var_wdgts[3].text())
                kwargs = str(var_wdgts[4].toPlainText())
                if len(args) > 0:
                    args = [yaml.load(x, Loader=yaml.FullLoader) for x in args.split(',')]
                    variable['args'] = args
                if len(kwargs) > 0:
                    _rows = [yaml.load(x, Loader=yaml.FullLoader) for x in kwargs.replace('=', ' : ').split('\n')]
                    print(_rows)
                    kwargs = {list(x.keys())[0]: x[list(x.keys())[0]] for x in _rows}
                    variable['kwargs'] = kwargs
            elif prop != '':
                variable['property'] = prop
            else:
                raise ValueError('Neither a function nor a property variable given at %d' % idx)

            minval, maxval = [np.float(x.text()) for x in var_wdgts[5:7]]
            steps = int(var_wdgts[7].text())
            mode = str(var_wdgts[8].currentText())
            variable['values'] = [mode, minval, maxval, steps]
            variables += [variable]

        measurements = []
        for idx, mes_wdgts in enumerate(self.measurements):
            instrument, prop, func = [x.text() for x in mes_wdgts[:3]]
            measurement = dict(instrument=instrument)
            if prop != '' and func != '':
                raise ValueError('Both a function and a property variable given at %d' % idx)
            elif func != '':
                measurement['function'] = func
                args = str(mes_wdgts[3].text())
                kwargs = str(mes_wdgts[4].toPlainText())
                if len(args) > 0:
                    args = [yaml.load(x, Loader=yaml.FullLoader) for x in args.split(',')]
                    measurement['args'] = args
                if len(kwargs) > 0:
                    _rows = [yaml.load(x, Loader=yaml.FullLoader) for x in kwargs.replace('=', ':').split('\n')]
                    kwargs = {x.keys()[0]: x.values()[0] for x in _rows}
                    measurement['kwargs'] = kwargs
            elif prop != '':
                measurement['property'] = prop
            else:
                raise ValueError('Neither a function nor a property variable given at %d' % idx)

            name = mes_wdgts[5].text()
            measurement['name'] = name
            measurement['save'] = bool(mes_wdgts[6].isChecked())
            measurements += [measurement]

        full_path = self.filename
        default_directory = get_data_directory()
        if os.path.commonpath([full_path, default_directory]) == default_directory:
            full_dict = dict(raw_data_file=os.path.relpath(full_path, default_directory))
        else:
            full_dict = dict(raw_data_file=full_path)

        series_name = str(self.lineEdit_seriesname.text())
        if series_name != '':
            full_dict['series_name'] = series_name
        full_dict['save_type'] = str(self.lineEdit_savetype.text())
        full_dict['variables'] = variables
        full_dict['measurements'] = measurements

        yaml_path = os.path.join(os.path.dirname(self.filename), '%s.yaml' % series_name)
        with open(yaml_path, 'w') as file:
            yaml.dump(full_dict, file)
        self.yaml_path = yaml_path

        self.done(1)


def get_data_from_yamls(yaml_paths, series_names=None, average_datasets=False):
    if type(yaml_paths) == str:
        yaml_paths = [yaml_paths] * len(series_names)
    elif series_names is None:
        series_names = [None] * len(yaml_paths)

    data = []
    variables = []
    for idx, series_name, yaml_path in zip(range(len(series_names)), series_names, yaml_paths):
        scan = AnalysisScan(yaml_path)
        if series_name is not None:
            scan.series_name = series_name
        scan.extract_hierarchy()
        scan.run()
        keys = list(scan.analysed_data.keys())
        keys.sort(key=SortingKey)
        if average_datasets:
            scan_data = np.mean(np.array([scan.analysed_data[key] for key in keys], np.float), 0)
        else:
            scan_data = np.squeeze(np.array([scan.analysed_data[key] for key in keys], np.float))
        data += [scan_data]
        variables += [scan.variables]
    return data, variables
