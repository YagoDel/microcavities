# -*- coding: utf-8 -*-

from nplab.utils.log import create_logger
from microcavities.analysis.utils import SortingKey
import h5py
import numpy as np
from collections import OrderedDict

# TODO: make iterator function to run through an HDF5 or file structure. It should be aware of parameter values (and
# attributes) and allow one to decide what to do once you get to the dataset (e.g. extract into an ArrayWithAttrs, fit
# Gaussian/other function, extract PL from a region, measure energy)


LOGGER = create_logger('Analysis')


class HierarchicalAnalysis(object):
    """
    Base class for implementing data analysis on hierarchical datasets.
    By hierarchical datasets I mean datasets coming from experiments in which you make a grid scan of many different
    parameters, taking different types of data for different parameters. The hierarchy is given by the scan parameters.
    A GeneralScan run would produce such an dataset.

    It is designed thinking that datasets could be HDF5 datasets, or they could be folder directories, or anything else.
    We have abstracted those details by keeping a 'level' object and leaving the user to reimplement the necessary
    methods to address these objects.
    The user must reimplement self.variable_name_value, self.is_lowest_level, and self.next_levels

    # TODO: implement analysis at different depths
    """
    def __init__(self, hierarchical_object):
        super(HierarchicalAnalysis, self).__init__()

        self.data = hierarchical_object
        self.analysis_functions = OrderedDict()
        self.variables = OrderedDict()
        self.results = OrderedDict()

    def variable_name_value(self, level):
        """
        Returns the name and value of the parameter being set at the given level

        :param level:
        :return: 2-tuple of name and value
        """
        raise NotImplementedError

    def is_lowest_level(self, level):
        """
        Whether this is the level the user wants to perform analysis or not.

        :param level:
        :return: bool
        """
        raise NotImplementedError

    def next_levels(self, level):
        """
        Returns the sub-levels of a given level

        :param level:
        :return: iterable
        """
        raise NotImplementedError

    def run(self, level):
        self.variables = OrderedDict()
        self.results = OrderedDict()
        self._iterate(level)
        self._reshape_results()

    def _iterate(self, level):
        """
        Recursive function to iterate over a hierarchical dataset. Given an input level, it will either perform analysis
        on it, or perform recursion over it's sublevels

        :param level: level over which you want to act
        :return:
        """
        if self.is_lowest_level(level):
            LOGGER.debug('Analysing: %s' % level)
            self.analyse(level)
        else:
            LOGGER.debug('Iterating over: %s' % level)
            levels = self.next_levels(level)
            self._extract_variable(levels)
            for next_level in levels:
                self._iterate(next_level)

    def analyse(self, level):
        """
        Iterates over the analysis functions, applying them to this level and keeping the results (if any) as
        flat-arrays inside the self.results OrderedDict

        :param level:
        :return:
        """
        for name, parameters in self.analysis_functions.items():
            if 'args' not in parameters:
                args = []
            else:
                args = parameters['args']
            if 'kwargs' not in parameters:
                kwargs = dict()
            else:
                kwargs = parameters['kwargs']

            result = parameters['function'](level, *args, **kwargs)
            if result is not None:
                if name not in self.results:
                    self.results[name] = [result]
                else:
                    self.results[name] += [result]

    def _reshape_results(self):
        """
        Since self.analyse stores results as flat-arrays, and that's not terribly useful, this reshapes them into numpy
        arrays where each dimension corresponds to one of the parameters in the hierarchy

        :return:
        """
        LOGGER.debug('Reshaping results')

        new_shape = tuple([len(var) for var in self.variables.values()])
        LOGGER.debug('New shape: %s' % new_shape)

        for name, result in self.results.items():
            if not isinstance(result, np.ndarray):
                LOGGER.debug('Making into array')
                result = np.array(result)

            current_shape = result.shape
            LOGGER.debug('Current %s shape: %s' % (name, current_shape))

            self.results[name] = result.reshape(new_shape + current_shape[1:])

    def _extract_variable(self, levels):
        """
        Iterates over levels, checking that only one parameter is being changed, and storing the parameter values (if any)
        :param levels:
        :return:
        """
        LOGGER.debug('Extracting variables from: %s' % levels)

        var_name = None
        values = []
        for level in levels:
            name, value = self.variable_name_value(level)

            values += [value]
            if var_name is None:
                var_name = name
            elif var_name != name:
                raise ValueError('There are differently named variables in the same hierarchical level: %s %s' % (var_name, name))

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


class Hdf5Analysis(HierarchicalAnalysis):
    def __init__(self):
        super(Hdf5Analysis, self).__init__(None)
        self.analysis_functions = OrderedDict()

    def variable_name_value(self, level):
        name = level.name
        name = name.split('/')[-1]
        if '=' in name:
            splitname = name.split('=')
            name = splitname[0]
            value = float(splitname[1])
            return name, value
        else:
            LOGGER.debug('Not a recognised variable name: %s' % name)
            return None, None

    def is_lowest_level(self, level):
        if isinstance(level, h5py.Dataset):
            return True
        elif isinstance(level, h5py.Group):
            return False
        else:
            raise ValueError

    def next_levels(self, level):
        keys = level.keys()
        keys.sort(key=SortingKey)
        return [level[key] for key in keys]

    @staticmethod
    def dummy_analysis(level):
        return level[...]
