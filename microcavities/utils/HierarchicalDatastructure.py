# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os
# import xarray
import re
import string
import itertools
from microcavities.analysis.utils import SortingKey


class Datastructure(object):
    """
    General class that wraps h5py HierarchicalScans, xarrays and folder
    directories in the same data structure
    """
    available_data_types = ['h5py', 'folders', 'xarray']

    def __init__(self, data_type, top_level):
        """Short summary.

        :param type data_type: Description of parameter `data_type`.
        :param type top_level: Description of parameter `top_level`.
    :return: Description of returned object.
        :rtype: type

        """

        assert data_type in self.available_data_types
        self.data_type = data_type
        self.top_level = top_level

    def _parse_parameters(self, parameter_dictionary):
        # TODO: make it so that location can not just be a string, but a
        # dictionary of variables and values, using the xarray functionality on
        # folders and my HDF5
        return

    def save(self, location, data=None, *args, **kwargs):
        # TODO: handling of attributes
        # h5py and folders is easy. xarray is hard
        if self.data_type == 'h5py':
            with h5py.File(self.top_level) as df:
                if data is None:
                    self.create_group(location, *args, **kwargs)
                else:
                    # Add checks to ensure data has shape and dtype
                    df.create_dataset(location, data.shape, data.dtype, data)
        elif self.data_type == 'folders':
            full_path = self.top_level + location
            if data is None:
                os.makedirs(full_path)
            else:
                previous_level = os.path.dirname(full_path)
                if not os.path.exists(previous_level):
                    os.makedirs(previous_level)
                if full_path.endswith('.h5'):
                    if isinstance(data, xarray.Dataset) or isinstance(data, xarray.DataArray):
                        data.to_netcdf(full_path, *args, **kwargs)
                elif full_path.endswith('.npy'):
                    np.save(full_path, data, *args, **kwargs)
        elif self.data_type == 'xarray':
            if data is None:
                print('Nothing happened. Need to provide data to save as xarray')
            else:
                assert isinstance(data, xarray.Dataset)
                full_path = self.top_level + location

                if os.path.exists(full_path):
                    existing_file = xarray.load(full_path)
                    assert data.columns == existing_file.columns
                    assert 'column' in kwargs
                    xarray.concat([existing_file, data], dim=kwargs['column'])
        else:
            raise ValueError('This should never happen...')

    def load(self, location, *args, **kwargs):
        # TODO: extend so that this always returns an xarray, and we don't have
        # to give it the bottom location in the hierarchy
        # h5py and folders is hard. xarray is easy

        if self.data_type == 'h5py':
            with h5py.File(self.top_level) as df:
                data = df[location]
                assert isinstance(data, h5py.Dataset)
            return data
        elif self.data_type == 'folders':
            full_path = self.top_level + location

            if full_path.endswith('.h5'):
                # CHECK THIS
                data = xarray.load(full_path, *args, **kwargs)
            elif full_path.endswith('.npy'):
                data = np.load(full_path, data, *args, **kwargs)
            else:
                raise ValueError('Unrecognised file type: %s' % location)
            return data
        elif self.data_type == 'xarray':
            return xarray.load(full_path, *args, **kwargs)
        else:
            raise ValueError('This should never happen...')


def h5py_to_xarray(group):
    keys = group.keys()
    keys.sort(key=SortingKey)
    array = 0
    return array


def h5py_string_format(data_file_path, series_name, final_name=None):
    """Extracts the string format from a hierarchical h5py

    This is for h5py files that are structured as followed:
        series_name
            variable1=v1val1
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3
            variable1=v1val2
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3
            variable1=v1val3
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3
    This way, it can simply go down the hierarchy along one path to extract
        series_name/variable1={variable1}/variable2={variable2}

    :param data_file_path: str
    :param series_name: str
    :param final_name: str
    :return:
    """
    string_format = series_name
    dset_path = series_name
    variable_names = []
    with h5py.File(data_file_path, 'r') as dfile:
        while isinstance(dfile[dset_path], h5py.Group):
            old_dset_path = str(dset_path)
            keys = dfile[dset_path].keys()
            for key in keys:
                # print(key)
                if '=' in key:
                    variable_names += [re.findall('([^=]*)=', key)[0]]
                    dset_path += '/' + key
                    string_format += '/%s=${%s}' % (variable_names[-1], variable_names[-1])
                    break
                elif final_name == key:
                    dset_path += '/' + key
                    string_format += '/%s' % key

                    break
            if dset_path == old_dset_path:
                raise ValueError('Unrecognised hierarchy')
    return string_format


def h5py_get_data(data_file_path, series_name, final_name=None, *args, **kwargs):
    """Extracts the data in a hierarchical h5py.Group

    This is for h5py files that are structured as followed:
        series_name
            variable1=v1val1
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3
            variable1=v1val2
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3
            variable1=v1val3
                variable2=v2val1
                variable2=v2val2
                variable2=v2val3

    Examples:
        h5py_get_data(file_path, series_name) would return a numpy array of shape: (3, 3) + dataset.shape
        h5py_get_data(file_path, series_name, v1val1) would return a numpy array of shape: (3, ) + dataset.shape

    :param data_file_path:
    :param series_name:
    :param final_name:
    :param args:
    :return:
    """

    string_format = h5py_string_format(data_file_path, series_name, final_name)
    formatter = list(string.Formatter().parse(string_format))
    variable_names = [v[1] for v in formatter if v[1] is not None]
    depth = formatter[0][0].count('/')
    n_variables = len(variable_names)
    # print('Found %d variables: ' % n_variables, variable_names)

    if len(args) != n_variables:
        for idx in range(len(args), n_variables):
            if variable_names[idx] in kwargs:
                args += (kwargs[variable_names[idx]], )
            else:
                args += (None, )
        # print(args)
        # args += tuple([None] * (n_variables - len(args)))

    string_template = string.Template(string_format)
    with h5py.File(data_file_path, 'r') as dfile:
        variable_values = []
        for idx, arg in enumerate(args):
            if arg is None:
                _string_format = string_template.safe_substitute({key: '%g' % value[0] for key, value in zip(variable_names, variable_values)})
                try:
                    _keys = list(dfile['/'.join(_string_format.split('/')[:(idx+depth)])].keys())
                    keys = [key for key in _keys if '=' in key]
                except KeyError as e:
                    print(variable_names, variable_values, _string_format)
                    print('/'.join(_string_format.split('/')[:(idx+1)]))
                    raise(e)
                variable_values += [np.sort([float(re.findall('=([^=]*)', x)[0]) for x in keys])]
            else:
                try:
                    # len(arg)
                    variable_values += [np.array(arg)]
                except:
                    variable_values += [np.array([arg])]
        # print('Found variables_values: ', variable_values)

        variables = itertools.product(*variable_values)
        data = []
        for variable in variables:
            dset_name = string_template.safe_substitute({key: '%g' % value for key, value in zip(variable_names, variable)})
            try:
                data += [dfile[dset_name][...]]
            except Exception as e:
                print(dset_name)
                print(list(variables))
                raise(e)
        data = np.array(data)
    data = np.reshape(data, [len(x) for x in variable_values] + list(data.shape[1:]))
    return np.squeeze(data), {key: value for key, value in zip(variable_names, variable_values)}


if __name__ == '__main__':
    powers = [0, 1]
    positions = [2, 3]
    for power in powers:
        for pos in positions:
            os.makedirs('Desktop/testing/power=%g/position=%g' % (power, pos))
