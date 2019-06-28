# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os
import xarray


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
                previous_level = '/'.join(full_path.split('/')[:-1])
                if not os.path.exists(previous_level):
                    os.makedirs(previous_level)

                if full_path.endswith('.h5'):
                    # CHECK THIS
                    xarray.save(data, *args, **kwargs)
                elif full_path.endswith('.npy'):
                    np.save(full_path, data, *args, **kwargs)
        elif self.data_type == 'xarray':
            if data is None:
                print 'Nothing happened. Need to provide data to save as xarray'
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
