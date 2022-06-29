# -*- coding: utf-8 -*-

import numpy as np
import yaml
from io import IOBase
import os
import subprocess
import datetime
import warnings


def depth(lst):
    """
    Returns the depth of a list by recursively checking whether the sub-elements are lists:
        depth([0,1,2,3,4]) is 1
        depth([[0,1],[2,3]]) is 2

    :param lst: list or list of lists
    :return: int. Depth of the list of lists provided
    """
    return isinstance(lst, list) and max(list(map(depth, lst))) + 1


def run_once(f):
    """
    Given a function that can run many times, returns a function that can run only once

    :param f: function
    :return:
    """
    def wrapper(*args, **kwargs):
        if args[1] not in list(wrapper.dicc.keys()):
            wrapper.dicc[args[1]] = False
        if not wrapper.dicc[args[1]]:
            wrapper.dicc[args[1]] = True
            return f(*args, **kwargs)

    wrapper.dicc = {}
    return wrapper


def string_to_number(s):
    """Given a string, tries to convert it to an integer, and if it fails tries to convert it to a float

    :param string s: Input string.
    :return: number.
    :rtype: type

    """
    try:
        return int(s)
    except ValueError:
        return float(s)


def is_prime(integer):
    """
    Checks whether integer is divisible by any other integer smaller than itself.

    :param integer: int
    :return: bool. True if a is prime, False otherwise
    """
    return all(integer % i for i in range(2, integer))


def square(integer, _iter=0):
    """
    Finds the two integers that most closely make a square of the input integer.
    E.g. square(9) = (3, 3); square(10) = (5, 2); square(3) = (3, 1); square(11) = (4, 3)

    :param integer: int
    :param _iter: int. Keeps track of what level of recursion we are at to avoid diverging.
    :return: two-tuple
    """
    square_root = np.sqrt(integer)
    a = np.round(square_root) + _iter
    if is_prime(integer) and integer > 10:
        return square(integer + 1, _iter)

    if np.mod(square_root, 1) == 0:
        return int(square_root), int(square_root)
    elif np.mod(integer, a) != 0 and _iter < integer:
        return square(integer, _iter + 1)
    else:
        return np.max([int(a), int(integer / a)]), np.min([int(a), int(integer / a)])


def yaml_loader(input_yaml):
    """Parsing for creating dictionaries from yaml file paths, files, or dictionaries

    :param input_yaml:
    :return:
    """
    if isinstance(input_yaml, str):
        if not os.path.isabs(input_yaml):
            input_yaml = os.path.join(get_data_directory(), input_yaml)
        with open(input_yaml, 'r') as yaml_file:
            output_yaml = yaml.full_load(yaml_file)
    elif isinstance(input_yaml, dict):
        output_yaml = input_yaml
    elif isinstance(input_yaml, IOBase):
        output_yaml = yaml.full_load(input_yaml)
    else:
        raise TypeError("yaml type cannot be %s. Needs to be str, dict or file" % type(input_yaml))

    return output_yaml


def get_data_directory():
    """Utility function to parse the settings.yaml"""
    yml_dict = yaml_loader(os.path.join(os.path.dirname(__file__), '..', 'settings.yaml'))
    if os.sys.platform == 'win32':
        computer_name = os.environ['COMPUTERNAME']
        # computer_name = os.getenv('HOSTNAME', os.getenv('COMPUTERNAME', platform.node())).split('.')[0]
        home_path = os.environ['HOMEPATH']
    elif os.sys.platform == 'darwin':
        cmd = "system_profiler SPHardwareDataType | grep 'Serial Number' | awk '{print $4}'"
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
        computer_name = result.stdout.strip().decode()
        # computer_name = os.uname()[1]
        home_path = os.environ['HOME']
    else:
        raise ValueError('Unrecognised platform %s' % os.sys.platform)
    computer_names = yml_dict['data_directory'].keys()

    directory = None
    if computer_name in computer_names:
        _directories = yml_dict['data_directory'][computer_name]
        if type(_directories) == str:
            directory = _directories
        else:
            for _directory in _directories:
                if os.path.exists(_directory):
                    directory = _directory
                    break

    if directory is None:
        warnings.warn('No default directory provided for the computer: %s. Using default home path' % computer_name)
        directory = os.path.abspath(home_path)

    return os.path.normpath(directory)


def get_data_path(filename=None, create_folder=True):
    """Returns a path

    Utility function for returning default file locations when working with different computers
    Looks inside a settings.yaml that should be placed one level higher in the directories for default data directories

    :param filename: by default points to an HDF5 file called raw_data.h5 inside a folder called year_month_day
    :return:
    """
    directory = get_data_directory()

    if filename is None:
        filename = os.path.join(datetime.date.today().strftime('%Y_%m_%d'), 'raw_data.h5')

    full_path = os.path.normpath(os.path.join(directory, filename))
    
    if create_folder:
        folder_name = os.path.dirname(full_path)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    return full_path


def interpolated_array(array, axes=None, **kwargs):
    from scipy.interpolate import LinearNDInterpolator
    # from scipy.interpolate import interpn
    # if axes is None: axes = [np.arange(x) for x in array.shape]
    # return partial(interpn, axes, array, **kwargs)
    if axes is None: axes = [np.arange(x) for x in array.shape]
    axes_arrays = [x.flatten() for x in np.meshgrid(*axes, indexing='ij')]
    return LinearNDInterpolator(list(zip(*axes_arrays)), array.flatten(), **kwargs)


def random_choice(array, axes=(0,), return_indices=False):
    """Returns a randomly-selected part of the array

    :param array:
    :param axes:
    :param return_indices: bool
    :return:
    """
    indices = ()
    for idx in range(len(array.shape)):
        if idx in axes:
            indices += (np.random.randint(array.shape[idx]), )
        else:
            indices += (slice(0, array.shape[idx]), )
    if return_indices:
        return array[indices], indices
    else:
        return array[indices]


def normalize(array, percentiles=(0, 100), axis=None, cut=False):
    mn = np.percentile(array, percentiles[0], axis=axis, keepdims=True)
    mx = np.percentile(array, percentiles[1], axis=axis, keepdims=True)
    normed = (array - mn) / (mx - mn)
    if cut:
        normed[normed > 1] = 1
        normed[normed < 0] = 0
    return normed


def apply_along_axes(func, axes, array, n_outputs=None):
    shape = np.array(array.shape)
    mask = np.full(len(shape), True)
    mask[(axes, )] = False
    mask2 = np.full(len(shape), np.s_[:])

    outputs = [()] * n_outputs
    indices = np.ndindex(tuple(shape[mask]))
    for idx in indices:
        mask2[mask] = idx
        try:
            results = func(array[tuple(mask2)])
        except Exception as e:
            print('Failed at indices: ', idx)
            raise e
        for idx2, result in enumerate(results):  # TODO: this fails if the output of func is not an iterable
            outputs[idx2] += (result, )
    outputs = [np.array(output) for output in outputs]
    reshaped_outputs = [np.reshape(output, tuple(shape[mask]) + output.shape[1:]) for output in outputs]
    return reshaped_outputs


def around_step(value, step):
    """Round a float to the nearest step
    :param value:
    :param step:
    :return:
    """
    up_down = (value % step // (step/2))
    if up_down:
        offset = -value % step
    else:
        offset = -(value % step)
    return value + offset
