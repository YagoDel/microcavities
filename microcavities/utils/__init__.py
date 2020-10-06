# -*- coding: utf-8 -*-
import numpy as np
import yaml
from io import IOBase
import os
import datetime


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
        with open(input_yaml, 'r') as yaml_file:
            output_yaml = yaml.full_load(yaml_file)
    elif isinstance(input_yaml, dict):
        output_yaml = input_yaml
    elif isinstance(input_yaml, IOBase):
        output_yaml = yaml.full_load(input_yaml)
    else:
        raise TypeError("yaml type cannot be %s. Needs to be str, dict or file" % type(input_yaml))

    return output_yaml


def get_data_path(filename=None, create_folder=True):
    """Returns a path

    Utility function for returning default file locations when working with different computers
    Looks inside a settings.yaml that should be placed one level higher in the directories for default data directories

    :param filename: by default points to an HDF5 file called raw_data.h5 inside a folder called year_month_day
    :return:
    """
    yml_dict = yaml_loader(os.path.join(os.path.dirname(__file__), '..', 'settings.yaml'))
    if os.sys.platform == 'win32':
        computer_name = os.environ['COMPUTERNAME']
        home_path = os.environ['HOMEPATH']
    elif os.sys.platform == 'darwin':
        computer_name = os.uname()[1]
        home_path = os.environ['HOME']
    else:
        raise ValueError('Unrecognised platform %s' % os.sys.platform)
    computer_names = yml_dict['data_directory'].keys()
    if computer_name not in computer_names:
        print('No default directory provided for the computer: %s. Using default home path' % computer_name)
        directory = os.path.abspath(home_path)
    else:
        directory = yml_dict['data_directory'][computer_name]

    if filename is None:
        filename = os.path.join(datetime.date.today().strftime('%Y_%m_%d'), 'raw_data.h5')

    full_path = os.path.join(directory, filename)
    
    if create_folder:
        folder_name = os.path.dirname(full_path)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    return full_path


def label_grid(figure, grid, label, position, offset=0.07):
    """Simple labelling of matplotlib.gridspec grids

    :param figure: matplotlib.figure
    :param grid: matplotlib.gridspec
    :param label: string
    :param position: string
    :param offset: float
    :return:
    """
    _pos = grid.get_grid_positions(figure)
    if position == 'bottom':
        figure.text(np.mean(_pos[2:]), _pos[0][-1]-offset, label, va='top', ha='center')
    elif position == 'top':
        figure.text(np.mean(_pos[2:]), _pos[1][0]+offset, label, va='bottom', ha='center')
    elif position == 'left':
        figure.text(_pos[2][0]-offset, np.mean(_pos[:2]), label, va='center', ha='right', rotation=90)
    elif position == 'right':
        figure.text(_pos[3][-1]+offset, np.mean(_pos[:2]), label, va='center', ha='left', rotation=-90)
