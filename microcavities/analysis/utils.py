# -*- coding: utf-8 -*-

import re
import ast
import numpy as np


def SortingKey(pattern):
    """
    To be used to sort the names of folders, assuming the folder names follow the convention 'ParamName=ParamValue'
    or 'ParamName=ParamValue-ParamIteration' if there is more than one dataset with the same parameter value.
    Sorting is done in order of ParamValue, and then ParamIteration
    Args:
        pattern: name of the folder
    Returns:
        Either a single value or a two-tuple that is used for the sort
    """
    splitpat = pattern.split('=')
    returnval = (bool(len(splitpat) - 1),)
    if len(splitpat) > 1:
        if re.match('.\d*-\d+|.*0\.\d*-\d+', splitpat[1]):
            returnval += tuple([np.real(ast.literal_eval(x)) for x in re.findall('(.+)-(\d+)', splitpat[1])[0]])
        elif re.match('\(\d+\.*\d*, *\d+.*\d*\)', splitpat[1]):
            returnval += tuple([np.real(ast.literal_eval(x)) for x in re.findall('(\d+\.*\d*),(\d+\.*\d*)', splitpat[1])[0]])
        else:
            val = ast.literal_eval(splitpat[1])
            returnval += (np.sign(np.real(val)) * np.abs(val),)
    return returnval


def fit_function(name):
    """
    Given a name, return a lmfit Model with a guess method implemented
    :param name:
    :return:
    """


def normalize(array, percentiles=(0, 100), axis=None, cut=False):
    mn = np.percentile(array, percentiles[0], axis=axis, keepdims=True)
    mx = np.percentile(array, percentiles[1], axis=axis, keepdims=True)
    normed = (array - mn) / (mx - mn)
    if cut:
        normed[normed > 1] = 1
        normed[normed < 0] = 0
    return normed
