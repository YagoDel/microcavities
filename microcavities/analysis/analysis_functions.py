# -*- coding: utf-8 -*-
import numpy as np


def roi(data, apply_function, roi_params=None, axes=-1):

    # if roi_params is None:
    #     popup a window with a plot and a button. Plot a random set of data and allow the button to change it.
    #     popup a ROI and allow the user to move it. 1D or 2D
    #     Also add a checkbox to the GUI to either lock the ROI to image pixels or not
    #
    # iterate over data, select a roi and apply_function
    # roid = pyqtgraph.affineSlice(data, shape, origin, vectors, axes, order=1, returnCoords=False, **kargs)
    # np.apply_over_axes(apply_function, roid, axes)

    return 1


def roi_intensity(data, roi_params=None):
    return np.sum(data), [roi_params], {}


def fit_gaussian(data):
    return np.sum(data, -1), [], {}
