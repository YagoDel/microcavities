# -*- coding: utf-8 -*-
import numpy as np
from microcavities.utils import depth


def gaussian(x, x0, a, gam):
    return a * np.exp(-(x-x0)**2 / (2*gam**2))


def gaussianNd(x, x0, a, gam):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x])
    X = np.meshgrid(*x)
    try:
        iter(gam)
    except:
        gam = np.array([gam] * len(X), dtype=np.float)
    gam /= 2*np.sqrt(2*np.log(2))
    try:
        iter(x0)
    except:
        x0 = np.array([x0] * len(X), dtype=np.float)
    exponent = np.zeros(X[0].shape)
    for _x, _x0, _gam in zip(X, x0, gam):
        exponent -= (_x - _x0)**2 / (2*_gam**2)
    return a * np.exp(exponent)


def lorentzian(x, x0, a, gam):
    return a * gam ** 2 / (gam ** 2 + (x - x0) ** 2)


def lorentzianNd(x, x0, a, gam):
    try:
        len(x[0])
    except:
        x = [x]
    X = np.meshgrid(*x)
    try:
        iter(gam)
    except:
        gam = np.array([gam] * len(X), dtype=np.float)
    try:
        iter(x0)
    except:
        x0 = np.array([x0] * len(X), dtype=np.float)
    quotient = np.zeros(X[0].shape)
    for _x, _x0, _gam in zip(X, x0, gam):
        quotient += ((_x - _x0) / (_gam / 2)) ** 2
    return a / (quotient + 1)
