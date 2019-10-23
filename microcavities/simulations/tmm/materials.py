# -*- coding: utf-8 -*-


def static_AlGaAs(fraction):
    """Non-dispersive refractive index of AlGaAs alloys"""
    n = 3.3-0.53*fraction+0.09*fraction**2
    return n

