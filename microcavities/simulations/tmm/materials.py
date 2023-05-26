# -*- coding: utf-8 -*-
import numpy as np


def static_AlGaAs(fraction, source='michael'):
    """Non-dispersive refractive index of AlGaAs alloys

    :param fraction: of aluminium. From 0 to 100
    :param source: str.
        ioffe  -  http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/basic.html
    :return:
    """
    if source == 'michael':
        return 3.3-0.53*(fraction/100)+0.09*(fraction/100)**2
    elif source == 'ioffe':
        return np.sqrt(10.89 - 2.73 * (fraction/100))


def AlGaAs_bandgap(fraction, k_point='gamma', source='casey'):
    """

    :param fraction:
    :param k_point:
    :param source:
    :return:
    """
    if source == 'casey':
        if k_point == 'gamma':
            if fraction <= 0.45:
                return 1.424 + 1.247 * fraction
            else:
                return 1.424 + 1.247 * fraction + 1.147 * (fraction - 0.45)**2
        elif k_point == 'X':
            return 1.9 + 0.125 * fraction + 0.143 * fraction**2
        elif k_point == 'L':
            return 1.708 + 0.642 * fraction
    elif source == 'lee':
        if k_point == 'gamma':
            return 1.425 + 1.155 * fraction + 0.37 * fraction**2
        elif k_point == 'X':
            return 1.911 + 0.005 * fraction + 0.245 * fraction**2
        elif k_point == 'L':
            return 1.734 + 0.574 * fraction + 0.055 * fraction**2


def dispersive_AlGaAs(fraction, wavelength, source='batop'):
    """Dispersive refractive index of AlGaAs alloys


    :param fraction: of aluminium. From 0 to 100
    :param wavelength:
    :param source: str.
        batop  -  https://www.batop.de/information/n_AlGaAs.html
        ohta  -  https://doi.org/10.1063/1.364443
        deri  -  https://doi.org/10.1063/1.359434
        adachi -  https://doi.org/10.1063/1.336070
    :return:
    """
    fraction = np.asarray(fraction) / 100
    h = 4.135667696e-15  # eV * s
    c = 299792458 * 1e9  # nm / s
    energy = h * c / wavelength

    if source == 'batop':
        spin_orbit_splitting = 1.765-1.425

        def chi(wvl, energy):
            return h * c / (wvl * energy)

        def f(_chi):
            return (2 - np.sqrt(1+_chi) - np.sqrt(1-_chi)) / (_chi**2)

        def A(x):
            return 6.3 + 19 * x

        def B(x):
            return 9.4 - 10.2 * x

        def bandgap_gamma(x):
            return 1.425 + 1.555 * x + 0.37 * x**2

        E0 = bandgap_gamma(fraction)
        ED0 = bandgap_gamma(fraction) + spin_orbit_splitting
        f_chi = f(chi(wavelength, E0))
        f_chi_s0 = f(chi(wavelength, ED0))
        return np.sqrt(A(fraction) * (f_chi + (f_chi_s0 / 2) * (E0 / ED0) ** (3 / 2)) + B(fraction))
    elif source == 'ohta':
        assert all(np.array(np.logical_and(fraction >= 0, fraction <= 0.38)))
        assert all(np.array([np.logical_and(np.array(energy) > 1.2, np.array(energy) < 1.8)]))
        return 3.3 + 0.09 * fraction - (0.08 + 0.7 * fraction) * energy + (
                    0.19 + 0.16 * fraction) * energy ** 2 + 0.00023 / ((energy - (1.42 + 1.25 * fraction)) ** 2 + 0.003)
    elif source == 'deri':
        raise NotImplementedError('Basic testing shows this is not working as expected')
        assert all(np.array([np.logical_and(np.array(wavelength) >= 750, np.array(wavelength) <= 1500)]))

        def f(y):
            return (2 - np.sqrt(1 + y) - np.sqrt(1 - y)) / (y ** 2)
        def g(e, e1, e1c=6.4):
            return (e/e1)**2 + np.log((1-(e/e1)**2)/(1-(e/e1c)**2))
        def a(x):
            return 5.6684 + 10.464 * x + 1.450 * np.exp(3.5584 * x)
        def b(x):
            return 9.1813 - 4.5059 * x - 1.4304 * x**2 + 2.2388 * x**3
        def e0(x):
            return 2.1582 + 0.80331 * x - 0.0911 * x**2 - 2.6906 * x**3
        def E1(x):
            return 2.926 + 0.6717 * x - 0.3242 * x**2 + 0.6172 * x**3
        def E0(x):
            try:
                if x > 0.45:
                    return 1.422 + 1.544 * x - 0.826 * x**2 + 0.878 * x**3
                else:
                    return 1.424 + 1.45 * x - 0.25 * x**2
            except ValueError:
                return np.array([E0(y) for y in x])
        def Es(x):
            return E0(x) + 0.34 - 0.02 * x

        square_index = e0(fraction)
        square_index += a(fraction) * ((f(energy / E0(fraction)) / E0(fraction)) ** 1.5 + (f(energy / Es(fraction)) / Es(fraction)) ** 1.5)
        square_index += b(fraction) * g(energy, E1(fraction))
        return np.sqrt(square_index)
    elif source == 'adachi':
        def f(y):
            return (2 - np.sqrt(1 + y) - np.sqrt(1 - y)) / (y ** 2)
        def a(x):
            return 6.3 + 19 * x
        def b(x):
            return 9.4 - 10.2 * x
        def chi(_wvl, _energy):
            return h * c / (_wvl * _energy)
        def e0(x):
            return 1.425 + 1.155 * x + 0.37 * x**2
        def e0d0(x):
            return 1.765 + 1.115 * x + 0.37 * x**2

        # print(e0(fraction))
        # print(chi(e0(fraction))

        permitivitty = b(fraction)
        permitivitty += a(fraction) * (f(chi(wavelength, e0(fraction))) +
                                       0.5*f(chi(wavelength, e0d0(fraction)))*(e0(fraction)/e0d0(fraction))**1.5)
        return np.sqrt(permitivitty)


def dispersive_InGaAs(fraction, wavelength=800, source='batop'):
    """Dispersive refractive index of AlGaAs alloys


    :param fraction: of aluminium. From 0 to 100
    :param wavelength:
    :param source: str.
        batop  -  https://www.batop.de/information/n_InGaAs.html
    :return:
    """
    fraction = np.asarray(fraction) / 100
    h = 4.135667696e-15  # eV * s
    c = 299792458 * 1e9  # nm / s
    energy = h * c / wavelength

    if source == 'batop':
        # spin_orbit_splitting = 1.765-1.425
        #
        # def chi(wvl, energy):
        #     return h * c / (wvl * energy)
        #
        # def f(_chi):
        #     return (2 - np.sqrt(1+_chi) - np.sqrt(1-_chi)) / (_chi**2)
        #
        # def A(x):
        #     return 6.3 + 19 * x
        #
        # def B(x):
        #     return 9.4 - 10.2 * x

        def bandgap_gamma(x):
            return 1.424 - 1.501 * x + 0.436 * x**2

        A = 8.950
        B = 2.054
        C = 0.6245

        E0 = bandgap_gamma(fraction)
        # ED0 = bandgap_gamma(fraction) + spin_orbit_splitting
        # f_chi = f(chi(wavelength, E0))
        # f_chi_s0 = f(chi(wavelength, ED0))
        return np.sqrt(A + B / (1 - (C*1.424/(wavelength*E0))**2))

