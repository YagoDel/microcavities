# -*- coding: utf-8 -*-

import numpy as np
import yaml
import tmm


class Structure(object):
    """Base class for microcavity structures

    Contains a list of refractive indices, thicknesses, wavelengths and angles, and wraps the tmm functionality into
    useful functions dispersion and normal_incidence
    """

    def __init__(self, wavelengths=np.linspace(730, 870, 100), angles=np.linspace(730, 870, 100)):
        super(Structure, self).__init__()

        self.n_list = None
        self.d_list = None
        self.wavelengths = wavelengths
        self.angles = angles
        self.results = dict()

    @staticmethod
    def _redo_linspace(array, min_val, max_val, n_vals):
        if min_val is None:
            min_val = np.min(array)
        if max_val is None:
            max_val = np.max(array)
        if n_vals is None:
            n_vals = len(array)
        return np.linspace(min_val, max_val, n_vals)

    def _pad_lists(self):
        """Makes sure calculations are done with semi-infinite volumes on either end"""
        if self.d_list[0] != np.inf:
            self.d_list = [np.inf] + self.d_list
            self.n_list = [1] + self.n_list

        if self.d_list[-1] != np.inf:
            self.d_list = self.d_list + [np.inf]
            self.n_list = self.n_list + [1]

    def dispersion(self, polarisation='s',
                   min_angle=None, max_angle=None, n_angles=None,
                   min_wvl=None, max_wvl=None, n_wvls=None):
        """Spectra for a range of angles

        :param polarisation: s or p
        :param min_angle:
        :param max_angle:
        :param n_angles:
        :param min_wvl:
        :param max_wvl:
        :param n_wvls:
        :return: 2D array of reflection coefficients
        """
        self.wavelengths = self._redo_linspace(self.wavelengths, min_wvl, max_wvl, n_wvls)
        self.angles = self._redo_linspace(self.angles, min_angle, max_angle, n_angles)
        self._pad_lists()
        reflec = []
        for angle in self.angles:
            _R = []
            for wvl in self.wavelengths:
                _R += [tmm.coh_tmm(polarisation, self.n_list, self.d_list, angle, wvl)['R']]
            reflec += [_R]
        self.results['dispersion'] = np.array(reflec)
        return self.results['dispersion']

    def normal_incidence(self, polarisation='s', min_wvl=None, max_wvl=None, n_wvls=None):
        """Normal incidence spectrum

        :param polarisation: s or p
        :param min_wvl:
        :param max_wvl:
        :param n_wvls:
        :return: 1D array of reflection coefficients
        """
        self.wavelengths = self._redo_linspace(self.wavelengths, min_wvl, max_wvl, n_wvls)
        self._pad_lists()
        reflec = []
        for wvl in self.wavelengths:
            reflec += [tmm.coh_tmm(polarisation, self.n_list, self.d_list, 0, wvl)['R']]
        self.results['normal_incidence'] = np.array(reflec)
        return self.results['normal_incidence']


class DBR(Structure):
    def __init__(self, layers, refractive_indices, thicknesses=None, center_wavelength=None):
        assert any(map(lambda x: x is not None, [thicknesses, center_wavelength]))
        assert not all(map(lambda x: x is not None, [thicknesses, center_wavelength]))
        super(DBR, self).__init__()

        refractive_indices = np.array(refractive_indices)  # to make use of distributive maths
        if thicknesses is None:
            thicknesses = center_wavelength / (4 * refractive_indices)

        self.n_list, self.d_list = self._make_lists(layers,
                                                    refractive_indices[0], refractive_indices[1],
                                                    thicknesses[0], thicknesses[1])

    @staticmethod
    def _make_lists(layers, n1, n2, d1, d2):
        n_list = [n1, n2] * layers
        d_list = [d1, d2] * layers
        return n_list, d_list


class Microcavity(Structure):
    def __init__(self, structure_yaml):
        super(Microcavity, self).__init__()
        if isinstance(structure_yaml, str):
            full_yaml = yaml.load(open(structure_yaml, 'r'))
        elif isinstance(structure_yaml, dict):
            full_yaml = structure_yaml
        elif isinstance(structure_yaml, file):
            full_yaml = yaml.load(structure_yaml)
        else:
            raise TypeError("structure_yaml cannot be %s. Needs to be str, dict or file" % type(structure_yaml))
        print full_yaml

        dbr1_kwargs = full_yaml['DBR1']
        if 'thicknesses' not in dbr1_kwargs and 'center_wavelength' not in dbr1_kwargs:
            dbr1_kwargs['center_wavelength'] = full_yaml['center_wavelength']

        dbr2_kwargs = full_yaml['DBR2']
        if 'thicknesses' not in dbr2_kwargs and 'center_wavelength' not in dbr2_kwargs:
            dbr2_kwargs['center_wavelength'] = full_yaml['center_wavelength']
        dbr1 = DBR(**dbr1_kwargs)
        dbr2 = DBR(**dbr2_kwargs)

        cavity = full_yaml['cavity']
        cavity_index = cavity['refractive_index']
        if 'thickness' not in cavity:
            if 'center_wavelength' not in cavity:
                cavity['center_wavelength'] = full_yaml['center_wavelength']
            n1 = dbr1_kwargs['refractive_indices'][0]
            n2 = dbr1_kwargs['refractive_indices'][1]
            n3 = cavity['refractive_index']
            l_dbr = (cavity['center_wavelength'] * n1 * n2) / (2 * n3 * (n1 - n2))
            cavity_thickness = np.abs(cavity['center_wavelength'] * cavity['fraction'] - l_dbr)
        else:
            cavity_thickness = cavity['thickness']

        self.n_list = dbr1.n_list + [cavity_index] + dbr2.n_list
        self.d_list = dbr1.d_list + [cavity_thickness] + dbr2.d_list
