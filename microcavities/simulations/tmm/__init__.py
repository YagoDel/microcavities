# -*- coding: utf-8 -*-

import numpy as np
import yaml
import tmm
import matplotlib.pyplot as plt


class Structure(object):
    """Base class for microcavity structures

    Contains a list of refractive indices, thicknesses, wavelengths and angles, and wraps the tmm functionality into
    useful functions dispersion and normal_incidence
    """

    def __init__(self, wavelengths=np.linspace(730, 870, 100), angles=np.linspace(0, np.pi/4, 100)):
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
        self.results['dispersion_wavelengths'] = np.copy(self.wavelengths)
        self.results['dispersion_angles'] = np.copy(self.angles)
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
        self.results['normal_incidence'] = np.array([self.wavelengths, reflec])
        return self.results['normal_incidence']

    def field_distribution(self, wavelength, polarisation='s', n_points=1000):
        """Electric field distribution inside the structure

        :param wavelength:
        :param polarisation:
        :param n_points:
        :return:
        """
        positions = np.linspace(0, np.sum(np.array(self.d_list)[1:-1]), n_points)
        results = tmm.coh_tmm(polarisation, self.n_list, self.d_list, 0, wavelength)
        electric_field = []
        refractive_index = []
        for pos in positions:
            layer, d_in_layer = tmm.find_in_structure_with_inf(self.d_list, pos)
            vals = tmm.position_resolved(layer, d_in_layer, results)
            refractive_index += [self.n_list[layer]]
            electric_field += [np.abs(vals['Ex']) ** 2 + np.abs(vals['Ey']) ** 2]
        self.results['field_distribution'] = np.array([positions, electric_field])
        self.results['index_distribution'] = np.array([positions, refractive_index])

    def plot(self):
        if 'normal_incidence' in self.results:
            fig, ax = plt.subplots(1, 1)
            ax.plot(*self.results['normal_incidence'])
            ax.set_xlabel('Wavelength / nm')
            ax.set_ylabel('Reflectivity')
            ax.set_title('Normal incidence')
        if 'dispersion' in self.results:
            fig, ax = plt.subplots(1, 1)
            angles = self.results['dispersion_angles']
            wvls = self.results['dispersion_wavelengths']
            ax.imshow(self.results['dispersion'].transpose(), vmin=0, vmax=1, aspect='auto',
                      extent=[np.min(angles), np.max(angles), np.max(wvls), np.min(wvls)])
            ax.set_xlabel('Angle / rad')
            ax.set_ylabel('Wavelength / nm')
            ax.set_title('Dispersion')
        if 'field_distribution' in self.results:
            fig, ax = plt.subplots(1, 1)
            ax.plot(*self.results['field_distribution'])
            ax.set_ylabel('Electric field')
            ax.set_xlabel('Structure depth / nm')
            ax.set_title('Field distribution')
            ax2 = ax.twinx()
            ax2.plot(*self.results['index_distribution'], color='r')
            ax2.set_ylabel('Refractive index', color='r')
            ax2.tick_params(axis='y', colors='r')
            ax2.set_ylim(bottom=0)


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

    def dbr_damage(self, layers, damage, mode='contrast'):
        """Simulates ion damage on the DBR layers

        Multiple options of doing this:
            - Reducing the contrast at those layers (only one implemented)
            - Creating intermediary layers that smoothen the transition from low to high refractive index

        :param layers: start and end indices of the layers to be damaged
        :param damage: contrast reduction factor
        :param mode:
        :return:
        """

        if mode == 'contrast':
            self.n_list = np.array(self.n_list)
            subsel = self.n_list[layers[0]:layers[1]]
            low_index = np.min(subsel)
            high_index = np.max(subsel)
            contrast = high_index - low_index
            damaged_contrast = contrast * (1 - damage)
            subsel[subsel == low_index] += (contrast - damaged_contrast) / 2
            subsel[subsel == high_index] -= (contrast - damaged_contrast) / 2
            self.n_list = list(self.n_list)


if __name__ == '__main__':
    mcav = Microcavity('microcavity_example.yaml')
    mcav.normal_incidence()
    mcav.dispersion()
    mcav.field_distribution(800)
    mcav.plot()
    plt.show()
