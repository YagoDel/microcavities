# -*- coding: utf-8 -*-

from nplab.instrument.stage.SigmaKoki import HIT
import numpy as np


class SampleMovement(HIT):
    axis_names = ['x', 'y']
    axis_LUT = dict(list(zip(['x', 'y'], [2, 1])))
    # axis_lim = dict(zip(['x', 'y'], [None, (3420000, 4420000)]))
    axis_lim = dict(list(zip(['x', 'y'], [None, None]))) #(-789000, 789000)]))
    # axis_lim = dict(zip(['x', 'y'], [None, None]))

    def __init__(self, address):
        super(SampleMovement, self).__init__(address)
        for ax in self.axis_names:
            self.set_speed(ax, 1, 100000, 1000)

    def move(self, counts, axes=None, relative=False, wait=True, safe=True):
        if axes is None:
            axes = self.axis_names
        elif not hasattr(axes, '__iter__'):
            axes = (axes, )
        if not hasattr(counts, '__iter__'):
            counts = [counts] * len(axes)
        if safe:
            for count, ax in zip(counts, axes):
                if self.axis_lim[ax] is not None:
                    lim = self.axis_lim[ax]
                    if relative:
                        pos = self.get_position((self.axis_LUT[ax], ))[0]
                        assert lim[0] < pos + count < lim[1]
                    else:
                        assert lim[0] < count < lim[1]
        super(SampleMovement, self).move(counts, axes, relative, wait)

    def mechanical_home(self, axes):
        assert 'y' not in axes
        super(SampleMovement, self).mechanical_home(axes)

    def define_line(self, xz_points):
        xz_points = np.array(xz_points, dtype=np.float)
        point = xz_points[0]
        vec = xz_points[0]-xz_points[1]
        slope = vec[1] / vec[0]
        offset = point[1] - point[0] * slope
        def line(x):
            return slope * x + offset
        self.line = line

    def define_plane(self, xyz_points):
        xyz_points = np.array(xyz_points, dtype=np.float)
        point = xyz_points[0]
        vec1 = xyz_points[0]-xyz_points[1]
        vec2 = xyz_points[0]-xyz_points[2]
        normal = np.cross(vec1, vec2)
        print(point, vec1, vec2, normal, normal[:2])
        def z_func(x, y):
            return (np.dot(point, normal) - np.dot([x,y], normal[:2])) / normal[2]
        self.plane = z_func

    def move_correct(self, x, relative=False):
        if hasattr(x, "__iter__"):
            HIT.move(self, x[0], 2, relative)
            HIT.move(self, x[1], 1, relative)
            HIT.move(self, self.plane(*x), 1, relative)
        else:
            HIT.move(self, x, 2, relative)
            HIT.move(self, self.line(x), 1, relative)


class Stages(HIT):

    def __init__(self, address):
        super(Stages, self).__init__(address)
        self.axis_names = ("spectrometer_lens", "k_lens", "filter_x", "filter_y", "stokes", "streak_lens")
        self.axis_LUT = dict(list(zip(self.axis_names, (3, 0, 4, 5, 1, 2))))
        for axis in range(6):
            self.set_speed(axis, 1, 500000, 1000)

        self.axis_toggle = dict(k_lens=dict(on=2550000, off=7000000), stokes=dict(on=2430000, off=7000000),
                                filter_y=dict(off=8604180, small=338640, medium=3394640, big=6475000),
                                filter_x=dict(off=6000000, small=2204000, medium=1983000, big=2030000))
        # 20um pinhole x,y = [2204000, 338640]
        # 50um pinhole x, y = [1983000, 3394640]
        # 100um pinhole x, y = [2030000, 6475000] [1955000, 6460000] [2106000, 6448000]
        self._tomography_limits = [6586000, 7246000]

    def toggle(self, axis, state):
        if axis in self.axis_toggle:
            dictionary = self.axis_toggle[axis]
            if state in dictionary:
                self.move(dictionary[state], axis)
            else:
                self._logger.warn('Unrecognised state %s. Needs to be one of %s' % (state, list(dictionary.keys())))
        else:
            self._logger.warn('Axis %s does not have toggle values' % axis)

    def tomography(self, kvalue):
        """

        :param kvalue: float. In inverse micron
        :return:
        """
        counts = np.interp(kvalue, np.linspace(-4, 4), np.linspace(self._tomography_limits[0],
                                                                   self._tomography_limits[1]))
        self.move(counts, 'spectrometer_lens')