# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtGui, QtCore, get_qt_app
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.utils import gui_generator
import yaml
import numpy as np
import os
from functools import partial


# TODO:
#   Autofocus:
#       A simple one simply calculating the required exposure to saturate (assuming linear trends).
#       A more complicated one where it adjusts according to an LUT.
#   AutoWL


# full_settings = yaml.load(open("settings.yaml", 'r'))
# settings = full_settings['instruments']


class Experiment(object, ShowGUIMixin):

    def __init__(self, settings_file="settings.yaml"):
        super(Experiment, self).__init__()
        self.settings = self._parse_settings(settings_file)
        self.instr_dict = self._open_instruments()
        print self.instr_dict

    @staticmethod
    def _parse_settings(path):
        """
        The yaml file should be a dictionary with at least one key called "instruments". Under "instruments" there
        should be another dictionary where the keys are the names you want the instrument instances to have, and the
        values are another dictionary with up to four keys:
            class: full name of the instrument class e.g. nplab.instrument.camera.lumenera.LumeneraCamera
            args: list of arguments to be passed to the instance. Not required
            kwargs: list of named arguments to be passed to the instance. Not required
            use: boolean on whether to use the instrument in the current experiment or not. Not required

        Example, two instruments a camera and a spectrometer:
            instruments:
                camera:
                    class: nplab.instrument.camera.Andor.Andor
                    use: True
                spectrometer:
                    class: nplab.instrument.spectrometer.Triax.Triax
                    args: ["COM11"]
                    use: True

        :param path: str
        :return:
        """
        full_settings = yaml.load(open(path, 'r'))
        return full_settings['instruments']

    @staticmethod
    def _open_instrument(setting):
        if 'args' not in setting:
            setting['args'] = []
        if 'kwargs' not in setting:
            setting['kwargs'] = {}
        assert isinstance(setting['kwargs'], dict)
        assert isinstance(setting['args'], list)

        try:
            full_class_name = setting['class']
            location = '.'.join(full_class_name.split('.')[:-1])
            class_name = full_class_name.split('.')[-1]
            exec("from %s import %s" % (location, class_name))
            return eval(class_name)(*setting['args'], **setting['kwargs'])
        except Exception as e:
            print 'Failed to open %s because %s' % (setting['class'], e)

    def _open_instruments(self):
        _instr_dict = dict()
        for name, setting in self.settings.items():
            if 'use' in setting and not setting['use']:
                continue
            else:
                print 'Opening %s' % name
                instr = Experiment._open_instrument(setting)
                if instr is not None:
                    _instr_dict[name] = instr
        print 'Instrument set up finished'
        return _instr_dict

    def get_qt_ui(self):
        return ExperimentGUI(self, dock_settings_path="docks.npy")


class ExperimentGUI(gui_generator.GuiGenerator):
    def __init__(self, experiment, *args, **kwargs):
        # self._instr_dict = self._open_instruments()
        # super(ExperimentGUI, self).__init__(weakref.WeakValueDictionary(self._instr_dict), *args, **kwargs)
        super(ExperimentGUI, self).__init__(experiment.instr_dict, *args, **kwargs)

        self.settings = experiment.settings

        self._abort_scan = False
        self._pause_scan = False
        self._setup_scan()

        self._setup_calibrations()
        # self._setup_instrument_actions()

    def _setup_scan(self):
        icon_path = os.path.dirname(os.path.abspath(gui_generator.__file__)) + r'\icons'
        scan_abort = QtWidgets.QAction(QtGui.QIcon(icon_path + '/ScanStop.png'), 'scan_abort', self)
        scan_abort.setCheckable(True)
        self.toolBar.addAction(scan_abort)
        self.actionScanAbort = scan_abort
        scan_abort.triggered.connect(self.abort_scan)

        scan_pause = QtWidgets.QAction(QtGui.QIcon(icon_path + '/ScanPause.png'), 'scan_pause', self)
        scan_pause.setCheckable(True)
        self.toolBar.addAction(scan_pause)
        self.actionScanPause = scan_pause
        scan_pause.triggered.connect(self.pause_scan)
        scan_play = QtWidgets.QAction(QtGui.QIcon(icon_path + '/ScanPlay.png'), 'scan_play', self)
        scan_play.setCheckable(True)
        self.toolBar.addAction(scan_play)
        self.actionScanPlay = scan_play
        scan_play.triggered.connect(self.play_scan)

    def _setup_calibrations(self):
        self.menuCalibration = QtWidgets.QMenu('Calibrations')
        self.menubar.addMenu(self.menuCalibration)
        # self.calibrationActions = ()
        # for name in ['kk', 'rr', 'Ek', 'Er']:
        #     action = QtWidgets.QAction(name, self)
        #     self.menuCalibration.addAction(action)
        #     action.triggered.connect(lambda: self.calibrate(name))
        #     self.calibrationActions += (action, )

        self.action_calibrate_kspace = QtWidgets.QAction('kk', self)
        self.menuCalibration.addAction(self.action_calibrate_kspace)
        self.action_calibrate_kspace.triggered.connect(lambda: self.calibrate('kk'))

        self.action_calibrate_rspace = QtWidgets.QAction('rr', self)
        self.menuCalibration.addAction(self.action_calibrate_rspace)
        self.action_calibrate_rspace.triggered.connect(lambda: self.calibrate('rr'))

        self.action_calibrate_Ekspace = QtWidgets.QAction('Ek', self)
        self.menuCalibration.addAction(self.action_calibrate_Ekspace)
        self.action_calibrate_Ekspace.triggered.connect(lambda: self.calibrate('Ek'))

        self.action_calibrate_Erspace = QtWidgets.QAction('Er', self)
        self.menuCalibration.addAction(self.action_calibrate_Erspace)
        self.action_calibrate_Erspace.triggered.connect(lambda: self.calibrate('Er'))

    def calibrate(self, name):
        try:
            if name == 'kk':
                # foci = [0.01, 0.25, 0.1, 0.1, 0.15, 0.15, 0.2]
                # mag = (9.93126844893427e-10, [1.2414085561167838e-09, 1.2414085561167838e-09,
                #       4.965634224467135e-10, 7.448451336700703e-10, 9.93126844893427e-10])
                # 1px = 0.008055 um-1
                self.instr_dict['andor'].x_axis = np.linspace(-4.03555, 4.03555, 1002)
                self.instr_dict['andor'].y_axis = np.linspace(-4.04361, 4.04361, 1004)
                self.instr_dict['andor'].units = (u"\u03BCm<sup>-1</sup>",
                                                  u"\u03BCm<sup>-1</sup>")
            elif name == 'rr':
                # foci = [0.01, 0.25, 0.1 , 0.1 , 0.15, 0.2 ]
                # mag = (33.333333333333336, [1, 25.0, 25.0, 33.333333333333336])
                # 1px = 0.24 um
                self.instr_dict['andor'].x_axis = np.linspace(-120.24, 120.24, 1002)
                self.instr_dict['andor'].y_axis = np.linspace(-120.48, 120.48, 1004)
                self.instr_dict['andor'].units = (u"\u03BCm", u"\u03BCm")
            elif name[0] == 'E':
                wvl = self.instr_dict['spectrometer'].wavelength
                pixels = np.arange(1340)
                self.instr_dict['pvcam'].x_axis = (-7.991E-06 * wvl + 2.454E-02) * pixels + \
                                                  (-2.131E-04 * wvl + 1.937E-01) + wvl
                self.instr_dict['pvcam'].units = ("nm", )

                if name[1] == 'k':
                    # foci = [0.01, 0.25, 0.1, 0.1, 0.2]
                    # mag = (9.93126844893427e-10, [1.2414085561167838e-09, 1.2414085561167838e-09,
                    #        4.965634224467135e-10, 9.93126844893427e-10])
                    # 1px = 0.008055 um-1
                    self.instr_dict['pvcam'].y_axis = np.linspace(-1.611, 1.611, 400)
                    self.instr_dict['pvcam'].units += (u"\u03BCm<sup>-1</sup>",)
                elif name[1] == 'r':
                    # foci = [0.01, 0.25, 0.1, 0.2]
                    # mag = (50.0, [1, 25.0, 50.0])
                    # 1px = 0.4 um
                    self.instr_dict['pvcam'].y_axis = np.linspace(-80, 80, 400)
                    self.instr_dict['pvcam'].units += (u"\u03BCm",)
            else:
                print 'Nope'
        except Exception as e:
            print 'Failed calibrating: ', e

    def abort_scan(self):
        self._abort_scan = self.actionScanAbort.isChecked()

    def pause_scan(self):
        self._pause_scan = self.actionScanPause.isChecked()
        self.actionScanPlay.setChecked(not self._pause_scan)

    def play_scan(self):
        self._pause_scan = not self.actionScanPlay.isChecked()
        self.actionScanPause.setChecked(self._pause_scan)

    def _close_instrument(self, name):
        dock = self.allDocks[name]
        dock.close()
        dock.setParent(None)
        dock.label.setParent(None)
        # QtCore.QCoreApplication.instance().processEvents()

        del self.dockWidgetArea.docks[name]
        # del self.dockwidgetArea.docks[name]
        del self.instr_dict[name]
        self.terminalWindow.execute_command('del %s' % name)
        del self.allWidgets[name]
        del self.allDocks[name]
        del self.actions['Views'][name]

    def _setup_instrument_actions(self):
        self.menuInstruments = QtWidgets.QMenu('Instruments')
        self.menubar.addMenu(self.menuInstruments)

        for name, setting in self.settings.items():
            action = QtWidgets.QAction(name, self)
            self.menuInstruments.addAction(action)
            action.setCheckable(True)
            if 'use' in setting:
                action.setChecked(setting['use'])
            else:
                action.setChecked(1)
            action.triggered.connect(partial(self._toggle_instr, name))
            self.actions['Instruments'][name] = action

    def _toggle_instr(self, name):
        print 'Togelling %s' % name
        if name in self.instr_dict:
            self._close_instrument(name)
            # del self.instr_dict[name]
            # self.terminalWindow.execute_command('del %s' % name)
            # self.terminalWindow.execute_command("del exper['%s']" % name)
        else:
            self.instr_dict[name] = self._open_instrument(self.settings[name])
            self.terminalWindow.push_vars({name: self.instr_dict[name]})
            # self._open_one_gui(name)


exper = Experiment()
exper.show_gui()


# from sympy import sympify, Symbol
# from sympy.matrices import Matrix
# from sympy import pprint
# import sys
# sys.displayhook = pprint
#
# f1 = Symbol('f1')
# f2 = Symbol('f2')
# f3 = Symbol('f3')
# d1 = Symbol('d1')
# d2 = Symbol('d2')
# d3 = Symbol('d3')
#
#
# def lens(f):
#     return Matrix([[1, 0], [-1./f, 0]])
# def dist(d):
#     return Matrix([[1, d], [0, 1]])
#
# mat1 = dist(d1)*lens(f1)
# mat2 = lens(f2) * mat1
# mat3 = dist(d2) * mat2
# full = lens(f3) * mat3


#
def magnification(focus_array, wavelength=780e-9):
    assert len(focus_array) > 1

    if len(focus_array) % 2:
        # For an odd number of lenses, you are measuring k-space, so we use the wavelength to get the wavenumber
        kp = 2 * np.pi / wavelength  # wavenumber in m-1
        m = focus_array[0] / kp
        m_array = [m]
        m2, m_array2 = magnification(focus_array[1:])
        m_array += list(map(lambda x: m * x, m_array2))
        m *= m2
    else:
        # For an even number of lenses, you are measuring real space
        m = 1
        m_array = [m]
        for idx in range(len(focus_array)/2):
            m *= focus_array[2*idx + 1] / focus_array[2*idx]
            m_array += [m]
    return m, m_array

# def calcmag(foc_arr, lam):
#     from numpy import pi
#     # all focal lengths in foc_arr are in m
#     kp = 2 * pi / lam  # wavenumber in m-1
#
#     assert len(foc_arr) > 1
#     if len(foc_arr) == 2:
#         return foc_arr[1] / foc_arr[0], None
#     else:
#         mr = [foc_arr[1] / foc_arr[0]]
#         mk = [foc_arr[2] / foc_arr[1] * foc_arr[0] / kp]
#         for idx, focus in enumerate(foc_arr):
#             if idx == 1:
#                 mr = [foc_arr[idx] / foc_arr[idx - 1]]
#             elif idx == 2:
#                 mk = [foc_arr[2] / foc_arr[1] * foc_arr[0] / kp]
#             elif idx > 2:
#                 if idx % 2:
#                     mk += [mk[-1] * foc_arr[idx] / foc_arr[idx - 1]]
#                 else:
#                     mr += [mr[-1] * foc_arr[idx] / foc_arr[idx - 1]]
#     return mr, mk
#
#     if len(foc_arr) == 5:
#         Mr1 = foc_arr['fa'] / foc_arr['fo']
#         Mk2 = foc_arr['fb'] / foc_arr['fa'] * foc_arr['fo'] / kp
#         Mr3 = Mr1 * foc_arr['fc'] / foc_arr['fb']
#         Mrs = Mr1 * foc_arr['fd'] / foc_arr['fb']
#         Mks = Mk2 * foc_arr['fd'] / foc_arr['fc']
#         M_arr = {'Mr1': Mr1, 'Mk2': Mk2, 'Mr3': Mr3, 'Mrs': Mrs, 'Mks': Mks, }
#
#     return M_arr
#
# m_per_pix = 8 / M_arr['Mrs']
#
# {'fo':10.0e-3,'fa':250e-3,'fb':100e-3,'fc':100e-3,'fd':200e-3,}
# {'Mk2': 4.965634224467135e-10,
#  'Mks': 9.93126844893427e-10,
#  'Mr1': 25.0,
#  'Mr3': 25.0,
#  'Mrs': 50.0}
#
#
# [10, 250, 100, 100, 150, 150, 200]