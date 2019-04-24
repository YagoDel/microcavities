# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtGui, QtCore, get_qt_app
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.utils import gui_generator
from microcavities.experiment.instruments import PvcamServer, AndorServer
import yaml
import numpy as np
import os
from functools import partial


# TODO:
#   Functions:
#       Autoexposure:
#           A simple one simply calculating the required exposure to saturate (assuming linear trends).
#           A more complicated one where it adjusts according to an LUT.
#       Autofocus:
#           Find a place on the sample that we know is in focus
#           Focus a laser on the surface manually
#           Auto-focus will maximise emission on the laser position / minimise laser shape
#           Check that the laser is properly focused. Add an offset correction if necessary
#           Need to somehow automatically switch
#       AutoWL
#       Threshold
#           Power fitting
#           Energy fitting
#   Experiment yaml
#       Lenses -> magnification -> calibrations


class Experiment(object, ShowGUIMixin):

    def __init__(self, settings_file="settings.yaml"):
        super(Experiment, self).__init__()
        self._parse_settings(settings_file)
        self.instr_dict = self._open_instruments()

    def _parse_settings(self, path):
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
        self.settings = full_settings['instruments']
        self.gui_settings = {}
        for variable in ['working_directory', 'scripts_path', 'dock_settings_path']:
            if variable in full_settings:
                self.gui_settings[variable] = full_settings[variable]

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
                    if isinstance(instr, PvcamServer) or isinstance(instr, AndorServer):
                        instr.run(False, True)
                        _instr_dict[name] = instr.instrument
                    else:
                        _instr_dict[name] = instr
        print 'Instrument set up finished'
        return _instr_dict

    def get_qt_ui(self):
        return ExperimentGUI(self, **self.gui_settings)


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

    def makeScriptMenu(self):
        """Generate a menu for running the scripts found in the scripts path locationlocation """
        from functools import partial

        if self.script_menu is None:
            script_menu = self.menuBar().addMenu('&Scripts')
        else:
            script_menu = self.script_menu

        menus = {self.scripts_path: script_menu}

        for dirpath, dirnames, filenames in os.walk(self.scripts_path):
            # print filenames
            current = menus[dirpath]
            for dn in dirnames:
                menus[os.path.join(dirpath, dn)] = current.addMenu(dn)
            for fn in filenames:
                if fn.endswith('.py'):
                    if fn != '__init__.py':
                        menuitem = current.addAction(fn)
                        menuitem.triggered.connect(partial(self.menuScriptClicked, fn))

        script_menu.addSeparator()
        refreshScripts = script_menu.addAction('Refresh')
        refreshScripts.triggered.connect(self.refreshScriptMenu)
        self.script_menu = script_menu


exper = Experiment()
exper.show_gui()
