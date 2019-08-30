# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtGui, QtCore, get_qt_app
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.utils import gui_generator
from microcavities.experiment.instruments import PvcamServer, AndorServer
from microcavities.experiment.utils import magnification
import yaml
import numpy as np
import os
from functools import partial


class Experiment(ShowGUIMixin):

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
        full_settings = yaml.full_load(open(path, 'r'))
        self.settings = full_settings
        self.instrument_settings = full_settings['instruments']
        self.gui_settings = {}
        for variable in ['working_directory', 'scripts_path', 'dock_settings_path', 'file_path']:
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
            print(('Failed to open %s because %s' % (setting['class'], e)))

    def _open_instruments(self):
        _instr_dict = dict()
        for name, setting in list(self.instrument_settings.items()):
            if 'use' in setting and not setting['use']:
                continue
            else:
                print(('Opening %s' % name))
                instr = Experiment._open_instrument(setting)
                if instr is not None:
                    if isinstance(instr, PvcamServer) or isinstance(instr, AndorServer):
                        instr.run(False, True)
                        _instr_dict[name] = instr.instrument
                    else:
                        _instr_dict[name] = instr
        print('Instrument set up finished')
        return _instr_dict

    def get_qt_ui(self):
        return ExperimentGUI(self, **self.gui_settings)


class ExperimentGUI(gui_generator.GuiGenerator):
    def __init__(self, experiment, *args, **kwargs):
        # self._instr_dict = self._open_instruments()
        # super(ExperimentGUI, self).__init__(weakref.WeakValueDictionary(self._instr_dict), *args, **kwargs)
        super(ExperimentGUI, self).__init__(experiment.instr_dict, *args, **kwargs)

        self.instrument_settings = experiment.instrument_settings
        self.experiment_settings = experiment.settings

        self._abort_scan = False
        self._pause_scan = False
        self._setup_scan()

        self._setup_calibrations()
        # self._setup_instrument_actions()

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

        for name, setting in list(self.instrument_settings.items()):
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
        print(('Togelling %s' % name))
        if name in self.instr_dict:
            self._close_instrument(name)
            # del self.instr_dict[name]
            # self.terminalWindow.execute_command('del %s' % name)
            # self.terminalWindow.execute_command("del exper['%s']" % name)
        else:
            self.instr_dict[name] = self._open_instrument(self.instrument_settings[name])
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

    # CCD axis calibrations
    def _setup_calibrations(self):
        self.menuCalibration = QtWidgets.QMenu('Calibrations')
        self.menubar.addMenu(self.menuCalibration)

        if 'calibrations' in self.experiment_settings:
            self.calibration_actions = []
            for camera, cam_props in list(self.experiment_settings['calibrations'].items()):
                if camera in self.instr_dict:
                    for name, cal_props in list(cam_props['calibrations'].items()):
                        action = self._calibration_button(camera, name, cam_props, cal_props)
                        self.calibration_actions += [action]

    def _calibration_button(self, camera, name, cam_props, cal_props):
        """ Utility function for creating action buttons dynamically in a loop

        :param camera: str camera name
        :param name: str name of the calibration
        :param cam_props: dictionary of camera properties (see yaml)
        :param cal_props: dictionary of calibration properties (see yaml)
        :return:
        """
        self._logger.debug('Making action: %s_%s' % (camera, name))
        action = QtWidgets.QAction('%s_%s' % (camera, name), self)
        self.menuCalibration.addAction(action)
        action.triggered.connect(lambda: self.calibrate(camera, cam_props, cal_props))
        return action

    def calibrate(self, camera_name, camera_properties, calibration_properties):
        self._logger.debug('Calibrating: %s, %s' % (camera_name, calibration_properties))
        camera = self.instr_dict[camera_name]
        pixel = camera_properties['pixel_size'] / 1e6  # Transforming from micron to SI
        shape = camera_properties['detector_shape']
        axes = dict(x=0, y=1)
        axes_names = dict(x='bottom', y='left')

        for ax, props in list(calibration_properties.items()):
            ax_idx = axes[ax]
            name = axes_names[ax]
            if props == 'spectrometer':
                wvl = self.instr_dict['spectrometer'].wavelength
                pixels = np.arange(shape[ax_idx])
                setattr(camera, '%s_axis' % ax,
                        (-7.991E-06 * wvl + 2.454E-02) * pixels + (-2.131E-04 * wvl + 1.937E-01) + wvl)
                camera.axis_units[name] = "nm"
            else:
                mag, _ = magnification(props['lenses'])
                ratio = pixel / mag
                if 'units' in props:
                    unit = props['units']
                    if unit == 'micron':
                        ratio /= 1e-6  # Converting from SI to micron
                    elif unit == 'inverse_micron':
                        ratio *= 1e-6  # Converting from SI to micron
                else:
                    unit = ''
                self._logger.debug('Calibration parameters %s axis: %s mag, %s pixel, %s shape, %s ratio, %s unit' %
                                   (ax, mag, shape, pixel, ratio, unit))

                setattr(camera, '%s_axis' % ax, np.linspace(-shape[ax_idx] * ratio / 2,
                                                            shape[ax_idx] * ratio / 2,
                                                            shape[ax_idx]))
                camera.axis_units[name] = unit

    # Scanning functionality
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

    def abort_scan(self):
        self._abort_scan = self.actionScanAbort.isChecked()

    def pause_scan(self):
        self._pause_scan = self.actionScanPause.isChecked()
        self.actionScanPlay.setChecked(not self._pause_scan)

    def play_scan(self):
        self._pause_scan = not self.actionScanPlay.isChecked()
        self.actionScanPause.setChecked(self._pause_scan)


exper = Experiment()
exper.show_gui()
