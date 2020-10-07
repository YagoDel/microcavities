# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtCore, uic, get_qt_app
from microcavities.utils.HierarchicalScan import ExperimentScan
from microcavities.utils import get_data_path, get_data_directory
from microcavities.analysis.characterisation import dispersion_power_series
import os
import yaml
import pymsgbox


data_path = get_data_path()
data_directory = get_data_directory()
app = get_qt_app()


def qt_wait(max_wait=120.0, interval=0.1):
    if max_wait > 0:
        QtCore.QTimer.singleShot(interval*1e3, lambda: qt_wait(max_wait-interval, interval))
    else:
        print('hey')


class PowerSeries(QtWidgets.QWidget):
    """GUI for multi-section power series data taking and analysis
    Should be run inside the kernel of a microcavities.experiment.run.Experiment

    Can set the instrument and variable names for changing the power and the camera name and function name for taking
    images.
    Each section in the power series has a different range of powers, and a different exposure value for the camera.
    Background images still need to be taken afterwards.
    """

    def __init__(self):
        super(PowerSeries, self).__init__()
        self.n_sections = 0
        self.lineedits = []
        self.labels = None
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'powerseries.ui'), self)

        # Set default data path
        self.lineEdit_filename.setText(data_path.lstrip(data_directory))

        # Add column labels
        labels = ['Minimum', 'Maximum', '# measurements', '# images', 'exposure']
        self.labels = [QtWidgets.QLabel(name) for name in labels]
        [self.gridLayout_sections.addWidget(lbl, 0, idx) for lbl, idx in zip(self.labels, range(len(self.labels)))]

        self._connect_gui()

    def _connect_gui(self):
        self.spinBox_sections.valueChanged.connect(self._update_sections_gui)
        self.pushButton_yamls.pressed.connect(self.make_yamls)
        self.pushButton_run.pressed.connect(self.run)
        self.pushButton_analyse.pressed.connect(self.analyse)

    def _update_sections_gui(self):
        """Add/remove rows for each of the power series sections"""
        try:
            exposure = exper[self.lineEdit_camera.text()].exposure
        except:
            exposure = 0.5
        defaults = ['0.01', '0.07', '7', '3', '%.2g' % exposure]
        n_sections = int(self.spinBox_sections.value())
        if n_sections > self.n_sections:
            for row in range(self.n_sections, n_sections):
                lineedit = [QtWidgets.QLineEdit(name) for name in defaults]
                self.lineedits += [lineedit]
                [self.gridLayout_sections.addWidget(wdgt, row+1, col) for wdgt, col in
                 zip(lineedit, range(len(lineedit)))]
        else:
            for row in range(n_sections, self.n_sections):
                [wdgt.deleteLater() for wdgt in self.lineedits[row]]
            self.lineedits = self.lineedits[:n_sections]
        self.n_sections = int(n_sections)

    @property
    def filename(self):
        return os.path.normpath(self.lineEdit_filename.text())

    def make_yamls(self):
        for index in range(self.n_sections):
            self._make_yaml(index)

    def _make_yaml(self, index):
        lineedit = self.lineedits[index]
        measurements = [dict(instrument=str(self.lineEdit_camera.text()),
                             function=str(self.lineEdit_variable2.text()),
                             name='img%d' % idx, save=True) for idx in range(1, int(lineedit[3].text()) + 1)]
        if str(self.lineEdit_variable2.text()) == 'raw_image':
            for measurement in measurements:
                measurement['kwargs'] = dict(update_latest_frame=True)
        dictionary = dict(raw_data_file=self.filename, series_name='dispersion_%d' % index,
                          variables=[dict(instrument=str(self.lineEdit_instrument.text()),
                                          property=str(self.lineEdit_variable.text()),
                                          values=['linear', float(lineedit[0].text()),
                                                  float(lineedit[1].text()), int(lineedit[2].text())])],
                          measurements=measurements
                          )
        filename = os.path.basename(self.filename).rstrip('.h5')
        if os.path.isabs(self.filename):
            directory = os.path.dirname(self.filename)
        else:
            directory = os.path.dirname(os.path.join(data_directory, self.filename))
        if not os.path.exists(os.path.join(directory, 'yamls')):
            os.mkdir(os.path.join(directory, 'yamls'))
        with open(os.path.join(directory, 'yamls', '%s_series_%d.yaml' % (filename, index)), 'w') as file:
            yaml.dump(dictionary, file)
        return

    def run(self):
        self.make_yamls()
        directory = os.path.dirname(self.filename)
        filename = os.path.basename(self.filename).rstrip('.h5')
        for index in range(self.n_sections):
            exposure = float(self.lineedits[index][-1].text())
            exper[self.lineEdit_camera.text()].exposure = exposure
            try:
                yaml_path = os.path.join(directory, 'yamls', '%s_series_%d.yaml' % (filename, index))
                scan = ExperimentScan(yaml_path, exper, gui)
                scan.run()
            except Exception as e:
                print('Failed running at %d because: %s' % (index, e))

            toggle = pymsgbox.confirm('Prepare next run: nd filters', 'Confirm abort', ["Continue", 'Stop'])
            if toggle == "Stop":
                break

    def analyse(self):
        directory = os.path.dirname(self.filename)
        filename = os.path.basename(self.filename).rstrip('.h5')
        yaml_paths = [os.path.join(directory, 'yamls', '%s_series_%d.yaml' % (filename, index)) for index in range(self.n_sections)]
        if self.lineEdit_analysistype.text() == 'dispersion':
            fig, axs = dispersion_power_series(yaml_paths, None, 0, exper['spectrometer'].wavelength, '1200')
            fig.show()


ps = PowerSeries()
ps.show()
