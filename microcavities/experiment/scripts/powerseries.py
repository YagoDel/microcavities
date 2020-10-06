# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtCore, uic, get_qt_app
from microcavities.utils.HierarchicalScan import ExperimentScan
from microcavities.utils import get_data_path
import time
import os
import sys
import yaml


data_path = get_data_path()
app = get_qt_app()


def qt_wait(max_wait=120.0, interval=0.1):
    if max_wait > 0:
        QtCore.QTimer.singleShot(interval*1e3, lambda: qt_wait(max_wait-interval, interval))
    else:
        print('hey')


class BasicInput(QtWidgets.QWidget):
    finished_setup = QtCore.Signal()

    def __init__(self):
        super(BasicInput, self).__init__()
        self.n_sections = 0
        self.lineedits = []
        self.labels = None
        uic.loadUi((os.path.dirname(__file__) + '/k_powerseries.ui'), self)
        self.lineEdit_filename.setText(data_path)

        labels = ['Minimum', 'Maximum', '# measurements', '# images', 'exposure']
        self.labels = [QtWidgets.QLabel(name) for name in labels]
        [self.gridLayout_sections.addWidget(lbl, 0, idx) for lbl, idx in zip(self.labels, range(len(self.labels)))]

        self.spinBox_sections.valueChanged.connect(self.update_sections)
        self.pushButton_yamls.pressed.connect(self.make_yamls)
        self.pushButton_run.pressed.connect(self.run)

        # self.setup_ui()
        # self.finished_setup.connect(self.run, type=QtCore.Qt.QueuedConnection)

    def update_sections(self):
        try:
            exposure = exper[self.lineEdit_camera.text()].exposure
        except:
            exposure = 0.5
        defaults = ['0.01', '0.07', '7', '3', '%.2g' % exposure]
        lineedit = [QtWidgets.QLineEdit(name) for name in defaults]
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
        return str(self.lineEdit_filename.text())
    #
    # def setup_ui(self):
    #     centralWidget = QtWidgets.QWidget()
    #     layout = QtWidgets.QGridLayout()
    #     centralWidget.setLayout(layout)
    #     labels = ['Instrument', 'Minimum', 'Maximum', '# measurements', '# images', 'exposure']
    #     defaults = ['power_wheel', '0.01', '0.15', '8', '1']
    #     self.lineedits = [QtWidgets.QLineEdit(name) for name in defaults]
    #     self.labels = [QtWidgets.QLabel(name) for name in labels]
    #
    #     for idx, wdgt, lbl in zip(list(range(5)), self.lineedits, self.labels):
    #         layout.addWidget(wdgt, idx, 1)
    #         layout.addWidget(lbl, idx, 0)
    #     self.setCentralWidget(centralWidget)

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
        directory = os.path.dirname(self.filename)
        with open(os.path.join(directory, 'yamls', 'dispersion_%d.yaml' % index), 'w') as file:
            yaml.dump(dictionary, file)
        return

    def run(self):
        self.make_yamls()
        directory = os.path.dirname(self.filename)
        for index in range(self.n_sections):
            exposure = float(self.lineedits[index][-1].text())
            exper[self.lineEdit_camera.text()].exposure = exposure
            try:
                yaml_path = os.path.join(directory, 'yamls', 'dispersion_%d.yaml' % index)
                scan = ExperimentScan(yaml_path, exper, gui)
                scan.run()
            except Exception as e:
                print('Failed running at %d because: %s' % (index, e))
        return


bi = BasicInput()
bi.show()
# sys.exit(app.exec_())

# while not bi.isHidden():
#     time.sleep(0.1)
#     app.processEvents()
#
# values = [float(x.text()) for x in bi.lineedits[1:-1]]
# instr = bi.lineedits[0].text()
# n_images = int(bi.lineedits[-1].text())
#
# params = dict(save_type="local",
#               variables=[dict(instrument=instr, property='power', values=['linear'] + values)],
#               measurements=[
#                   dict(instrument='pvcam', function='raw_image', name='img%d' % (x + 1),
#                        kwargs=dict(update_latest_frame=True), save=True) for x in range(n_images)
#               ],
#               analysis_functions=[
#                   dict(data_name='img%d' % (x + 1), function_name='raw_img%d' % (x + 1)) for x in range(n_images)
#               ]
#               )
# scan = ExperimentScan(params, exper, gui)
# scan.run_modally()
# data = np.array([scan.results['img%d' % (x+1)] for x in range(n_images)])
#
# # The following presumes you have run the dispersion script, which defines k0 and energy_axis
# mean_data = np.mean(data, 0)
# k0_data = mean_data[:, k0]
#
# normalised = []
# for row in k0_data:
#     row = np.copy(row)
#     row -= np.percentile(row, 0.1)
#     row /= np.percentile(row, 99.9)
#     normalised += [row]
# normalised = np.array(normalised)
#
# plt.imshow(normalised.transpose(), aspect='auto', extent=[values[0], values[1], energy_axis[-1], energy_axis[0]])
#
