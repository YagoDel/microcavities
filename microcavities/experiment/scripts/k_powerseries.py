# -*- coding: utf-8 -*-
from nplab.utils.gui import QtWidgets, QtCore, uic, get_qt_app
from microcavities.utils.HierarchicalScan import ExperimentScan
import time


class BasicInput(QtWidgets.QMainWindow):
    def __init__(self):
        super(BasicInput, self).__init__()
        self.lineedits = None
        self.labels = None
        self.setup_ui()

    def setup_ui(self):
        centralWidget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        centralWidget.setLayout(layout)
        labels = ['Instrument', 'Minimum', 'Maximum', '# measurements', '# images']
        defaults = ['power_wheel', '0.01', '0.15', '8', '1']
        self.lineedits = [QtWidgets.QLineEdit(name) for name in defaults]
        self.labels = [QtWidgets.QLabel(name) for name in labels]

        for idx, wdgt, lbl in zip(list(range(5)), self.lineedits, self.labels):
            layout.addWidget(wdgt, idx, 1)
            layout.addWidget(lbl, idx, 0)
        self.setCentralWidget(centralWidget)


app = get_qt_app()
bi = BasicInput()
bi.show()

while not bi.isHidden():
    time.sleep(0.1)
    app.processEvents()

values = [float(x.text()) for x in bi.lineedits[1:-1]]
instr = bi.lineedits[0].text()
n_images = int(bi.lineedits[-1].text())

params = dict(save_type="local",
              variables=[dict(instrument=instr, property='power', values=['linear'] + values)],
              measurements=[
                  dict(instrument='pvcam', function='raw_image', name='img%d' % (x + 1),
                       kwargs=dict(update_latest_frame=True), save=True) for x in range(n_images)
              ],
              analysis_functions=[
                  dict(data_name='img%d' % (x + 1), function_name='raw_img%d' % (x + 1)) for x in range(n_images)
              ]
              )
scan = ExperimentScan(params, exper, gui)
scan.run_modally()
data = np.array([scan.results['img%d' % (x+1)] for x in range(n_images)])

# The following presumes you have run the dispersion script, which defines k0 and energy_axis
mean_data = np.mean(data, 0)
k0_data = mean_data[:, k0]

normalised = []
for row in k0_data:
    row = np.copy(row)
    row -= np.percentile(row, 0.1)
    row /= np.percentile(row, 99.9)
    normalised += [row]
normalised = np.array(normalised)

plt.imshow(normalised.transpose(), aspect='auto', extent=[values[0], values[1], energy_axis[-1], energy_axis[0]])

