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
        labels = ['Instrument', 'Minimum', 'Maximum', '# measurements']
        defaults = ['power_wheel', '0.01', '0.15', '8']
        self.lineedits = [QtWidgets.QLineEdit(name) for name in defaults]
        self.labels = [QtWidgets.QLabel(name) for name in labels]

        for idx, wdgt, lbl in zip(range(4), self.lineedits, self.labels):
            layout.addWidget(wdgt, idx, 1)
            layout.addWidget(lbl, idx, 0)
        self.setCentralWidget(centralWidget)

print 1

app = get_qt_app()
bi = BasicInput()
bi.show()

while not bi.isHidden():
    time.sleep(0.1)
    app.processEvents()

values = [float(x.text()) for x in bi.lineedits[1:]]
instr = bi.lineedits[0].text()

print 2

params = dict(save_type="local",
              variables=[dict(instrument=instr, property='power', values=['linear'] + values)],
              measurements=[
                  dict(instrument='pvcam', function='raw_image', name='img1', kwargs=dict(update_latest_frame=True),
                       save=True),
                  # dict(instrument='pvcam', function='raw_image', name='img2', kwargs=dict(update_latest_frame=True),
                  #      save=True),
                  # dict(instrument='pvcam', function='raw_image', name='img3', kwargs=dict(update_latest_frame=True),
                  #      save=True)
              ],
              analysis_functions=[
                  dict(data_name='img1', function_name='raw_img1'),
                  # dict(data_name='img2', function_name='raw_img1'),
                  # dict(data_name='img3', function_name='raw_img1')
              ]
              )

print 3
# TODO: analysis for finding k0 and plotting the k=0 spectra as a function of power
scan = ExperimentScan(params, exper, gui)

print 4

scan.run()

print 5