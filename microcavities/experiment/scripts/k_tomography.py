# -*- coding: utf-8 -*-

from nplab.utils.gui import QtWidgets, QtCore, uic, get_qt_app
from microcavities.utils.HierarchicalScan import ExperimentScan

stages.toggle('k_lens', 'on')


class BasicInput(QtWidgets.QMainWindow):
    def __init__(self):
        super(BasicInput, self).__init__()
        self.lineedits = None
        self.labels = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        central_widget.setLayout(layout)
        labels = ['Minimum', 'Maximum', '# measurements', '# images']
        defaults = ['0.01', '0.15', '8', '1']
        self.lineedits = [QtWidgets.QLineEdit(name) for name in defaults]
        self.labels = [QtWidgets.QLabel(name) for name in labels]

        for idx, wdgt, lbl in zip(range(4), self.lineedits, self.labels):
            layout.addWidget(wdgt, idx, 1)
            layout.addWidget(lbl, idx, 0)
        self.setCentralWidget(central_widget)


app = get_qt_app()
bi = BasicInput()
bi.show()

values = [float(x.text()) for x in bi.lineedits[:-1]]
n_images = int(bi.lineedits[-1].text())

params = dict(save_type="local",
              variables=[dict(instrument='stages', function='move', values=['linear'] + values,
                              kwargs=dict(axes='spectrometer_lens'))],
              measurements=[
                  dict(instrument='pvcam', function='raw_image', name='img%d' % (x+1),
                       kwargs=dict(update_latest_frame=True), save=True) for x in range(n_images)
              ],
              analysis_functions=[
                  dict(data_name='img%d' % (x+1), function_name='raw_img%d' % (x+1)) for x in range(n_images)
              ]
              )

scan = ExperimentScan(exper, gui, params)
scan.run()

# TODO: analysis for interactive plotting of the tomography data
