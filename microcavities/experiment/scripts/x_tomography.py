# -*- coding: utf-8 -*-

from nplab.utils.gui import QtWidgets, QtCore, uic, get_qt_app
from microcavities.utils.HierarchicalScan import ExperimentScan
from microcavities.experiment.utils import magnification

center_steps = 6617980  # This value might need to be changed if the setup is realigned

stages.toggle('k_lens', 'off')
stages.move(center_steps, 'spectrometer_lens')

mag = magnification([0.01, 0.25, 0.1, 0.2])[0]
x_fac = 20 * 1e-6 / mag  # Converting to SI and dividing by magnification
x_fac /= 1e-6  # converting to micron
stage_steps_to_inverse_micron = 150*x_fac/300000.  # 150 pixels is 300000 steps


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
        # labels = ['Minimum', 'Maximum', '# measurements', '# images']
        # defaults = ['6568000', '7228000', '100', '1']
        labels = ['x_range (micron)', '# measurements', '# images']
        defaults = ['100', '100', '1']
        self.lineedits = [QtWidgets.QLineEdit(name) for name in defaults]
        self.labels = [QtWidgets.QLabel(name) for name in labels]

        for idx, wdgt, lbl in zip(list(range(len(labels))), self.lineedits, self.labels):
            layout.addWidget(wdgt, idx, 1)
            layout.addWidget(lbl, idx, 0)
        self.setCentralWidget(central_widget)


app = get_qt_app()
bi = BasicInput()
bi.show()

x_range = float(bi.lineedits[0].text())
min_steps = int(center_steps - x_range/(2*stage_steps_to_inverse_micron))
max_steps = int(center_steps + x_range/(2*stage_steps_to_inverse_micron))
n_steps = int(bi.lineedits[1].text())
n_images = int(bi.lineedits[-1].text())

params = dict(save_type="local",
              variables=[dict(instrument='stages', function='move', values=['linear', min_steps, max_steps, n_steps],
                              kwargs=dict(axes='spectrometer_lens'))],
              measurements=[
                  dict(instrument='pvcam', function='raw_image', name='img%d' % (x+1),
                       kwargs=dict(update_latest_frame=True), save=True) for x in range(n_images)
              ],
              analysis_functions=[
                  dict(data_name='img%d' % (x+1), function_name='raw_img%d' % (x+1)) for x in range(n_images)
              ]
              )

scan = ExperimentScan(params, exper, gui)
scan.run_modally()


data = np.array([scan.results['img%d' % (x+1)] for x in range(n_images)])
mean_data = np.mean(data, 0)
tst = np.rollaxis(mean_data, 2, 0)
img = pyqtgraph.image(tst)  # Aspect auto
img.getView().setAspectLocked(False)
