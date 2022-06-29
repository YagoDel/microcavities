# -*- coding: utf-8 -*-
from microcavities.utils.plotting import *
from microcavities.utils.plotting import _make_axes
from microcavities.utils import depth
from microcavities.experiment.utils import spectrometer_calibration, magnification
from nplab.utils.gui import QtWidgets, uic
from nplab.utils.log import create_logger
from nplab.utils.show_gui_mixin import ShowGUIMixin
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import least_squares
import pyqtgraph as pg
import pymsgbox
from matplotlib.colors import LinearSegmentedColormap
from functools import partial
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy

LOGGER = create_logger("analysis_functions")

cdict = {'red': [(0.0, 0.0, 1.0),
                 (0.25, 0.0, 0.0),
                 (0.5, 1.0, 1.0),
                 (0.75, 1.0, 1.0),
                 (1.0, 0.0, 0.0)],
         'green': [(0.0, 0.0, 1.0),
                   (0.25, 0.4, 0.4),
                   (0.5, 1.0, 1.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)],
         'blue': [(0.0, 1.0, 1.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]}
mycmap = LinearSegmentedColormap('Michael', cdict, 256)
plt.register_cmap(cmap=mycmap)


def gui_checkplot():
    plt.show()
    reply = pymsgbox.confirm('Are you happy to continue?',
                             'Check plot',
                             ["Yes", 'No'])
    if reply == 'No':
        raise RuntimeError("Unsatisfactory plotting")


# IMAGE ANALYSIS
# Functions to be used on real- and k-space images

class roi2d_GUI(QtWidgets.QMainWindow):
    """ Multi region-of-interest GUI

    Base class for creating GUIs for analysing images where you want to create
    multiple ROIs and extract information about those ROIs
    """

    def __init__(self, images, results, **kwargs):
        super(roi2d_GUI, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'roi_gui.ui'), self)

        self.images = images
        self.results = results
        self.kwargs = dict(kwargs)

        self.spinBoxNoROI.valueChanged.connect(self._make_ROIs)
        self.pushButtonSetROI.clicked.connect(self.SetROIs)
        self.pushButtonAutoMove.clicked.connect(self._move_ROIs)
        self.pushButtonNewImage.clicked.connect(self.new_image)

        if len(self.images.shape) <= 2:
            self.pushButtonNewImage.hide()
            self._current_image = self.images
            self.graphicsView.setImage(self.images)

        self.rois = []

    def _make_single_ROI(self, pen):
        print("ROI type: ", self.comboBox_roitype.text())
        if self.comboBox_roitype.text() == 'Square':
            roi = pg.ROI([50, 10], [3, 3], pen=pen)
            roi.addScaleHandle([1, 0.5], [0.5, 0.5])
            roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            roi.addRotateHandle([0, 0], [0.5, 0.5])
            return roi
        if self.comboBox_roitype.text() == 'Round':
            return pg.CircleROI([10, 10], [3, 3], pen=pen)

    def _make_ROIs(self):
        n_rois = self.spinBoxNoROI.value()

        for roi in self.rois:
            self.graphicsView.getView().removeItem(roi)

        self.rois = []
        for n in range(n_rois):
            pen = pg.mkPen(n)
            self.rois += [self._make_single_ROI(pen)]

        for roi in self.rois:
            self.graphicsView.getView().addItem(roi)

    def SetROIs(self):
        affineSliceParams = []
        for roi in self.rois:
            affineSliceParams += [roi.getAffineSliceParams(self._current_image,
                                                           self.graphicsView.getImageItem())]

        self.results['affineSliceParams'] = affineSliceParams

    def _move_ROIs(self):
        pass

    def new_image(self):
        self._index = np.random.randint(0, self.images.shape[0])
        self._current_image = self.images[self._index]
        self.graphicsView.setImage(self.images[self._index])


class roi2d(ShowGUIMixin):
    def __init__(self, images):
        super(roi2d, self).__init__()

        self.images = images
        self.results = dict()

    def get_qt_ui(self):
        return roi2d_GUI(self.images, self.results)


def roi_intensity(data, roi_params=None):
    return np.sum(data), [roi_params], {}
