# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app, QtGui, QtCore, QtWidgets, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.ui.widgets.imageview import ExtendedImageView
# from nplab.instrument.camera.camera_scaled_roi import DisplayWidgetRoiScale, ArbitraryAxis, Crosshair
import numpy as np
import pyqtgraph as pg
# import os
from microcavities.utils.HierarchicalScan import AnalysisScan
from microcavities.experiment.utils import spectrometer_calibration
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients


class myROI(pg.PolyLineROI):
    """pyqtgraph ROI. Collection of user-modifiable line segments that returns (N-1)D slices from an ND array"""
    def __init__(self, *args, **kwargs):
        super(myROI, self).__init__(*args, **kwargs)

    def getArrayRegion(self, data, img, axes=(0, 1), **kwds):
        selection = None
        for segment in self.segments:
            sel = segment.getArrayRegion(data, img, axes, **kwds)
            if selection is None:
                selection = np.copy(sel)
            else:
                selection = np.append(selection, sel, 1)
        return selection


class TomographyDisplay(ExtendedImageView):
    def __init__(self, data, wavelength, *args, **kwargs):
        super(TomographyDisplay, self).__init__(*args, **kwargs)
        self.wavelength = wavelength
        z_axis = spectrometer_calibration(wavelength=wavelength)
        spectra = np.mean(data, (1, 2))

        self.setImage(data, xvals=z_axis)
        self.spectra = self.ui.roiPlot.plot(z_axis, spectra, pen=pg.mkPen('r'))
        self.getHistogramWidget().gradient.restoreState(list(Gradients.values())[1])

        self.checkbox_logy = QtWidgets.QCheckBox('log y')
        self.tools.gridLayout.addWidget(self.checkbox_logy, 0, 3, 1, 1)
        self.checkbox_logy.stateChanged.connect(
            lambda: self.ui.roiPlot.getPlotItem().setLogMode(y=self.checkbox_logy.isChecked())
            )


class Tomography(ShowGUIMixin):
    def __init__(self, yaml_path):
        super(Tomography, self).__init__()
        scan = AnalysisScan(yaml_path)
        scan.run()

        keys = scan.analysed_data.keys()
        images = scan.analysed_data[keys[0]]
        self.images = np.swapaxes(images, 0, -1)

        data, attrs = scan.get_data(scan.get_random_group(scan.series_name))
        if 'x_axis' in attrs:
            xaxis = attrs['x_axis']
        else:
            xaxis = np.linspace(-1, 1, data.shape[0])
        if 'y_axis' in attrs:
            yaxis = attrs['y_axis']
        else:
            yaxis = np.linspace(-1, 1, data.shape[1])
        zaxis = list(scan.variables.items())[0][1]

        self.Im3D = None
        self.Im2D = None

    def get_qt_ui(self):
        win = QtWidgets.QMainWindow()
        win.resize(800, 800)
        l = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        win.setCentralWidget(l)

        self.Im3D = TomographyDisplay(self.images, 800)  #DisplayWidgetRoiScale()
        self.Im3D.imageItem.axisOrder = 'col-major'
        l.addWidget(self.Im3D)

        self.Im2D = pg.ImageView()
        l.addWidget(self.Im2D)
        roi = myROI([[0, 10], [10, 10], [10, 30]])
        self.Im3D.addItem(roi)
        roi.sigRegionChanged.connect(self.update)

        return win

    def update(self, _roi):
        imitem = self.Im3D.getImageItem()
        img = _roi.getArrayRegion(self.images, imitem, axes=(1, 2))
        self.Im2D.setImage(img)
        self.Im2D.autoRange()


if __name__ == '__main__':
    path = r'D:\DATA\2019_05_30/tomography.yaml'
    tomo = Tomography(path)
    tomo.show_gui()