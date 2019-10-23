# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app, QtGui, QtCore, QtWidgets, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.ui.widgets.imageview import ExtendedImageView
import numpy as np
import pyqtgraph as pg
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
        self.imageItem.axisOrder = 'col-major'  # without this, the LineROIs don't select the right region (not sure why)

        self.setImage(data, xvals=z_axis)
        self.spectra = self.ui.roiPlot.plot(z_axis, spectra, pen=pg.mkPen('r'))
        self.getHistogramWidget().gradient.restoreState(list(Gradients.values())[1])

        self.checkbox_logy = QtWidgets.QCheckBox('log y')
        self.tools.gridLayout.addWidget(self.checkbox_logy, 0, 3, 1, 1)
        self.checkbox_logy.stateChanged.connect(
            lambda: self.ui.roiPlot.getPlotItem().setLogMode(y=self.checkbox_logy.isChecked())
            )

        self.Im2D = pg.ImageView()
        self.ui.splitter.addWidget(self.Im2D)
        roi = myROI([[0, 10], [10, 10], [10, 30]])
        self.view.addItem(roi)
        roi.sigRegionChanged.connect(self.update)

    def update(self, _roi):
        imitem = self.getImageItem()
        try:
            img = _roi.getArrayRegion(self.image, imitem, axes=(1, 2))
            self.Im2D.setImage(img)
            self.Im2D.autoRange()
        except Exception as e:
            print e


class Tomography(ShowGUIMixin):
    def __init__(self, yaml_path):
        """Extracting data and axes from a yaml location for a tomography scan"""
        super(Tomography, self).__init__()

        # DEBUGGING CODE
        # x = np.linspace(-10, 10, 101)
        # y = np.linspace(-10, 10, 101)
        # z = np.linspace(-2, 10, 1340)
        # X, Y, Z = np.meshgrid(x, y, z)
        # images = np.zeros(X.shape)
        # images[np.abs(0.5 * (X-5)**2 + 2 * (Y+5)**2 - Z**2) < 1] = 1

        scan = AnalysisScan(yaml_path)
        scan.run()

        keys = scan.analysed_data.keys()
        images = scan.analysed_data[keys[0]]
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

        self.images = np.swapaxes(images, 0, -1)

        self.Im3D = None
        self.Im2D = None

    def get_qt_ui(self):
        return TomographyDisplay(self.images, 800)


if __name__ == '__main__':
    path = r'D:\DATA\2019_09_30/yamls/tomography.yaml'
    tomo = Tomography(path)
    tomo.show_gui()