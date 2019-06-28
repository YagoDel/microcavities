# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app, QtGui, QtCore, QtWidgets
from nplab.utils.show_gui_mixin import ShowGUIMixin
import numpy as np
import pyqtgraph as pg
from microcavities.utils.HierarchicalScan import AnalysisScan
from nplab.instrument.camera.camera_scaled_roi import DisplayWidgetRoiScale


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


class Tomography(object, ShowGUIMixin):
    def __init__(self, yaml_path):
        super(Tomography, self).__init__()
        scan = AnalysisScan(yaml_path)
        scan.run()

        images = scan.analysed_data['raw1']
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
        zaxis = scan.variables.items()[0][1]

        self.Im3D = None
        self.Im2D = None

    def get_qt_ui(self):
        win = QtWidgets.QMainWindow()
        win.resize(800, 800)
        l = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        win.setCentralWidget(l)

        self.Im3D = DisplayWidgetRoiScale()
        self.Im3D.ImageDisplay.imageItem.axisOrder = 'col-major'
        self.Im3D.ImageDisplay.setImage(self.images)
        l.addWidget(self.Im3D)

        self.Im2D = pg.ImageView()
        l.addWidget(self.Im2D)
        roi = myROI([[0, 10], [10, 10], [10, 30]])
        self.Im3D.ImageDisplay.addItem(roi)
        roi.sigRegionChanged.connect(self.update)

        return win

    def update(self, _roi):
        imitem = self.Im3D.ImageDisplay.getImageItem()
        img = _roi.getArrayRegion(self.images, imitem, axes=(1, 2))
        self.Im2D.setImage(img)
        self.Im2D.autoRange()


if __name__ == '__main__':
    path = r'D:\DATA\2019_05_30/tomography.yaml'
    tomo = Tomography(path)
    tomo.show_gui()