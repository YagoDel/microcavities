# -*- coding: utf-8 -*-

from nplab.utils.gui import get_qt_app, QtGui, QtCore, QtWidgets, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.instrument.camera.camera_scaled_roi import DisplayWidgetRoiScale, ArbitraryAxis, Crosshair
import numpy as np
import pyqtgraph as pg
import os
from microcavities.utils.HierarchicalScan import AnalysisScan
from microcavities.experiment.utils import spectrometer_calibration


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


class ExtendedImageView(pg.ImageView):
    """
    Extension of the pyqtgraph ImageView so that it's possible to put percentile levels instead of playing around with
    the histogram. Also adds the possibility of normalising each image when given a 3D array, instead of normalising to
    the maximum of the whole array.
    """
    def __init__(self, *args, **kwargs):
        self.axis_values = dict(bottom=None, left=None, top=None, right=None)
        self.axis_units = dict(bottom=None, left=None, top=None, right=None)
        kwargs['view'] = pg.PlotItem(axisItems=dict(bottom=ArbitraryAxis(orientation="bottom"),
                                                    left=ArbitraryAxis(orientation="left"),
                                                    top=ArbitraryAxis(orientation="top"),
                                                    right=ArbitraryAxis(orientation="right")))
        super(ExtendedImageView, self).__init__(*args, **kwargs)
        self.imageItem.axisOrder = 'row-major'

        # Setting up the autoleveling GUI
        self.level_percentiles = None
        self.levelGroup = uic.loadUi(os.path.join(os.path.dirname(__file__), 'autolevel.ui'))
        self.ui.gridLayout_3.addWidget(self.levelGroup, 2, 0, 1, 1)
        self.levelGroup.setVisible(False)

        self.levelGroup.checkbox_singleimagelevel.stateChanged.connect(self.set_level_percentiles)
        self.levelGroup.lineEdit_minLevel.returnPressed.connect(self.set_level_percentiles)
        self.levelGroup.lineEdit_maxLevel.returnPressed.connect(self.set_level_percentiles)
        self.levelGroup.pushButton_reset.pressed.connect(self.reset)

        # Setting up the additional tools GUI
        self.tools = uic.loadUi(os.path.join(os.path.dirname(__file__), 'imageview_tools.ui'))
        self.ui.splitter.addWidget(self.tools)
        self.tools.checkbox_tools.stateChanged.connect(self.show_tools)
        self.tools.checkbox_aspectratio.stateChanged.connect(
            lambda: self.view.setAspectLocked(self.tools.checkbox_aspectratio.isChecked()))
        self.tools.checkbox_axes.stateChanged.connect(self.hide_axes)

        # Setting up the crosshairs
        for idx, color in enumerate(['r', 'g']):
            crosshair = Crosshair(color)
            self.getView().addItem(crosshair)
            crosshair.Released.connect(self.crosshair_moved)
            setattr(self, 'CrossHair%d' % (idx + 1), crosshair)
        self.label_crosshairpos = QtWidgets.QLabel()
        self.ui.gridLayout.addWidget(self.label_crosshairpos, 2, 0, 1, 3)
        self.label_crosshairpos.hide()
        self.crosshair_moved()

    def show_tools(self):
        boolean = self.tools.checkbox_tools.isChecked()
        if boolean:
            self.getHistogramWidget().show()
            self.ui.roiBtn.show()
            self.ui.menuBtn.show()
        else:
            self.getHistogramWidget().hide()
            self.ui.roiBtn.hide()
            self.ui.menuBtn.hide()

    def roiClicked(self):
        """Ensures that the new widget in the splitter is displayed"""
        super(ExtendedImageView, self).roiClicked()
        if self.hasTimeAxis() and not self.ui.roiBtn.isChecked():
            self.ui.splitter.setSizes([self.height()-70, 35, 35])

    def buildMenu(self):
        """Adds an action to the existing pyqtgraph.ImageView menu to toggle the visibility of the new GUI"""
        super(ExtendedImageView, self).buildMenu()
        # Percentiles
        self.levelAction = QtWidgets.QAction("Autolevel", self.menu)
        self.levelAction.setCheckable(True)
        self.levelAction.toggled.connect(lambda boolean: self.levelGroup.setVisible(boolean))
        self.menu.addAction(self.levelAction)
        # Crosshair label
        self.labelAction = QtWidgets.QAction("Crosshair label", self.menu)
        self.labelAction.setCheckable(True)
        self.labelAction.toggled.connect(lambda boolean: self.label_crosshairpos.setVisible(boolean))
        self.menu.addAction(self.labelAction)

    # Scaled axis functions
    def get_axes(self):
        """Returns the AxisItems"""
        axes_dict = self.getView().axes
        names = ["bottom", "left", "top", "right"]  # Ensures its always in the same order
        axs = [axes_dict[name]['item'] for name in names]
        return axs

    def hide_axes(self):
        boolean = self.tools.checkbox_axes.isChecked()
        if boolean:
            for ax in self.get_axes():
                ax.hide()
        else:
            for ax in self.get_axes():
                ax.show()

    # Percentile functions
    def getProcessedImage(self):
        """Checks if we want to autolevel for each image and does it"""
        image = super(ExtendedImageView, self).getProcessedImage()
        if self.levelGroup.checkbox_singleimagelevel.isChecked() and self.hasTimeAxis():
            cur_image = image[self.currentIndex]
            self.levelMin, self.levelMax = self._percentile(cur_image, self.level_percentiles)
            self.autoLevels()  # sets the histogram setLevels(self.levelMin, self.levelMax)
        return image

    def set_level_percentiles(self):
        """
        Reads the GUI lineEdits and sets the level percentiles. If not normalising each image, it also finds the levels
        and sets them
        :return:
        """
        min_level = float(self.levelGroup.lineEdit_minLevel.text())
        max_level = float(self.levelGroup.lineEdit_maxLevel.text())

        self.level_percentiles = [min_level, max_level]
        if not self.levelGroup.checkbox_singleimagelevel.isChecked():
            image = self.getProcessedImage()
            self.levelMin, self.levelMax = self._percentile(image, self.level_percentiles)
            self.autoLevels()
        self.updateImage()

    def reset(self):
        self.levelGroup.lineEdit_minLevel.setText('0')
        self.levelGroup.lineEdit_maxLevel.setText('100')
        self.set_level_percentiles()

    def _percentile(self, image, percentiles):
        """Simple percentiles for getting the levels of an image

        For some reason the following is more than 20x faster than the direct numpy implementation:
        >>> levelmin = np.percentile(image, percentiles[0])
        >>> levelmax = np.percentile(image, percentiles[1])
        And it's also faster if you change the self.quickMinMax with two calls to np.min and np.max

        :param image: array
        :param percentiles: two-iterable of the percentiles
        :return:
        """
        minval, maxval = self.quickMinMax(image)
        rng = maxval - minval
        levelmin = minval + rng * percentiles[0] / 100.
        levelmax = minval + rng * percentiles[1] / 100.

        return levelmin, levelmax

    # Crosshairs
    def pos_to_unit(self, positions):
        """
        Given an iterable of positions (bottom, left, top, right) returns the scaled values on those axes

        :param positions: 2- or 4-tuple of floats.
        :return:
        """
        axs = self.get_axes()
        units = ()
        # If only 2-tuple given, it corresponds to (bottom, left) axes
        if len(positions) == 2:
            axs = axs[:2]
        for ax, pos in zip(axs, positions):
            if hasattr(ax, 'pos_to_unit'):
                units += (ax.pos_to_unit(pos), )
            else:
                units += (pos, )

        return units

    def crosshair_moved(self):
        """
        Displays the current position of the two cross-hairs, as well as the distance between them, in pixels and in
        units (when given)
        :return:
        """
        try:
            # First gets the crosshair positions, and finds the distance between them
            positions = ()
            for idx in [1, 2]:
                xhair = getattr(self, "CrossHair%d" % idx)
                pos = tuple(xhair.referenced_pos())
                positions += pos
            diff = np.linalg.norm(np.array(positions[:2]) - np.array(positions[2:]))
            positions += (diff, )

            display_string = "Pixels: <span style='color: red'>[%i,%i] </span> " \
                             "<span style='color: green'> [%i,%i] </span> " \
                             u"\u0394px=%g" % positions

            # If any units are given, get the positions and scale them using pos_to_unit
            if any([self.axis_units[x] is not None for x in ['bottom', 'left']]):
                scaled_positions = ()
                for idx in [1, 2]:
                    xhair = getattr(self, "CrossHair%d" % idx)
                    pos = tuple(xhair.referenced_pos())
                    scaled_positions += self.pos_to_unit(pos)
                units = ()
                for ax in ['bottom', 'left']:
                    if self.axis_units[ax] is None:
                        units += ('px', )
                    else:
                        units += (self.axis_units[ax],)
                display_string += "\t(%s, %s):" \
                                  "<span style='color: red'> (%g, %g)</span> " \
                                  "<span style='color: green'> (%g, %g)</span> " % (units + scaled_positions)

                # If the bottom and left axis have the same units, display the distance between the crosshairs
                if self.axis_units['bottom'] == self.axis_units['left']:
                    difft = np.linalg.norm(np.array(scaled_positions[:2]) - np.array(scaled_positions[2:]))
                    unit = self.axis_units['bottom']
                    display_string += u"\u0394%s=%g" % (unit, difft)

            self.label_crosshairpos.setText(display_string)
        except Exception as e:
            print('Failed updating crosshair position: %s' % e)

    def get_roi(self):
        """
        Pixel positions of the edges of the rectangle bound by the crosshairs
        :return: 4-tuple of integers. left, right, top, and bottom edges
        """
        assert hasattr(self, 'CrossHair1')
        assert hasattr(self, 'CrossHair2')

        pos1 = self.CrossHair1.referenced_pos()
        pos2 = self.CrossHair2.referenced_pos()
        if pos1 == pos2:
            return None

        min_x, max_x = [int(x) for x in (min(pos1[0], pos2[0]), max(pos1[0], pos2[0]))]
        min_y, max_y = [int(x) for x in (min(pos1[1], pos2[1]), max(pos1[1], pos2[1]))]

        return min_x, max_x, min_y, max_y


class TomographyDisplay(ExtendedImageView):
    def __init__(self, data, wavelength, *args, **kwargs):
        super(TomographyDisplay, self).__init__(*args, **kwargs)
        self.wavelength = wavelength
        z_axis = spectrometer_calibration(wavelength=wavelength)
        spectra = np.mean(data, (1, 2))

        self.setImage(data, xvals=z_axis)
        self.spectra = self.ui.roiPlot.plot(z_axis, spectra, pen=pg.mkPen('r'))

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