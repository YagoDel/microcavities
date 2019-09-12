# -*- coding: utf-8 -*-
from nplab.utils.gui import QtCore, QtGui, QtWidgets, get_qt_app, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
import numpy as np
import re
import os
from functools import partial
from PIL import Image
from scipy.ndimage import median_filter
from scipy.ndimage.measurements import center_of_mass
import pyqtgraph as pg
from pyqtgraph.functions import affineSlice
from matplotlib import cm
from microcavities.utils import square


def open_image(path, smooth=False):
    """Opens a data file created by the Hamamatsu streak software

    Parameters
    ----------
    path : string
        full or partial path to the data file
    smooth: bool
        if True, it applies a median filter, removing hot pixels

    Returns
    -------
    numpy array
        2D array with the image data.
    dictionary
        Contains all the image attributes saved by the streak software
    """
    header_dict = {}
    if path.endswith('.dat'):
        data_image = None
        for line in open(path, 'r'):
            data_line = line.rstrip()
            data_list = re.findall(r"(\d+)\t", data_line)
            data_numbers = map(lambda x: float(x), data_list)
            if data_image is None:
                data_image = np.array([data_numbers])
            else:
                data_image = np.append(data_image, [data_numbers], axis=0)
    elif path.endswith('.tif'):
        tif_image = Image.open(path)
        raw_header = tif_image.tag[270]
        sections = re.findall(r'\[.+?\].+?(?=\r|\[|$)', raw_header[0])
        for section in sections:
            group_name = re.findall(r'(?<=\[).+(?=\])', section)[0]
            attrs = re.findall(r'(?:[^,]+=".*?"(?=,|$))|(?:[^,]+=.+?(?=,|$))',
                               section)
            header_dict[group_name] = {}
            for attr in attrs:
                name = str(attr).split('=')[0]
                value = str(attr).split('=')[1]
                header_dict[group_name][name] = value
        data_image = np.array(tif_image)
    else:
        raise ValueError('Image type not recognised')

    if smooth:
        data_image = median_filter(data_image, 5)
    return data_image, header_dict


def open_images(directory, smooth=False):
    all_files = [file_name for file_name in os.listdir(directory) if
                 file_name.endswith('.tif')]
    # Sort according to whatever number is in the filename
    all_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    images = ()
    for fil in all_files:
        if os.path.isfile(directory + '/' + fil):
            image, header = open_image(directory + '/' + fil, smooth)
            images += (image, )
    images = np.asarray(images)
    return images


# Fitting program for k-filtered, spatial images on the streak
class FittingLinear(ShowGUIMixin):
    def __init__(self, images, masks=None):
        super(FittingLinear, self).__init__()
        self._original_shape = images.shape
        self.images = self.flatten(images)
        self.lines = self.find_centers(self.images)
        if masks is None:
            self.masks = np.ones(self.lines.shape, dtype=bool)
        else:
            self.masks = masks

    @staticmethod
    def flatten(images):
        return np.reshape(images,
                          (np.prod(images.shape[:-3]), ) + images.shape[-3:])

    def reshape(self, images):
        return np.reshape(images, self._original_shape)

    def get_qt_ui(self):
        return FittingLinearUi(self)

    @staticmethod
    def _none_array(shape):
        if len(shape) > 1:
            return [FittingLinear._none_array(shape[1:])] * shape[0]
        else:
            return [np.nan] * shape[0]

    @staticmethod
    def find_centers(images):
        lines = np.apply_along_axis(center_of_mass, -1, images)[..., 0]
        lines[lines > images.shape[-1]] = np.nan
        lines[lines < 0] = np.nan
        return lines

    def make_fits_array(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )
        # [None] * self.images.shape[0]
        self.fits = np.array(self._none_array(self.lines.shape[:-1] + shape))
        # [None] * self.images.shape[0]
        self.points = np.array(self._none_array(self.lines.shape[:-1] + shape))


class FittingLinearUi(QtWidgets.QMainWindow):
    def __init__(self, fl):
        super(FittingLinearUi, self).__init__()
        uipath = os.path.join(os.path.dirname(__file__), 'LineFitting.ui')
        uic.loadUi(uipath, self)

        self.object = fl
        self.object.make_fits_array(2)
        self.lines = fl.lines
        self._indxs = [0] * (len(self.lines.shape) - 2)

        self.ImageDisplay = pg.ImageView()
        pos = [0, 0.5, 1]
        col = [[255, 0, 0], [255, 255, 255], [0, 0, 255]]
        cmap = pg.ColorMap(pos, col)
        self.ImageDisplay.setColorMap(cmap)
        self.ImageDisplay.imageItem.axisOrder = 'row-major'
        # self.ImageDisplay.getHistogramWidget().hide()
        # self.ImageDisplay.ui.roiBtn.hide()
        # self.ImageDisplay.ui.menuBtn.hide()
        self.splitter.insertWidget(0, self.ImageDisplay)

        self.roi = pg.LinearRegionItem([10, 100], pg.LinearRegionItem.Vertical)
        self.graphics_linear.addItem(self.roi)
        self.img_roi = pg.LinearRegionItem([10, 100],
                                           pg.LinearRegionItem.Horizontal)
        self.ImageDisplay.addItem(self.img_roi)
        self.roi.sigRegionChanged.connect(lambda x: self.update_plot_roi('line'))
        self.img_roi.sigRegionChanged.connect(lambda x: self.update_plot_roi('img'))
        self.roi.sigRegionChangeFinished.connect(self.fit)

        self.data_lines = []
        self.fit_lines = []
        colors = 255*cm.jet(np.linspace(0, 1, self.lines.shape[-2]))
        for color in colors:
            self.data_lines += [self.graphics_linear.plot(pen=pg.mkPen(color=color[:3], width=1))]
            self.fit_lines += [self.graphics_linear.plot(pen=pg.mkPen(
                color='w', width=2, style=QtCore.Qt.DashDotDotLine))]
        self._plot()

        self.button_next.clicked.connect(self.next_image)
        self.button_save.clicked.connect(self.save)
        self.button_proceed.clicked.connect(self.fast_button)
        self.button_fitlinear.clicked.connect(self.fit)

        self.indx_spinboxes = []
        for idx in range(len(self._indxs)):
            sb = QtWidgets.QSpinBox()
            sb.setMaximum(999)
            self.img_explorer.layout().addWidget(sb)
            sb.valueChanged.connect(self.new_image)
            self.indx_spinboxes += [sb]

    def update_plot_roi(self, axis):
        if axis == 'line':
            region = self.roi.getRegion()
            # new_region = np.exp(region)
            self.img_roi.setRegion(region)
        elif axis == 'img':
            region = self.img_roi.getRegion()
            # new_region = np.log(region)
            self.roi.setRegion(region)
            # self.image_roi.setRegion(region)

    def _plot(self):
        # print 'Plotting'
        try:
            lines = self.lines[tuple(self._indxs)]
            for idx, ydata in enumerate(lines):
                xdata = np.arange(len(ydata))
                self.data_lines[idx].setData(x=xdata, y=ydata)
            # data = self.get_data(self.checkbox_masked.isChecked())
            # self.data_line.setData(x=data[0], y=data[1])
            img = np.mean(self.object.images[tuple(self._indxs)], 0)
            self.ImageDisplay.setImage(img)
            lvl = np.max(np.abs([np.percentile(img, 5), np.percentile(img, 95)]))
            self.ImageDisplay.getHistogramWidget().setLevels(-lvl, lvl)
        except Exception as e:
            print e
            raise e

    def new_image(self):
        # print 'New image'
        self._indxs = []
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val > self.lines.shape[idx] - 1:
                self._indxs += [self.lines.shape[idx] - 1]
                sb.setValue(self.lines.shape[idx] - 1)
            elif val < 0:
                self._indxs += [0]
                sb.setValue(0)
            else:
                self._indxs += [val]
        self._plot()

    def next_image(self, breaker=0):
        # print "Next image"
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val < self.lines.shape[idx] - 1:
                sb.setValue(val + 1)
                break
            else:
                sb.setValue(0)
        if hasattr(self.object, "fits"):
            if not np.isnan(self.object.fits[tuple(self._indxs) + (0, 0)]):
                if breaker < self.object.fits.shape[0]:
                    self.next_image(breaker+1)
                else:
                    print "You've saved everything"

    def prev_image(self):
        # print "Previous image"
        if self.image_indx > 0:
            self.image_indx -= 1
            self._plot()

    def save(self):
        # print 'Saving'
        try:
            # print self.fit_results.shape
            # print self.object.fits.shape
            # print self.object.fits[tuple(self._indxs)].shape
            # print type(self.fl.fits)
            # print self.image_indxs
            self.object.fits[tuple(self._indxs)] = self.fit_results

            # for idx, roi in enumerate(self.rois):
            #     points = roi.listPoints()
            #     slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
            #     self.fl.fits[tuple(self.image_indxs) + (idx, )] = slope
            #     self.fl.points[tuple(self.image_indxs) + (idx, )] = points
            #
            #     # slopes += [slope]
            # # print slopes
            # # print self.image_indxs
            # # print self.fl.fits.shape
            # # self.fl.fits[tuple(self.image_indxs)] = slopes
            # print 'Saved: ', self.image_indxs
        except Exception as e:
            print 'Failed saving: ', e

    @staticmethod
    def _select_array(array, indxs):
        if len(indxs) > 1:
            return FittingLinearUi._select_array(array[indxs[0]], indxs[1:])
        else:
            return np.copy(array[indxs[0]])

    def get_data(self, masked=True, roi=False):
        # print 'Getting data'
        ydata = self._select_array(self.lines, self._indxs)
        xdata = np.arange(ydata.shape[0])
        mask = self._select_array(self.object.masks, self._indxs)
        if roi:
            roi = tuple(map(int, self.roi.getRegion()))
            lims = [max(0, roi[0]), min(roi[1], self.lines.shape[1])]
            ydata = ydata[lims[0]:lims[1]]
            xdata = xdata[lims[0]:lims[1]]
            mask = mask[lims[0]:lims[1]]
        if masked:
            ydata = ydata[mask]
            xdata = xdata[mask]

        return xdata, ydata

    def fast_button(self):
        # print 'Fast button'
        self.save()
        try:
            self.next_image()
        except Exception as e:
            print e
            raise e
        self.fit()

    def fit(self):
        # print 'Fitting'
        try:
            lines = self.lines[tuple(self._indxs)]
            roi = tuple(map(int, self.roi.getRegion()))
            # print 'ROI: ', roi
            lims = [max(0, roi[0]), min(roi[1], self.lines.shape[-1])]
            # print 'lims: ', lims
            lines = lines[:, lims[0]:lims[1]]

            xdata = np.arange(lines.shape[-1])
            xdata += roi[0]
            # xdata, ydata = self.get_data(roi=True)
            newxdata = np.linspace(0.9*xdata[0], 1.1*xdata[-1], 2)
            results = []
            for idx, ydata in enumerate(lines):
                result = np.polyfit(xdata, ydata, 1)
                results += [result]
                newfunc = np.poly1d(result)
                newydata = newfunc(newxdata)
                self.fit_lines[idx].setData(x=newxdata, y=newydata)
                self.fit_lines[idx].setZValue(100)
                # self.label_speed.setText(str(results[0]))
            results = np.array(results)
            self.fit_results = results
            # newfunc = np.poly1d(results)
            # newydata = newfunc(newxdata)
            # self.fit_line.setData(x=newxdata, y=newydata)
            # self.fit_line.setZValue(100)
            self.label_speed.setText(str(results[:, 0]))
        except Exception as e:
            pass
            # print e
            # raise e


# Fitting program for linear wavefronts (thresholding)
class FittingWavefronts(object, ShowGUIMixin):
    def __init__(self, images, backgrounds=None, image_graphics_settings=None):
        super(FittingWavefronts, self).__init__()
        self.images = images
        if backgrounds is None:
            self.backgrounds = self.make_backgrounds()
        else:
            self.backgrounds = backgrounds
        self.img_grph_sttngs = image_graphics_settings

    def get_qt_ui(self):
        return FittingWavefrontsUi(self, self.img_grph_sttngs)

    @staticmethod
    def _none_array(shape):
        if len(shape) > 1:
            return [FittingWavefronts._none_array(shape[1:])] * shape[0]
        else:
            return [np.nan] * shape[0]

    def make_backgrounds(self):
        bkgs = np.copy(self.images[..., :5, :])
        bkgs = np.mean(bkgs, -2)
        bkgs = np.expand_dims(bkgs, -2)

        return np.repeat(bkgs, self.images.shape[-2], -2)

    def make_fits_array(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )
        return np.array(self._none_array(self.images.shape[:-2] + shape))


class FittingWavefrontsUi(QtWidgets.QMainWindow):
    def __init__(self, fitting_instance, image_graphics_settings=None):
        super(FittingWavefrontsUi, self).__init__()
        uipath = os.path.join(os.path.dirname(__file__), 'WavefrontFitting.ui')
        uic.loadUi(uipath, self)

        # Setting up empty class variables
        self.lineplot_widgets = ()
        self._savebuttons = None
        self.plots_lin = None
        self.fit_lines = None
        self.rois_lin = None
        self.iso_lines = ()
        self.iso_levels = ()
        self.xdata = None
        self.ydatas = None
        self.fit_results = None
        self.current_image = None

        if image_graphics_settings is None:
            self.image_graphics_settings = dict(colormap=pg.ColorMap([0, 0.5, 1], [[1.0, 0.0, 0.0, 1.],
                                                                                   [1.0, 1.0, 1.0, 1.],
                                                                                   [0.0, 0.0, 1.0, 1.]]),
                                                levels=[-0.5, 0.5]
                                                )
        else:
            self.image_graphics_settings = image_graphics_settings

        self.fitting_instance = fitting_instance
        self.images = fitting_instance.images
        self.image_indxs = [0] * (len(self.images.shape) - 2)
        self.roi = pg.RectROI([0, 0], [30, 30], pen='r')
        self.roi.sigRegionChanged.connect(self.update_roi)
        self.roi.sigRegionChangeFinished.connect(self.update_line_plots)
        self.graphics_image.addItem(self.roi)
        if 'colormap' in self.image_graphics_settings:
            self.graphics_image.setColorMap(self.image_graphics_settings['colormap'])

        self.graphics_imagethresholded = pg.ImageView()
        self.verticallayout_left.addWidget(self.graphics_imagethresholded)
        if 'colormap' in self.image_graphics_settings:
            self.graphics_imagethresholded.setColorMap(self.image_graphics_settings['colormap'])

        self.button_next.clicked.connect(self.next_image)
        self.button_previous.clicked.connect(self.previous_image)
        self.button_save.clicked.connect(partial(self.save, None))
        self.button_proceed.clicked.connect(self.proceed)
        self.button_fitlinear.clicked.connect(self.fit)
        self.button_plot.clicked.connect(self.update_line_plots)
        self.spinbox_reps.valueChanged.connect(self.setup_line_plots)
        self.spinBox_noIsolines.valueChanged.connect(self.setup_isolines)
        self.checkbox_randomize.stateChanged.connect(self.randomize)
        self.checkbox_analysistype.stateChanged.connect(self.analysis_type)

        self.indx_spinboxes = []
        for idx in range(len(self.image_indxs)):
            sb = QtWidgets.QSpinBox()
            self.img_explorer.layout().addWidget(sb)
            sb.valueChanged.connect(self.update_image_indices)
            self.indx_spinboxes += [sb]

        self.setup_isolines()
        self.update_image_indices()

    @staticmethod
    def clear_layout(layout):
        """Deletes all widgets in a layout"""
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

    def randomize(self):
        """Toggles between having randomized thresholds or not"""
        state = self.checkbox_randomize.isChecked()
        self.lineEdit_thresholdvar.setEnabled(state)
        self.spinbox_reps.setEnabled(state)
        self.setup_line_plots()
        self.setup_save_array()

    def analysis_type(self):
        """Toggles between plotting lineplots and having manual fits with lines on images"""
        self.graphics_imagethresholded.deleteLater()

        state = self.checkbox_analysistype.isChecked()
        self.checkbox_randomize.setEnabled(state)

        self.graphics_imagethresholded = pg.ImageView()
        if 'colormap' in self.image_graphics_settings:
            self.graphics_imagethresholded.setColorMap(self.image_graphics_settings['colormap'])
        if state:
            self.verticallayout_left.addWidget(self.graphics_imagethresholded)
        else:
            self.clear_layout(self.layout_plots)
            self.layout_plots.addWidget(self.graphics_imagethresholded)

        state = self.checkbox_randomize.isChecked()
        self.lineEdit_thresholdvar.setEnabled(state)
        self.spinbox_reps.setEnabled(state)
        self.setup_isolines()

    def setup_isolines(self):
        """Creates a number of IsocurveItems (given by the spinbox). Places
        them in the ROI image and links them to an InfiniteLine in the
        HistogramWidget of the image"""
        view = self.graphics_imagethresholded.getView()
        hist = self.graphics_imagethresholded.getHistogramWidget()
        for line, level in zip(self.iso_lines, self.iso_levels):
            view.removeItem(level)
            hist.vb.removeItem(line)

        n_lines = self.spinBox_noIsolines.value()
        self.iso_lines = ()
        self.iso_levels = ()
        for idx in range(n_lines):
            pen = pg.mkPen(pg.intColor(idx, n_lines))
            iso_level = pg.IsocurveItem(level=0.0,
                                        pen=pen)
            view.addItem(iso_level)
            iso_level.setZValue(1000)
            iso_line = pg.InfiniteLine(angle=0, movable=True, pen=pen)
            hist.vb.addItem(iso_line)
            iso_line.setValue(0.0)
            iso_line.setZValue(1000)  # bring iso line above contrast controls
            iso_line.sigDragged.connect(partial(self.update_isocurve, idx))
            iso_line.sigPositionChangeFinished.connect(self.update_line_plots)
            self.iso_lines += (iso_line, )
            self.iso_levels += (iso_level, )
        self.update_isocurve()
        self.setup_line_widgets()

    def setup_line_widgets(self):
        """
        For lineplot analysis, creates a number of PlotWidgets (given by spinbox). Places them in as close to a square
        shape as possible inside the self.widget_replaceable.
        For manually fitting lines, creates a number of LineSegmentROI and places them on the graphics_imagethresholded
        Creates saving buttons, places them in the same arrangement, and links them to the save function with the
        appropriate index
        """
        n_lines = self.spinBox_noIsolines.value()
        state = self.checkbox_analysistype.isChecked()
        roishape = self.get_roi().shape

        for plot in self.lineplot_widgets:
            plot.deleteLater()
        self.clear_layout(self.layout_savebuttons)
        if state:
            self.clear_layout(self.layout_plots)
        else:
            for widgt in self.lineplot_widgets:
                self.graphics_imagethresholded.removeItem(widgt)

        self.lineplot_widgets = ()
        self._savebuttons = ()
        idx = 0
        shape = square(n_lines)
        for i in range(shape[0]):
            for j in range(shape[1]):
                color = pg.intColor(idx, n_lines)
                pen = pg.mkPen(color, width=3, style=QtCore.Qt.DashLine)
                if state:
                    widgt = pg.PlotWidget(pen=pen)
                    self.lineplot_widgets += (widgt, )
                    self.layout_plots.addWidget(widgt, i, j)
                else:
                    widgt = pg.LineSegmentROI([[0, -5 * idx], [roishape[0], -5 * idx]], pen=pen)
                    self.lineplot_widgets += (widgt, )
                    self.graphics_imagethresholded.addItem(widgt)

                widgt = QtWidgets.QCheckBox('Save %d' % idx)
                widgt.setStyleSheet("background-color: rgba%s" % (color.getRgb(), ))
                self.layout_savebuttons.addWidget(widgt, i, j)
                self._savebuttons += (widgt, )
                idx += 1
        if state:
            self.setup_line_plots()
        self.setup_save_array()

    def setup_line_plots(self):
        """Makes one (or more, if random thresholds are active) line plots
        inside each ot the PlotWidgets. Also makes a single fit line for each
        widget and a single linear ROI.
        Should only get called when performing line fits
        """
        if self.checkbox_randomize.isChecked():
            n_thresholds = self.spinbox_reps.value()
        else:
            n_thresholds = 1

        self.plots_lin = ()
        self.fit_lines = ()
        self.rois_lin = ()
        for idx, widgt in enumerate(self.lineplot_widgets):
            widgt.clear()
            pen = pg.mkPen(pg.intColor(idx, len(self.lineplot_widgets)))
            plots = ()
            for idx2 in range(n_thresholds):
                plots += (widgt.plot(pen=pen), )
            self.plots_lin += (plots, )

            pen = pg.mkPen(color='w', width=3, style=QtCore.Qt.DashDotDotLine)
            self.fit_lines += (widgt.plot(pen=pen), )

            if self.xdata is None:
                roi_image = self.get_roi()
                lims = [0, roi_image.shape[1]]
            else:
                lims = [np.min(self.xdata), np.max(self.xdata)]
            linear_roi = pg.LinearRegionItem(lims)
            widgt.addItem(linear_roi)
            linear_roi.sigRegionChangeFinished.connect(self.fit)
            self.rois_lin += (linear_roi, )
        self.plots_lin = np.array(self.plots_lin)

    def setup_save_array(self):
        """Sets up an array of the appropriate size to store the results of the analysis. Note that this resets the
        arrays, so you should not do this half-way through analysing stuff"""
        n_lines = self.spinBox_noIsolines.value()
        n_thresholds = self.spinbox_reps.value()
        self.fitting_instance.thresholds = self.fitting_instance.make_fits_array((n_lines, ))

        if self.checkbox_analysistype.isChecked() and self.checkbox_randomize.isChecked():
            self.fitting_instance.fits = self.fitting_instance.make_fits_array((n_lines, n_thresholds, 2))
        elif not self.checkbox_analysistype.isChecked():
            self.fitting_instance.fits = self.fitting_instance.make_fits_array((n_lines, 2))
        else:
            self.fitting_instance.fits = self.fitting_instance.make_fits_array((n_lines, 1, 2))

    def update_isocurve(self, index=None):
        """Links the threshold value in the histogram widgets to the
        IsocurveItems"""
        if index is None:
            for level, line in zip(self.iso_levels, self.iso_lines):
                level.setLevel(line.value())
        else:
            self.iso_levels[index].setLevel(self.iso_lines[index].value())
        self.update_image()  # ensures the data of the iso_levels is updated

    def update_image(self):
        """Selects an image according to self.image_indxs and displays it"""
        img = self._select_image(self.images, self.image_indxs)
        if self.checkbox_bkg.isChecked():
            img -= self._select_image(self.fitting_instance.backgrounds,
                                      self.image_indxs)
        if self.checkbox_transpose.isChecked():
            img = img.transpose()
        self.current_image = img
        self.graphics_image.setImage(img, False, False)
        if 'levels' in self.image_graphics_settings:
            self.graphics_image.setLevels(*self.image_graphics_settings['levels'])
            self.graphics_image.getHistogramWidget().setHistogramRange(*self.image_graphics_settings['levels'])
        self.update_roi()

    def update_roi(self):
        """Updates the GraphicsItem displaying the image ROI as wells as the
        IsocurveItem's data"""
        roi = self.get_roi()
        widgt = self.graphics_imagethresholded
        if 'levels' in self.image_graphics_settings:
            widgt.setImage(roi, levels=self.image_graphics_settings['levels'])
            widgt.getHistogramWidget().setHistogramRange(*self.image_graphics_settings['levels'])
        else:
            widgt.setImage(roi)
        for iso_level in self.iso_levels:
            iso_level.setData(roi)

    def update_line_plots(self):
        """Thresholds the ROI image and creates the ydata for the line plots"""
        state = self.checkbox_analysistype.isChecked()
        if not state:
            return
        try:
            n_lines = self.spinBox_noIsolines.value()
            if self.checkbox_randomize.isChecked():
                n_thresholds = self.spinbox_reps.value()
                thrsh_var = float(self.lineEdit_thresholdvar.text())
            else:
                n_thresholds = 1
                thrsh_var = 0

            self.ydatas = ()
            for idx in range(n_lines):
                iso_line = self.iso_lines[idx]
                plots_lin = self.plots_lin[idx]
                threshold = iso_line.value()

                variation = np.random.uniform(-thrsh_var, thrsh_var,
                                              n_thresholds)
                ydatas = ()

                for idx2, plot_lin in zip(range(n_thresholds), plots_lin):
                    roi_image = self.get_roi()

                    minval = np.min(roi_image) - 1
                    roi_image[roi_image < (threshold+variation[idx2])] = minval
                    roi_image[roi_image != minval] = 1
                    roi_image[roi_image == minval] = 0
                    ydata = np.sum(roi_image, 0)
                    if self.xdata is None or len(self.xdata) != len(ydata):
                        self.xdata = range(1, len(ydata)+1)

                    plot_lin.setData(x=self.xdata, y=ydata)

                    ydatas += (ydata, )
                self.ydatas += (ydatas, )
            self.ydatas = np.array(self.ydatas)

        except Exception as e:
            raise e

    def fit(self):
        """Iterates over all of the line plots and fits lines inside the
        regions determined by the ROIs"""
        state = self.checkbox_analysistype.isChecked()
        if not state:
            return
        self.fit_results = ()
        for roi_lin, ydatas, fit_line in zip(self.rois_lin,
                                             self.ydatas,
                                             self.fit_lines):
            roi = tuple(map(int, roi_lin.getRegion()))
            # Linear fit
            xdata = self.xdata[roi[0]:roi[1]]
            newxdata = np.linspace(0.9*xdata[0], 1.1*xdata[-1], 2)
            results = ()
            for ydata in ydatas[:, roi[0]:roi[1]]:
                results += (np.polyfit(xdata, ydata, 1), )
            results = np.array(results)
            self.fit_results += (results, )
            newfunc = np.poly1d(np.mean(results, 0))
            newydata = newfunc(newxdata)
            fit_line.setData(x=newxdata, y=newydata)
            fit_line.setZValue(100)
        self.fit_results = np.array(self.fit_results)
        # Averaging over thresholds, display all the speeds
        self.label_speed.setText(str(np.mean(self.fit_results, 1)[:, 0]))
        # print 'Results shape: %s' % (self.fit_results.shape, )

    def update_image_indices(self):
        """Called every time the indx_spinboxes are triggered"""
        self.image_indxs = []
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val > self.images.shape[idx] - 1:
                self.image_indxs += [self.images.shape[idx] - 1]
                sb.setValue(self.images.shape[idx] - 1)
            elif val < 0:
                self.image_indxs += [0]
                sb.setValue(0)
            else:
                self.image_indxs += [val]
        self.update_image()

    def next_image(self):
        """Finds the first spinbox to which we can add +1. If a spinbox has a
        value already corresponding to the size of the array, it sets it back
        to zero
        """
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val < self.images.shape[idx] - 1:
                sb.setValue(val + 1)
                break
            else:
                sb.setValue(0)

    def previous_image(self):
        """Opposite of next_image"""
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val > 0:
                sb.setValue(val - 1)
                break
            else:
                sb.setValue(self.images.shape[idx] - 1)

    def save(self):
        """Saves the current fitting results at the correct indices inside the
        FittingWavefronts instance
        """
        try:
            indxs = tuple(self.image_indxs)
            # print 'Saving: %s, %s' % (str(indxs), line_index)
            thresholds = np.array([iso_line.value() for iso_line in self.iso_lines])
            state = self.checkbox_analysistype.isChecked()
            if state:
                data_to_save = np.copy(self.fit_results)
            else:
                data_to_save = []
                for line in self.lineplot_widgets:
                    handles = line.getHandles()
                    limits = np.array([(h.pos().x(), h.pos().y()) for h in handles], dtype=np.float)
                    diff = np.diff(limits, axis=0)[0]
                    slope = diff[1] / diff[0]
                    offset = limits[0, 1] - slope * limits[0, 0]
                    data_to_save += [(slope, offset)]
                data_to_save = np.array(data_to_save)

            for idx, widgt in enumerate(self._savebuttons):
                indices = indxs + (idx,)
                if widgt.isChecked():
                    self.fitting_instance.fits[indices] = data_to_save[idx]
                    self.fitting_instance.thresholds[indices] = thresholds[idx]
                else:
                    self.fitting_instance.fits[indices] = np.nan
                    self.fitting_instance.thresholds[indices] = np.nan
        except Exception as e:
            print 'Failed saving: ', e

    def proceed(self):
        """Convenience function for quickly proceeding through a dataset"""
        self.save()
        self.next_image()
        self.update_line_plots()
        self.fit()

    @staticmethod
    def _select_image(images, indxs):
        """Given a set of indices and a large array, return the sub-array
        determined by the indices"""
        if len(indxs) > 1:
            return FittingWavefrontsUi._select_image(images[indxs[0]],
                                                     indxs[1:])
        else:
            return np.copy(images[indxs[0]])

    def get_roi(self, image=None):
        """Returns the sub-image determined by the GUIs ROI"""
        if image is None:
            image = self.current_image
        roi = self.roi.getAffineSliceParams(image,
                                            self.graphics_image.getImageItem())
        roi_image = affineSlice(image, shape=roi[0], vectors=roi[1],
                                origin=roi[2], axes=(0, 1))
        return roi_image
