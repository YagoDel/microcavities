# -*- coding: utf-8 -*-
from nplab.utils.gui import QtCore, QtGui, QtWidgets, get_qt_app, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
import numpy as np
import re
import os
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
class FittingLinear(object, ShowGUIMixin):
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
    def __init__(self, images, backgrounds=None, repeat_ax=0):
        super(FittingWavefronts, self).__init__()
        self.images = images
        if backgrounds is None:
            self.backgrounds = self.make_backgrounds()
        else:
            self.backgrounds = backgrounds

    def get_qt_ui(self):
        return FittingWavefrontsUi(self)

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
        self.fits = np.array(self._none_array(self.images.shape[:-2] + shape))
        self.points = np.array(self._none_array(self.images.shape[:-2] + shape))


cmap = pg.ColorMap([0, 0.5, 1], [[1.0, 0.0, 0.0, 1.],
                                 [1.0, 1.0, 1.0, 1.],
                                 [0.0, 0.0, 1.0, 1.]])


class FittingWavefrontsUi(QtWidgets.QMainWindow):
    def __init__(self, fl):
        super(FittingWavefrontsUi, self).__init__()
        uipath = os.path.join(os.path.dirname(__file__), 'SumFitting.ui')
        uic.loadUi(uipath, self)

        self.fl = fl
        self.fl.make_fits_array(3)
        self.images = fl.images
        self.image_indxs = [0] * (len(self.images.shape) - 2)
        self.roi = pg.RectROI([0, 0], [30, 30], pen='r')
        self.roi.sigRegionChanged.connect(self.update_roi)
        self.graphics_image.addItem(self.roi)
        self.graphics_image.setColorMap(cmap)
        self.graphics_imagethresholded.setColorMap(cmap)

        # self.iso_level = pg.IsocurveItem(level=0.0, pen='g')
        # self.graphics_imagethresholded.getView().addItem(self.iso_level)
        # self.iso_level.setZValue(1000)
        # self.iso_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
        # self.graphics_imagethresholded.getHistogramWidget().vb.addItem(self.iso_line)
        # self.iso_line.setValue(0.0)
        # self.iso_line.setZValue(1000)  # bring iso line above contrast controls
        # self.iso_line.sigDragged.connect(self.update_isocurve)

        # self.linear_roi = pg.LinearRegionItem([400, 700])
        # self.graphics_linear.addItem(self.linear_roi)
        # self.log_roi = pg.LinearRegionItem([400, 700])
        # self.graphics_loglog.addItem(self.log_roi)
        # self.linear_roi.sigRegionChanged.connect(lambda x: self.update_plot_roi)

        self.button_next.clicked.connect(self.next_image)
        self.button_save.clicked.connect(self.save)
        self.button_proceed.clicked.connect(self.fast_button)
        self.button_fitlinear.clicked.connect(self.fit)
        self.button_plot.clicked.connect(self.plot_lines)
        self.spinbox_reps.valueChanged.connect(self.create_roi)
        self.spinBox_noIsolines.valueChanged.connect(self.create_isolines)

        self.indx_spinboxes = []
        for idx in range(len(self.image_indxs)):
            sb = QtWidgets.QSpinBox()
            self.img_explorer.layout().addWidget(sb)
            sb.valueChanged.connect(self.new_image)
            self.indx_spinboxes += [sb]
        self.iso_lines = ()
        self.iso_levels = ()
        self.xdata = None
        self.create_isolines()
        self.new_image()
        self.create_roi()

    def create_roi(self):
        # self.graphics_linear.clear()
        # self.graphics_loglog.clear()
        # self.graphics_linear.addItem(self.linear_roi)
        # self.graphics_loglog.addItem(self.log_roi)
        # self.fit_linear = self.graphics_linear.plot(pen=pg.mkPen(
            # color='w', width=3, style=QtCore.Qt.DashDotDotLine))
        # self.fit_log = self.graphics_loglog.plot(pen=pg.mkPen(
            # color='w', width=3, style=QtCore.Qt.DashDotDotLine))

        new_number = self.spinbox_reps.value()
        # self.plots_lin = ()
        # self.plots_log = ()
        # # for idx2 in range(new_number):
        # #     color = pg.intColor(idx2, new_number)
        # #     plots_lin = self.graphics_linear.plot(pen=color)
        # #     plots_log = self.graphics_loglog.plot(pen=color)
        # #     self.plots_lin += (plots_lin, )
        # #     self.plots_log += (plots_log, )
        # self.plots_lin = np.array(self.plots_lin)
        # self.plots_log = np.array(self.plots_log)
        self.fl.make_fits_array((new_number, 2))

    def create_isolines(self):
        n_lines = self.spinBox_noIsolines.value()
        print 'Creating %d lines' % n_lines
        for line, level  in zip(self.iso_lines, self.iso_levels):
            self.graphics_imagethresholded.getView().removeItem(level)
            self.graphics_imagethresholded.getHistogramWidget().vb.removeItem(line)

        self.iso_lines = ()
        self.iso_levels = ()
        for idx in range(n_lines):
            pen = pg.mkPen(pg.intColor(idx, n_lines))
            iso_level = pg.IsocurveItem(level=0.0,
                                        pen=pen)
            self.graphics_imagethresholded.getView().addItem(iso_level)
            iso_level.setZValue(1000)
            iso_line = pg.InfiniteLine(angle=0, movable=True, pen=pen)
            self.graphics_imagethresholded.getHistogramWidget().vb.addItem(iso_line)
            iso_line.setValue(0.0)
            iso_line.setZValue(1000)  # bring iso line above contrast controls
            iso_line.sigDragged.connect(self.update_isocurve)
            self.iso_lines += (iso_line, )
            self.iso_levels += (iso_level, )
        self.update_isocurve()
        self.setup_lineplots()

    def setup_lineplots(self):
        n_lines = self.spinBox_noIsolines.value()
        n_thresholds = self.spinbox_reps.value()

        layout = self.widget_replaceable.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

        self.lineplot_widgets = ()
        self.plots_lin = ()
        shape = square(n_lines)
        idx = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                widgt = pg.PlotWidget()
                self.lineplot_widgets += (widgt, )
                layout.addWidget(widgt, i, j)

                pen = pg.mkPen(pg.intColor(idx, n_lines))
                plots = ()
                for idx2 in range(n_thresholds):
                    plots += (widgt.plot(pen=pen), )
                self.plots_lin += (plots, )
                idx += 1

        # self.plots_lin = ()
        # # self.plots_log = ()
        # for idx in range(n_lines):
        #     plots_lin = ()
        #     for idx2 in range(n_thresholds)
        # #     color = pg.intColor(idx2, new_number)
        # #     plots_lin = self.graphics_linear.plot(pen=color)
        # #     plots_log = self.graphics_loglog.plot(pen=color)
        # #     self.plots_lin += (plots_lin, )
        # #     self.plots_log += (plots_log, )
        # self.plots_lin = np.array(self.plots_lin)
        # self.plots_log = np.array(self.plots_log)

    def update_plot_roi(self, axis):
        region = self.linear_roi.getRegion()
        new_region = np.log(region)
        self.log_roi.setRegion(new_region)

    def update_isocurve(self):
        for level, line in zip(self.iso_levels, self.iso_lines):
            level.setLevel(line.value())
        self._plot()  # ensures the data of the iso_levels is updated

    def update_roi(self):
        roi = self.get_roi()
        self.graphics_imagethresholded.setImage(roi, levels=(-0.5, 0.5))
        self.graphics_imagethresholded.getHistogramWidget().setHistogramRange(-0.5, 0.5)
        for iso_level in self.iso_levels:
            iso_level.setData(roi)

    def _plot(self):
        img = self._select_image(self.images, self.image_indxs)
        if self.checkbox_bkg.isChecked():
            img -= self._select_image(self.fl.backgrounds, self.image_indxs)
        if self.checkbox_transpose.isChecked():
            img = img.transpose()
        self.current_image = img
        self.graphics_image.setImage(img, False, False)
        self.graphics_image.setLevels(-0.5, 0.5)
        self.graphics_image.getHistogramWidget().setHistogramRange(-0.5, 0.5)
        self.update_roi()

    def new_image(self):
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
        self._plot()

    def next_image(self, breaker=0):
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val < self.images.shape[idx] - 1:
                sb.setValue(val + 1)
                break
            else:
                sb.setValue(0)

        # if self.fl.fits[tuple(self.image_indxs) + (0, )] != np.nan:
        #     if breaker < np.prod(self.images.shape[:-2]):
        #         self.next_image(breaker+1)
        #     else:
        #         print "You've saved everything"

    def prev_image(self):
        if self.image_indx > 0:
            self.image_indx -= 1
            self._plot()

    def save(self):
        try:
            self.fl.fits[tuple(self.image_indxs)] = self.fit_results
        except Exception as e:
            print 'Failed saving: ', e

    @staticmethod
    def _select_image(images, indxs):
        if len(indxs) > 1:
            return FittingWavefrontsUi._select_image(images[indxs[0]],
                                                     indxs[1:])
        else:
            return np.copy(images[indxs[0]])

    def fast_button(self):
        self.save()
        self.next_image()
        self.plot_lines()
        self.fit()

    def get_roi(self, image=None):
        if image is None:
            image = self.current_image
        roi = self.roi.getAffineSliceParams(image,
                                            self.graphics_image.getImageItem())
        roi_image = affineSlice(image, shape=roi[0], vectors=roi[1],
                                origin=roi[2], axes=(0, 1))
        return roi_image

    def plot_lines(self):
        try:
            threshold = self.iso_line.value()
            thrsh_var = self.lineEdit_thresholdvar.text()
            if len(thrsh_var) == 0:
                thrsh_var = 0.1
                self.lineEdit_thresholdvar.setText(str(thrsh_var))
            else:
                thrsh_var = float(thrsh_var)

            reps = self.spinbox_reps.value()
            variation = np.random.uniform(-thrsh_var, thrsh_var, reps)
            reverse = self.checkbox_reverse.isChecked()
            self.ydatas = ()
            self.logydatas = ()
            for idx, plot_lin, plot_log in zip(range(reps),
                                               self.plots_lin,
                                               self.plots_log):
                roi_image = self.get_roi()

                minval = np.min(roi_image) - 1
                if reverse:
                    roi_image[roi_image > (threshold+variation[idx])] = minval
                    roi_image[roi_image != minval] = 1
                    roi_image[roi_image == minval] = 0
                else:
                    roi_image[roi_image < (threshold+variation[idx])] = minval
                    roi_image[roi_image != minval] = 1
                    roi_image[roi_image == minval] = 0
                ydata = np.sum(roi_image, 0)
                if self.xdata is None or len(self.xdata) != len(ydata):
                    self.xdata = range(1, len(ydata)+1)
                    bnds = np.array([np.min(self.xdata), np.max(self.xdata)])
                    self.linear_roi.setBounds(bnds)
                    self.linear_roi.setRegion(0.9 * bnds)
                    self.log_roi.setBounds(np.log(bnds))
                    self.log_roi.setRegion(0.9 * np.log(bnds))
                plot_lin.setData(x=self.xdata, y=ydata)
                plot_log.setData(x=np.log(self.xdata), y=np.log(ydata))

                self.ydatas += (ydata, )
                self.logydatas += (np.log(ydata), )
            self.ydatas = np.array(self.ydatas)
            self.logydatas = np.array(self.logydatas)

        except Exception as e:
            raise e

    def fit(self):
        roi = tuple(map(int, self.linear_roi.getRegion()))
        self.fit_results = ()
        # Linear fit
        xdata = self.xdata[roi[0]:roi[1]]
        newxdata = np.linspace(0.9*xdata[0], 1.1*xdata[-1], 2)
        results = ()
        for ydata in self.ydatas[:, roi[0]:roi[1]]:
            results += (np.polyfit(xdata, ydata, 1), )
        results = np.array(results)
        self.fit_results += (results, )
        newfunc = np.poly1d(np.mean(results, 0))
        newydata = newfunc(newxdata)
        self.fit_linear.setData(x=newxdata, y=newydata)
        self.fit_linear.setZValue(100)
        self.label_speed.setText(str(np.mean(results, 0)[0]))

        # Loglog fit
        xdata = np.log(self.xdata[roi[0]:roi[1]])
        newxdata = np.linspace(0.9 * xdata[0], 1.1 * xdata[-1], 2)
        results = ()
        for ydata in self.logydatas[:, roi[0]:roi[1]]:
            results += (np.polyfit(xdata, ydata, 1),)
        results = np.array(results)
        # self.fit_results += (results, )
        newfunc = np.poly1d(np.mean(results, 0))
        newydata = newfunc(newxdata)
        self.fit_log.setData(x=newxdata, y=newydata)
        self.fit_log.setZValue(100)
        self.label_exponent.setText(str(np.mean(results, 0)[0]))

        self.fit_results = np.array(self.fit_results)
