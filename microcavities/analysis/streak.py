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
from matplotlib import cm


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
        return np.reshape(images,(np.prod(images.shape[:-3]), ) + images.shape[-3:])

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
        self.fits = np.array(self._none_array(self.lines.shape[:-1] + shape)) #[None] * self.images.shape[0]
        self.points = np.array(self._none_array(self.lines.shape[:-1] + shape)) #[None] * self.images.shape[0]


class FittingLinearUi(QtWidgets.QMainWindow):
    def __init__(self, fl):
        super(FittingLinearUi, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'LineFitting.ui'), self)

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
        self.img_roi = pg.LinearRegionItem([10, 100], pg.LinearRegionItem.Horizontal)
        self.ImageDisplay.addItem(self.img_roi)
        self.roi.sigRegionChanged.connect(lambda x: self.update_plot_roi('line'))
        self.img_roi.sigRegionChanged.connect(lambda x: self.update_plot_roi('img'))
        self.roi.sigRegionChangeFinished.connect(self.fit)

        self.data_lines = []
        self.fit_lines = []
        colors = 255*cm.jet(np.linspace(0, 1, self.lines.shape[-2]))
        for color in colors:
            self.data_lines += [self.graphics_linear.plot(pen=pg.mkPen(color=color[:3], width=1))]
            self.fit_lines += [self.graphics_linear.plot(pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.DashDotDotLine))]
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