# -*- coding: utf-8 -*-
from nplab.utils.gui import QtCore, QtWidgets, uic
from nplab.utils.show_gui_mixin import ShowGUIMixin
from nplab.ui.widgets.imageview import ExtendedImageView
import numpy as np
import os
import pyqtgraph as pg


class InteractiveBase(ShowGUIMixin):
    """
    Scrolling through images
    Button to apply analysis function and save the results
    Reshape results into the original image shape
    """
    def __init__(self, images, variables=None):
        super(InteractiveBase, self).__init__()
        self._original_shape = images.shape
        self.images = images

        if variables is None:
            names = ['var %d' % x for x in range(len(self._original_shape) - 2)]
            values = [list(range(x)) for x in self._original_shape[:-2]]
            self.variables = {key: value for key, value in zip(names, values)}
        else:
            self.variables = variables

    def get_qt_ui(self):
        return InteractiveBaseUi(self)

    @staticmethod
    def _none_array(shape):
        if len(shape) > 1:
            return np.array([InteractiveBase._none_array(shape[1:])] * shape[0])
        else:
            return [np.nan] * shape[0]

    def reshape(self, images):
        return np.reshape(images, self._original_shape)


class InteractiveBaseUi(QtWidgets.QMainWindow):
    def __init__(self, interactive_base):
        super(InteractiveBaseUi, self).__init__()
        uipath = os.path.join(os.path.dirname(__file__), 'InteractiveBase.ui')
        uic.loadUi(uipath, self)

        self.object = interactive_base
        self._indxs = [0] * (len(self.object._original_shape) - 2)

        self.ImageDisplay = ExtendedImageView()
        # pos = [0, 0.5, 1]
        # col = [[255, 0, 0], [255, 255, 255], [0, 0, 255]]
        # cmap = pg.ColorMap(pos, col)
        # self.ImageDisplay.setColorMap(cmap)
        self.ImageDisplay.imageItem.axisOrder = 'row-major'
        self.splitter.insertWidget(0, self.ImageDisplay)
        self._plot()

        self.button_next.clicked.connect(self.next_image)
        self.button_previous.clicked.connect(self.prev_image)
        self.button_save.clicked.connect(self.save)
        self.button_analyse.clicked.connect(self.analyse)

        self.indx_spinboxes = []
        for idx, (key, value) in enumerate(zip(self.object.variables.keys(), self.object.variables.values())):
            label = QtWidgets.QLabel(key)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            sb = QtWidgets.QSpinBox()
            sb.setMaximum(999)
            splitter.addWidget(label)
            splitter.addWidget(sb)
            self.layout_spinboxes.addWidget(splitter)
            sb.valueChanged.connect(self.new_image)
            self.indx_spinboxes += [sb]

    def _plot(self):
        try:
            img = self.object.images[tuple(self._indxs)]
            self.ImageDisplay.setImage(img)
            # lvl = np.max(np.abs([np.percentile(img, 5), np.percentile(img, 95)]))
            # self.ImageDisplay.getHistogramWidget().setLevels(-lvl, lvl)
        except Exception as e:
            print(e)
            raise e

    def new_image(self):
        for idx, sb in enumerate(self.indx_spinboxes):
            val = sb.value()
            if val >= self.object._original_shape[idx]:
                val = self.object._original_shape[idx] - 1
            elif val < 0:
                val = 0
            self._indxs[idx] = val
            sb.setValue(val)
        self._plot()

    def next_image(self):
        current_indices = np.array([self._indxs]).transpose()
        flat_index = np.ravel_multi_index(current_indices, self.object._original_shape[:-2])[0]
        if flat_index < np.prod(self.object._original_shape[:-2]) - 1:
            next_indices = np.unravel_index(flat_index + 1, self.object._original_shape[:-2])
            for idx, (old, new) in enumerate(zip(current_indices.flatten(), next_indices)):
                if old != new:
                    self.indx_spinboxes[idx].setValue(new)

    def prev_image(self):
        current_indices = np.array([self._indxs]).transpose()
        flat_index = np.ravel_multi_index(current_indices, self.object._original_shape[:-2])[0]
        if flat_index > 0:
            next_indices = np.unravel_index(flat_index - 1, self.object._original_shape[:-2])
            for idx, (old, new) in enumerate(zip(current_indices.flatten(), next_indices)):
                if old != new:
                    self.indx_spinboxes[idx].setValue(new)

    def save(self):
        raise NotImplementedError

    def analyse(self):
        raise NotImplementedError


class InteractiveCrosshairs(InteractiveBase):
    def __init__(self, images, variables=None):
        super(InteractiveCrosshairs, self).__init__(images, variables)

    def get_qt_ui(self):
        return InteractiveCrosshairsUI(self)


class InteractiveCrosshairsUI(InteractiveBaseUi):
    def __init__(self, *args, **kwargs):
        super(InteractiveCrosshairsUI, self).__init__(*args, **kwargs)
        self.object.crosshair_positions = self.object._none_array(self.object._original_shape[:-2] + (2, 2))

    def analyse(self):
        positions = []
        for idx in [1, 2]:
            xhair = getattr(self.ImageDisplay, "CrossHair%d" % idx)
            positions += [xhair.referenced_pos()]
        return np.array(positions)

    def save(self):
        self.object.crosshair_positions[tuple(self._indxs)] = self.analyse()


class InteractiveLines(InteractiveBase):
    def __init__(self, images, variables=None, n_lines=1):
        super(InteractiveLines, self).__init__(images, variables)
        self.n_lines = n_lines

    def get_qt_ui(self):
        return InteractiveLinesUI(self)


class InteractiveLinesUI(InteractiveBaseUi):
    def __init__(self, *args, **kwargs):
        super(InteractiveLinesUI, self).__init__(*args, **kwargs)
        x_size, y_size = self.object.images.shape[-2], self.object.images.shape[-1]
        self.lines = []
        n_lines = self.object.n_lines
        for idx in range(n_lines):
            line = pg.LineSegmentROI([[0, idx * y_size/n_lines],
                                      [x_size, idx * y_size/n_lines]], pen=pg.mkPen(pg.intColor(idx, n_lines),
                                                                                    width=3, style=QtCore.Qt.DashLine) )
            self.ImageDisplay.addItem(line)
            self.lines += [line]

        self.object.line_positions = self.object._none_array(self.object._original_shape[:-2] + (n_lines, 2, 2))

    def analyse(self):
        positions = []
        n_lines = self.object.n_lines
        for idx in range(n_lines):
            point_list = self.lines[idx].getState()['points']
            positions += [[(p.x(), p.y()) for p in point_list]]
        return np.array(positions)

    def save(self):
        self.object.line_positions[tuple(self._indxs)] = self.analyse()


def test_interactive():
    data = np.random.random((2, 3, 100, 100))
    # interactive = InteractiveCrosshairs(data)
    interactive = InteractiveLines(data, n_lines=3)
    interactive.show_gui()
    print(interactive.line_positions[0,0])


if __name__ == '__main__':
    test_interactive()
