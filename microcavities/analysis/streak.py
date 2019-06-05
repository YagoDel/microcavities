# -*- coding: utf-8 -*-
import numpy as np
import re
import os
from PIL import Image
from scipy.ndimage import median_filter


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
