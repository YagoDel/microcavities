# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image


def tile_array(image_array, position_array=None, offsets=None, normalize=True):
    """
    TODO:
        - Average the overlap between arrays, instead of replacing one with the other
        - Make a sub-pixel resolution image, so the alignment can be even better
        - If making a linear tiling, popup a GUI to determine offsets
        - Make one that takes a 4D image array, and uses offsets in the two directions to tile
        
    Examples:
        tiled = tile_array(images, position_array=np.array(zip(np.zeros((21,), dtype=np.int), 
                                                                  np.linspace(1000, 4000, 21, dtype=np.int))))
        tiled = tile_array(images)
        tiled = tile_array(images, offsets=(-50, 50))
        
        plt.imshow(tiled)

    

    :param image_array: 3D array (MxNxP). M images of NxP shape. First dimension corresponds to different images. The 
    next two are the dimensions along which the tiling is going to occur. If normalize=False, values should go between 0 
    and 255.
    :param position_array: 2D array (Mx2). First dimension is the same as for image_array. Then two values (x, y) can be 
    given corresponding to the pixel position of the top-left corner of an image. If None, it automatically decides how 
    to stack the images (vertically by default).
    :param offsets: int or 2-tuple of ints. How many pixels offset there should be between one tile and the next. Can be 
    used for simple tiling.
    :param normalize: bool. Whether to normalise each image before tiling.
    :return: 
    """
    if offsets is None and position_array is None:
        # By default stacks all of the images vertically, with no overlap
        vert_positions = np.linspace(0, (image_array.shape[0] - 1) * image_array.shape[1], image_array.shape[0])
        position_array = np.array(list(zip(np.zeros_like(vert_positions), vert_positions)))
    elif position_array is None:
        if hasattr(offsets, '__iter__'):
            # You can give two values for the offsets (x and y). Stacking still occurs vertically, but also provides horizontal offset
            vert_positions = np.linspace(0, (image_array.shape[0] - 1) * (image_array.shape[1] + offsets[1]),
                                         image_array.shape[0], dtype=np.int)
            horz_positions = np.linspace(0, (image_array.shape[0] - 1) * (1 + offsets[0]), image_array.shape[0],
                                         dtype=np.int)
        else:
            # Equivalent to above if the horizontal offset is 0
            vert_positions = range(0, image_array.shape[0] * (image_array.shape[1] + offsets),
                                   image_array.shape[1] + offsets)
            horz_positions = np.zeros_like(vert_positions)
        position_array = np.array(zip(horz_positions, vert_positions))

    # Ensures the position array starts at 0 so it doesn't go outside of the frame
    position_array = np.array(zip(position_array[:, 0] - np.min(position_array[:, 0]),
                                  position_array[:, 1] - np.min(position_array[:, 1])))

    image_size = (image_array.shape[2] + np.max(position_array[:, 0]) - np.min(position_array[:, 0]),
                  image_array.shape[1] + np.max(position_array[:, 1]) - np.min(position_array[:, 1]))
    base_image = Image.new('F', image_size)
    base_image = base_image.convert('RGBA')

    for img, pos in zip(image_array, position_array):
        if normalize:
            img -= np.min(img)
            img /= np.max(img)
            img *= 255.
        pil_image = Image.fromarray(img)
        pil_image = pil_image.convert('RGBA')
        base_image.paste(pil_image, tuple(pos))
    return base_image



