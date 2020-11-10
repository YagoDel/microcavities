# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from microcavities.utils import square


def diverging_imshow(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2)
    else:
        fig = ax.figure

    val = np.max(np.abs([np.max(img), np.min(img)]))
    im = ax.imshow(img, vmin=-val, vmax=val, cmap='RdBu')
    fig.colorbar(im, ax=ax)
    return fig, ax


def imshows(imgs, diverging=False, *args, **kwargs):
    n_images = imgs.shape[0]
    a, b = square(n_images)
    figshape = (b/a) * (imgs.shape[1]/imgs.shape[2])
    fig = plt.figure(figsize=(8, 8*figshape))
    gs = gridspec.GridSpec(b, a)
    axs = []
    for idx in range(n_images):
        img = imgs[idx]
        ax = plt.subplot(gs[idx % b, idx // b])
        if diverging:
            diverging_imshow(img, ax)
        else:
            ax.imshow(img, *args, **kwargs)
        axs += [ax]
    return fig, axs


def colorbar_imshow(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2)
    else:
        fig = ax.figure

    return fig, ax
