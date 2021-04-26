# -*- coding: utf-8 -*-
import lmfit
import numpy as np
from microcavities.utils.plotting import *
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter


def fit_peak(spectra, xaxis=None, find_peaks_kwargs=None, n_peaks=None, axplot=None):
    spectra = gaussian_filter(np.copy(spectra), 2)

    if xaxis is None:
        xaxis = np.arange(len(spectra))
    if find_peaks_kwargs is None:
        find_peaks_kwargs = dict()
    default_kwargs = dict(distance=10, width=1, prominence=10)
    [find_peaks_kwargs.update({key: value}) for key, value in default_kwargs.items() if key not in find_peaks_kwargs]

    peak_indices, peak_properties = find_peaks(spectra, **find_peaks_kwargs)
    if n_peaks is None:
        n_peaks = len(peak_indices)
    if len(peak_indices) > 1:
        # Sort by height and clip
        heights = spectra[peak_indices]
        sorter = np.argsort(heights)[::-1]
        peak_indices = peak_indices[sorter][:n_peaks]
        # Sort by index (so things are always ordered)
        sorter = np.argsort(peak_indices)
        peak_indices = peak_indices[sorter][::-1]

    centers = []
    for idx in range(n_peaks):
        try:
            peak = peak_indices[idx]
            _spectra = spectra[peak-3:peak+3]
            _xaxis = xaxis[peak-3:peak+3]
            fit = np.polyfit(_xaxis, _spectra, 2)
            centers += [-fit[1]/(2*fit[0])]
        except IndexError:
            centers += [np.nan]
    if axplot is not None:
        axplot.plot(xaxis, spectra)
        colours = [cm.get_cmap('tab10')(x % 10) for x in range(len(peak_indices))]
        axplot.vlines(centers, -np.max(spectra)/20, np.max(spectra)/20, colours)
    return np.array(centers)


def fit_spectra(spectra, xaxis=None, find_peaks_kwargs=None, n_peaks=None, fit_in_one=False, plot=None,
                peak_model='VoigtModel'):
    spectra = gaussian_filter(np.copy(spectra), 1)
    if xaxis is None:
        xaxis = np.arange(len(spectra))
    if find_peaks_kwargs is None:
        find_peaks_kwargs = dict()
    default_kwargs = dict(distance=10, width=1, prominence=10)
    [find_peaks_kwargs.update({key: value}) for key, value in default_kwargs.items() if key not in find_peaks_kwargs]

    peak_indices, peak_properties = find_peaks(spectra, **find_peaks_kwargs)
    # print(peak_indices)
    if plot is not None:
        # print(peak_indices)
        # fig, ax = plt.subplots(1, 1)
        ax = plot
        ax.plot(xaxis, spectra)
        ax.vlines(peak_indices, 0, np.max(spectra) / 10, 'k')
    if len(peak_indices) > 1:
        if n_peaks is not None:
            # Sort by height and select limit
            heights = spectra[peak_indices]
            sorter = np.argsort(heights)[::-1]
            peak_indices = peak_indices[sorter][:n_peaks]
            new_properties = dict()
            for key, value in peak_properties.items():
                new_properties[key] = value[sorter][:n_peaks]
            peak_properties = new_properties
        sorter = np.argsort(peak_indices)
        peak_indices = peak_indices[sorter]
        new_properties = dict()
        for key, value in peak_properties.items():
            new_properties[key] = value[sorter]
        peak_properties = new_properties

    if plot is not None:
        ax.vlines(peak_indices, -np.max(spectra) / 10, 0, 'r')

    if 'prominences' in peak_properties:
        prominences = (peak_properties['prominences'], (peak_properties['left_bases'], peak_properties['right_bases']))
    else:
        prominences = peak_prominences(spectra, peak_indices)
    if 'widths' in peak_properties:
        widths = (peak_properties['widths'], peak_properties['width_heights'], (peak_properties['left_ips'], peak_properties['right_ips']))
    else:
        widths = peak_widths(spectra, peak_indices, prominence_data=prominences)
    # guess = dict(background=np.percentile(spectra, 10))
    if fit_in_one:
        guess = dict()
        for idx, peak in enumerate(peak_indices):
            guess['peak%d_center' % idx] = xaxis[peak]
            width = widths[0][idx]
            guess['peak%d_width' % idx] = width
            prominence = prominences[0][idx]
            guess['peak%d_amplitude' % idx] = prominence * width
        print(guess)
        # model = lmfit.models.ConstantModel()
        model = None
        for idx, _ in enumerate(peak_indices):
            if model is None:
                model = lmfit.models.VoigtModel(prefix='peak%d_' % idx)
            else:
                model += lmfit.models.VoigtModel(prefix='peak%d_' % idx)
        print(model.param_names)
        params_guess = model.make_params(**guess)
        fit = model.fit(spectra, params_guess, x=xaxis)
        plt.plot(xaxis, spectra)
        plt.plot(xaxis, fit.init_fit)
        plt.plot(xaxis, fit.best_fit)
    else:
        best_fit = dict()
        for idx, peak in enumerate(peak_indices):
            pk_model = getattr(lmfit.models, peak_model)
            # model = lmfit.models.PolynomialModel(1) + pk_model(prefix='peak%d_' % idx)
            model = lmfit.models.ConstantModel() + pk_model(prefix='peak%d_' % idx)
            guess = dict()
            width = widths[0][idx]
            prominence = prominences[0][idx]
            # print(len(peak_indices), idx)
            if len(peak_indices) == 1:
                roi = [np.max([0, int(peak - 2 * width)]),
                       np.min([len(xaxis), int(np.round(peak + 2 * width))])]
            else:
                if len(peak_indices) - 1 > idx > 0:
                    roi = [np.max([int(peak_indices[idx-1]+width), int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width)), int(peak_indices[idx+1]-width)])]
                elif idx > 0:
                    roi = [np.max([int(peak_indices[idx-1]+width), int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width))])]
                else:
                    roi = [np.max([0, int(peak-2*width)]),
                           np.min([len(xaxis), int(np.round(peak+2*width)), int(peak_indices[idx+1]-width)])]

            guess['peak%d_center' % idx] = xaxis[peak]
            guess['peak%d_sigma' % idx] = width
            guess['peak%d_amplitude' % idx] = prominence * width
            guess['c'] = np.percentile(spectra[roi[0]:roi[1]], 1)
            # guess['c0'] = np.percentile(spectra[roi[0]:roi[1]], 1)
            # guess['c1'] = 0
            # guess['c2'] = 0
            # guess['c3'] = 0
            params_guess = model.make_params(**guess)
            if plot is not None:
                # print(model.param_names)
                ax.plot(xaxis[roi[0]:roi[1]], model.eval(params_guess, x=xaxis[roi[0]:roi[1]]), 'k--')
            try:
                fit = model.fit(spectra[roi[0]:roi[1]], params_guess, x=xaxis[roi[0]:roi[1]])
                for key, value in fit.best_values.items():
                    if 'peak' in key:
                        best_fit[key] = value
            except Exception as e:
                print('Failed fit: ', idx, peak, peak_indices, guess, e)
                for key, value in guess.items():
                    if 'peak' in key:
                        best_fit[key] = np.nan
            if plot is not None:
                _best_fit = dict(best_fit)
                try:
                    _best_fit['c0'] = fit.best_fit['c']
                    # _best_fit['c0'] = fit.best_fit['c0']
                    # _best_fit['c1'] = fit.best_fit['c1']
                except:
                    _best_fit['c'] = 0
                    # _best_fit['c0'] = 0
                    # _best_fit['c1'] = 0
                _best_fit = model.make_params(**_best_fit)
                # print(model.eval(_best_fit, x=xaxis[roi[0]:roi[1]]))
                ax.plot(xaxis[roi[0]:roi[1]], model.eval(_best_fit, x=xaxis[roi[0]:roi[1]]), 'k.-')
                try:
                    ax.plot(xaxis[roi[0]:roi[1]], fit.best_fit, 'r')
                except Exception as e:
                    pass
    return best_fit


def fit_ground_state(band, xaxis=None, debug=False):
    if xaxis is None:
        xaxis = np.arange(len(band), dtype=np.float)
        xaxis -= np.mean(xaxis)

    # Remove np.nan
    indices = np.where(~np.isnan(band))
    band = band[indices]
    xaxis = xaxis[indices]

    # brillouin, _ = find_peaks(band, distance=10, width=5)
    if debug:
        # print(brillouin)
        plt.figure()
        plt.plot(xaxis, band)
        # plt.plot(xaxis[brillouin], band[brillouin], 'x')
    linear = np.polyfit(xaxis, band, 1)

    model = lmfit.models.LinearModel() + lmfit.models.SineModel()
    # print(model.param_names)
    # print(np.squeeze(np.diff(xaxis[brillouin])))
    guess = dict(slope=linear[0], intercept=linear[1], amplitude=(np.max(band) - np.min(band))/2,
                 # frequency=2*np.pi/np.squeeze(np.diff(xaxis[brillouin])),
                 frequency=2*np.pi/35,
                 shift=-np.pi/2)
    params_guess = model.make_params(**guess)
    fit = model.fit(band, params_guess, x=xaxis)
    if debug:
        plt.plot(xaxis, fit.init_fit, 'k--')
        plt.plot(xaxis, fit.best_fit)
    return fit.best_values, fit.best_fit, xaxis
