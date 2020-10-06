# -*- coding: utf-8 -*-

from microcavities.utils.HierarchicalScan import AnalysisScan
from microcavities.experiment.utils import spectrometer_calibration, magnification
from microcavities.analysis.analysis_functions import find_k0, dispersion, find_mass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

"""
Utility functions that wrap underlying analysis functionality
"""


def dispersion_power_series(yaml_path, series_names=None, bkg=0, wavelength=780, grating='1200', known_sample_parameters=None):
    """For experiments with multiple ExperimentScan power series of energy-resolved, k-space images (at different exposures)

    Extracts the polariton mass and plots the normalised k=0 spectra as a function of power

    TODO: prevent overlapping power values

    :param yaml_path: str. Location of the yaml used to run the ExperimentScan
    :param series_names:
    :param bkg:
    :param wavelength:
    :param grating:
    :param known_sample_parameters:
    :return:
    """
    photolum, powers = get_dispersion_data(yaml_path, series_names, bkg)

    k0_energy, lifetimes, masses, exciton_fractions, dispersion_img, energy_axis, k_axis = get_calibrated_mass(photolum, wavelength, grating, known_sample_parameters)
    _, quad_fit = find_mass(np.mean(dispersion_img, 0), energy_axis, k_axis, return_fit=True)

    k0_img, xaxis = get_k0_image(list(map(lambda x: np.mean(x, 0), photolum)), powers)
    yaxis = (energy_axis - np.mean(k0_energy))

    indx = np.argmin(np.abs(yaxis))
    lims = [np.max([indx - 200, 0]), np.min([indx + 40, photolum[0].shape[-1]-1])]
    indxs = np.argmax(k0_img, 1)
    lims2 = [np.min(indxs)-40, np.max(indxs) + 40]

    condensate_img = np.mean(photolum[-1][:, -1, :, lims[0]:lims[1]], 0).transpose()
    dispersion_img = np.mean((dispersion_img[..., lims[0]:lims[1]]), 0).transpose()
    dispersion_img -= np.percentile(dispersion_img, 1)
    condensate_img -= np.percentile(condensate_img, 1)
    dispersion_img /= np.percentile(dispersion_img, 99.9)
    condensate_img /= np.percentile(condensate_img, 99.9)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    img = np.rollaxis(np.array([dispersion_img, condensate_img, np.zeros(condensate_img.shape)]), 0, 3)
    axs[0].imshow(img, aspect='auto',
                  extent=[np.min(k_axis), np.max(k_axis), energy_axis[lims[1]], energy_axis[lims[0]]])
    axs[0].set_xlabel(u'Wavevector / \u00B5m$^{-1}$')
    axs[0].set_ylabel('Energy / eV')
    axs[0].text(0, energy_axis[lims[0] + 18], r'(%.4g $\pm$ %.1g) m$_e$' % (np.mean(masses), np.std(masses)),
                ha='center', color='w')
    axs[0].text(0, energy_axis[lims[0] + 30], r'$\Gamma$=%.4gps  $|X|^2$=%g' % (np.mean(lifetimes), np.mean(exciton_fractions)),
                ha='center', color='w')
    axs[0].text(0, energy_axis[lims[0] + 42], 'Low power', ha='center', color='r')
    axs[0].text(0, energy_axis[lims[0] + 54], 'High power', ha='center', color='g')
    k0_idx = np.argmin(np.abs(k_axis))
    axs[0].plot(k_axis[k0_idx-70:k0_idx+70], np.poly1d(quad_fit)(k_axis[k0_idx-70:k0_idx+70]), 'w')
    axs[1].imshow(k0_img[:, lims2[0]:lims2[1]].transpose(), aspect='auto', extent=[np.min(xaxis), np.max(xaxis),
                                                                                   yaxis[lims2[1]],
                                                                                   yaxis[lims2[0]]])
    axs[1].set_xlabel('CW power / W')
    axs[1].set_ylabel('Blueshift / meV')
    fig.tight_layout()
    return fig, axs


def get_dispersion_data(yaml_paths, series_names, bkg=0, average=False):
    try:
        if len(bkg) == len(series_names):
            pass
        elif len(bkg.shape) == 3:
            pass
        elif len(bkg.shape) == 2:
            bkg = [bkg] * len(series_names)
    except:
        bkg = [bkg] * len(series_names)

    if type(yaml_paths) == str:
        yaml_paths = [yaml_paths] * len(series_names)
    elif series_names is None:
        series_names = [None] * len(yaml_paths)

    photolum = []
    powers = []
    for idx, series_name, yaml_path in zip(range(len(series_names)), series_names, yaml_paths):
        scan = AnalysisScan(yaml_path)
        if series_name is not None:
            scan.series_name = series_name
        scan.extract_hierarchy()
        scan.run()
        scan_data = np.array([scan.analysed_data['raw_img%d' % (x+1)] for x in range(len(scan.analysed_data.keys()))],
                             np.float)
        scan_data -= bkg[idx]
        if average:
            scan_data = np.mean(scan_data, 0)
        photolum += [scan_data]
        powers += [list(scan.variables['power_wheel_power'])]
    return photolum, powers


def get_calibrated_mass(photolum, wavelength=780, grating='1200', known_sample_parameters=None):
    dispersion_img = np.copy(photolum[0][:, 0])

    wvls = spectrometer_calibration(wavelength=wavelength, grating=grating)
    energy_axis = 1240 / wvls

    mag = magnification([0.01, 0.25, 0.1, 0.1, 0.2])[0]
    k0 = int(np.mean(list(map(find_k0, dispersion_img))))
    k_axis = np.linspace(-200, 200, 400)  # pixel units
    k_axis -= -200 + k0
    k_axis *= 20 * 1e-6 / mag  # Converting to SI and dividing by magnification
    k_axis *= 1e-6  # converting to inverse micron

    energies = []
    lifetimes = []
    masses = []
    exciton_fractions = []
    for img in dispersion_img:
        results, args, kwargs = dispersion(img, k_axis, energy_axis, False, known_sample_parameters)
        energies += [results[0]]
        lifetimes += [results[1]]
        masses += [results[2]]
        if len(results) > 3:
            exciton_fractions += [results[3]]
    energies = np.array(energies)
    lifetimes = np.array(lifetimes)
    masses = np.array(masses)
    exciton_fractions = np.array(exciton_fractions)
    # print("Energy = %.4f (%.4f)" % (np.mean(energies), np.std(energies)))
    # print("Lifetime = %.4f (%.4f)" % (np.mean(lifetimes), np.std(lifetimes)))
    # print("Mass = %g (%g)" % (np.mean(masses), np.std(masses)))
    # print("X = %.4f (%.4f)" % (np.mean(exciton_fractions), np.std(exciton_fractions)))
    return energies, lifetimes, masses, exciton_fractions, dispersion_img, energy_axis, k_axis


def get_k0_image(photolum, powers):
    dispersion_img = np.copy(photolum[0][0])

    k0 = int(find_k0(dispersion_img))

    dummy = np.mean(np.copy(photolum[0][:, k0 - 5:k0 + 5]), 1)
    xaxis = np.copy(powers[0])
    for indx in range(len(powers) - 1):
        overlap_index = np.sum(powers[indx + 1] <= np.max(powers[indx]))
        dummy2 = np.mean(np.copy(photolum[indx + 1][overlap_index:, k0 - 5:k0 + 5]), 1)
        dummy = np.concatenate((dummy, dummy2), 0)
        xaxis = np.concatenate((xaxis, powers[indx + 1][overlap_index:]))

    normalised = []
    for dm in np.copy(dummy):
        dm -= np.percentile(dm, 0.1)
        dm /= np.percentile(dm, 99.9)
        normalised += [dm]
    normalised = np.array(normalised)
    return normalised, xaxis


def realspace_power_series(yaml_path, bkg=0):
    """For ExperimentScans that take a power series of real-spaces images

    Plots 4 example images, and the emission as a function of power

    :param yaml_path: str. Location of the yaml used to run the ExperimentScan
    :param bkg:
    :return:
    """
    scan = AnalysisScan(yaml_path)
    scan.run()
    dummy = np.asarray([scan.analysed_data['raw_img%d' % (x+1)] for x in range(len(scan.analysed_data))], np.float)
    realspace = dummy - bkg
    emission = np.percentile(realspace, 99.9, (-2, -1))
    powers = np.array(scan.variables['power_wheel_power']) * 1000

    yval = np.mean(emission, 0)
    yerr = np.std(emission, 0)

    shp = realspace.shape
    figratio = shp[-2] / float(shp[-1])
    fig = plt.figure(figsize=(8/figratio, 4))
    gs = gridspec.GridSpec(1, 2)
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0], hspace=0.01, wspace=0.01)

    ax1 = plt.subplot(gs[1])
    ax1.errorbar(powers, yval, yerr, fmt='o-', markersize=2, elinewidth=0.75, capsize=0.75, lw=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('CW power / mW')
    ax1.set_ylabel('Emission / a.u.')

    axs = []
    for idx, index in enumerate(map(int, np.linspace(0, len(powers)-1, 4))):
        ax = plt.subplot(gs_sub[idx])
        ax.imshow(np.mean(realspace[:, index], 0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        axs += [ax]
    fig.tight_layout()

    return fig, axs + [ax1]
