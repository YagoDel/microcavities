# -*- coding: utf-8 -*-

from microcavities.utils.HierarchicalScan import AnalysisScan
from microcavities.experiment.utils import spectrometer_calibration, magnification
from microcavities.analysis.analysis_functions import find_k0, dispersion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def dispersion_power_series(yaml_path, series_names, bkg=0):
    """For experiments with multiple ExperimentScan power series of energy-resolved, k-space images (at different exposures)

    Extracts the polariton mass and plots the normalised k=0 spectra as a function of power

    :param yaml_path: str. Location of the yaml used to run the ExperimentScan
    :param series_names:
    :param bkg:
    :return:
    """
    photolum, powers = get_dispersion_data(yaml_path, series_names, bkg)

    masses, k0_energy, dispersion_img, energy_axis, k_axis = get_calibrated_mass(photolum)

    k0_img, xaxis = get_k0_image(list(map(lambda x: np.mean(x, 0), photolum)), powers)
    yaxis = (energy_axis - np.mean(k0_energy) * 1000) * 1000

    indx = np.argmin(np.abs(yaxis))
    lims = [indx - 200, indx + 40]
    indxs = np.argmax(k0_img, 1)
    lims2 = [np.min(indxs)-40, np.max(indxs) + 40]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(np.mean(np.log(dispersion_img[..., lims[0]:lims[1]]), 0).transpose(),  aspect='auto',
                  extent=[np.min(k_axis), np.max(k_axis), energy_axis[lims[1]], energy_axis[lims[0]]])
    axs[0].set_xlabel(u'Wavevector / \u00B5m$^{-1}$')
    axs[0].set_ylabel('Energy / eV')
    axs[0].text(0, energy_axis[lims[0] + 50], r'(%.4f $\pm$ %.4f) m$_e$' % (np.mean(masses), np.std(masses)), ha='center')
    axs[1].imshow(k0_img[:, lims2[0]:lims2[1]].transpose(), aspect='auto', extent=[np.min(xaxis), np.max(xaxis),
                                                                                   yaxis[lims2[1]],
                                                                                   yaxis[lims2[0]]])
    axs[1].set_xlabel('CW power / W')
    axs[1].set_ylabel('Blueshift / meV')
    fig.tight_layout()
    return fig, axs


def get_dispersion_data(yaml_path, series_names, bkg=0, average=False):
    photolum = []
    powers = []
    for series_name in series_names:
        scan = AnalysisScan(yaml_path)
        scan.series_name = series_name
        scan.run()
        scan_data = np.array([scan.analysed_data['raw_img%d' % (x+1)] for x in range(len(scan.analysed_data.keys()))])
        scan_data -= bkg
        if average:
            scan_data = np.mean(scan_data, 0)
        photolum += [scan_data]
        powers += [list(scan.variables['power_wheel_power'])]
    return photolum, powers


def get_calibrated_mass(photolum):
    dispersion_img = np.copy(photolum[0][:, 0])

    wvls = spectrometer_calibration(wavelength=780)
    energy_axis = 1240 / wvls

    mag = magnification([0.01, 0.25, 0.1, 0.1, 0.2])[0]
    k0 = int(np.mean(list(map(find_k0, dispersion_img))))
    k_axis = np.linspace(-200, 200, 400)
    k_axis -= -200 + k0
    k_axis *= 20 * 1e-6 / mag  # Converting to SI and dividing by magnification
    k_axis *= 1e-6  # converting to inverse micron

    masses = []
    energies = []
    for img in dispersion_img:
        results, args, kwargs = dispersion(img, k_axis, energy_axis, True)
        masses += [results[-1]]
        energies += [results[0]]
    masses = np.array(masses)
    energies = np.array(energies)
    print("Mass = %.4f (%.4f)" % (np.mean(masses), np.std(masses)))
    return masses, energies, dispersion_img, energy_axis, k_axis


def get_k0_image(photolum, powers):
    dispersion_img = np.copy(photolum[0][0])

    k0 = int(find_k0(dispersion_img))

    dummy = np.mean(np.copy(photolum[0][:, k0 - 5:k0 + 5]), 1)
    xaxis = np.copy(powers[0])
    for indx in range(len(powers) - 1):
        overlap_index = np.sum(powers[indx + 1] <= np.max(powers[indx]))
        # print(overlap_index)
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
    powers = scan.variables['power_wheel_power'] * 1000

    yval = np.mean(emission, 0)
    yerr = np.std(emission, 0)

    shp = realspace.shape
    figratio = shp[-2] / float(shp[-1])
    print(figratio, shp)
    fig = plt.figure(figsize=(8, 4*figratio))
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