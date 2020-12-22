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


def realspace_power_series(yaml_paths, series_names=None, bkgs=0, exposures=None, nd_filters=None):
    photolum, powers = get_data_from_yamls(yaml_paths, series_names, bkgs)
    mag = magnification([0.01, 0.25, 0.1, 0.2], 805e-9)
    center = [40, 38]
    size = 20
    cropped = photolum[..., center[0] - size:center[0] + size, center[1] - size:center[1] + size]
    x_axis = np.arange(cropped.shape[-1], dtype=np.float)
    x_axis -= np.mean(x_axis)
    x_axis *= (20 / mag[0])
    _x, _y = np.meshgrid(x_axis, x_axis)
    _r = np.sqrt(_x ** 2 + _y ** 2)
    rmax = 3
    mask = _r < rmax
    extent = [np.min(x_axis), np.max(x_axis), np.max(x_axis), np.min(x_axis)]
    # pl = np.sum(cropped[..., int(size/2-5):int(size/2+5), int(size/2-5):int(size/2+5)], (2, 3))
    pl = np.array([np.sum(cropped[x, :, mask], 0) for x in range(3)])

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0])
    ax0 = plt.subplot(gs2[0])
    ax1 = plt.subplot(gs2[1])
    ax2 = plt.subplot(gs[1])
    ax0.imshow(np.mean(cropped[:, 0], 0), extent=extent)
    x_tst = np.linspace(-rmax, rmax, 50)
    y_tst = np.sqrt(rmax ** 2 - x_tst ** 2)
    y_tst2 = -np.sqrt(rmax ** 2 - x_tst ** 2)
    ax0.plot(np.append(x_tst, x_tst[::-1]), np.append(y_tst, y_tst2[::-1]), 'w--')
    ax1.plot(np.append(x_tst, x_tst[::-1]), np.append(y_tst, y_tst2[::-1]), 'w--')
    ax1.imshow(np.mean(cropped[:, -1], 0), extent=extent)
    ax2.semilogy(scan.variables['power_wheel_power'] * 1e3, pl.transpose())
    ax2.set_xlabel('Power / mW')
    ax2.set_ylabel('PL / a.u.')
    ax1.set_xlabel('Position / um')
    [ax.set_ylabel('Position / um') for ax in [ax0, ax1]]

    return


def dispersion_power_series(yaml_paths, series_names=None, bkg=0, energy_axis=(780, '1200'), k_axis=None,
                            known_sample_parameters=None):
    """For experiments with multiple ExperimentScan power series of energy-resolved, k-space images (at different exposures)

    Extracts the polariton mass and plots the normalised k=0 spectra as a function of power

    TODO: prevent overlapping power values

    :param yaml_paths: str or list. Location(s) of the yaml used to run the ExperimentScan
    :param series_names:
    :param bkg:
    :param wavelength:
    :param grating:
    :param known_sample_parameters:
    :return:
    """
    photolum, powers = get_data_from_yamls(yaml_paths, series_names, bkg)

    k0_energy, lifetimes, masses, exciton_fractions, dispersion_img, energy_axis, k_axis = get_calibrated_mass(np.copy(photolum[0][:, 0]), energy_axis, k_axis, known_sample_parameters)
    _, quad_fit = find_mass(np.mean(dispersion_img, 0), energy_axis, k_axis, return_fit=True)

    k0_img, xaxis = get_k0_image(list(map(lambda x: np.mean(x, 0), photolum)), powers)
    yaxis = (energy_axis - np.mean(k0_energy))

    indx = np.argmin(np.abs(yaxis))
    lims = [np.max([indx - 200, 0]), np.min([indx + 40, photolum[0].shape[-1]-1])]
    indxs = np.argmax(k0_img, 1)
    lims2 = [np.max([0, np.min(indxs)-40]), np.min([np.max(indxs) + 40, k0_img.shape[-1]-1])]

    condensate_img = np.mean(photolum[-1][:, -1, :, lims2[0]:lims2[1]], 0).transpose()
    dispersion_img = np.mean((dispersion_img[..., lims2[0]:lims2[1]]), 0).transpose()
    dispersion_img -= np.percentile(dispersion_img, 1)
    condensate_img -= np.percentile(condensate_img, 1)
    dispersion_img /= np.percentile(dispersion_img, 99.9)
    condensate_img /= np.percentile(condensate_img, 99.9)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    img = np.rollaxis(np.array([dispersion_img, condensate_img, np.zeros(condensate_img.shape)]), 0, 3)
    axs[0].imshow(img, aspect='auto',
                  extent=[np.min(k_axis), np.max(k_axis), energy_axis[lims2[1]], energy_axis[lims2[0]]])
    axs[0].set_xlabel(u'Wavevector / \u00B5m$^{-1}$')
    axs[0].set_ylabel('Energy / eV')
    axs[0].text(0, energy_axis[lims2[0] + 18], r'(%.4g $\pm$ %.1g) m$_e$' % (np.mean(masses), np.std(masses)),
                ha='center', color='w')
    axs[0].text(0, energy_axis[lims2[0] + 30], r'$\Gamma$=%.4gps  $|X|^2$=%g' % (np.mean(lifetimes), np.mean(exciton_fractions)),
                ha='center', color='w')
    axs[0].text(0, energy_axis[lims2[0] + 42], 'Low power', ha='center', color='r')
    axs[0].text(0, energy_axis[lims2[0] + 54], 'High power', ha='center', color='g')
    k0_idx = np.argmin(np.abs(k_axis))
    axs[0].plot(k_axis[k0_idx-70:k0_idx+70], np.poly1d(quad_fit)(k_axis[k0_idx-70:k0_idx+70]), 'w')
    axs[1].imshow(k0_img[:, lims2[0]:lims2[1]].transpose(), aspect='auto', extent=[np.min(xaxis), np.max(xaxis),
                                                                                   yaxis[lims2[1]],
                                                                                   yaxis[lims2[0]]])
    axs[1].set_xlabel('CW power / W')
    axs[1].set_ylabel('Blueshift / meV')
    fig.tight_layout()
    return fig, axs


def get_data_from_yamls(yaml_paths, series_names, bkg=0, average=False):
    if type(yaml_paths) == str:
        yaml_paths = [yaml_paths] * len(series_names)
    elif series_names is None:
        series_names = [None] * len(yaml_paths)

    try:
        if len(bkg) == len(series_names):
            pass
        elif len(bkg.shape) == 3:
            pass
        elif len(bkg.shape) == 2:
            bkg = [bkg] * len(series_names)
    except:
        bkg = [bkg] * len(series_names)

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


def get_calibrated_mass(dispersion_img, energy_axis=(780, '1200'), k_axis=None, known_sample_parameters=None):
    # dispersion_img = np.copy(photolum[0][:, 0])

    if len(energy_axis) <= 2:
        wavelength = energy_axis[0]
        if len(energy_axis) == 1:
            grating = '1200'
        else:
            grating = energy_axis[1]
        wvls = spectrometer_calibration(wavelength=wavelength, grating=grating)
        energy_axis = 1240 / wvls

    if k_axis is None:
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


def realspace_power_series(yaml_paths, series_names=None, bkg=0, corrections=None):
    """For ExperimentScans that take a power series of real-spaces images

    Plots 4 example images, and the emission as a function of power

    :param yaml_path: str. Location of the yaml used to run the ExperimentScan
    :param bkg:
    :return:
    """

    photolum, powers = get_data_from_yamls(yaml_paths, series_names, bkg)
    all_powers = np.concatenate(powers)

    if corrections is None:
        # TODO: make an automatic thing that matches the values with common powers
        corrections = [1] * len(photolum)

    normed = [x*y for x, y in zip(photolum, corrections)]
    # scan = AnalysisScan(yaml_path)
    # scan.run()
    # dummy = np.asarray([scan.analysed_data['raw_img%d' % (x+1)] for x in range(len(scan.analysed_data))], np.float)
    # realspace = dummy - bkg
    emission = [np.percentile(x, 99.9, (-2, -1)) for x in normed]
    # powers = np.array(scan.variables['power_wheel_power']) * 1000

    yval = [np.mean(x, 0) for x in emission]
    yerr = [np.std(x, 0) for x in emission]

    shp = normed[0][0,0].shape
    figratio = shp[-2] / float(shp[-1])
    fig = plt.figure(figsize=(8/figratio, 4))
    gs = gridspec.GridSpec(1, 2)
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0], hspace=0.01, wspace=0.01)

    ax1 = plt.subplot(gs[1])
    [ax1.errorbar(x, y, e, fmt='o-', markersize=2, elinewidth=0.75, capsize=0.75, lw=0.5) for x,y,e in zip(powers, yval, yerr)]
    ax1.set_yscale('log')
    ax1.set_xlabel('CW power / mW')
    ax1.set_ylabel('Emission / a.u.')

    # TODO: make the selection of positions a bit more clever (find the threshold)
    # TODO: add micron units to the axis
    # TODO: add cropping. Add selection of the region over which the integration happens or options for what to extract (maximum, total, cropped)
    axs = []
    for idx, index in enumerate(map(int, np.linspace(0, len(all_powers)-1, 4))):
        power = all_powers[index]
        power_len = [len(x) for x in powers]
        cum_len = np.cumsum(power_len)
        idx2 = np.min(np.argwhere(index < cum_len))
        if idx2 > 0:
            index -= cum_len[idx2-1]
        image = photolum[idx2][:, index]
        ax = plt.subplot(gs_sub[idx])
        ax.imshow(np.mean(image, 0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5, 1, '%gW' % power, ha='center', va='top', transform=ax.transAxes)
        axs += [ax]
    fig.tight_layout()

    return fig, axs + [ax1]
