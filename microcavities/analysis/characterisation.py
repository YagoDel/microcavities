# -*- coding: utf-8 -*-

from microcavities.utils.HierarchicalScan import get_data_from_yamls
from microcavities.experiment.utils import spectrometer_calibration, magnification
from microcavities.analysis.analysis_functions import find_k0, dispersion, fit_quadratic_dispersion
from microcavities.analysis.utils import normalize
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from microcavities.utils.plotting import pcolormesh, colorful_imshow

"""
Utility functions that wrap underlying analysis functionality
"""


def powerseries_remove_overlap(images, powers, correction_type='sum'):
    calculated_intensity_correction = []
    new_images = np.copy(images[0])
    new_powers = np.copy(powers[0])
    # Removes common indices
    for indx in range(len(powers) - 1):
        overlap_index = np.sum(powers[indx + 1] <= np.max(powers[indx]))
        np_func = getattr(np, correction_type)
        _dummy = np.copy(images[indx + 1][overlap_index:])
        new_images = np.concatenate((new_images, _dummy), 0)
        if overlap_index > 0:
            new_powers = np.concatenate((new_powers, powers[indx + 1][overlap_index:]))
            calculated_intensity_correction += [np.mean(np_func(images[indx+1][:overlap_index], (1, 2)) / np_func(images[indx][-overlap_index:], (1, 2)))]
        else:
            new_powers = np.concatenate((new_powers, powers[indx + 1]))
            calculated_intensity_correction += [np_func(images[indx+1][0], (0, 1)) / np_func(images[indx][-1], (0, 1))]
    return new_images, new_powers, calculated_intensity_correction


def get_k0_image(photolum, powers):
    """From a sequence of energy-momentum images at different powers, extracts the power-ordered k=0 spectra"""
    # We assume that the lowest power in the lowest power scan is sufficiently low to extract k=0 by quadratic fitting
    dispersion_img = np.copy(photolum[0][0])
    k0 = int(find_k0(dispersion_img))
    print(k0, photolum[0].shape, powers)
    new_images, xaxis, _ = powerseries_remove_overlap([x[:, k0-5:k0+5] for x in photolum], powers)

    normalised = []
    for dm in np.copy(np.mean(new_images, 1)):
        dm -= np.percentile(dm, 0.1)
        dm /= np.percentile(dm, 99.9)
        normalised += [dm]
    normalised = np.array(normalised)
    return normalised, xaxis


def dispersion_power_series(yaml_paths, series_names=None, bkg=0, energy_axis=('rotation_acton', 780, '2'), k_axis=None,
                            known_sample_parameters=None, intensity_corrections=None, fig_ax=None,
                            powers_to_imshow=None):
    """For experiments with multiple ExperimentScan power series of energy-resolved, k-space images (at different exposures)

    Extracts the polariton mass and plots the normalised k=0 spectra as a function of power

    :param yaml_paths: str or list. Location(s) of the yaml used to run the ExperimentScan
    :param series_names:
    :param bkg:
    :param energy_axis:
    :param k_axis:
    :param known_sample_parameters:
    :param intensity_corrections:
    :param fig_ax:
    :return:
    """

    photolum, powers = get_data_from_yamls(yaml_paths, series_names)
    try:
        if len(bkg) == len(series_names):
            pass
        elif len(bkg.shape) == 3:
            pass
        elif len(bkg.shape) == 2:
            bkg = [bkg] * len(series_names)
    except:
        bkg = [bkg] * len(photolum)
    print(len(photolum), photolum[0].shape)
    photolum = [p - b for p, b in zip(photolum, bkg)]
    print(powers)
    powers = [p['power_wheel_power'] for p in powers]

    if powers_to_imshow is None:
        powers_to_imshow = [powers[-1][-1]]

    # Fitting the low-power dispersion with a quadratic
    k0_energy, lifetimes, masses, exciton_fractions, dispersion_img, energy_axis, k_axis = get_calibrated_mass(np.copy(photolum[0]), energy_axis, k_axis, known_sample_parameters)
    quad_fit = fit_quadratic_dispersion(np.mean(dispersion_img, 0), energy_axis, k_axis)

    # Extracting the k=0 spectra at each power
    k0_img, xaxis = get_k0_image(photolum, powers)
    yaxis = (energy_axis - np.mean(k0_energy))

    # Finding the energy range that needs to be plotted by finding the energy of maximum emission at each power, and
    # adding a 40 pixel pad around it
    indxs = np.argmax(k0_img, 1)
    lims = [np.max([0, np.min(indxs)-40]), np.min([np.max(indxs) + 40, k0_img.shape[-1]-1])]

    # Cropping the dispersion and condensate images and thresholding between 1 and 99.9
    condensate_imgs = []
    for power in powers_to_imshow:
        section_idx = np.argmin([np.min(np.abs(np.array(x) - power)) for x in powers])
        dset_idx = np.argmin(np.abs(np.array(powers[section_idx]) - power))
        print('Here: ', photolum[section_idx].shape)
        img = photolum[section_idx][dset_idx, :, lims[0]:lims[1]].transpose()
        img -= np.percentile(img, 1)
        img /= np.percentile(img, 99.9)
        condensate_imgs += [img]
    condensate_imgs = np.array(condensate_imgs)

    dispersion_img = np.mean((dispersion_img[..., lims[0]:lims[1]]), 0).transpose()
    dispersion_img -= np.percentile(dispersion_img, 1)
    dispersion_img /= np.percentile(dispersion_img, 99.9)

    # Extracting the total emission
    k0 = int(find_k0(dispersion_img.transpose()))
    _, _, sum_corrections = powerseries_remove_overlap(photolum, powers, 'sum')
    _, _, max_corrections = powerseries_remove_overlap([x[..., k0-5:k0+5, :] for x in photolum], powers, 'max')

    if intensity_corrections is None:
        intensity_corrections = [1] + list(np.mean([sum_corrections, max_corrections], 0))
    total_emission = [np.sum(x, (1, 2)) / correction for x, correction in zip(photolum, intensity_corrections)]
    max_k0_emission = [np.max(x[..., k0-5:k0+5, :], (1, 2)) / correction for x, correction in zip(photolum, intensity_corrections)]

    # Creating a figure if not given
    if fig_ax is None:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
    else:
        fig, axs = fig_ax
    # Plotting RGB image of lower-polariton and condensate dispersion
    colorful_imshow(np.concatenate(([dispersion_img], condensate_imgs)), ax=axs[0], aspect='auto', from_black=False,
                    xaxis=k_axis, yaxis=energy_axis[lims[0]:lims[1]]) #[::-1])
    axs[0].set_xlabel(u'Wavevector / \u00B5m$^{-1}$')
    axs[0].set_ylabel('Energy / eV')
    axs[0].text(0.5, 1, (r'(%.4g $\pm$ %.1g) m$_e$' % (np.mean(masses), np.std(masses)) + '\n' +
                         (r'$\Gamma$=%.4gps  $|X|^2$=%g' % (np.mean(lifetimes), np.mean(exciton_fractions)))),
                ha='center', va='center', color='k', transform=axs[0].transAxes)
    axs[0].text(k_axis[k0], energy_axis[lims[0]:lims[1]][np.argmax(np.sum(dispersion_img, 1))+10], '%gW' % powers[0][0],
                ha='center', va='top', color='r')
    for idx, power in enumerate(powers_to_imshow):
        axs[0].text(k_axis[k0], energy_axis[lims[0]:lims[1]][np.argmax(np.sum(condensate_imgs[idx], 1))+10], '%gW' % power,
                    ha='center', va='top', color='g')
    k0_idx = np.argmin(np.abs(k_axis + quad_fit[1] / (2 * quad_fit[0])))
    axs[0].plot(k_axis[k0_idx-70:k0_idx+70], np.poly1d(quad_fit)(k_axis[k0_idx-70:k0_idx+70]), 'w')

    # Plotting the k=0 spectra as a function of power
    pcolormesh(k0_img[:, lims[0]:lims[1]].transpose(), axs[1], xaxis, yaxis[lims[0]:lims[1]], diverging=False, cbar=False, cmap='Greys')
    axs[1].set_xlabel('CW power / W')
    axs[1].set_ylabel('Blueshift / meV')

    # Plotting the total (over all k) and maximal (at k=0) emission as a function of power
    custom_cycler = cycler(linestyle=['-', '--', ':', '-.'])
    axs[2].set_prop_cycle(custom_cycler)
    [axs[2].semilogy(x, y, color='k') for x, y in zip(powers, total_emission)]
    axs[2].set_xlabel('CW power / W')
    axs[2].set_ylabel('Total emission / a.u.')
    ax3 = axs[2].twinx()
    ax3.set_prop_cycle(custom_cycler)
    [ax3.semilogy(x, y, color='r') for x, y in zip(powers, max_k0_emission)]
    ax3.set_ylabel('Max emission / a.u.')
    ax3.spines['right'].set_color('red')
    ax3.tick_params(axis='y', colors='red', which='both')
    ax3.yaxis.label.set_color('red')

    # Plotting the momenta vs power
    new_images, new_powers, _ = powerseries_remove_overlap(photolum, powers)
    img = np.array([normalize(x, (1, 99.9)) for x in np.sum(new_images[..., lims[0]:lims[1]], -1)])
    pcolormesh(img.transpose(), axs[3], new_powers, k_axis, diverging=False, cbar=False, cmap='Greys')

    # Plotting the momentum centroid on the same axes
    summed_energies = [np.sum(x[..., lims[0]:lims[1]], -1) for x in photolum]
    k0s = [[np.average(k_axis, weights=x) for x in summed_energy] for summed_energy in summed_energies]
    # axs[3].set_prop_cycle(custom_cycler)
    legends = [r'$\langle k_y \rangle$'] + ['_nolegend_'] * (len(powers)-1)
    [axs[3].plot(x, y, 'b', label=legend) for x, y, legend in zip(powers, k0s, legends)]
    axs[3].set_ylabel(r'$k_y$')
    axs[3].set_xlabel(r'CW power / W')
    axs[3].legend()

    fig.tight_layout()
    return fig, axs


def get_calibrated_mass(dispersion_imgs, energy_axis=('rotation_acton', 780, '2'), k_axis=None, known_sample_parameters=None, plotting=False):
    if len(energy_axis) <= 5:
        try:
            wvls = spectrometer_calibration(*energy_axis)
        except Exception as e:
            print(energy_axis)
            raise e
        #     wavelength = energy_axis[0]
        #     if len(energy_axis) == 1:
        #         grating = '1200'
        #     else:
        #         grating = energy_axis[1]
        #     wvls = spectrometer_calibration_old(wavelength=wavelength, grating=grating)
        energy_axis = 1240 / wvls

    if k_axis is None:
        k_axis = ('rotation_pvcam', 'k_space')
    if len(k_axis) <= 2:
        # try:
        mag = magnification(*k_axis)[0]
        mag *= 1e-6  # using the default 20um pixel size
        # except:
        #     mag = magnification_old(camera=('pvcam', 'k_space'))[0]
        k0 = np.mean(list(map(find_k0, dispersion_imgs)))
        _k_axis = np.arange(dispersion_imgs.shape[1], dtype=np.float)  # pixel units
        try:
            _k_axis = _k_axis[k_axis[0]:k_axis[1]]
            _k_axis -= k0 + k_axis[0]
        except:
            pass
        _k_axis *= mag
        k_axis = np.copy(_k_axis)

    energies = []
    lifetimes = []
    masses = []
    exciton_fractions = []
    for img in dispersion_imgs:
        results, args, kwargs = dispersion(img, k_axis, energy_axis, plotting, known_sample_parameters)
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
    return energies, lifetimes, masses, exciton_fractions, dispersion_imgs, energy_axis, k_axis
