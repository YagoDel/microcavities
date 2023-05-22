# -*- coding: utf-8 -*-
"""
Utility functions to analyse condensation characteristics

The main function is dispersion_power_series, which takes a series of file paths to YAML files that have been used to
take data using the microcavities.utils.HierarchicalScan.ExperimentScan and then extracts the saved data, analyses it
and plots it
"""

from microcavities.analysis import *
from microcavities.utils.HierarchicalScan import get_data_from_yamls
from microcavities.analysis.dispersion import find_k0, dispersion, fit_quadratic_dispersion
from cycler import cycler


def powerseries_remove_overlap(images, powers):
    # TODO: replace this with a multidimensional stich_dataset function
    new_images = np.copy(images[0])
    new_powers = np.copy(powers[0])
    # Removes common indices
    for indx in range(len(powers) - 1):
        overlap_index = np.sum(powers[indx + 1] <= np.max(powers[indx]))
        _dummy = np.copy(images[indx + 1][overlap_index:])
        new_images = np.concatenate((new_images, _dummy), 0)
        if overlap_index > 0:
            new_powers = np.concatenate((new_powers, powers[indx + 1][overlap_index:]))
        else:
            new_powers = np.concatenate((new_powers, powers[indx + 1]))
    return new_images, new_powers


def get_k0_image(photolum, powers):
    """From a sequence of energy-momentum images at different powers, extracts the power-ordered k=0 spectra"""
    # We assume that the lowest power in the lowest power scan is sufficiently low to extract k=0 by quadratic fitting
    dispersion_img = np.copy(photolum[0][0])
    k0 = int(find_k0(dispersion_img, plotting=False))
    new_images, xaxis = powerseries_remove_overlap([x[:, k0-5:k0+5] for x in photolum], powers)

    normalised = []
    for dm in np.copy(np.mean(new_images, 1)):
        dm -= np.percentile(dm, 0.1)
        dm /= np.percentile(dm, 99.9)
        normalised += [dm]
    normalised = np.array(normalised)
    return normalised, xaxis


def dispersion_power_series(yaml_paths, series_names=None, bkg=0, energy_axis=('rotation_acton', 780, '2'), k_axis=None,
                            dispersion_fit_parameters=None, intensity_corrections=None, fig_ax=None,
                            powers_to_imshow=None):
    """For experiments with multiple ExperimentScan power series of energy-resolved, k-space images (at different exposures)

    Extracts the polariton mass and plots the normalised k=0 spectra as a function of power

    :param yaml_paths: str or list. Location(s) of the yaml used to run the ExperimentScan
    :param series_names:
    :param bkg:
    :param energy_axis:
    :param k_axis:
    :param dispersion_fit_parameters:
    :param intensity_corrections:
    :param fig_ax:
    :param powers_to_imshow: list of floats. Values of power to display in the colorful_imshow subplot
    :return:
    """

    # Creating a figure if not given
    if fig_ax is None:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
    else:
        fig, axs = fig_ax

    # Extracting the data and removing the background. Can give different backgrounds for each series
    photolum, powers = get_data_from_yamls(yaml_paths, series_names)
    key = list(powers[0].keys())[0]
    try:
        if len(bkg) == len(series_names):
            pass
        elif len(bkg.shape) == 3:
            pass
        elif len(bkg.shape) == 2:
            bkg = [bkg] * len(series_names)
    except:
        bkg = [bkg] * len(photolum)
    photolum = [p - b for p, b in zip(photolum, bkg)]
    powers = [p[key] for p in powers]
    dispersion_imgs = np.copy(photolum[0][0])
    if len(dispersion_imgs.shape) == 2:
        dispersion_imgs = dispersion_imgs[np.newaxis]

    # Analysing the dispersion images
    results = [dispersion(img, k_axis, energy_axis, False, dispersion_fit_parameters) for img in dispersion_imgs]
    k0_energies = [d['polariton_energy'] for d in results]
    lifetimes = [d['polariton_lifetime'] for d in results]
    masses = [d['polariton_mass'] for d in results]
    exciton_fractions = [d['exciton_fraction'] for d in results]

    """subplot(0, 1) pcolormesh of the normalised emission spectra at k~0 at all powers"""
    # Extracting the k=0 spectra at each power
    k0_img, xaxis = get_k0_image(photolum, powers)
    yaxis = (energy_axis - np.mean(k0_energies))

    # Finding the energy range that needs to be plotted by finding the energy of maximum emission at each power, and
    # adding a 40 pixel pad around it
    indxs = np.argmax(k0_img, 1)
    lims = [np.max([0, np.min(indxs)-40]), np.min([np.max(indxs) + 40, k0_img.shape[-1]-1])]

    # Plotting
    pcolormesh(k0_img[:, lims[0]:lims[1]].transpose(), axs[1], xaxis, yaxis[lims[0]:lims[1]], diverging=False, cbar=False, cmap='Greys')
    axs[1].set_xlabel('CW power / W')
    axs[1].set_ylabel('Blueshift / meV')

    """subplot(0,0) colorful_imshow of:
     - The lowest power image, which is assumed to be low power enough to fit the mass/lifetime
     - Selected condensate dispersions"""
    k0 = int(np.mean([find_k0(img, plotting=False) for img in dispersion_imgs]))
    if powers_to_imshow is None:
        powers_to_imshow = [powers[-1][-1]]

    # Cropping the dispersion and condensate images using the same energy range as subplot(0, 1)
    condensate_imgs = []
    for power in powers_to_imshow:
        section_idx = np.argmin([np.min(np.abs(np.array(x) - power)) for x in powers])
        dset_idx = np.argmin(np.abs(np.array(powers[section_idx]) - power))
        img = photolum[section_idx][dset_idx, :, lims[0]:lims[1]].transpose()
        condensate_imgs += [img]
    condensate_imgs = np.array(condensate_imgs)
    dispersion_img = np.mean((dispersion_imgs[..., lims[0]:lims[1]]), 0).transpose()

    # Plotting
    joined_images = np.concatenate(([dispersion_img], condensate_imgs))
    _, colours = colorful_imshow(joined_images, ax=axs[0], norm_args=(1, 99.99),
                                 aspect='auto', from_black=False, xaxis=k_axis, yaxis=energy_axis[lims[0]:lims[1]])
    quad_fit = fit_quadratic_dispersion(np.mean(dispersion_imgs, 0), energy_axis, k_axis, plotting=False)
    axs[0].plot(k_axis[k0 - 70:k0 + 70], np.poly1d(quad_fit)(k_axis[k0 - 70:k0 + 70]), 'w')

    label_axes(axs[2], u'Wavevector [\u00B5m$^{-1}$]', 'Energy [meV]')
    for img, power, c in zip(joined_images, [powers[0][0]]+powers_to_imshow, colours):
        axs[0].text(k_axis[k0], energy_axis[lims[0]:lims[1]][np.argmax(np.sum(img, 1)) + 10], '%gW' % power,
                    ha='center', va='top', color=c)
    axs[0].text(0.5, 1, (r'(%.4g $\pm$ %.1g) m$_e$' % (np.mean(masses), np.std(masses)) + '\n' +
                         (r'$\Gamma$=(%.4g $\pm$ %.1g)ps  $|X|^2$=%g' % (np.mean(lifetimes), np.std(lifetimes), np.mean(exciton_fractions)))),
                ha='center', va='center', color='k', transform=axs[0].transAxes)

    """subplot(1, 0) line plots of the condensate emission"""
    # Analysing the total emission intensity over all k and the maximum emission at around k~0
    total_emissions = [np.sum(x, (1, 2)) for x in photolum]
    max_k0_emissions = [np.max(x[..., k0-5:k0+5, :], (1, 2)) for x in photolum]

    if intensity_corrections is None:  # if not given, we guess it by making the lines overlap
        # create interpolated arrays that cover all given power values
        x_new = np.sort(np.concatenate(powers))

        # Corrections for the total emission
        interpolated_datasets = [interp1d(x, y, bounds_error=False)(x_new) for x, y in zip(powers, total_emissions)]
        intensity_corrections = [1]
        for idx in range(len(interpolated_datasets)-1):
            # The emission ratio between adjacent datasets to be used to scale them
            emission_ratio = np.nanmean(interpolated_datasets[idx]/interpolated_datasets[idx+1])
            # Each correction needs to take into account previous scalings
            intensity_corrections += [np.prod(intensity_corrections) * emission_ratio]
        total_emissions = [e*c for e, c in zip(total_emissions, intensity_corrections)]

        # Corrections for the maximum emission around k~0
        interpolated_datasets = [interp1d(x, y, bounds_error=False)(x_new) for x, y in zip(powers, max_k0_emissions)]
        intensity_corrections = [1]
        for idx in range(len(interpolated_datasets)-1):
            emission_ratio = np.nanmean(interpolated_datasets[idx]/interpolated_datasets[idx+1])
            intensity_corrections += [np.prod(intensity_corrections) * emission_ratio]
        max_k0_emissions = [e*c for e, c in zip(max_k0_emissions, intensity_corrections)]
    else:
        # intensity_corrections should be given if there are changes to the setup, like ND filters
        total_emissions = [e/c for e, c in zip(total_emissions, intensity_corrections)]
        max_k0_emissions = [e/c for e, c in zip(max_k0_emissions, intensity_corrections)]

    # Plotting
    custom_cycler = cycler(linestyle=['-', '--', ':', '-.'])
    axs[2].set_prop_cycle(custom_cycler)
    [axs[2].semilogy(x, y, color='k') for x, y in zip(powers, total_emissions)]
    label_axes(axs[2], 'CW power [W]', 'Total emission [a.u.]')

    ax3 = axs[2].twinx()
    ax3.set_prop_cycle(custom_cycler)
    [ax3.semilogy(x, y, color='r') for x, y in zip(powers, max_k0_emissions)]
    ax3.set_ylabel('Max emission [a.u.]')
    colour_axes(ax3, 'red', 'y', 'right')

    """subplot(1, 1)"""
    # Plotting the momenta vs power
    new_images, new_powers = powerseries_remove_overlap(photolum, powers)
    img = np.array([normalize(x, (1, 99.9)) for x in np.sum(new_images[..., lims[0]:lims[1]], -1)])
    pcolormesh(img.transpose(), axs[3], new_powers, k_axis, diverging=False, cbar=False, cmap='Greys')

    # Plotting the momentum centroid on the same axes
    summed_energies = [np.sum(x[..., lims[0]:lims[1]], -1) for x in photolum]
    k0s = [[np.average(k_axis, weights=x) for x in summed_energy] for summed_energy in summed_energies]
    legends = [r'$\langle k_y \rangle$'] + ['_nolegend_'] * (len(powers)-1)
    [axs[3].plot(x, y, 'b', label=legend) for x, y, legend in zip(powers, k0s, legends)]
    label_axes(axs[3], r'$k_y$', 'CW power [W]')
    axs[3].legend()

    fig.tight_layout()
    return fig, axs
