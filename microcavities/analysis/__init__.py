# -*- coding: utf-8 -*-

from microcavities.utils.plotting import *


hbar = 6.582119569 * 10 ** (-16) * 10 ** 3 * 10 ** 12   # Planck constant   meV.ps
c = 3 * 10 ** 14 * 10 ** -12                            # Speed of Light    um/ps
m_e = (0.511 * 1e9) / c**2                              # Electron mass     meV/c**2


LOGGER = create_logger('Analysis')


def photon_density(camera_count, nd_filter=1, exposure_time=1e-4, lifetime=15e-12, hopfield=0.23, alpha=4,
                   camera_qe=0.4, optical_losses=None):
    """Converting camera counts into photon density

    :param camera_count: float. # of photon counts
    :param nd_filter: float. ND filter losses
    :param exposure_time: float. Camera exposure time in s
    :param lifetime: float. Polariton lifetime in s
    :param hopfield: float. Photon hopfield coefficient
    :param alpha: float. Electrons per count (from the manual)
    :param camera_qe: float. Quantum efficiency of the camera
    :param optical_losses: float. Loss from reflections at the optical elements from the microcavity to the camera
    :return: float
    """
    if optical_losses is None:
        optical_losses = 0.97  # at the objective
        optical_losses *= 0.97  # at the dichroic
        optical_losses *= 0.99  # at the bandpass filter
        optical_losses *= (0.97**2)  # estimated reflection from 2 mirrors
        optical_losses *= 0.3125  # directly measured up to the camera

    efficiency = nd_filter * camera_qe * optical_losses / 2  # factor of two from both directions in the mcav
    photon_flux = camera_count * alpha / (efficiency * exposure_time)
    polariton = photon_flux * lifetime / hopfield

    return polariton


def photons_per_mw(power, wavelength=0.805):
    """Returns the number of photons per picosecond, given a power and wavelength
    :param power: in watts
    :param wavelength: in micron
    :return:
    """
    single_photon_energy = 2 * np.pi * hbar * c / wavelength  # in meV
    power *= 6.242e21   # joules to meV
    power /= single_photon_energy  # number of photons / s
    return power * 1e-12  # number of photons / ps
