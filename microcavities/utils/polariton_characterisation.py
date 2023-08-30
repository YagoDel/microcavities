# -*- coding: utf-8 -*-

import numpy as np


def hopfield_coefficients(rabi_splitting=None, detuning=None, exciton_energy=None, photon_energy=None,
                          exciton_mass=None, photon_mass=None, polariton_mass=None):
    if rabi_splitting is not None:
        if detuning is None:
            detuning = exciton_energy - photon_energy
        exciton_fraction = 0.5 * (1 + detuning / np.sqrt(detuning**2 + rabi_splitting**2))
    elif exciton_mass is not None:
        exciton_fraction = (exciton_mass * (photon_mass - polariton_mass)) / (polariton_mass * (photon_mass - exciton_mass))
    else:
        raise ValueError('Need to give either energies or masses')
    photon_fraction = 1 - exciton_fraction
    return exciton_fraction, photon_fraction


def exciton_binding_energy():
    # 10meV
    return 10


def exciton_bohr_radius():
    # 10nm, in micron
    return 10e-3


def exciton_interaction():
    # In meV micron**2
    return 6 * exciton_binding_energy() * (exciton_bohr_radius() ** 2)


def polariton_interaction(rabi_splitting, detuning, number_of_quantum_wells):
    x, _ = hopfield_coefficients(rabi_splitting, detuning)

    return (x**2) * exciton_interaction() / (2 * number_of_quantum_wells)


def polariton_blueshift(density, interaction=None, interaction_args=None):
    if interaction is None:
        interaction = polariton_interaction(*interaction_args)

    return density * interaction


def mott_transition():
    # In micron**-2
    return 1 / (np.pi*exciton_bohr_radius()**2)
