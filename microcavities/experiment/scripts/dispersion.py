# -*- coding: utf-8 -*-

from microcavities.analysis.analysis_functions import dispersion, find_k0
from microcavities.experiment.utils import spectrometer_calibration, magnification

wvl = spectrometer.wavelength
dispersion_img = pvcam.raw_image()
bkg = pvcam.background
try:
    dispersion_img -= bkg
except Exception as e:
    print 'Failed to background image. Should not matter'

wvls = spectrometer_calibration(wavelength=wvl)
energy_axis = 1240 / wvls

mag = magnification([0.01, 0.25, 0.1, 0.1, 0.2])[0]
k0 = int(find_k0(dispersion_img))
k_axis = np.linspace(-200, 200, 400)
k_axis -= -200+k0
k_axis *= 20 * 1e-6 / mag  # Converting to SI and dividing by magnification
k_axis *= 1e-6  # converting to inverse micron

results, args, kwargs = dispersion(dispersion_img, k_axis, energy_axis, True)

energy = results[0]
lifetime = results[1]
mass = results[2]

print "Energy: %g" % results[0]
print "Lifetime: %g" % results[1]
print "Mass: %g" % results[2]
