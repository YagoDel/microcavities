# Analysis

## Dispersion analysis
Utility code to analyse low-power dispersion PL images of polaritons. Main result is the following figure, which is the
result of [dispersion](../microcavities/analysis/dispersion.py).dispersion, and calculates the lower polariton mass, 
lifetime, energy and (if possible) exciton fraction:

<img src="figures/analysis_dispersion.png" width="500">


## Condensation analysis
Utility code to analyse the condensation of polaritons. Relies on having taken a power series scan using 
[HierarchicalScan](../microcavities/utils/HierarchicalScan.py).ExperimentScan with YAML files. 
Main result is the following figure, showing a few dispersion images, and three different quantities vs power: the k~0
spectra, the total/maximum emission, and the momentum distribution:

<img src="figures/analysis_condensation.png" width="500">
