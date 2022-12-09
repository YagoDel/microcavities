# Simulations
Useful numerical procedures:
- [Transfer matrix simulations](#transfer-matrix-method) 
- [Linear Schrodinger solver](#linear-schrdinger-solver)

## Transfer Matrix Method
Utility functions for microcavity design using [tmm](https://github.com/sbyrnes321/tmm) to solve the transfer matrix 
equations. Cavity parameters can be given as a yaml, and the normal incidence reflection, angular dispersion, and field 
distribution inside the cavity can be easily and quickly calculated and plotted.

![](figures/tmm_normalincidence.png)

![](figures/tmm_dispersion.png)

![](figures/tmm_fielddistribution.png)


## Linear Schrödinger solver
Direct diagonalisation solvers for polariton systems in one and two dimensions.

As a 1D example, you can get the farfield emission pattern from a harmonic well that modifies the photon component of the 
polariton:
```
from microcavities.simulation.linear.one_d.realspace import *
test_farfield_harmonic_potential()
```
![](figures/simulations_linear_1D_QHO.png)

As a 2D example, you can get the free-space dispersion relation from farfield emission of realspace wavefunctions and
compare it to analytical results (it'll take ~5min to run):
```
from microcavities.simulations.linear.polariton_realspace import *
test_hamiltonian_x()
```
![](figures/simulations_linear_2d_freespace.png)
