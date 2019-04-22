# -*- coding: utf-8 -*-

Given an initial file, containing parameters (either single values, ranges or lists),
the type of equations to be used (by name, spinor/reservoir...), and the type of
solver wanted (by name, RK4/ABM/splitstep), the run file should be able to plot
simulations dynamically (if wanted), run C/CUDA-kernels for speed, run different
parameters in parallel, and save everything in HDF5.
    Example 1:
        equations: scalar_reservoir

        parameters:
            mass: 1
            nonlinearity: 0.01
            feedrate: 0.1
            loss: 0.2
            power:
                min: 0.1
                max: 0.5
                steps: 10
            reservoir_loss: [0.1, 0.3, 0.5]
            repetitions: 10

        solver:
            name: RK4
            stepsize: 0.1
            steps: 10000

        options:
            parallelize_parameters: True
            plot: False

    Example 2:
        equations: spinor_RK4_C  # this is a C-kernel

        parameters:
            mass: 1
            alpha1: 0.01
            alpha2: -0.001
            loss: 0.2
            splitting: 0.05
            power:
                min: 0.1
                max: 0.5
                steps: 10

        solver:
            stepsize: 0.1
            steps: 10000

        options:
            parallelize_parameters: True
            plot: False


Then we should have an Analysis class able to take this HDF5 and plot the
multidimensional datasets. Again, we should give it a yaml that specifies which
quantities to plot (spin, intensity)
