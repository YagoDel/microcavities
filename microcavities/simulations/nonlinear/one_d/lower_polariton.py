# -*- coding: utf-8 -*-

from microcavities.simulations.nonlinear.one_d import *
from microcavities.simulations.linear.one_d.lower_polariton import kinetic_matrix, polariton_mass


def farfield(hamiltonian, time_range, starting_vectors=None):
    """Calculates the farfield emission pattern of a one dimensional Hamiltonian

    Given a list of starting wavefunctions, evolves them in time using a Runge Kutta stepper and extracts the farfield
    emission of each wavefunction by Fourier transforming them in space and time

    :param hamiltonian: function. Takes one argument (time), returns a 2D matrix
    :param time_range: list of times to evaluate the Hamiltonian at
    :param starting_vectors: list of vectors, integer or None. If None, If an integer,
    :return:
    """
    n_points = hamiltonian(0, 0).shape[0]
    size = int(n_points)
    if starting_vectors is None:
        starting_vectors = [None]
    elif type(starting_vectors) == int:
        starting_vectors = [None] * starting_vectors

    rho = np.zeros((size, len(time_range)))
    wavefunctions = []
    for vec in tqdm(starting_vectors, 'farfield'):
        psi = solve_timerange(hamiltonian, time_range, vec)
        wavefunctions += [psi]
        photon_field = np.reshape(psi, (size, len(time_range)))

        psi_kw = np.fft.fftshift(np.fft.fft2(photon_field))
        rho += np.abs(psi_kw) ** 2
        if np.isnan(rho).any():
            break

    dE = hbar * np.pi / time_range[-1]
    theory_eax = (np.linspace(-dE, dE, len(time_range)) * len(time_range) / 2)[::-1]

    return rho, theory_eax  #, np.array(wavefunctions)


def test_uniform_condensation():
    def hamiltonian(t, psi, gain, loss, saturation, mass=1e-4, xax=None):
        try:
            len(psi)
        except TypeError:
            psi = np.full(len(xax), psi)

        T = kinetic_matrix(len(xax), mass, np.diff(xax)[0])

        gain = (gain - saturation * np.abs(psi)**2 / 2 - loss) / 2
        U = np.diag(1j * gain)
        return T + U, xax

    times = np.linspace(0, 80, 10001)
    xax = make_ax()
    kax = make_k_ax(xax)
    loss = 0.2
    saturation = 100
    threshold = loss * saturation
    gains = np.append(np.insert(np.linspace(0.5, 3, 6), 0, 0.1), 6) * threshold
    wavefunctions = []
    densities = []
    for gain in tqdm(gains):
        _ham = partial(hamiltonian, xax=xax, gain=gain, loss=loss, saturation=saturation)
        ham = lambda t, psi: _ham(t, psi)[0]
        wavefunctions += [solve_timerange(ham, times)]

        density, e_ax = farfield(ham, times)
        densities += [density]
    wavefunctions = np.array(wavefunctions)
    densities = np.array(densities)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(wspace=0.3))
    colors = ['C%d' % x for x in range(6)]
    [axs[0].semilogy(times, np.sum(np.abs(t) ** 2, 0), c=c) for t, c in zip(wavefunctions, colors)]
    label_axes(axs[0], 'Time [ps]', '%s$_x|%s(t)|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    axs[1].loglog(gains/threshold, np.sum(np.abs(wavefunctions[..., -1]) ** 2, 1), '.-')
    label_axes(axs[1], '$P/P_{th}$', '%s$_x|%s(t_{final})|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    fig = figure(figsize=(12, 8))
    b, a = square(len(gains))
    gs = gridspec.GridSpec(a, b, fig, hspace=0.02, wspace=0.02)
    axs = gs.subplots(sharex=True, sharey=True)
    for ax, d, g in zip(axs.flatten(), densities, gains/threshold):
        imshow(d.transpose(), ax, diverging=False, norm=LogNorm(), cbar=False, xaxis=kax, yaxis=e_ax)
        ax.set_ylim(-5, 10)
        ax.text(0.5, 0.99, '%.1f $P/P_{th}$' % g, transform=ax.transAxes, ha='center', va='top')
    label_grid(gs, '$k_x$ [%sm$^{-1}$]' % greek_alphabet['mu'], 'bottom', 0.04)
    label_grid(gs, 'E [meV]', 'left', 0.04)


def test_dissipative_edges():
    def hamiltonian(t, psi, gain, loss, saturation, mass=1e-4, xax=None):
        if xax is None: xax = make_ax()

        try:
            len(psi)
        except TypeError:
            psi = np.full(len(xax), psi)

        T = kinetic_matrix(len(xax), mass, np.diff(xax)[0])

        loss = dissipative_edges(loss, xax)

        gain = (gain / (saturation + np.abs(psi) ** 2) - loss) / 2
        U = np.diag(1j * gain)
        return T + U, xax

    xax = make_ax()
    starting_wavefunction = 1e-6 * np.exp(-(xax ** 2 / (2 * 0.5 ** 2)))
    times = np.linspace(0, 50, 5001)

    _ham1 = partial(hamiltonian, gain=0.01, loss=0.1, saturation=0.01)
    ham1 = lambda a, b: _ham1(a, b)[0]

    solved1 = solve_timerange(ham1, times, starting_wavefunction)

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    axs = axs.flatten()

    axs[0].plot(xax, np.imag(np.diag(ham1(0, 0))))
    label_axes(axs[0], 'x [%sm]' % greek_alphabet['mu'], 'Loss [meV]')

    axs[1].semilogy(times, np.sum(np.abs(solved1) ** 2, 0))
    label_axes(axs[1], 'Time [ps]', '%s$_x|%s(t)|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    axs[2].plot(xax, np.abs(solved1[:, -1])**2)
    label_axes(axs[2], 'x [%sm]' % greek_alphabet['mu'], '%s$_x|%s(t_{final})|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    imshow(np.abs(solved1) ** 2 + 1e-19, axs[3],
           diverging=False, norm=LogNorm(), xaxis=times, yaxis=xax, cbar=False, interpolation='none')
    label_axes(axs[3], 'time [ps]', 'x [%sm]' % greek_alphabet['mu'])

    fig.suptitle('Butterworth edges')

    def hamiltonian(t, psi, gain, loss, saturation, mass=1e-4, xax=None):
        if xax is None: xax = make_ax()

        try:
            len(psi)
        except TypeError:
            psi = np.full(len(xax), psi)

        T = kinetic_matrix(len(xax), mass, np.diff(xax)[0])

        loss = dissipative_edges(loss, xax, 'exp')

        gain = (gain / (saturation + np.abs(psi) ** 2) - loss) / 2
        U = np.diag(1j * gain)
        return T + U, xax

    xax = make_ax()
    starting_wavefunction = 1e-6 * np.exp(-(xax ** 2 / (2 * 0.5 ** 2)))
    times = np.linspace(0, 50, 5001)

    _ham1 = partial(hamiltonian, gain=0.01, loss=0.1, saturation=0.01)
    ham1 = lambda a, b: _ham1(a, b)[0]

    solved1 = solve_timerange(ham1, times, starting_wavefunction)

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    axs = axs.flatten()

    axs[0].plot(xax, np.imag(np.diag(ham1(0, 0))))
    label_axes(axs[0], 'x [%sm]' % greek_alphabet['mu'], 'Loss [meV]')

    axs[1].semilogy(times, np.sum(np.abs(solved1) ** 2, 0))
    label_axes(axs[1], 'Time [ps]', '%s$_x|%s(t)|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    axs[2].plot(xax, np.abs(solved1[:, -1]) ** 2)
    label_axes(axs[2], 'x [%sm]' % greek_alphabet['mu'],
               '%s$_x|%s(t_{final})|^2$' % (greek_alphabet['Sigma'], greek_alphabet['psi']))

    imshow(np.abs(solved1) ** 2 + 1e-19, axs[3],
           diverging=False, norm=LogNorm(), xaxis=times, yaxis=xax, cbar=False, interpolation='none')
    label_axes(axs[3], 'time [ps]', 'x [%sm]' % greek_alphabet['mu'])

    fig.suptitle('Exponential edges')
