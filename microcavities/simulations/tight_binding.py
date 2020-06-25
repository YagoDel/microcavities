# -*- coding: utf-8 -*-

import numpy as np
import sys  # for exiting
import copy  # for deepcopying
import itertools
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.patches import Polygon
from fractions import Fraction
from scipy.stats import binned_statistic
from pythtb import tb_model


class tb_model_NH(tb_model):
    def __init__(self, *args, **kwargs):
        super(tb_model_NH, self).__init__(*args, **kwargs)
        self._nh_hoppings = []

    def set_nh_hop(self, hop_amp, ind_i, ind_j, ind_R=None, mode="set", allow_conjugate_pair=False):
        r"""

        Defines hopping parameters between tight-binding orbitals. In
        the notation used in section 3.1 equation 3.6 of
        :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>` this function specifies the
        following object

        .. math::

          H_{ij}({\bf R})= \langle \phi_{{\bf 0} i}  \vert H  \vert \phi_{{\bf R},j} \rangle

        Where :math:`\langle \phi_{{\bf 0} i} \vert` is i-th
        tight-binding orbital in the home unit cell and
        :math:`\vert \phi_{{\bf R},j} \rangle` is j-th tight-binding orbital in
        unit cell shifted by lattice vector :math:`{\bf R}`. :math:`H`
        is the Hamiltonian.

        (Strictly speaking, this term specifies hopping amplitude
        for hopping from site *j+R* to site *i*, not vice-versa.)

        Hopping in the opposite direction is automatically included by
        the code since

        .. math::

          H_{ji}(-{\bf R})= \left[ H_{ij}({\bf R}) \right]^{*}

        .. warning::

           There is no need to specify hoppings in both :math:`i
           \rightarrow j+R` direction and opposite :math:`j
           \rightarrow i-R` direction since that is done
           automatically. If you want to specifiy hoppings in both
           directions, see description of parameter
           *allow_conjugate_pair*.

        .. warning:: In previous version of PythTB this function was
          called *add_hop*. For backwards compatibility one can still
          use that name but that feature will be removed in future
          releases.

        :param hop_amp: Hopping amplitude; can be real or complex
          number, equals :math:`H_{ij}({\bf R})`. If *nspin* is *2*
          then hopping amplitude can be given either as a single
          number, or as an array of four numbers, or as 2x2 matrix. If
          a single number is given, it is interpreted as hopping
          amplitude for both up and down spin component.  If an array
          of four numbers is given, these are the coefficients of I,
          sigma_x, sigma_y, and sigma_z (that is, the 2x2 identity and
          the three Pauli spin matrices) respectively. Finally, full
          2x2 matrix can be given as well.

        :param ind_i: Index of bra orbital from the bracket :math:`\langle
          \phi_{{\bf 0} i} \vert H \vert \phi_{{\bf R},j} \rangle`. This
          orbital is assumed to be in the home unit cell.

        :param ind_j: Index of ket orbital from the bracket :math:`\langle
          \phi_{{\bf 0} i} \vert H \vert \phi_{{\bf R},j} \rangle`. This
          orbital does not have to be in the home unit cell; its unit cell
          position is determined by parameter *ind_R*.

        :param ind_R: Lattice vector (integer array, in reduced
          coordinates) pointing to the unit cell where the ket
          orbital is located.  The number of coordinates must equal
          the dimensionality in real space (*dim_r* parameter) for
          consistency, but only the periodic directions of ind_R are
          used. If reciprocal space is zero-dimensional (as in a
          molecule), this parameter does not need to be specified.

        :param mode: Similar to parameter *mode* in function *set_onsite*.
          Speficies way in which parameter *hop_amp* is
          used. It can either set value of hopping term from scratch,
          reset it, or add to it.

          * "set" -- Default value. Hopping term is set to value of
            *hop_amp* parameter. One can use "set" for each triplet of
            *ind_i*, *ind_j*, *ind_R* only once.

          * "reset" -- Specifies on-site energy to given value. This
            function can be called multiple times for the same triplet
            *ind_i*, *ind_j*, *ind_R*.

          * "add" -- Adds to the previous value of hopping term This
            function can be called multiple times for the same triplet
            *ind_i*, *ind_j*, *ind_R*.

          If *set_hop* was ever called with *allow_conjugate_pair* set
          to True, then it is possible that user has specified both
          :math:`i \rightarrow j+R` and conjugate pair :math:`j
          \rightarrow i-R`.  In this case, "set", "reset", and "add"
          parameters will treat triplet *ind_i*, *ind_j*, *ind_R* and
          conjugate triplet *ind_j*, *ind_i*, *-ind_R* as distinct.

        :param allow_conjugate_pair: Default value is *False*. If set
          to *True* code will allow user to specify hopping
          :math:`i \rightarrow j+R` even if conjugate-pair hopping
          :math:`j \rightarrow i-R` has been
          specified. If both terms are specified, code will
          still count each term two times.

        Example usage::

          # Specifies complex hopping amplitude between first orbital in home
          # unit cell and third orbital in neigbouring unit cell.
          tb.set_hop(0.3+0.4j, 0, 2, [0, 1])
          # change value of this hopping
          tb.set_hop(0.1+0.2j, 0, 2, [0, 1], mode="reset")
          # add to previous value (after this function call below,
          # hopping term amplitude is 100.1+0.2j)
          tb.set_hop(100.0, 0, 2, [0, 1], mode="add")

        """
        #
        if self._dim_k != 0 and (ind_R is None):
            raise Exception("\n\nNeed to specify ind_R!")
        # if necessary convert from integer to array
        if self._dim_k == 1 and type(ind_R).__name__ == 'int':
            tmpR = np.zeros(self._dim_r, dtype=int)
            tmpR[self._per] = ind_R
            ind_R = tmpR
        # check length of ind_R
        if self._dim_k != 0:
            if len(ind_R) != self._dim_r:
                raise Exception("\n\nLength of input ind_R vector must equal dim_r! Even if dim_k<dim_r.")
        # make sure ind_i and ind_j are not out of scope
        if ind_i < 0 or ind_i >= self._norb:
            raise Exception("\n\nIndex ind_i out of scope.")
        if ind_j < 0 or ind_j >= self._norb:
            raise Exception("\n\nIndex ind_j out of scope.")
            # do not allow onsite hoppings to be specified here because then they
        # will be double-counted
        if self._dim_k == 0:
            if ind_i == ind_j:
                raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        else:
            if ind_i == ind_j:
                all_zer = True
                for k in self._per:
                    if int(ind_R[k]) != 0:
                        all_zer = False
                if all_zer == True:
                    raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        #
        # make sure that if <i|H|j+R> is specified that <j|H|i-R> is not!
        if allow_conjugate_pair == False:
            for h in self._nh_hoppings:
                if ind_i == h[2] and ind_j == h[1]:
                    if self._dim_k == 0:
                        raise Exception( \
                            """\n
                            Following matrix element was already implicitely specified:
                               i=""" + str(ind_i) + " j=" + str(ind_j) + """
    Remember, specifying <i|H|j> automatically specifies <j|H|i>.  For
    consistency, specify all hoppings for a given bond in the same
    direction.  (Or, alternatively, see the documentation on the
    'allow_conjugate_pair' flag.)
    """)
                    elif False not in (np.array(ind_R)[self._per] == (-1) * np.array(h[3])[self._per]):
                        raise Exception( \
                            """\n
                            Following matrix element was already implicitely specified:
                               i=""" + str(ind_i) + " j=" + str(ind_j) + " R=" + str(ind_R) + """
    Remember,specifying <i|H|j+R> automatically specifies <j|H|i-R>.  For
    consistency, specify all hoppings for a given bond in the same
    direction.  (Or, alternatively, see the documentation on the
    'allow_conjugate_pair' flag.)
    """)
        # convert to 2by2 matrix if needed
        hop_use = self._val_to_block(hop_amp)
        # hopping term parameters to be stored
        if self._dim_k == 0:
            new_hop = [hop_use, int(ind_i), int(ind_j)]
        else:
            new_hop = [hop_use, int(ind_i), int(ind_j), np.array(ind_R)]
        #
        # see if there is a hopping term with same i,j,R
        use_index = None
        for iih, h in enumerate(self._nh_hoppings):
            # check if the same
            same_ijR = False
            if ind_i == h[1] and ind_j == h[2]:
                if self._dim_k == 0:
                    same_ijR = True
                else:
                    if False not in (np.array(ind_R)[self._per] == np.array(h[3])[self._per]):
                        same_ijR = True
            # if they are the same then store index of site at which they are the same
            if same_ijR == True:
                use_index = iih
        #
        # specifying hopping terms from scratch, can be called only once
        if mode.lower() == "set":
            # make sure we specify things only once
            if use_index != None:
                raise Exception(
                    "\n\nHopping energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
            else:
                self._nh_hoppings.append(new_hop)
        # reset value of hopping term, without adding to previous value
        elif mode.lower() == "reset":
            if use_index != None:
                self._nh_hoppings[use_index] = new_hop
            else:
                self._nh_hoppings.append(new_hop)
        # add to previous value
        elif mode.lower() == "add":
            if use_index != None:
                self._nh_hoppings[use_index][0] += new_hop[0]
            else:
                self._nh_hoppings.append(new_hop)
        else:
            raise Exception("\n\nWrong value of mode parameter")

    def _gen_ham(self, k_input=None):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt = np.array(k_input)
        if not (k_input is None):
            # if kpnt is just a number then convert it to an array
            if len(kpnt.shape) == 0:
                kpnt = np.array([kpnt])
            # check that k-vector is of corect size
            if kpnt.shape != (self._dim_k,):
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self._dim_k != 0:
                raise Exception("\n\nHave to provide a k-vector!")
        # zero the Hamiltonian matrix
        if self._nspin == 1:
            ham = np.zeros((self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
            ham = np.zeros((self._norb, 2, self._norb, 2), dtype=complex)
        # modify diagonal elements
        for i in range(self._norb):
            if self._nspin == 1:
                ham[i, i] = self._site_energies[i]
            elif self._nspin == 2:
                ham[i, :, i, :] = self._site_energies[i]
        # go over all hoppings
        for hopping in self._hoppings:
            # get all data for the hopping parameter
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)
            i = hopping[1]
            j = hopping[2]
            # in 0-dim case there is no phase factor
            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)
                # vector from one site to another
                rv = -self._orb[i, :] + self._orb[j, :] + ind_R
                # Take only components of vector which are periodic
                rv = rv[self._per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase = np.exp((2.0j) * np.pi * np.dot(kpnt, rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate
            if self._nspin == 1:
                ham[i, j] += amp
                ham[j, i] += amp.conjugate()
            elif self._nspin == 2:
                ham[i, :, j, :] += amp
                ham[j, :, i, :] += amp.T.conjugate()
        for hopping in self._nh_hoppings:
            # get all data for the hopping parameter
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)
            i = hopping[1]
            j = hopping[2]
            # in 0-dim case there is no phase factor
            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)
                # vector from one site to another
                rv = -self._orb[i, :] + self._orb[j, :] + ind_R
                # Take only components of vector which are periodic
                rv = rv[self._per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase = np.exp((2.0j) * np.pi * np.dot(kpnt, rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate
            if self._nspin == 1:
                ham[i, j] += amp
                ham[j, i] += amp
            elif self._nspin == 2:
                ham[i, :, j, :] += amp
                ham[j, :, i, :] += amp.T
        return ham

    def _sol_ham(self, ham, eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        if self._nspin == 1:
            ham_use = ham
        elif self._nspin == 2:
            ham_use = ham.reshape((2 * self._norb, 2 * self._norb))
        # # check that matrix is hermitian
        # if np.max(ham_use - ham_use.T.conj()) > 1.0E-9:
        #     raise Exception("\n\nHamiltonian matrix is not hermitian?!")

        # solve matrix
        if eig_vectors == False:  # only find eigenvalues
            eval = np.linalg.eigvals(ham_use)
            # sort eigenvalues and convert to real numbers
            eval = _nicefy_eig_nh(eval)
            return eval
        else:  # find eigenvalues and eigenvectors
            (eval, eig) = np.linalg.eig(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig = eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval, eig) = _nicefy_eig_nh(eval, eig)
            # reshape eigenvectors if doing a spinfull calculation
            if self._nspin == 2:
                eig = eig.reshape((self._nsta, self._norb, 2))
            return (eval, eig)

    def solve_all(self, k_list=None, eig_vectors=False):
        r"""
        Solves for eigenvalues and (optionally) eigenvectors of the
        tight-binding model on a given one-dimensional list of k-vectors.

        .. note::

           Eigenvectors (wavefunctions) returned by this
           function and used throughout the code are exclusively given
           in convention 1 as described in section 3.1 of
           :download:`notes on tight-binding formalism
           <misc/pythtb-formalism.pdf>`.  In other words, they
           are in correspondence with cell-periodic functions
           :math:`u_{n {\bf k}} ({\bf r})` not
           :math:`\Psi_{n {\bf k}} ({\bf r})`.

        .. note::

           In some cases class :class:`pythtb.wf_array` provides a more
           elegant way to deal with eigensolutions on a regular mesh of
           k-vectors.

        :param k_list: One-dimensional array of k-vectors. Each k-vector
          is given in reduced coordinates of the reciprocal space unit
          cell. For example, for real space unit cell vectors [1.0,0.0]
          and [0.0,2.0] and associated reciprocal space unit vectors
          [2.0*pi,0.0] and [0.0,pi], k-vector with reduced coordinates
          [0.25,0.25] corresponds to k-vector [0.5*pi,0.25*pi].
          Dimensionality of each vector must equal to the number of
          periodic directions (i.e. dimensionality of reciprocal space,
          *dim_k*).
          This parameter shouldn't be specified for system with
          zero-dimensional k-space (*dim_k* =0).

        :param eig_vectors: Optional boolean parameter, specifying whether
          eigenvectors should be returned. If *eig_vectors* is True, then
          both eigenvalues and eigenvectors are returned, otherwise only
          eigenvalues are returned.

        :returns:
          * **eval** -- Two dimensional array of eigenvalues for
            all bands for all kpoints. Format is eval[band,kpoint] where
            first index (band) corresponds to the electron band in
            question and second index (kpoint) corresponds to the k-point
            as listed in the input parameter *k_list*. Eigenvalues are
            sorted from smallest to largest at each k-point seperately.

            In the case when reciprocal space is zero-dimensional (as in a
            molecule) kpoint index is dropped and *eval* is of the format
            eval[band].

          * **evec** -- Three dimensional array of eigenvectors for
            all bands and all kpoints. If *nspin* equals 1 the format
            of *evec* is evec[band,kpoint,orbital] where "band" is the
            electron band in question, "kpoint" is index of k-vector
            as given in input parameter *k_list*. Finally, "orbital"
            refers to the tight-binding orbital basis function.
            Ordering of bands is the same as in *eval*.

            Eigenvectors evec[n,k,j] correspond to :math:`C^{n {\bf
            k}}_{j}` from section 3.1 equation 3.5 and 3.7 of the
            :download:`notes on tight-binding formalism
            <misc/pythtb-formalism.pdf>`.

            In the case when reciprocal space is zero-dimensional (as in a
            molecule) kpoint index is dropped and *evec* is of the format
            evec[band,orbital].

            In the spinfull calculation (*nspin* equals 2) evec has
            additional component evec[...,spin] corresponding to the
            spin component of the wavefunction.

        Example usage::

          # Returns eigenvalues for three k-vectors
          eval = tb.solve_all([[0.0, 0.0], [0.0, 0.2], [0.0, 0.5]])
          # Returns eigenvalues and eigenvectors for two k-vectors
          (eval, evec) = tb.solve_all([[0.0, 0.0], [0.0, 0.2]], eig_vectors=True)

        """
        # if not 0-dim case
        if not (k_list is None):
            nkp = len(k_list)  # number of k points
            # first initialize matrices for all return data
            #    indices are [band,kpoint]
            ret_eval = np.zeros((self._nsta, nkp), dtype=complex)
            #    indices are [band,kpoint,orbital,spin]
            if self._nspin == 1:
                ret_evec = np.zeros((self._nsta, nkp, self._norb), dtype=complex)
            elif self._nspin == 2:
                ret_evec = np.zeros((self._nsta, nkp, self._norb, 2), dtype=complex)
            # go over all kpoints
            for i, k in enumerate(k_list):
                # generate Hamiltonian at that point
                ham = self._gen_ham(k)
                # solve Hamiltonian
                if eig_vectors == False:
                    eval = self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eval[:, i] = eval[:]
                else:
                    (eval, evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eval[:, i] = eval[:]
                    if self._nspin == 1:
                        ret_evec[:, i, :] = evec[:, :]
                    elif self._nspin == 2:
                        ret_evec[:, i, :, :] = evec[:, :, :]
            # return stuff
            if eig_vectors == False:
                # indices of eval are [band,kpoint]
                return ret_eval
            else:
                # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
                return (ret_eval, ret_evec)
        else:  # 0 dim case
            # generate Hamiltonian
            ham = self._gen_ham()
            # solve
            if eig_vectors == False:
                eval = self._sol_ham(ham, eig_vectors=eig_vectors)
                # indices of eval are [band]
                return eval
            else:
                (eval, evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
                # indices of eval are [band] and of evec are [band,orbital,spin]
                return (eval, evec)

    def plot_eigen_surface(self, krange=None, kpoints=21, bands=None, fig_ax=None, continuity_threshold=0.01):
        if krange is None:
            krange = [[-0.5, 0.5], [-0.5, 0.5]]
        kx = np.linspace(krange[0][0], krange[0][1], kpoints)
        ky = np.linspace(krange[1][0], krange[1][1], kpoints)
        klist = list(itertools.product(kx, ky))
        sol = self.solve_all(klist)
        reciprocal = np.linalg.inv(self._lat).T
        reciprocal /= np.linalg.norm(reciprocal, axis=0)
        cartesian_k_points = np.dot(reciprocal, np.array(klist).T)
        # Kx, Ky = np.meshgrid(kx, ky)

        if fig_ax is None:
            if len(self._nh_hoppings) > 0:
                fig = plt.figure(figsize=(8, 4))
                gs = gridspec.GridSpec(1, 2)
                ax1 = plt.subplot(gs[0], projection='3d')
                ax2 = plt.subplot(gs[1], projection='3d')
            else:
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(1, 1)
                ax1 = plt.subplot(gs[0], projection='3d')
                ax2 = None
        else:
            if len(self._nh_hoppings) > 0:
                fig, ax1, ax2 = fig_ax
            else:
                fig, ax1 = fig_ax
                ax2 = None
        if bands is None:
            bands = [0, self.get_num_orbitals()]
        elif type(bands) == int:
            bands = [0, bands]
        for band in range(*bands):
            energies = sol[band].real #np.reshape(sol[band].real, (kpoints, kpoints))
            lifetimes = sol[band].imag # np.reshape(sol[band].imag, (kpoints, kpoints))
            # grad = np.sum(np.array(np.gradient(lifetimes)), 0)
            # mask = np.abs(grad) > continuity_threshold
            # energies[mask] = np.inf
            # lifetimes[mask] = np.inf
            # ax1.plot_surface(Kx, Ky, energies)
            ax1.plot_trisurf(cartesian_k_points[0], cartesian_k_points[1], energies)
            if ax2 is not None:
                # ax2.plot_surface(Kx, Ky, lifetimes)
                ax2.plot_trisurf(cartesian_k_points[0], cartesian_k_points[1], lifetimes)

        ax1.set_xlabel('k$_x$')
        ax1.set_ylabel('k$_y$')
        ax1.set_zlabel('Energy')
        if ax2 is not None:
            ax2.set_xlabel('k$_x$')
            ax2.set_ylabel('k$_y$')
            ax2.set_zlabel('Lifetime')
        return fig, (ax1, ax2)

    def plot_bands(self, angle=0, krange=None, kpoints=101, bands=None, fig_ax=None, plot_lines=False):
        if krange is None:
            krange = [[-0.5, 0.5], [-0.5, 0.5]]
        krange = np.array(krange)
        kx = np.linspace(krange[0, 0], krange[0, 1], kpoints)
        ky = np.linspace(krange[1, 0], krange[1, 1], kpoints)
        klist = list(itertools.product(kx, ky))
        sol = self.solve_all(klist)
        reciprocal = np.linalg.inv(self._lat).T
        cartesian_k_points = np.dot(reciprocal, np.array(klist).T).T
        # print(np.min(cartesian_k_points, axis=0))
        # print(np.max(cartesian_k_points, axis=0))

        direction = np.array([np.cos(angle), np.sin(angle)])
        projected_k = np.sum(cartesian_k_points * direction, 1)
        # print(np.min(projected_k, axis=0))
        # print(np.max(projected_k, axis=0))
        # k = np.linspace(np.sum(krange[:, 0] * direction), np.sum(krange[:, 1] * direction), kpoints)
        # k = np.linspace(np.min(projected_k), np.max(projected_k), kpoints)
        # print(np.min(k), np.max(k))
        if fig_ax is None:
            if len(self._nh_hoppings) > 0:
                fig = plt.figure(figsize=(8, 4))
                gs = gridspec.GridSpec(1, 2)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
            else:
                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(1, 1)
                ax1 = plt.subplot(gs[0])
                ax2 = None
        else:
            if len(self._nh_hoppings) > 0:
                fig, ax1, ax2 = fig_ax
            else:
                fig, ax1 = fig_ax
                ax2 = None
        if bands is None:
            bands = [0, self.get_num_orbitals()]
        for band in range(*bands):
            energies = sol[band].real
            lifetimes = sol[band].imag
            min_band, bin_edges1, _ = binned_statistic(projected_k, energies, 'min', kpoints, (np.min(projected_k), np.max(projected_k)))  # np.min(energies, axis)
            max_band, bin_edges2, _ = binned_statistic(projected_k, energies, 'max', kpoints, (np.min(projected_k), np.max(projected_k)))  # np.max(energies, axis)
            bincenter1 = np.mean([bin_edges1, np.roll(bin_edges1, 1)], 0)[1:]
            bincenter2 = np.mean([bin_edges2, np.roll(bin_edges2, 1)], 0)[1:]
            k = np.mean([bincenter1, bincenter2], 0)
            ax1.fill_between(k, min_band, max_band, alpha=0.5)
            if plot_lines:
                ax1.plot(projected_k, energies)
            if ax2 is not None:
                min_band = binned_statistic(projected_k, lifetimes, 'min', kpoints, (np.min(k), np.max(k)))  # np.min(energies, axis)
                max_band = binned_statistic(projected_k, lifetimes, 'max', kpoints, (np.min(k), np.max(k)))  # np.max(energies, axis)
                ax2.fill_between(k, min_band, max_band, alpha=0.5)
        return fig, (ax1, ax2)

    def plot_main_directions(self, path=None, labels=None, k_points=101, bands=None, fig_ax=None, plot_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = dict()

        if path is None:
            path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
        if labels is None:
            labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
        assert len(labels) == len(path)

        if bands is None:
            bands = self.get_num_orbitals()
        (k_vec, k_dist, k_node) = self.k_path(path, k_points, report=False)
        _idx = np.argmin(np.sum(k_vec**2, 1))  # finds the index of the origin. TODO: this fails if the origin is not in the path
        k_ax = k_dist - k_dist[_idx]
        k_node -= k_dist[_idx]
        evals = self.solve_all(k_vec)
        if fig_ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig, ax = fig_ax
        # ax.set_xlim(k_node[0], k_node[-1])
        ax.set_xticks(k_node)
        ax.set_xticklabels(labels)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k')
        for idx in range(bands):
            ax.plot(k_ax, evals[idx], **plot_kwargs)
        ax.set_ylabel('Energy')
        ax.set_xlabel('Momentum')
        return fig, ax

# NEED TO FIND THE CONNECTION BETWEEN REAL SPACE DIRECTION (WHICH IS WHAT WE CUT), AND RECIPROCAL SPACE DIRECTION (WHICH IS WHAT WE CALCULATE)

    def cut_piece(self, num, fin_dir, glue_edgs=False):
        r"""
        Constructs a (d-1)-dimensional tight-binding model out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. The real-space
        lattice vectors of the returned model are the same as those of
        the original model; only the dimensionality of reciprocal space
        is reduced.

        :param num: How many times to repeat the unit cell.

        :param fin_dir: Index of the real space lattice vector along
          which you no longer wish to maintain periodicity.

        :param glue_edgs: Optional boolean parameter specifying whether to
          allow hoppings from one edge to the other of a cut model.

        :returns:
          * **fin_model** -- Object of type
            :class:`pythtb.tb_model` representing a cutout
            tight-binding model. Orbitals in *fin_model* are
            numbered so that the i-th orbital of the n-th unit
            cell has index i+norb*n (here norb is the number of
            orbitals in the original model).

        Example usage::

          A = tb_model(3, 3, ...)
          # Construct two-dimensional model B out of three-dimensional
          # model A by repeating model along second lattice vector ten times
          B = A.cut_piece(10, 1)
          # Further cut two-dimensional model B into one-dimensional model
          # A by repeating unit cell twenty times along third lattice
          # vector and allow hoppings from one edge to the other
          C = B.cut_piece(20, 2, glue_edgs=True)

        See also these examples: :ref:`haldane_fin-example`,
        :ref:`edge-example`.


        """
        if self._dim_k == 0:
            raise Exception("\n\nModel is already finite")
        if type(num).__name__ != 'int':
            raise Exception("\n\nArgument num not an integer")

        # check value of num
        if num < 1:
            raise Exception("\n\nArgument num must be positive!")
        if num == 1 and glue_edgs == True:
            raise Exception("\n\nCan't have num==1 and glueing of the edges!")

        # generate orbitals of a finite model
        fin_orb = []
        onsite = []  # store also onsite energies
        for i in range(num):  # go over all cells in finite direction
            for j in range(self._norb):  # go over all orbitals in one cell
                # make a copy of j-th orbital
                orb_tmp = np.copy(self._orb[j, :])
                # change coordinate along finite direction
                orb_tmp[fin_dir] += float(i)
                # add to the list
                fin_orb.append(orb_tmp)
                # do the onsite energies at the same time
                onsite.append(self._site_energies[j])
        onsite = np.array(onsite)
        fin_orb = np.array(fin_orb)

        # generate periodic directions of a finite model
        fin_per = copy.deepcopy(self._per)
        # find if list of periodic directions contains the one you
        # want to make finite
        if fin_per.count(fin_dir) != 1:
            raise Exception("\n\nCan not make model finite along this direction!")
        # remove index which is no longer periodic
        fin_per.remove(fin_dir)

        # generate object of tb_model type that will correspond to a cutout
        fin_model = tb_model_NH(self._dim_k - 1,
                             self._dim_r,
                             copy.deepcopy(self._lat),
                             fin_orb,
                             fin_per,
                             self._nspin)

        # remember if came from w90
        fin_model._assume_position_operator_diagonal = self._assume_position_operator_diagonal

        # now put all onsite terms for the finite model
        fin_model.set_onsite(onsite, mode="reset")

        # put all hopping terms
        for c in range(num):  # go over all cells in finite direction
            for h in range(len(self._hoppings)):  # go over all hoppings in one cell
                # amplitude of the hop is the same
                amp = self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._hoppings[h][3])
                jump_fin = ind_R[fin_dir]  # store by how many cells is the hopping in finite direction
                if fin_model._dim_k != 0:
                    ind_R[fin_dir] = 0  # one of the directions now becomes finite

                # index of "from" and "to" hopping indices
                hi = self._hoppings[h][1] + c * self._norb
                #   have to compensate  for the fact that ind_R in finite direction
                #   will not be used in the finite model
                hj = self._hoppings[h][2] + (c + jump_fin) * self._norb

                # decide whether this hopping should be added or not
                to_add = True
                # if edges are not glued then neglect all jumps that spill out
                if glue_edgs == False:
                    if hj < 0 or hj >= self._norb * num:
                        to_add = False
                # if edges are glued then do mod division to wrap up the hopping
                else:
                    hj = int(hj) % int(self._norb * num)

                # add hopping to a finite model
                if to_add == True:
                    if fin_model._dim_k == 0:
                        fin_model.set_hop(amp, hi, hj, mode="add", allow_conjugate_pair=True)
                    else:
                        fin_model.set_hop(amp, hi, hj, ind_R, mode="add", allow_conjugate_pair=True)
            for h in range(len(self._nh_hoppings)):  # go over all hoppings in one cell
                # amplitude of the hop is the same
                amp = self._nh_hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._nh_hoppings[h][3])
                jump_fin = ind_R[fin_dir]  # store by how many cells is the hopping in finite direction
                if fin_model._dim_k != 0:
                    ind_R[fin_dir] = 0  # one of the directions now becomes finite

                # index of "from" and "to" hopping indices
                hi = self._nh_hoppings[h][1] + c * self._norb
                #   have to compensate  for the fact that ind_R in finite direction
                #   will not be used in the finite model
                hj = self._nh_hoppings[h][2] + (c + jump_fin) * self._norb

                # decide whether this hopping should be added or not
                to_add = True
                # if edges are not glued then neglect all jumps that spill out
                if glue_edgs == False:
                    if hj < 0 or hj >= self._norb * num:
                        to_add = False
                # if edges are glued then do mod division to wrap up the hopping
                else:
                    hj = int(hj) % int(self._norb * num)

                # add hopping to a finite model
                if to_add == True:
                    if fin_model._dim_k == 0:
                        fin_model.set_nh_hop(amp, hi, hj, mode="add", allow_conjugate_pair=True)
                    else:
                        fin_model.set_nh_hop(amp, hi, hj, ind_R, mode="add", allow_conjugate_pair=True)

        return fin_model

    def make_supercell(self, sc_red_lat, return_sc_vectors=False, to_home=True):
        r"""

        Returns tight-binding model :class:`pythtb.tb_model`
        representing a super-cell of a current object. This function
        can be used together with *cut_piece* in order to create slabs
        with arbitrary surfaces.

        By default all orbitals will be shifted to the home cell after
        unit cell has been created. That way all orbitals will have
        reduced coordinates between 0 and 1. If you wish to avoid this
        behavior, you need to set, *to_home* argument to *False*.

        :param sc_red_lat: Array of integers with size *dim_r*dim_r*
          defining a super-cell lattice vectors in terms of reduced
          coordinates of the original tight-binding model. First index
          in the array specifies super-cell vector, while second index
          specifies coordinate of that super-cell vector.  If
          *dim_k<dim_r* then still need to specify full array with
          size *dim_r*dim_r* for consistency, but non-periodic
          directions must have 0 on off-diagonal elemets s and 1 on
          diagonal.

        :param return_sc_vectors: Optional parameter. Default value is
          *False*. If *True* returns also lattice vectors inside the
          super-cell. Internally, super-cell tight-binding model will
          have orbitals repeated in the same order in which these
          super-cell vectors are given, but if argument *to_home*
          is set *True* (which it is by default) then additionally,
          orbitals will be shifted to the home cell.

        :param to_home: Optional parameter, if *True* will
          shift all orbitals to the home cell. Default value is *True*.

        :returns:
          * **sc_tb** -- Object of type :class:`pythtb.tb_model`
            representing a tight-binding model in a super-cell.

          * **sc_vectors** -- Super-cell vectors, returned only if
            *return_sc_vectors* is set to *True* (default value is
            *False*).

        Example usage::

          # Creates super-cell out of 2d tight-binding model tb
          sc_tb = tb.make_supercell([[2, 1], [-1, 2]])

        """

        # Can't make super cell for model without periodic directions
        if self._dim_r == 0:
            raise Exception("\n\nMust have at least one periodic direction to make a super-cell")

        # convert array to numpy array
        use_sc_red_lat = np.array(sc_red_lat)

        # checks on super-lattice array
        if use_sc_red_lat.shape != (self._dim_r, self._dim_r):
            raise Exception("\n\nDimension of sc_red_lat array must be dim_r*dim_r")
        if use_sc_red_lat.dtype != int:
            raise Exception("\n\nsc_red_lat array elements must be integers")
        for i in range(self._dim_r):
            for j in range(self._dim_r):
                if (i == j) and (i not in self._per) and use_sc_red_lat[i, j] != 1:
                    raise Exception("\n\nDiagonal elements of sc_red_lat for non-periodic directions must equal 1.")
                if (i != j) and ((i not in self._per) or (j not in self._per)) and use_sc_red_lat[i, j] != 0:
                    raise Exception("\n\nOff-diagonal elements of sc_red_lat for non-periodic directions must equal 0.")
        if np.abs(np.linalg.det(use_sc_red_lat)) < 1.0E-6:
            raise Exception("\n\nSuper-cell lattice vectors length/area/volume too close to zero, or zero.")
        if np.linalg.det(use_sc_red_lat) < 0.0:
            raise Exception("\n\nSuper-cell lattice vectors need to form right handed system.")

        # converts reduced vector in original lattice to reduced vector in super-cell lattice
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(np.array(use_sc_red_lat.T, dtype=float),
                                   np.array(red_vec_orig, dtype=float))

        # conservative estimate on range of search for super-cell vectors
        max_R = np.max(np.abs(use_sc_red_lat)) * self._dim_r

        # candidates for super-cell vectors
        # this is hard-coded and can be improved!
        sc_cands = []
        if self._dim_r == 1:
            for i in range(-max_R, max_R + 1):
                sc_cands.append(np.array([i]))
        elif self._dim_r == 2:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    sc_cands.append(np.array([i, j]))
        elif self._dim_r == 3:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    for k in range(-max_R, max_R + 1):
                        sc_cands.append(np.array([i, j, k]))
        elif self._dim_r == 4:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    for k in range(-max_R, max_R + 1):
                        for l in range(-max_R, max_R + 1):
                            sc_cands.append(np.array([i, j, k, l]))
        else:
            raise Exception("\n\nWrong dimensionality of dim_r!")

        # find all vectors inside super-cell
        # store them here
        sc_vec = []
        eps_shift = np.sqrt(2.0) * 1.0E-8  # shift of the grid, so to avoid double counting
        #
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = to_red_sc(vec).tolist()
            # check if in the interior
            inside = True
            for t in tmp_red:
                if t <= -1.0 * eps_shift or t > 1.0 - eps_shift:
                    inside = False
            if inside == True:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(use_sc_red_lat)))) != num_sc:
            raise Exception("\n\nSuper-cell generation failed! Wrong number of super-cell vectors found.")

        # cartesian vectors of the super lattice
        sc_cart_lat = np.dot(use_sc_red_lat, self._lat)

        # orbitals of the super-cell tight-binding model
        sc_orb = []
        for cur_sc_vec in sc_vec:  # go over all super-cell vectors
            for orb in self._orb:  # go over all orbitals
                # shift orbital and compute coordinates in
                # reduced coordinates of super-cell
                sc_orb.append(to_red_sc(orb + cur_sc_vec))

        # create super-cell tb_model object to be returned
        sc_tb = tb_model_NH(self._dim_k, self._dim_r, sc_cart_lat, sc_orb, per=self._per, nspin=self._nspin)

        # remember if came from w90
        sc_tb._assume_position_operator_diagonal = self._assume_position_operator_diagonal

        # repeat onsite energies
        for i in range(num_sc):
            for j in range(self._norb):
                sc_tb.set_onsite(self._site_energies[j], i * self._norb + j)

        # set hopping terms
        for c, cur_sc_vec in enumerate(sc_vec):  # go over all super-cell vectors
            for h in range(len(self._hoppings)):  # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp = self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part = np.floor(to_red_sc(ind_R + cur_sc_vec))  # round down!
                sc_part = np.array(sc_part, dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part = ind_R + cur_sc_vec - np.dot(sc_part, use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind = None
                for p, pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec == orig_part):
                        if pair_ind != None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind = p
                if pair_ind == None:
                    raise Exception("\n\nDid not find super cell vector!")

                # index of "from" and "to" hopping indices
                hi = self._hoppings[h][1] + c * self._norb
                hj = self._hoppings[h][2] + pair_ind * self._norb

                # add hopping term
                sc_tb.set_hop(amp, hi, hj, sc_part, mode="add", allow_conjugate_pair=True)

        # set nh hopping terms
        for c, cur_sc_vec in enumerate(sc_vec):  # go over all super-cell vectors
            for h in range(len(self._nh_hoppings)):  # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp = self._nh_hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._nh_hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part = np.floor(to_red_sc(ind_R + cur_sc_vec))  # round down!
                sc_part = np.array(sc_part, dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part = ind_R + cur_sc_vec - np.dot(sc_part, use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind = None
                for p, pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec == orig_part):
                        if pair_ind != None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind = p
                if pair_ind == None:
                    raise Exception("\n\nDid not find super cell vector!")

                # index of "from" and "to" hopping indices
                hi = self._nh_hoppings[h][1] + c * self._norb
                hj = self._nh_hoppings[h][2] + pair_ind * self._norb

                # add hopping term
                sc_tb.set_nh_hop(amp, hi, hj, sc_part, mode="add", allow_conjugate_pair=True)

        # put orbitals to home cell if asked for
        if to_home == True:
            sc_tb._shift_to_home()

        # return new tb model and vectors if needed
        if return_sc_vectors == False:
            return sc_tb
        else:
            return (sc_tb, sc_vec)

    def visualize(self, dir_first, dir_second=None, eig_dr=None, draw_hoppings=True, ph_color="black", fig_ax=None):
        # fig, ax = super(tb_model_NH, self).visualize(dir_first, dir_second, eig_dr, draw_hoppings, ph_color, fig_ax)
        r"""

        Rudimentary function for visualizing tight-binding model geometry,
        hopping between tight-binding orbitals, and electron eigenstates.

        If eigenvector is not drawn, then orbitals in home cell are drawn
        as red circles, and those in neighboring cells are drawn with
        different shade of red. Hopping term directions are drawn with
        green lines connecting two orbitals. Origin of unit cell is
        indicated with blue dot, while real space unit vectors are drawn
        with blue lines.

        If eigenvector is drawn, then electron eigenstate on each orbital
        is drawn with a circle whose size is proportional to wavefunction
        amplitude while its color depends on the phase. There are various
        coloring schemes for the phase factor; see more details under
        *ph_color* parameter. If eigenvector is drawn and coloring scheme
        is "red-blue" or "wheel", all other elements of the picture are
        drawn in gray or black.

        :param dir_first: First index of Cartesian coordinates used for
          plotting.

        :param dir_second: Second index of Cartesian coordinates used for
          plotting. For example if dir_first=0 and dir_second=2, and
          Cartesian coordinates of some orbital is [2.0,4.0,6.0] then it
          will be drawn at coordinate [2.0,6.0]. If dimensionality of real
          space (*dim_r*) is zero or one then dir_second should not be
          specified.

        :param eig_dr: Optional parameter specifying eigenstate to
          plot. If specified, this should be one-dimensional array of
          complex numbers specifying wavefunction at each orbital in
          the tight-binding basis. If not specified, eigenstate is not
          drawn.

        :param draw_hoppings: Optional parameter specifying whether to
          draw all allowed hopping terms in the tight-binding
          model. Default value is True.

        :param ph_color: Optional parameter determining the way
          eigenvector phase factors are translated into color. Default
          value is "black". Convention of the wavefunction phase is as
          in convention 1 in section 3.1 of :download:`notes on
          tight-binding formalism  <misc/pythtb-formalism.pdf>`.  In
          other words, these wavefunction phases are in correspondence
          with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
          not :math:`\Psi_{n {\bf k}} ({\bf r})`.

          * "black" -- phase of eigenvectors are ignored and wavefunction
            is always colored in black.

          * "red-blue" -- zero phase is drawn red, while phases or pi or
            -pi are drawn blue. Phases in between are interpolated between
            red and blue. Some phase information is lost in this coloring
            becase phase of +phi and -phi have same color.

          * "wheel" -- each phase is given unique color. In steps of pi/3
            starting from 0, colors are assigned (in increasing hue) as:
            red, yellow, green, cyan, blue, magenta, red.

        :returns:
          * **fig** -- Figure object from matplotlib.pyplot module
            that can be used to save the figure in PDF, EPS or similar
            format, for example using fig.savefig("name.pdf") command.
          * **ax** -- Axes object from matplotlib.pyplot module that can be
            used to tweak the plot, for example by adding a plot title
            ax.set_title("Title goes here").

        Example usage::

          # Draws x-y projection of tight-binding model
          # tweaks figure and saves it as a PDF.
          (fig, ax) = tb.visualize(0, 1)
          ax.set_title("Title goes here")
          fig.savefig("model.pdf")

        See also these examples: :ref:`edge-example`,
        :ref:`visualize-example`.

        """
        # check the format of eig_dr
        if not (eig_dr is None):
            if eig_dr.shape != (self._norb,):
                raise Exception("\n\nWrong format of eig_dr! Must be array of size norb.")

        # check that ph_color is correct
        if ph_color not in ["black", "red-blue", "wheel"]:
            raise Exception("\n\nWrong value of ph_color parameter!")

        # check if dir_second had to be specified
        if dir_second == None and self._dim_r > 1:
            raise Exception("\n\nNeed to specify index of second coordinate for projection!")

        # start a new figure
        import matplotlib.pyplot as plt
        if fig_ax is None:
            fig = plt.figure(figsize=[plt.rcParams["figure.figsize"][0],
                                      plt.rcParams["figure.figsize"][0]])
            ax = fig.add_subplot(111, aspect='equal')
        else:
            fig, ax = fig_ax

        def proj(v):
            "Project vector onto drawing plane"
            coord_x = v[dir_first]
            if dir_second == None:
                coord_y = 0.0
            else:
                coord_y = v[dir_second]
            return [coord_x, coord_y]

        def to_cart(red):
            "Convert reduced to Cartesian coordinates"
            return np.dot(red, self._lat)

        # define colors to be used in plotting everything
        # except eigenvectors
        if (eig_dr is None) or ph_color == "black":
            c_cell = "b"
            c_orb = "r"
            c_nei = [0.85, 0.65, 0.65]
            c_hop = "g"
        else:
            c_cell = [0.4, 0.4, 0.4]
            c_orb = [0.0, 0.0, 0.0]
            c_nei = [0.6, 0.6, 0.6]
            c_hop = [0.0, 0.0, 0.0]

        # determine color scheme for eigenvectors
        def color_to_phase(ph):
            if ph_color == "black":
                return "k"
            if ph_color == "red-blue":
                ph = np.abs(ph / np.pi)
                return [1.0 - ph, 0.0, ph]
            if ph_color == "wheel":
                if ph < 0.0:
                    ph = ph + 2.0 * np.pi
                ph = 6.0 * ph / (2.0 * np.pi)
                x_ph = 1.0 - np.abs(ph % 2.0 - 1.0)
                if ph >= 0.0 and ph < 1.0: ret_col = [1.0, x_ph, 0.0]
                if ph >= 1.0 and ph < 2.0: ret_col = [x_ph, 1.0, 0.0]
                if ph >= 2.0 and ph < 3.0: ret_col = [0.0, 1.0, x_ph]
                if ph >= 3.0 and ph < 4.0: ret_col = [0.0, x_ph, 1.0]
                if ph >= 4.0 and ph < 5.0: ret_col = [x_ph, 0.0, 1.0]
                if ph >= 5.0 and ph <= 6.0: ret_col = [1.0, 0.0, x_ph]
                return ret_col

        # draw origin
        ax.plot([0.0], [0.0], "o", c=c_cell, mec="w", mew=0.0, zorder=7, ms=4.5)

        # first draw unit cell vectors which are considered to be periodic
        for i in self._per:
            # pick a unit cell vector and project it down to the drawing plane
            vec = proj(self._lat[i])
            ax.plot([0.0, vec[0]], [0.0, vec[1]], "-", c=c_cell, lw=1.5, zorder=7)

        # now draw all orbitals
        for i in range(self._norb):
            # find position of orbital in cartesian coordinates
            pos = to_cart(self._orb[i])
            pos = proj(pos)
            ax.plot([pos[0]], [pos[1]], "o", c=c_orb, mec="w", mew=0.0, zorder=10, ms=4.0)

        # draw hopping terms
        if draw_hoppings == True:
            for h in self._hoppings:
                # draw both i->j+R and i-R->j hop
                for s in range(2):
                    # get "from" and "to" coordinates
                    pos_i = np.copy(self._orb[h[1]])
                    pos_j = np.copy(self._orb[h[2]])
                    # add also lattice vector if not 0-dim
                    if self._dim_k != 0:
                        if s == 0:
                            pos_j[self._per] = pos_j[self._per] + h[3][self._per]
                        if s == 1:
                            pos_i[self._per] = pos_i[self._per] - h[3][self._per]
                    # project down vector to the plane
                    pos_i = np.array(proj(to_cart(pos_i)))
                    pos_j = np.array(proj(to_cart(pos_j)))
                    # add also one point in the middle to bend the curve
                    prcnt = 0.05  # bend always by this ammount
                    pos_mid = (pos_i + pos_j) * 0.5
                    dif = pos_j - pos_i  # difference vector
                    orth = np.array([dif[1], -1.0 * dif[0]])  # orthogonal to difference vector
                    orth = orth / np.sqrt(np.dot(orth, orth))  # normalize
                    pos_mid = pos_mid + orth * prcnt * np.sqrt(
                        np.dot(dif, dif))  # shift mid point in orthogonal direction
                    # draw hopping
                    all_pnts = np.array([pos_i, pos_mid, pos_j]).T
                    ax.plot(all_pnts[0], all_pnts[1], "-", c=c_hop, lw=0.75, zorder=8)
                    # draw "from" and "to" sites
                    ax.plot([pos_i[0]], [pos_i[1]], "o", c=c_nei, zorder=9, mew=0.0, ms=4.0, mec="w")
                    ax.plot([pos_j[0]], [pos_j[1]], "o", c=c_nei, zorder=9, mew=0.0, ms=4.0, mec="w")

        # now draw the eigenstate
        if not (eig_dr is None):
            for i in range(self._norb):
                # find position of orbital in cartesian coordinates
                pos = to_cart(self._orb[i])
                pos = proj(pos)
                # find norm of eigenfunction at this point
                nrm = (eig_dr[i] * eig_dr[i].conjugate()).real
                # rescale and get size of circle
                nrm_rad = 2.0 * nrm * float(self._norb)
                # get color based on the phase of the eigenstate
                phase = np.angle(eig_dr[i])
                c_ph = color_to_phase(phase)
                ax.plot([pos[0]], [pos[1]], "o", c=c_ph, mec="w", mew=0.0, ms=nrm_rad, zorder=11, alpha=0.8)

        # center the image
        #  first get the current limit, which is probably tight
        xl = ax.set_xlim()
        yl = ax.set_ylim()
        # now get the center of current limit
        centx = (xl[1] + xl[0]) * 0.5
        centy = (yl[1] + yl[0]) * 0.5
        # now get the maximal size (lengthwise or heightwise)
        mx = max([xl[1] - xl[0], yl[1] - yl[0]])
        # set new limits
        extr = 0.05  # add some boundary as well
        ax.set_xlim(centx - mx * (0.5 + extr), centx + mx * (0.5 + extr))
        ax.set_ylim(centy - mx * (0.5 + extr), centy + mx * (0.5 + extr))

        # Plots the non-Hermitian part
        def proj(v):
            "Project vector onto drawing plane"
            coord_x = v[dir_first]
            if dir_second == None:
                coord_y = 0.0
            else:
                coord_y = v[dir_second]
            return [coord_x, coord_y]

        def to_cart(red):
            "Convert reduced to Cartesian coordinates"
            return np.dot(red, self._lat)

        if draw_hoppings == True:
            for h in self._nh_hoppings:
                # draw both i->j+R and i-R->j hop
                for s in range(2):
                    # get "from" and "to" coordinates
                    pos_i = np.copy(self._orb[h[1]])
                    pos_j = np.copy(self._orb[h[2]])
                    # add also lattice vector if not 0-dim
                    if self._dim_k != 0:
                        if s == 0:
                            pos_j[self._per] = pos_j[self._per] + h[3][self._per]
                        if s == 1:
                            pos_i[self._per] = pos_i[self._per] - h[3][self._per]
                    # project down vector to the plane
                    pos_i = np.array(proj(to_cart(pos_i)))
                    pos_j = np.array(proj(to_cart(pos_j)))
                    # add also one point in the middle to bend the curve
                    prcnt = 0.05  # bend always by this ammount
                    pos_mid = (pos_i + pos_j) * 0.5
                    dif = pos_j - pos_i  # difference vector
                    orth = np.array([dif[1], -1.0 * dif[0]])  # orthogonal to difference vector
                    orth = orth / np.sqrt(np.dot(orth, orth))  # normalize
                    pos_mid = pos_mid + orth * prcnt * np.sqrt(np.dot(dif, dif))  # shift mid point in orthogonal direction
                    pos_mid2 = pos_mid - 2 * orth * prcnt * np.sqrt(np.dot(dif, dif))
                    colors = ['r', 'b']
                    poly = Polygon([pos_i, pos_mid, pos_j, pos_mid2], color=colors[int((np.sign(h[0].imag)+1)/2)], alpha=0.3)
                    ax.add_patch(poly)

    def plot_all(self, n_lattices=3, krange=None, kpoints=101, bands=None, continuity_threshold=0.01):
        if len(self._nh_hoppings) > 0:
            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], projection='3d')
            ax2 = plt.subplot(gs[2], projection='3d')
            self.plot_eigen_surface(krange, kpoints, bands, (fig, ax1, ax2), continuity_threshold)
        else:
            fig = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(1, 2)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], projection='3d')
            ax2 = None
            self.plot_eigen_surface(krange, kpoints, bands, (fig, ax1), continuity_threshold)
        tmp_model = self.cut_piece(n_lattices, 0, glue_edgs=False)
        fin_model = tmp_model.cut_piece(n_lattices, 1, glue_edgs=False)
        fin_model.visualize(0, 1, fig_ax=(fig, ax0))
        return fig, (ax0, ax1, ax2)

    def cut_angle(self, num, angle, glue_edges=False, order=5):
        """Cuts at an angle wrt to the first lattice vector

        :param num:
        :param angle:
        :param glue_edges:
        :param order:
        :return:
        """
        direction = [np.cos(angle), np.sin(angle)]
        v1, v2 = np.linalg.inv(self.get_lat()).transpose()
        proj1 = np.sum(direction * v1)
        proj2 = np.sum(direction * v2)
        # mn = np.min([proj1, proj2])
        # mx = np.max([proj1, proj2])
        print(proj1, proj2)
        if np.abs(proj1) >= np.abs(proj2):
            frc = Fraction(proj2 / proj1).limit_denominator(order)
            b, a = frc.numerator, frc.denominator
        else:
            frc = Fraction(proj1 / proj2).limit_denominator(order)
            a, b = frc.numerator, frc.denominator
        # frc = Fraction(proj1/proj2).limit_denominator(order)
        # a, b = frc.numerator, frc.denominator
        print(a,b)
        if b == 0:
            return self.cut_piece(num, 0, glue_edges)
        else:
            try:
                cell = self.make_supercell([[1, 0], [a, b]])
            except:
                cell = self.make_supercell([[a, b], [1, 0]])
            return cell.cut_piece(num, 0, glue_edges)


def _nicefy_eig_nh(eval, eig=None):
    # TODO: round off small numerical errors
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    energy = np.array(eval.real, dtype=float)
    # sort by energy
    args = energy.argsort()
    eval = eval[args]
    if not (eig is None):
        eig = eig[args]
        return (eval, eig)
    return eval


if __name__ == '__main__':
    # Square lattice
    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0]]
    sqr = tb_model_NH(2, 2, lat, orb)

    forw = -1
    back = -1j
    sqr.set_hop(forw, 0, 0, [0, 1])
    sqr.set_hop(forw, 0, 0, [1, 0])

    sqr.set_hop(forw, 0, 0, [0, 1], allow_conjugate_pair=True)
    sqr.set_hop(back, 0, 0, [0, -1], allow_conjugate_pair=True)
    sqr.set_hop(forw, 0, 0, [1, 0], allow_conjugate_pair=True)
    sqr.set_hop(back, 0, 0, [-1, 0], allow_conjugate_pair=True)

    sqr.plot_eigen_surface()

    # Kagome
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[0, 0], [0.5, 0], [0, 0.5]]
    kgm = tb_model_NH(2, 2, lat, orb)
    t1 = 1
    t2 = 0.05j
    kgm.set_hop(t1, 0, 1, [0, 0])
    kgm.set_hop(t1, 1, 0, [1, 0])
    kgm.set_hop(t1, 0, 2, [0, 0])
    kgm.set_hop(t1, 2, 0, [0, 1])
    kgm.set_hop(t1, 1, 2, [0, 0])
    kgm.set_hop(t1, 2, 1, [-1, 1])
    kgm.plot_eigen_surface()
    kgm.set_nh_hop(t2, 0, 2, [0, 0])
    kgm.set_nh_hop(t2, 2, 0, [0, 1])
    kgm.set_nh_hop(-t2, 1, 2, [0, 0])
    kgm.set_nh_hop(-t2, 2, 1, [-1, 1])
    kgm.plot_eigen_surface()

    plt.show()
