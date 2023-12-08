import numpy as np
import scipy

from .music import *


class GevdMUSIC(MUSIC):
    """
    Class to apply the Generalized Eigenvalue Decomposition (GEVD) based MUSIC
    (GEVD-MUSIC) direction-of-arrival (DoA) for a particular microphone array,
    extending the capabilities of the original MUSIC algorithm.

    .. note:: Run locate_source() to apply the GEVD-MUSIC algorithm.
    """

    def _process(self, X, X_noise, display, save, auto_identify):
        # compute steered response
        self.spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        # Compute source and noise correlation matrices
        R = self._compute_correlation_matricesvec(X)
        K = self._compute_correlation_matricesvec(X_noise)
        # subspace decomposition
        noise_subspace = self._extract_noise_subspace(R, K, display=display, save=save,
                                                      auto_identify=auto_identify)
        # compute spatial spectrum
        self.spatial_spectrum = self._compute_spatial_spectrum(noise_subspace)

        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.spatial_spectrum, axis=1) / self.num_freq))

    def _extract_noise_subspace(self, R, K, display, save, auto_identify):
        # Initialize
        decomposed_values = np.empty(R.shape[:2], dtype=complex)
        decomposed_vectors = np.empty(R.shape, dtype=complex)

        # Step 1: Eigenvalue decomposition
        for i in range(self.num_freq):
            decomposed_values[i], decomposed_vectors[i] = scipy.linalg.eigh(R[i], K[i])
        decomposed_values = np.real(decomposed_values)

        # Step 2: Display if flag is True
        if display or save:
            self._plot_decomposed_values(decomposed_values, display, save)

        # Step 3: Auto-identify source and noise if flag is True
        if auto_identify:
            self.num_src = self._auto_identify(decomposed_values, save)

        # Step 4: Extract subspace
        noise_subspace = decomposed_vectors[..., :-self.num_src]

        return noise_subspace
