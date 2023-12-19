import numpy as np
import scipy

from .music import *
np.set_printoptions(linewidth=150)

class GevdMUSIC(MUSIC):
    """
    Class to apply the Generalized Eigenvalue Decomposition (GEVD) based MUSIC
    (GEVD-MUSIC) direction-of-arrival (DoA) for a particular microphone array,
    extending the capabilities of the original MUSIC algorithm.

    .. note:: Run locate_source() to apply the GEVD-MUSIC algorithm.
    """

    def _process(self, X, X_noise, auto_identify, **kwargs):
        # compute steered response
        self.spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        # compute source and noise correlation matrices
        R = self._compute_correlation_matricesvec(X)
        K = self._compute_correlation_matricesvec(X_noise)
        if kwargs.get("ncm_diff", False):
            # K = apply_error_to_hermitian_matrices(K, kwargs.get("ncm_diff", 0))
            K = apply_error_with_regularization_and_check(K, kwargs.get("ncm_diff", 0))
            print("K_modified", K.shape)
            np.save("K_modified.npy", K)
        for i in range(self.num_freq):
            print(is_positive_definite(K[i]))
        # subspace decomposition
        noise_subspace = self._extract_noise_subspace(R, K, auto_identify=auto_identify)
        # compute spatial spectrum
        self.spatial_spectrum = self._compute_spatial_spectrum(noise_subspace)

        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.spatial_spectrum, axis=1) / self.num_freq))
        self.spectra_storage.append(self.grid.values)

    def _extract_noise_subspace(self, R, K, auto_identify):
        decomposed_values = np.empty(R.shape[:2], dtype=complex)
        decomposed_vectors = np.empty(R.shape, dtype=complex)

        for i in range(self.num_freq):
            decomposed_values[i], decomposed_vectors[i] = scipy.linalg.eigh(R[i], K[i])
        decomposed_values = np.real(decomposed_values)

        # print(decomposed_values.shape, self.num_freq)
        self.decomposed_values_strage.append(decomposed_values)

        # if auto_identify:
        #     self.num_src = self._auto_identify(decomposed_values)

        noise_subspace = decomposed_vectors[..., :-self.num_src]

        return noise_subspace


def apply_error_to_hermitian_matrices(K, error_percentage):
    """
    Apply random errors to all Hermitian matrices in the given array.

    :param K: A numpy array of shape (N, 8, 8) containing N Hermitian matrices.
    :param error_percentage: The percentage of error to be applied.
    :return: A numpy array with the modified Hermitian matrices.
    """
    # Copy the original array to avoid modifying it directly
    modified_K = np.copy(K)

    # Function to add random error
    def add_random_error(value, percentage):
        error_percentage = np.random.uniform(-percentage, percentage)
        # print(error_percentage)
        error = error_percentage * value
        return value + error

    # Iterate over each 8x8 Hermitian matrix
    for matrix in modified_K:
        # Real and imaginary parts
        real_part = np.real(matrix)
        imag_part = np.imag(matrix)

        # Apply error to real part (upper triangular including diagonal)
        for i in range(8):
            for j in range(i, 8):
                real_part[i, j] = add_random_error(real_part[i, j], error_percentage)

        # Apply error to imaginary part (upper triangular excluding diagonal)
        for i in range(8):
            for j in range(i + 1, 8):
                imag_part[i, j] = add_random_error(imag_part[i, j], error_percentage)

        # Reflect the upper part to the lower part
        for i in range(1, 8):
            for j in range(i):
                real_part[i, j] = real_part[j, i]
                imag_part[i, j] = -imag_part[j, i]

        # 虚部の対角成分は0になっていることを確認
        # print(np.diag(imag_part))

        # Combine the real and imaginary parts
        matrix[:] = real_part + 1j * imag_part

    return modified_K


def is_positive_definite(matrix):
        return np.all(np.linalg.eigvalsh(matrix) > 0)


def apply_error_with_regularization_and_check(K, error_percentage, regularization_term=1e-6, max_attempts=1000):
    """
    Apply errors to Hermitian matrices and regularize them. If the matrix is not positive definite,
    retry the process up to a maximum number of attempts.

    :param K: A numpy array of shape (N, M, M) containing N Hermitian matrices.
    :param error_percentage: The percentage of error to be applied.
    :param regularization_term: A small positive term added to the diagonal to ensure positive definiteness.
    :param max_attempts: Maximum number of attempts to generate a positive definite matrix.
    :return: A numpy array with the modified Hermitian matrices.
    """
    def is_positive_definite(matrix):
        return np.all(np.linalg.eigvalsh(matrix) > 0)

    modified_K = np.copy(K)

    for idx, matrix in enumerate(modified_K):
        for attempt in range(max_attempts):
            # Apply error to the matrix
            modified_matrix = np.copy(matrix)
            real_part = np.real(modified_matrix)
            imag_part = np.imag(modified_matrix)

            for i in range(real_part.shape[0]):
                for j in range(i, real_part.shape[1]):
                    error = np.random.uniform(-error_percentage, error_percentage) * real_part[i, j]
                    real_part[i, j] += error
                    if i != j:
                        error = np.random.uniform(-error_percentage, error_percentage) * imag_part[i, j]
                        imag_part[i, j] += error

            for i in range(1, real_part.shape[0]):
                for j in range(i):
                    real_part[i, j] = real_part[j, i]
                    imag_part[i, j] = -imag_part[j, i]

            modified_matrix = real_part + 1j * imag_part
            np.fill_diagonal(modified_matrix, np.diag(modified_matrix) + regularization_term)

            # Check if the modified matrix is positive definite
            if is_positive_definite(modified_matrix):
                modified_K[idx] = modified_matrix
                break
            elif attempt == max_attempts - 1:
                raise ValueError(f"Failed to generate a positive definite matrix for index {idx} after {max_attempts} attempts.")

    return modified_K

# Note: To use this function, you need to provide the original K matrix (K_original) as an input.
# Example usage:
# K_modified_checked = apply_error_with_regularization_and_check(K_original, 0.05)
