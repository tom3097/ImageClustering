"""
Principal component analysis for Global Histogram-Based Color Image Clustering.
"""

import numpy as np


class Pca(object):
    """
    Principal component analysis for Global Histogram-Based Color Image Clustering.

    Parameters
    ----------

    n_components : int, default: 300
        The number of components to keep.
    """
    def __init__(self, n_components=300):
        self.__n_components = n_components

    def fit_transform(self, data):
        """ Fits the model with data and apply the dimensionality reduction on data. """
        original_dim_no = data.shape[1]
        data_transposed = data.T
        cov_mat = np.cov([data_transposed[i, :] for i in range(0, original_dim_no)])

        # Calculates eigenvalues and eigenvectors
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(original_dim_no)]

        # Sort eig_pairs by eigenvalues
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        matrix_w = np.hstack((eig_pairs[i][1].reshape(original_dim_no, 1) for i in range(self.__n_components)))

        reduced_data = matrix_w.T.dot(data_transposed).T
        assert reduced_data.shape == (data.shape[0], self.__n_components)

        return reduced_data
