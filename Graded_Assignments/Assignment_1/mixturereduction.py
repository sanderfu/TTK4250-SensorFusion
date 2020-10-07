from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights, shape=(N,)
    mean: np.ndarray,  # the mixture means, shape(N, n)
    cov: np.ndarray,  # the mixture covariances, shape (N, n, n)
    ) -> Tuple[np.ndarray, np.ndarray]:
         # the mean and covariance of of the mixture, shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""
    # mean
    mean_bar = np.average(mean, weights=w, axis=0)

    # covariance
    # # internal covariance
    cov_int = np.average(cov, weights=w, axis=0)

    # # spread of means
    N, n = mean.shape
    cov_ext = np.zeros((n,n))
    for i in range(N):
        mean_diff = (mean[i] - mean_bar).reshape((-1,1))
        cov_ext += (mean_diff @ mean_diff.T) * w[i]

    # # total covariance
    cov_bar = cov_int + cov_ext

    return mean_bar, cov_bar
