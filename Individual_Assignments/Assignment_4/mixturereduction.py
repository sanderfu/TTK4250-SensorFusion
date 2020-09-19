from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    mean_bar = np.average(mean,weights=w, axis=0)  # DONE
    
    """
    Comment (to myself, remove it later)
        - It was a bit unclear if the gaussian means were row or column vectors, but based
        on the shapes if mean and cov in the input i believe they are column-vectors (shape: (n,))
    """

    # covariance
    # # internal covariance
    cov_int = np.average(cov,axis=0,weights=w)  # DONE

    # # spread of means
    # Optional calc: mean_diff =
    mean_diff = mean-mean_bar #Tested this method with a matrix of mockmeans and a mockavg
    #cov_ext = np.average(mean_diff.T@mean_diff,axis=0,weights=w)  # DID NOT GET THIS WAY TO WORK
    cov_ext = np.zeros(shape=cov[0,:,:].shape)
    for i in range(np.size(mean,axis=0)):
        mean_diff_i = mean[i,:]-mean_bar
        cov_ext += w[i]*(mean_diff[i,:].T@mean_diff[i,:])
    """
    Comment to understand the code above: Had to take mean_diff.T@mean_diff instead of mean_diff@mean_diff.T
    which is in the formula 6.21 because we have row mean vectors and the equation assumes column vectors.
    """

    # # total covariance
    cov_bar = cov_int+cov_ext  # DONE

    return mean_bar, cov_bar
