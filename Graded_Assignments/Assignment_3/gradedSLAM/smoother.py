# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Optional

def rts_smooth(x_pred, P_pred, x_est, P_est, F):
    """
    Compute the Rauch-Tung-Striebel smoothed state estimates and estimate
    covariances for a Kalman filter.

    """
    state_count = len(x_pred)

    # Initialise with final posterior estimate

    smoothed_means: List[Optional[np.ndarray]] = [None] * state_count
    smoothed_covs: List[Optional[np.ndarray]] = [None] * state_count
    smoothed_means[-1] = x_pred[-1]
    smoothed_covs[-1] = P_pred[-1]
 

    # Work backwards from final state
    for k in range(state_count-2, -1, -1):
        process_mat = F[k+1]
        cmat = P_est[k].dot(process_mat.T).dot(
            np.linalg.inv(P_pred[k+1]))

        # Calculate smoothed state and covariance
        smoothed_means[k]=x_est[k] + cmat.dot(smoothed_means[k+1] -
                                           x_pred[k+1])
        smoothed_covs[k]=P_est[k] + cmat.dot(smoothed_covs[k+1] -
                                        P_pred[k+1]).dot(cmat.T)
        

    
    return smoothed_means, smoothed_covs