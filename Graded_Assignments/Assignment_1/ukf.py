"""
File name: ukf.py

Creation Date: So 18 Okt 2020

Description:

"""

# Standard Python Libraries
# -----------------------------------------------------------------------------------------

# Local Application Modules
# -----------------------------------------------------------------------------------------

"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z (capital) are mulitple measurements so that z = Z[k] at a given time step
v is the innovation z - h(x)
S is the innovation covariance
"""
# %% Imports
# types
from typing import Union, Any, Dict, Optional, List, Sequence, Tuple, Iterable
from typing_extensions import Final

# packages
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import scipy
import math

# local
import dynamicmodels as dynmods
import measurementmodels as measmods
from gaussparams import GaussParams, GaussParamList
from mixturedata import MixtureParameters
import mixturereduction
from singledispatchmethod import singledispatchmethod

# %% The EKF


@dataclass
class UKF:
    # A Protocol so duck typing can be used
    dynamic_model: dynmods.DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: measmods.MeasurementModel

    #_MLOG2PIby2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._MLOG2PIby2: Final[float] = self.sensor_model.m * \
            np.log(2 * np.pi) / 2

        self.n_dim = self.dynamic_model.n
        self.n_sig = 1 + self.n_dim*2
        
        self.beta = float(2)
        self.alpha = float(1)
        self.k = float(3-self.n_dim)
        self.lambd = math.pow(self.alpha, 2)*(self.n_dim+self.k) - self.n_dim
        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)
        
        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1.0 / (2.0*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1.0 / (2.0*(self.n_dim + self.lambd))

    def __get_sigmas(self, mean, cov):
        ret = np.zeros((self.n_sig, self.n_dim))
        tmp_mat = (self.n_dim + self.lambd)*cov
        spr_mat = scipy.linalg.sqrtm(tmp_mat)
        
        ret[0] = mean

        import ipdb

        for i in range(1, self.n_sig//2+1):
            ret[i] = mean + spr_mat[i-1]
            ret[i+self.n_dim] = mean-spr_mat[i-1]

        ipdb.set_trace()
        return ret

    def predict(self,
                ukfstate: GaussParams,
                # The sampling time in units specified by dynamic_model
                Ts: float,
                ) -> GaussParams:
        """Predict the EKF state Ts seconds ahead."""
        x, P = ukfstate  # tuple unpacking


        sigmas = self.__get_sigmas(x, P)
        sigmas_out = np.array([self.dynamic_model.f(sig,Ts) for sig in sigmas])

        x_pred = np.zeros(self.n_dim)
        P_pred = np.zeros((self.n_dim, self.n_dim))
        
        for i in range(self.n_sig):
            x_pred += self.mean_weights[i]*sigmas_out[i]

        for i in range(self.n_sig):
            diff = sigmas_out[i]-x_pred
            P_pred += self.covar_weights[i]*np.dot(diff, diff.T)

        Q = self.dynamic_model.Q(x, Ts)
        import ipdb
        ipdb.set_trace()
        P_pred += Q

        return GaussParams(x_pred, P_pred)



    def update(
        self, z: np.ndarray, ukfstate: GaussParams, sensor_state: Dict[str, Any] = None
    ) -> GaussParams:
        """Update ukfstate with z in sensor_state"""

        import ipdb
        x, P = ukfstate
        ipdb.set_trace()
        m = self.sensor_model.m
        sigmas = self.__get_sigmas(x, P)
        sigmas_out = np.array([self.sensor_model.h(sig) for sig in sigmas])

        z_upd = np.zeros(m)
        S_upd = np.zeros((m, m))
        for i in range(self.n_sig):
            z_upd += self.mean_weights[i]*sigmas_out[i]

        for i in range(self.n_sig):
            diff = sigmas_out[i]-z_upd
            S_upd += self.covar_weights[i]*np.dot(diff, diff.T)

        S_upd += self.sensor_model.R(x, sensor_state=sensor_state, z=z)

        Sigma_upd = np.zeros((self.n_dim, m))
        for i in range(self.n_sig):
            left_diff = sigmas[i]-x
            right_diff = sigmas_out[i]-z_upd
            Sigma_upd += self.covar_weights[i]*np.outer(left_diff, right_diff)

        kalman_gain = Sigma_upd@np.linalg.inv(S_upd)
        assert(kalman_gain.shape == (self.n_dim, m)), "Wrong shape for kalman gain"

        x_upd = x + kalman_gain@(z-z_upd)
        P_upd = P - kalman_gain@S_upd@kalman_gain.T

        ukfstate_upd = GaussParams(x_upd, P_upd)

        ipdb.set_trace()
        return ukfstate_upd

    def step(self,
             z: np.ndarray,
             ukfstate: GaussParams,
             # sampling time
             Ts: float,
             *,
             sensor_state: Dict[str, Any] = None,
             ) -> GaussParams:
        """Predict ukfstate Ts units ahead and then update this prediction with z in sensor_state."""

        ukfstate_pred = self.predict(ukfstate,Ts)
        ukfstate_upd = self.update(z,ukfstate_pred)
        
        return ukfstate_upd


    @classmethod

    def gate(self,
             z: np.ndarray,
             ukfstate: GaussParams,
             *,
             sensor_state: Dict[str, Any],
             gate_size_square: float,
             ) -> bool:
        """Check if z is inside sqrt(gate_sized_squared)-sigma ellipse of
        ukfstate in sensor_state """

        #Equation unnumbered, page 116.
        x, P = ukfstate
        innov = z-self.sensor_model.h(x)
        gated = innov@P@innov.T<gate_size_square
        return gated

    def loglikelihood(
        self,
        z: np.ndarray,
        ukfstate: GaussParams,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate the log likelihood of ukfstate at z in sensor_state
            Comment: This function was retrieved from Blackboard.
            Comment2: What they ask for here is log(p(z|x_k(posteriori))) for a 
            realization of z. I should modify this function to allow for using 
            a method that makes me understand better what is going on.
            Comment3: For completeness, realize that likelihood function is 
            the same thign as the measurement function (Eq. 4.6)
        """
    
        v, S = self.innovation(z, ukfstate, sensor_state=sensor_state)
    
        cholS = la.cholesky(S, lower=True)
    
        invcholS_v = la.solve_triangular(cholS, v, lower=True)
        NISby2 = (invcholS_v ** 2).sum() / 2
        # alternative self.NIS(...) /2 or v @ la.solve(S, v)/2
    
        logdetSby2 = np.log(cholS.diagonal()).sum()
        # alternative use la.slogdet(S)
    
        ll = -(NISby2 + logdetSby2 + self._MLOG2PIby2)
    
        # simplest overall alternative
        # ll = scipy.stats.multivariate_normal.logpdf(v, cov=S)
    
        return ll

    @classmethod
    def estimate(cls, ukfstate: GaussParams):
        """Get the estimate from the state with its covariance. (Compatibility method)"""
        # dummy function for compatibility with IMM class
        return ukfstate

    def estimate_sequence(
            self,
            # A sequence of measurements
            Z: Sequence[np.ndarray],
            # the initial KF state to use for either prediction or update (see start_with_prediction)
            init_ukfstate: GaussParams,
            # Time difference between Z's. If start_with_prediction: also diff before the first Z
            Ts: Union[float, Sequence[float]],
            *,
            # An optional sequence of the sensor states for when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # sets if Ts should be used for predicting before the first measurement in Z
            start_with_prediction: bool = False,
    ) -> Tuple[GaussParamList, GaussParamList]:
        """Create estimates for the whole time series of measurements."""

        # sequence length
        K = len(Z)

        # Create and amend the sampling array
        Ts_start_idx = int(not start_with_prediction)
        Ts_arr = np.empty(K)
        Ts_arr[Ts_start_idx:] = Ts
        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            Ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * K

        # initialize and allocate
        ukfupd = init_ukfstate
        n = init_ukfstate.mean.shape[0]
        ukfpred_list = GaussParamList.allocate(K, n)
        ukfupd_list = GaussParamList.allocate(K, n)

        # perform the actual predict and update cycle
        # DONE loop over the data and get both the predicted and updated states in the lists
        # the predicted is good to have for evaluation purposes
        # A potential pythonic way of looping through  the data
        for k, (zk, Tsk, ssk) in enumerate(zip(Z, Ts_arr, sensor_state_seq)):
            ukfpred = self.predict(ukfupd,Tsk)
            ukfupd = self.update(zk,ukfpred,Tsk)
            ukfpred_list[k]=ukfpred
            ukfupd_list[k]=ukfupd
        return ukfpred_list, ukfupd_list

    def performance_stats(
            self,
            *,
            z: Optional[np.ndarray] = None,
            ukfstate_pred: Optional[GaussParams] = None,
            ukfstate_upd: Optional[GaussParams] = None,
            sensor_state: Optional[Dict[str, Any]] = None,
            x_true: Optional[np.ndarray] = None,
            # None: no norm, -1: all idx, seq: a single norm for given idxs, seqseq: a norms for idxseq
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs, see np.linalg.norm ord argument.
            norms: Union[Iterable[int], int] = 2,
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate performance statistics available from the given parameters."""
        stats: Dict[str, Union[float, List[float]]] = {}

        # NIS, needs measurements
        if z is not None and ukfstate_pred is not None:
            stats['NIS'] = self.NIS(
                z, ukfstate_pred, sensor_state=sensor_state)

        # NEES and RMSE, needs ground truth
        if x_true is not None:
            # prediction
            if ukfstate_pred is not None:
                stats['NEESpred'] = self.NEES(ukfstate_pred, x_true)

                # distances
                err_pred = ukfstate_pred.mean - x_true
                if norm_idxs is None:
                    stats['dist_pred'] = np.linalg.norm(err_pred, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_pred'] = [
                        np.linalg.norm(err_pred[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]

            # update
            if ukfstate_upd is not None:
                stats['NEESupd'] = self.NEES(ukfstate_upd, x_true)

                # distances
                err_upd = ukfstate_upd.mean - x_true
                if norm_idxs is None:
                    stats['dist_upd'] = np.linalg.norm(err_upd, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_upd'] = [
                        np.linalg.norm(err_upd[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]
        return stats

    def performance_stats_sequence(
            self,
            # Sequence length
            K: int,
            *,
            # The measurements
            Z: Optional[Iterable[np.ndarray]] = None,
            ukfpred_list: Optional[Iterable[GaussParams]] = None,
            ukfupd_list: Optional[Iterable[GaussParams]] = None,
            # An optional sequence of all the sensor states when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # Optional ground truth for error checking
            X_true: Optional[Iterable[Optional[np.ndarray]]] = None,
            # Indexes to be used to calculate errornorms, multiple to separate the state space.
            # None: all idx, Iterable (eg. list): each element is an index sequence into the dimension of the state space.
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs (see numpy.linalg.norm ord argument).
            norms: Union[Iterable[int], int] = 2,
    ) -> np.ndarray:
        """Get performance metrics on a pre-estimated sequence"""

        None_list = [None] * K

        for_iter = []
        for_iter.append(Z if Z is not None else None_list)
        for_iter.append(ukfpred_list or None_list)
        for_iter.append(ukfupd_list or None_list)
        for_iter.append(sensor_state or None_list)
        for_iter.append(X_true if X_true is not None else None_list)

        stats = []
        for zk, ukfpredk, ukfupdk, ssk, xtk in zip(*for_iter):
            stats.append(
                self.performance_stats(
                    z=zk, ukfstate_pred=ukfpredk, ukfstate_upd=ukfupdk, sensor_state=ssk, x_true=xtk,
                    norm_idxs=norm_idxs, norms=norms
                )
            )

        # make structured array
        dtype = [(key, *((type(val[0]), len(val)) if isinstance(val, Iterable)
                         else (type(val),))) for key, val in stats[0].items()]
        stats_arr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

        return stats_arr

    def reduce_mixture(
        self, ukfstate_mixture: MixtureParameters[GaussParams]
    ) -> GaussParams:
        """Merge a Gaussian mixture into single mixture"""
        w = ukfstate_mixture.weights
        x = np.array([c.mean for c in ukfstate_mixture.components], dtype=float)
        P = np.array([c.cov for c in ukfstate_mixture.components], dtype=float)
        x_reduced, P_reduced = mixturereduction.gaussian_mixture_moments(w, x, P)
        return GaussParams(x_reduced, P_reduced)
    
    @singledispatchmethod
    def init_filter_state(self, init) -> None:
        raise NotImplementedError(
            f"EKF do not know how to make {init} into GaussParams"
        )
    
    @init_filter_state.register(GaussParams)
    def _(self, init: GaussParams) -> GaussParams:
        return init
    
    @init_filter_state.register(tuple)
    @init_filter_state.register(list)
    def _(self, init: Union[Tuple, List]) -> GaussParams:
        return GaussParams(*init)
    
    @init_filter_state.register(dict)
    def _(self, init: dict) -> GaussParams:
        got_mean = False
        got_cov = False
    
        for key in init:
            if not got_mean and key in ["mean", "x", "m"]:
                mean = init[key]
                got_mean = True
            if not got_cov and key in ["cov", "P"]:
                cov = init[key]
                got_cov = True
    
        assert (
            got_mean and got_cov
        ), f"EKF do not recognize mean and cov keys in the dict {init}."
    
        return GaussParams(mean, cov)

# %% End
