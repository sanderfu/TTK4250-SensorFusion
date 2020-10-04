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

# local
import dynamicmodels as dynmods
import measurementmodels as measmods
from gaussparams import GaussParams, GaussParamList
from mixturedata import MixtureParameters
import mixturereduction
from singledispatchmethod import singledispatchmethod

# %% The EKF


@dataclass
class EKF:
    # A Protocol so duck typing can be used
    dynamic_model: dynmods.DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: measmods.MeasurementModel

    #_MLOG2PIby2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._MLOG2PIby2: Final[float] = self.sensor_model.m * \
            np.log(2 * np.pi) / 2

    def predict(self,
                ekfstate: GaussParams,
                # The sampling time in units specified by dynamic_model
                Ts: float,
                ) -> GaussParams:
        """Predict the EKF state Ts seconds ahead."""

        x, P = ekfstate  # tuple unpacking

        F = self.dynamic_model.F(x, Ts)
        Q = self.dynamic_model.Q(x, Ts)

        x_pred = self.dynamic_model.f(x,Ts)
        
        #Predicted covariance matrix. Using definitin from Algorithm 1 p. 54
        P_pred = F@P@F.T+Q  # DONE
        
        #Packing predicted mean and predicted covariance matrix into GaussParams object.
        state_pred = GaussParams(x_pred, P_pred)

        return state_pred

    def innovation_mean(
            self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the innovation mean for ekfstate at z in sensor_state."""

        x = ekfstate.mean

        #Predicted measurement (Algorithm 1 p.54)
        zbar = self.sensor_model.h(x)
        
        #Innovation (Algorithm 1 p.54)
        v = z-zbar

        return v

    def innovation_cov(self,
                       z: np.ndarray,
                       ekfstate: GaussParams,
                       *,
                       sensor_state: Dict[str, Any] = None,
                       ) -> np.ndarray:
        """Calculate the innovation covariance for ekfstate at z in sensorstate."""

        #Tuple unpacking
        x, P = ekfstate
        
        
        H = self.sensor_model.H(x, sensor_state=sensor_state)
        R = self.sensor_model.R(x, sensor_state=sensor_state, z=z)
        
        #Innovation covariance (Algorithm 1, p.54)
        S = H@P@H.T+R

        return S

    def innovation(self,
                   z: np.ndarray,
                   ekfstate: GaussParams,
                   *,
                   sensor_state: Dict[str, Any] = None,
                   ) -> GaussParams:
        """Calculate the innovation for ekfstate at z in sensor_state."""

        v = self.innovation_mean(z,ekfstate)
        S = self.innovation_cov(z,ekfstate)

        innovationstate = GaussParams(v, S)

        return innovationstate

    def update(self,
               z: np.ndarray,
               ekfstate: GaussParams,
               sensor_state: Dict[str, Any] = None
               ) -> GaussParams:
        """Update ekfstate with z in sensor_state"""
        
        #Tuple unpacking of prediction
        x, P = ekfstate

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        H = self.sensor_model.H(x, sensor_state=sensor_state)
        
        #Calculate the Kalman Gain (Algorithm 1, p. 54)
        W = P@H.T@la.inv(S)

        #The mean and covariance update (Algorithm 1, p. 54)
        x_upd = x+W@v
        P_upd = (np.eye(4)-W@H)@P

        #Tuple packing
        ekfstate_upd = GaussParams(x_upd, P_upd)

        return ekfstate_upd

    def step(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             # sampling time
             Ts: float,
             *,
             sensor_state: Dict[str, Any] = None,
             ) -> GaussParams:
        """Predict ekfstate Ts units ahead and then update this prediction with z in sensor_state."""

        ekfstate_pred = self.predict(ekfstate,Ts)
        ekfstate_upd = self.update(z,ekfstate_pred)
        
        return ekfstate_upd

    def NIS(self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
            ) -> float:
        """Calculate the normalized innovation squared for ekfstate at z in sensor_state"""

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        #Equation 4.66 p.67
        NIS = v.T@la.inv(S)@v

        return NIS

    @classmethod
    def NEES(cls,
             ekfstate: GaussParams,
             # The true state to comapare against
             x_true: np.ndarray,
             ) -> float:
        """Calculate the normalized etimation error squared from ekfstate to x_true."""

        x, P = ekfstate

        x_diff = x-x_true
        
        #Equation 4.65 p. 67
        NEES = x_diff.T@la.inv(P)@x_diff
        
        return NEES

    def gate(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             *,
             sensor_state: Dict[str, Any],
             gate_size_square: float,
             ) -> bool:
        """Check if z is inside sqrt(gate_sized_squared)-sigma ellipse of
        ekfstate in sensor_state """

        #Equation unnumbered, page 116.
        gated = self.NIS(z,ekfstate,sensor_state=sensor_state)<gate_size_square
        return gated

    def loglikelihood(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate the log likelihood of ekfstate at z in sensor_state
            Comment: This function was retrieved from Blackboard.
            Comment2: What they ask for here is log(p(z|x_k(posteriori))) for a 
            realization of z. I should modify this function to allow for using 
            a method that makes me understand better what is going on.
            Comment3: For completeness, realize that likelihood function is 
            the same thign as the measurement function (Eq. 4.6)
        """
    
        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)
    
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
    def estimate(cls, ekfstate: GaussParams):
        """Get the estimate from the state with its covariance. (Compatibility method)"""
        # dummy function for compatibility with IMM class
        return ekfstate

    def estimate_sequence(
            self,
            # A sequence of measurements
            Z: Sequence[np.ndarray],
            # the initial KF state to use for either prediction or update (see start_with_prediction)
            init_ekfstate: GaussParams,
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
        ekfupd = init_ekfstate
        n = init_ekfstate.mean.shape[0]
        ekfpred_list = GaussParamList.allocate(K, n)
        ekfupd_list = GaussParamList.allocate(K, n)

        # perform the actual predict and update cycle
        # DONE loop over the data and get both the predicted and updated states in the lists
        # the predicted is good to have for evaluation purposes
        # A potential pythonic way of looping through  the data
        for k, (zk, Tsk, ssk) in enumerate(zip(Z, Ts_arr, sensor_state_seq)):
            ekfpred = self.predict(ekfupd,Tsk)
            ekfupd = self.update(zk,ekfpred,Tsk)
            ekfpred_list[k]=ekfpred
            ekfupd_list[k]=ekfupd
        return ekfpred_list, ekfupd_list

    def performance_stats(
            self,
            *,
            z: Optional[np.ndarray] = None,
            ekfstate_pred: Optional[GaussParams] = None,
            ekfstate_upd: Optional[GaussParams] = None,
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
        if z is not None and ekfstate_pred is not None:
            stats['NIS'] = self.NIS(
                z, ekfstate_pred, sensor_state=sensor_state)

        # NEES and RMSE, needs ground truth
        if x_true is not None:
            # prediction
            if ekfstate_pred is not None:
                stats['NEESpred'] = self.NEES(ekfstate_pred, x_true)

                # distances
                err_pred = ekfstate_pred.mean - x_true
                if norm_idxs is None:
                    stats['dist_pred'] = np.linalg.norm(err_pred, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_pred'] = [
                        np.linalg.norm(err_pred[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]

            # update
            if ekfstate_upd is not None:
                stats['NEESupd'] = self.NEES(ekfstate_upd, x_true)

                # distances
                err_upd = ekfstate_upd.mean - x_true
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
            ekfpred_list: Optional[Iterable[GaussParams]] = None,
            ekfupd_list: Optional[Iterable[GaussParams]] = None,
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
        for_iter.append(ekfpred_list or None_list)
        for_iter.append(ekfupd_list or None_list)
        for_iter.append(sensor_state or None_list)
        for_iter.append(X_true if X_true is not None else None_list)

        stats = []
        for zk, ekfpredk, ekfupdk, ssk, xtk in zip(*for_iter):
            stats.append(
                self.performance_stats(
                    z=zk, ekfstate_pred=ekfpredk, ekfstate_upd=ekfupdk, sensor_state=ssk, x_true=xtk,
                    norm_idxs=norm_idxs, norms=norms
                )
            )

        # make structured array
        dtype = [(key, *((type(val[0]), len(val)) if isinstance(val, Iterable)
                         else (type(val),))) for key, val in stats[0].items()]
        stats_arr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

        return stats_arr

    def reduce_mixture(
        self, ekfstate_mixture: MixtureParameters[GaussParams]
    ) -> GaussParams:
        """Merge a Gaussian mixture into single mixture"""
        w = ekfstate_mixture.weights
        x = np.array([c.mean for c in ekfstate_mixture.components], dtype=float)
        P = np.array([c.cov for c in ekfstate_mixture.components], dtype=float)
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