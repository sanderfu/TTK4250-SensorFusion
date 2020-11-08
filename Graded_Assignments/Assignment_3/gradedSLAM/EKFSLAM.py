from typing import Tuple
import numpy as np
from scipy.linalg import block_diag
import scipy.linalg as la
from utils import rotmat2d
from JCBB import JCBB
import utils

# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class EKFSLAM:
    def __init__(
        self,
        Q,
        R,
        do_asso=False,
        alphas=np.array([0.001, 0.0001]),
        sensor_offset=np.zeros(2),
    ):

        self.Q = Q
        self.R = R
        self.do_asso = do_asso
        self.alphas = alphas
        self.sensor_offset = sensor_offset

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        x_prev = x[0]
        y_prev = x[1]
        psi_prev = x[2]
        
        x = x_prev + u[0]*np.cos(psi_prev) - u[1]*np.sin(psi_prev)
        y = y_prev + u[0]*np.sin(psi_prev) + u[1]*np.cos(psi_prev)
        psi = utils.wrapToPi(psi_prev + u[2])
        xpred = np.array([x,y,psi]) # TODO, eq (11.7). Should wrap heading angle between (-pi, pi), see utils.wrapToPi

        assert xpred.shape == (3,), "EKFSLAM.f: wrong shape for xpred"
        return xpred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """

        psi_prev = x[2]
        row0 = np.array([1, 0, -u[0]*np.sin(psi_prev)-u[1]*np.cos(psi_prev)])
        row1 = np.array([0, 1, u[0]*np.cos(psi_prev)-u[1]*np.sin(psi_prev)])
        row2 = np.array([0, 0, 1])
        Fx = np.vstack((row0, row1, row2))  #DONE, eq (11.13)

        assert Fx.shape == (3, 3), "EKFSLAM.Fx: wrong shape"
        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        psi_prev = x[2]
        row0 = np.array([np.cos(psi_prev), -np.sin(psi_prev), 0])
        row1 = np.array([np.sin(psi_prev), np.cos(psi_prev), 0])
        row2 = np.array([0,0,1])
        Fu = np.vstack((row0,row1,row2)) #eq (11.14)

        assert Fu.shape == (3, 3), "EKFSLAM.Fu: wrong shape"
        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        etapred = np.empty_like(eta)

        x = eta[:3]
        etapred[:3] = self.f(x, z_odo) # Done robot state prediction
        etapred[3:] = eta[3:] # Done landmarks: no effect

        Fx = self.Fx(x, z_odo)# Done
        Fu = self.Fu(x, z_odo)# Done

        # evaluate covariance prediction in place to save computation
        # only robot state changes, so only rows and colums of robot state needs changing
        # cov matrix layout:
        # [[P_xx, P_xm],
        # [P_mx, P_mm]]
        M = (np.shape(eta)[0] - 3)/2.0
        #Eq. 11.18 (they used some G without mentioning what it was, assuming identity
        P[:3, :3] = Fx@P[:3, :3]@Fx.T + Fu@self.Q@Fu.T# Done robot cov prediction
        P[:3, 3:] = Fx@P[:3, 3:]# Done robot-map covariance prediction
        P[3:, :3] = P[:3, 3:].T# Done map-robot covariance: transpose of the above

        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: calculated shapes does not match"
        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """
        # extract states and map
        x = eta[0:3]
        pos = x[:2].reshape((2,1))
        ## reshape map (2, #landmarks), m[:,j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T #DONE

        Rot = rotmat2d(x[2])        
        # None as index ads an axis with size 1 at that position.
        # Numpy broadcasts size 1 dimensions to any size when needed
        delta_m = m - pos # Done, relative position of landmark to sensor on robot in world frame

            

        zpredcart = Rot.T@delta_m - self.sensor_offset.reshape((2,1))# Done, predicted measurements in cartesian coordinates, beware sensor offset for VP

        zpred_r = la.norm(zpredcart, axis=0)# Done, ranges
        zpred_theta = np.arctan2(zpredcart[1], zpredcart[0]) # Done, bearings
        zpred = np.vstack((zpred_r, zpred_theta)) # Done, the two arrays above stacked on top of each other vertically like 
        # [ranges; 
        #  bearings]
        # into shape (2, #lmrk)

        zpred = zpred.T.ravel() # stack measurements along one dimension, [range1 bearing1 range2 bearing2 ...]

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"
        return zpred

    def H(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = eta[0:3]
        pos = x[:2].reshape((2,1))
        ## reshape map (2, #landmarks), m[:,j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        numM = m.shape[1]

        Rot = rotmat2d(x[2])

        delta_m = m - pos  #DONE, relative position of landmark to robot in world frame. m - rho that appears in (11.15) and (11.16)

        zc = delta_m - Rot@self.sensor_offset.reshape((2,1)) #DONE, (2, #measurements), each measured position in cartesian coordinates like
        # [x coordinates;
        #  y coordinates]
        Rpihalf = rotmat2d(np.pi / 2)
        
        
        # Allocate H and set submatrices as memory views into H
        # You may or may not want to do this like this
        H = np.zeros((2 * numM, 3 + 2 * numM)) #DONE, see eq (11.15), (11.16), (11.17)
        Hx = H[:, :3]  # slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # slice view, setting elements of Hm will set H as well

        for i in range(numM):
            zc_i = zc[:,i]
            delta_mi = delta_m[:,i]
            zrange = zc_i.T/la.norm(zc_i,2)@np.hstack((-np.eye(2), np.array(-Rpihalf@delta_mi).reshape(2,1)))
            zbearing = zc_i.T@Rpihalf.T/la.norm(zc_i, 2)**2 @ np.hstack((-np.eye(2), np.array(-Rpihalf@delta_mi).reshape(2,1)))
            zpred_i = np.vstack((zrange,zbearing))
            #Note: zpred_i = H_x^i
            H_xi = zpred_i
            H_mi = -H_xi[:,:2]
            
            assert (np.shape(H_xi)==np.shape(Hx[2*i:2*i+2,:])), "Hfunc, H_xi incorrect shape"
            assert (np.shape(H_mi)==np.shape(Hm[2*i:2*i+2,2*i:2*i+2])), "Hfunc, H_mi incorrect shape"

            #Placing H_xi and H_mi in H
            Hx[2*i:2*i+2,:]=H_xi
            Hm[2*i:2*i+2,2*i:2*i+2]=H_mi

        # In what follows you can be clever and avoid making this for all the landmarks you _know_
        # you will not detect (the maximum range should be available from the data).
        # But keep it simple to begin with.
        assert(np.shape(H) == (2*numM, 3+ 2*numM))
        # TONO: You can set some assertions here to make sure that some of the structure in H is correct
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk = z.shape[0] // 2

        lmnew = np.empty_like(z)

        

        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))

        I2 = np.eye(2) # Preallocate, used for Gx
        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset # For transforming landmark position into world frame
        sensor_offset_world_der = rotmat2d(eta[2] + np.pi / 2) @ self.sensor_offset # Used in Gx

        psi = eta[2]
        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            zj = z[inds]
            zj_r = zj[0]
            zj_b = zj[1]

            rot = rotmat2d(zj_b+psi) # Done, rotmat in Gz
            
            #Comment: Should we add or subtract sensor_offset_world?
            lmnew[inds] = eta[0:2] + np.array([zj_r*np.cos(psi+zj_b),zj_r*np.sin(psi+zj_b)]) + sensor_offset_world

            Gx[inds, :2] = I2
            Gx[inds, 2] = zj_r*np.array([-np.sin(zj_b+psi),np.cos(zj_b+psi)]) + sensor_offset_world_der# Done

            Gz = rot@np.diag([1, zj_r])# Done

            Rall[inds, inds] = Gz@self.R@Gz.T# Done, Gz * R * Gz^T, transform measurement covariance from polar to cartesian coordinates

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"
        etaadded = np.concatenate((eta, lmnew))# Done, append new landmarks to state vector
        Padded = la.block_diag(P, Gx@P[:3,:3]@Gx.T+Rall)# Done, block diagonal of P_new, see problem text in 1g) in graded assignment 3
        Padded[:n, n:] = P[:,:3]@Gx.T# Done, top right corner of P_new
        Padded[n:, :n] = Padded[:n, n:].T# Done, transpose of above. Should yield the same as calcualion, but this enforces symmetry and should be cheaper

        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            [description]
        P : np.ndarray
            [description]
        z : np.ndarray, shape=(#detections, 2)
            [description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta) #Done
            H = self.H(eta)
            # Here you can use simply np.kron (a bit slow) to form the big (very big in VP after a while) R,
            # or be smart with indexing and broadcasting (3d indexing into 2d mat) realizing you are adding the same R on all diagonals
            R_large = np.kron(np.eye(numLmk),self.R)
            S = H@P@H.T+R_large
            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta #Done
                Pupd = P# Done
                #Comment: If no assoc, good idea to set NIS = E[NIS] = DoF
                #Comment2: A maybe better idea is to skip entirely
                NIS = 2 # Done: beware this one when analysing consistency.
                NIS_ranges = 2
                NIS_bearings = 2

            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update
                # S_cho_factors = la.cho_factor(Sa) # Optional, used in places for S^-1, see scipy.linalg.cho_factor and scipy.linalg.cho_solve

                Sa_cho_factor = la.cho_factor(Sa)

                #Calculating the innovation, already done
                #innovation = z-zpred

                #Eq 11.19
                W = P@la.cho_solve(Sa_cho_factor,Ha).T    #Done, Kalman gain, can use S_cho_factors
                etaupd = eta + W@v #Done, Kalman update
                
                # Kalman cov update: use Joseph form for stability
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1  # same as adding Identity mat
                Pupd = jo@P@jo.T+W@R_large[:len(v), :len(v)]@W.T# Done, Kalman update. This is the main workload on VP after speedups

                # calculate NIS, can use S_cho_factors
                v_ranges = v[::2]
                za_ranges = za[::2]
                v_bearings = v[1::2]
                za_bearings = v[1::2]
                
                Sa_cho_factor_ranges = la.cho_factor(Sa[::2, ::2])
                Sa_cho_factor_bearings = la.cho_factor(Sa[1::2, 1::2])
                NIS_ranges = v_ranges.T@la.cho_solve(Sa_cho_factor_ranges,v_ranges) 
                NIS_bearings = v_bearings.T@la.cho_solve(Sa_cho_factor_bearings,v_bearings) 


                NIS = v.T@la.cho_solve(Sa_cho_factor,v) #Done

                # When tested, remove for speed
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(
                    np.linalg.eigvals(Pupd) > 0
                ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 2 #Done: beware this one, you can change the value to for instance 1
            NIS_ranges = 2
            NIS_bearings = 2
            etaupd = eta
            Pupd = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new) # Done, add new landmarks.

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, NIS_ranges, NIS_bearings, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0 # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
