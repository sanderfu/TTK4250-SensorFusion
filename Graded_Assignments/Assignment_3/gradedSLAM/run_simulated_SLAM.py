# %% Imports
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
from utils import wrapToPi

#Imports for multithreading and logging results
import logging
import threading
from zipfile import ZipFile
import datetime
import re
import os

save_results = True

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 1,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
            "legend.loc" : "upper right",
            'legend.fontsize': 10,
            # Font
            "font.size" : 15,
            #Subplots and figure
            "figure.figsize" : [8,7],
            "figure.subplot.wspace" : 0.37,
            "figure.subplot.hspace" : 0.41,
            "figure.subplot.top" : 0.9,
            "figure.subplot.right" : 0.95,
            "figure.subplot.left" : 0.1,
        }
    )



from EKFSLAM import EKFSLAM
from plotting import ellipse

# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")

## NB: this is a MATLAB cell, so needs to "double index" to get out the measurements of a time step k:
#
# ex:
#
# z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
#
##
z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T
pose_dim = len(poseGT[0])
K = len(z)
M = len(landmarks)

# %% Initilize
Q = np.diag([0.05**2,0.009*2,(0.4*np.pi/180)**2]) #INITDONE
R = np.diag([0.07**2, (1*np.pi/180)**2]) #INITDONE

doAsso = True


'''
Explanation of JCBBalphas:
    Both alphas are used for gating in JCBB. Smaller alpha = larger gate generally.
    JCBBalpha[0]: For gating joint compatability.
    JCBBalpha[1]: For gating individual compatability
    
    Remark: Affects runtime significantly and slam accuracy also to some extent.
    Based on experience from this assignment, alphas must be low (large gates) 
    especially when the measurement noise is set low
'''
JCBBalphas = np.array([0.0001, 0.0001])

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

# Allocate matrices and vectors
eta_pred: List[Optional[np.ndarray]] = [None] * K
P_pred: List[Optional[np.ndarray]] = [None] * K
eta_hat: List[Optional[np.ndarray]] = [None] * K
P_hat: List[Optional[np.ndarray]] = [None] * K
a: List[Optional[np.ndarray]] = [None] * K
NIS = np.zeros(K)
NIS_ranges = np.zeros(K)
NIS_bearings = np.zeros(K)
NISnorm = np.zeros(K)
NISnorm_ranges = np.zeros(K)
NISnorm_bearings = np.zeros(K)

CI = np.zeros((K, 2))
CI_ranges_bearings = np.zeros((K, 2))
CInorm = np.zeros((K, 2))
CInorm_ranges_bearings = np.zeros((K, 2))
NEESes = np.zeros((K, 3))

# For consistency testing
alpha = 0.05
confprob = 1 - alpha

# Initial values
#Comment: We start at the correct position for reference
eta_pred[0] = poseGT[0]

#Comment: We say that we are very certain about this start position
P_pred[0] = 1e-4 * np.eye(3)

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = False
if doAssoPlot:
    figAsso, axAsso = plt.subplots(num=1, clear=True)

# %% Run simulation
N = 1000

print("starting simulation (" + str(N) + " iterations)")

for k, z_k in tqdm(enumerate(z[:N])):

    eta_hat[k], P_hat[k], NIS[k], NIS_ranges[k], NIS_bearings[k], a[k] = slam.update(eta_pred[k],np.copy(P_pred[k]),z_k)

    if k < K - 1:
        eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k],np.copy(P_hat[k]),odometry[k])

    assert (
        eta_hat[k].shape[0] == P_hat[k].shape[0]
    ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI[k] = chi2.interval(confprob, 2 * num_asso)
    CI_ranges_bearings[k] = chi2.interval(confprob, num_asso)

    if num_asso > 0:
        NISnorm[k] = NIS[k] / (2 * num_asso)
        NISnorm_ranges[k] = NIS_ranges[k]/num_asso
        NISnorm_bearings[k] = NIS_bearings[k]/num_asso
        CInorm[k] = CI[k] / (2 * num_asso)
        CInorm_ranges_bearings[k] = CI_ranges_bearings[k]/num_asso
    else:
        NISnorm[k] = 1
        NISnorm_ranges[k] = 1
        NISnorm_bearings[k] = 1
        CInorm[k].fill(1)
        CInorm_ranges_bearings[k].fill(1)
    NEESes[k] = slam.NEESes(eta_hat[k][:pose_dim],P_hat[k][:pose_dim, :pose_dim],poseGT[k]) #Done, use provided function slam.NEESes

    if doAssoPlot and k > 0:
        axAsso.clear()
        axAsso.grid()
        zpred = slam.h(eta_pred[k]).reshape(-1, 2)
        axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
        axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
        xcoords = np.block([[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
        ycoords = np.block([[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
        for x, y in zip(xcoords, ycoords):
            axAsso.plot(x, y, lw=3, c="r")
        axAsso.legend()
        axAsso.set_title(f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
        plt.draw()
        plt.pause(0.001)


print("Simulation complete")

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

fig2, ax2 = plt.subplots(num=2, clear=True)
# landmarks
ax2.scatter(*landmarks.T, c="r", marker="^",label="Ground truth landmarks")
ax2.scatter(*lmk_est_final.T, c="b", marker=".", label="Estimated landmarks")
# Draw covariance ellipsis of measurements
for l, lmk_l in enumerate(lmk_est_final):
    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
    rI = P_hat[N - 1][idxs, idxs]
    el = ellipse(lmk_l, rI, 5, 200)
    ax2.plot(*el.T, "b")

ax2.plot(*poseGT.T[:2], c="r", label="Ground truth position")
ax2.plot(*pose_est.T[:2], c="g", label="Estimated position")
ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="g")
ax2.set(title="Map", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
ax2.axis("equal")
ax2.grid()
plt.xlabel("[m]")
plt.ylabel("[m]")
ax2.legend()

# %% Consistency
print("\n----------\nConsistency results\n----------\n")

# NIS
insideCI = (CInorm[:N,0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N,1])

fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(CInorm[:N,0], '--')
ax3.plot(CInorm[:N,1], '--')
ax3.plot(NISnorm[:N], lw=0.5)

ax3.set_title(f'NIS, {np.round(insideCI.mean()*100,2)}% inside {confprob*100}% CI')
insideCI_ranges = (CInorm_ranges_bearings[:N,0] <= NISnorm_ranges[:N]) * (NISnorm_ranges[:N] <= CInorm_ranges_bearings[:N,1])
insideCI_bearings = (CInorm_ranges_bearings[:N,0] <= NISnorm_bearings[:N]) * (NISnorm_bearings[:N] <= CInorm_ranges_bearings[:N,1])

fig7, ax7 = plt.subplots(nrows=2, ncols=1,num=7, clear=True)

ax7[0].plot(CInorm[:N,0], '--')
ax7[0].plot(CInorm[:N,1], '--')
ax7[0].plot(NISnorm[:N], lw=0.5)

ax7[0].legend(['CI lower', 'CI upper', 'NIS'])

ax7[0].set_title(f'NIS, {np.round(insideCI.mean()*100,2)}% inside {confprob*100}% CI\n')
ax7[1].plot(CInorm_ranges_bearings[:N,0], '--', color='blue')
ax7[1].plot(CInorm_ranges_bearings[:N,1], '--', color='blue')
ax7[1].plot(NISnorm_ranges[:N], lw=0.5, color='purple')
ax7[1].plot(NISnorm_bearings[:N], lw=0.5, color='red')
ax7[1].legend(['CI lower', 'CI upper','NIS ranges', 'NIS bearings'])
ax7[1].set_title(f'NIS_ranges, {np.round(insideCI_ranges.mean()*100,2)}% inside {confprob*100}% CI\nNIS_bearings, {np.round(insideCI_bearings.mean()*100,2)}% inside {confprob*100}% CI')


ANIS = np.mean(NISnorm[:N])
CI_ANIS = np.array(chi2.interval(confprob,N))/N
print(f"ANIS: {ANIS}")
print(f"CI ANIS: {CI_ANIS}")

# NEES
fig4, ax4 = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), num=4, clear=True, sharex=True)
tags = ['pose', 'position', 'heading']
dfs = [3, 2, 1]

for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
    CI_NEES = chi2.interval(confprob, df)
    ax.plot(np.full(N, CI_NEES[0]), '--')
    ax.plot(np.full(N, CI_NEES[1]), '--')
    ax.plot(NEES[:N], lw=0.5)
    insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
    ax.set_title(f'NEES {tag}: {np.round(insideCI.mean()*100,2)}% inside {confprob*100}% CI')

    CI_ANEES = np.array(chi2.interval(confprob, df*N)) / N
    print(f"CI ANEES {tag}: {CI_ANEES}")
    print(f"ANEES {tag}: {NEES.mean()}")

# %% RMSE

ylabels = ['m', 'deg']
scalings = np.array([1, 180/np.pi])
fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), num=5, clear=True, sharex=True)
pos_err = np.linalg.norm(pose_est[:N,:2] - poseGT[:N,:2], axis=1)
heading_err = np.abs(wrapToPi(pose_est[:N,2] - poseGT[:N,2]))
errs = np.vstack((pos_err, heading_err))

for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
    ax.plot(err*scaling)
    ax.set_title(f"{tag}: RMSE {np.round(np.sqrt((err**2).mean())*scaling,2)} {ylabel}")
    ax.set_ylabel(f"[{ylabel}]")
    ax.grid()

# %% Movie time

if playMovie:
    try:
        print("recording movie...")

        from celluloid import Camera

        pauseTime = 0.05
        fig_movie, ax_movie = plt.subplots(num=6, clear=True)

        camera = Camera(fig_movie)

        ax_movie.grid()
        ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
        camera.snap()

        for k in tqdm(range(N)):
            ax_movie.scatter(*landmarks.T, c="r", marker="^")
            ax_movie.plot(*poseGT[:k, :2].T, "r-")
            ax_movie.plot(*pose_est[:k, :2].T, "g-")
            ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

            if k > 0:
                el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                ax_movie.plot(*el.T, "g")

            numLmk = lmk_est[k].shape[0]
            for l, lmk_l in enumerate(lmk_est[k]):
                idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                rI = P_hat[k][idxs, idxs]
                el = ellipse(lmk_l, rI, 5, 200)
                ax_movie.plot(*el.T, "b")

            camera.snap()
        animation = camera.animate(interval=10, blit=True, repeat=False)
        print("playing movie")

    except ImportError:
        print(
            "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
        )

# %% Save plots
the_time = str(datetime.datetime.now())
the_time = re.sub(r':',r';', the_time)
the_time = re.sub(r' ',r'_', the_time)
print(the_time)

if save_results:
    zipObj = ZipFile(f"test_simulated{the_time}.zip", 'w')
    for i in plt.get_fignums():
        filename = f"fig_simulated{i}{the_time}.pdf"
        plt.figure(i)
        plt.savefig(filename)
        zipObj.write(filename)
        os.remove(filename)
    zipObj.close()

plt.show()
# %%
