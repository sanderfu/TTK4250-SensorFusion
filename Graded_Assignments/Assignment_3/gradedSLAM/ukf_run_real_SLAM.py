# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]

import scipy.linalg as la
import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d
from EKFSLAM import EKFSLAM
from UKFSLAM import UKFSLAM

#Imports for multithreading and logging results
import logging
import threading
from zipfile import ZipFile
import datetime
import re
import os

save_results=True

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
            "figure.subplot.hspace" : 0.76,
            "figure.subplot.top" : 0.9,
            "figure.subplot.right" : 0.95,
            "figure.subplot.left" : 0.1,
        }
    )
# %% Load data
VICTORIA_PARK_PATH = "./victoria_park/"
realSLAM_ws = {
    **loadmat(VICTORIA_PARK_PATH + "aa3_dr"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_lsr2"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_gpsx"),
}

timeOdo = (realSLAM_ws["time"] / 1000).ravel()
timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

steering = realSLAM_ws["steering"].ravel()
speed = realSLAM_ws["speed"].ravel()
LASER = (
    realSLAM_ws["LASER"] / 100
)  # Divide by 100 to be compatible with Python implementation of detectTrees
La_m = realSLAM_ws["La_m"].ravel()
Lo_m = realSLAM_ws["Lo_m"].ravel()

K = timeOdo.size
mK = timeLsr.size
Kgps = timeGps.size

# %% Parameters

L = 2.83  # axel distance
H = 0.76  # center to wheel encoder
a = 0.95  # laser distance in front of first axel
b = 0.5  # laser distance to the left of center

car = Car(L, H, a, b)

sigmas = [0.01,0.008,(0.115*np.pi/180)]
CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)*1.5

# %% Initilize
#Q = np.diag([0.1**2,0.1**2,(np.pi/180)**2]) #INITDONE
R = np.diag([0.06**2, (0.12*np.pi/180)**2])*1.1 #INITDONE


JCBBalphas = np.array(
    [1e-7, 1e-7]  # INITDONE
)










sensorOffset = np.array([car.a + car.L, car.b])
doAsso = True

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)

# For consistency testing
alpha = 0.05
confidence_prob = 1 - alpha


ekfslam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)
ukfslam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)

xupd = np.zeros((mK, 3))
a = [None] * mK
NIS = np.zeros(mK)
NISnorm = np.zeros(mK)
CI = np.zeros((mK, 2))
CInorm = np.zeros((mK, 2))

NIS_ranges = np.zeros(mK)
NIS_bearings = np.zeros(mK)
NIS_gnss = np.zeros(Kgps)

NISnorm_ranges = np.zeros(mK)
NISnorm_bearings = np.zeros(mK)
NISnorm_gnss = np.zeros(Kgps)

CI_ranges_bearings = np.zeros((mK, 2))
CInorm_ranges_bearings = np.zeros((mK, 2))
CI_gnss = np.zeros((Kgps, 2))
CInorm_gnss = np.zeros((Kgps, 2))


# Initialize state
eta = np.array([Lo_m[0], La_m[1], 36 * np.pi / 180]) # you might want to tweak these for a good reference
P = 0.00001*np.eye(3)
P_cached = np.copy(P)

mk_first = 1  # first seems to be a bit off in timing
mk = mk_first
t = timeOdo[0]

# %%  run
print(K)
N = K#K

doPlot = False

lh_pose = None

if doPlot:
    fig, ax = plt.subplots(num=1, clear=True)

    lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
    sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
    sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")


#Warning: slam.predict modifies the P function it takes in, and without using np.copy this is a shallow copy meaning we modify the original!
do_raw_prediction = True
if do_raw_prediction:  # TODO: further processing such as plotting
    odos = np.zeros((K, 3))
    odox = np.zeros((K, 3))
    odox[0] = eta

    for k in range(min(N, K - 1)):
        odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
        odox[k + 1], _ = ekfslam.predict(odox[k], np.copy(P), odos[k + 1])

assert np.allclose(P,P_cached), "P has been modified in function!!"
squared_error = 0
do_gnss_update = True
k_gnss = 0
#R_gnss = np.diag([0.45**2,0.45**2])*30
R_gnss = np.diag([0.45**2,0.45**2])*30

for k in tqdm(range(N)):
    
    if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
        # Force P to symmetric: there are issues with long runs (>10000 steps)
        # seem like the prediction might be introducing some minor asymetries,
        # so best to force P symetric before update (where chol etc. is used).
        # TODO: remove this for short debug runs in order to see if there are small errors
        #P = (P + P.T) / 2
        dt = timeLsr[mk] - t
        if dt < 0:  # avoid assertions as they can be optimized avay?
            raise ValueError("negative time increment")

        t = timeLsr[mk]  # ? reset time to this laser time for next post predict
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        
        #Comment: Do not think the line below should be here.
        #eta, P = slam.predict(eta,P,odo) # Done predict

        z = detectTrees(LASER[mk])
        eta, P,  NIS[mk], NIS_ranges[mk], NIS_bearings[mk], a[mk] =  ekfslam.update(eta,P,z, True) # TODO update

        num_asso = np.count_nonzero(a[mk] > -1)

        if num_asso > 0:
            NISnorm[mk] = NIS[mk] / (2 * num_asso)
            NISnorm_ranges[mk] = NIS_ranges[mk] /num_asso
            NISnorm_bearings[mk] = NIS_bearings[mk]/num_asso
            
            CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                2 * num_asso
            )            
            CInorm_ranges_bearings[mk] = np.array(chi2.interval(confidence_prob, num_asso)) / (
                num_asso
            )
        else:
            NISnorm[mk] = 1
            NISnorm_ranges[mk] = 1
            NISnorm_bearings[mk] = 1
            
            CInorm[mk].fill(1)

            CInorm_ranges_bearings[mk].fill(1)

        xupd[mk] = eta[:3]

        if doPlot:
            sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
            if len(z) > 0:
                zinmap = (
                    rotmat2d(eta[2])
                    @ (
                        z[:, 0] * np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                        + slam.sensor_offset[:, None]
                    )
                    + eta[0:2, None]
                )
                sh_Z.set_offsets(zinmap.T)
            lh_pose.set_data(*xupd[mk_first:mk, :2].T)

            ax.set(
                xlim=[-200, 200],
                ylim=[-200, 200],
                title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
            )
            plt.draw()
            plt.pause(0.00001)

        mk += 1
    if k_gnss < Kgps-1 and timeGps[k_gnss]<=timeOdo[k+1]:
        z_gnss = np.array([Lo_m[k_gnss], La_m[k_gnss]])
        squared_error += la.norm(eta[:2]-z_gnss, 2)**2

        if do_gnss_update:
            eta, P, NIS_gnss[k_gnss] =  slam.updateGNSS(eta, P, z_gnss, R_gnss) # Done update
            NISnorm_gnss[k_gnss] = NIS_gnss[k_gnss]/2

            CInorm_gnss[k_gnss] = np.array(chi2.interval(confidence_prob, 2)) / 2 
        k_gnss +=1        
    if k < K - 1:
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = ukfslam.predict(eta,P,odo) # Done predict


#RMSE for pose, where GT is GPS measurements
RMSE = np.sqrt(squared_error/k_gnss)

# %% Consistency
# NIS
confprob = confidence_prob
insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * (NISnorm[:mk] <= CInorm[:mk, 1])
ANIS = np.mean(NISnorm[:mk])
CI_ANIS = np.array(chi2.interval(confidence_prob,mk))/mk
print(f"\nANIS: {ANIS}")
print(f"CI ANIS: {CI_ANIS}")
fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(CInorm[:mk, 0], "--")
ax3.plot(CInorm[:mk, 1], "--")
ax3.plot(NISnorm[:mk], lw=0.5)

ax3.set_title(f"NIS Laser Measurements, {insideCI.mean()*100:.2f}% inside CI")
insideCI_ranges = (CInorm_ranges_bearings[:mk,0] <= NISnorm_ranges[:mk]) * (NISnorm_ranges[:mk] <= CInorm_ranges_bearings[:mk,1])
insideCI_bearings = (CInorm_ranges_bearings[:mk,0] <= NISnorm_bearings[:mk]) * (NISnorm_bearings[:mk] <= CInorm_ranges_bearings[:mk,1])

fig7, ax7 = plt.subplots(nrows=3, ncols=1,num=7, clear=True)

ax7[0].plot(CInorm[:mk,0], '--')
ax7[0].plot(CInorm[:mk,1], '--')
ax7[0].plot(NISnorm[:mk], lw=0.5)
ax7[0].legend(['CI lower', 'CI upper', 'NIS_laserMeasurements'])
ax7[0].set_title(f'NIS, {np.round(insideCI.mean()*100,2)}% inside {confprob*100}% CI\n')

ax7[1].plot(CInorm_ranges_bearings[:mk,0], '--', color='blue')
ax7[1].plot(CInorm_ranges_bearings[:mk,1], '--', color='blue')
ax7[1].plot(NISnorm_ranges[:mk], lw=0.5, color='purple')
ax7[1].plot(NISnorm_bearings[:mk], lw=0.5, color='red')
ax7[1].legend(['CI lower', 'CI upper','NIS ranges', 'NIS bearings'])
ax7[1].set_title(f'NIS_ranges, {np.round(insideCI_ranges.mean()*100,2)}% inside {confprob*100}% CI\nNIS_bearings, {np.round(insideCI_bearings.mean()*100,2)}% inside {confprob*100}% CI')


insideCI_gnss = (CInorm_gnss[:k_gnss, 0] <= NISnorm_gnss[:k_gnss]) * (NISnorm_gnss[:k_gnss] <= CInorm_gnss[:k_gnss, 1])

ax7[2].plot(CInorm_gnss[:k_gnss,0], '--')
ax7[2].plot(CInorm_gnss[:k_gnss,1], '--')
ax7[2].plot(NISnorm_gnss[:k_gnss], lw=0.5)
ax7[2].legend(['CI lower', 'CI upper', 'NIS_gnss'])
ax7[2].set_title(f'NIS_gnss, {np.round(insideCI_gnss.mean()*100,2)}% inside {confprob*100}% CI\n')
# %% slam

if do_raw_prediction:
    fig5, ax5 = plt.subplots(num=5, clear=True)
    ax5.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
    ax5.plot(*odox[:N, :2].T, label="odom")
    ax5.plot(*xupd[mk_first:mk, :2].T, label="SLAM position")
    ax5.grid()
    ax5.set_title(f"Comparison of position. \nTotal RMSE for position: {np.round(RMSE,2)}")
    ax5.legend()

# %%
fig6, ax6 = plt.subplots(num=6, clear=True)
ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x")
ax6.plot(*xupd[mk_first:mk, :2].T)
ax6.set(
    title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
)


# %% Save plots
the_time = str(datetime.datetime.now())
the_time = re.sub(r':',r';', the_time)
the_time = re.sub(r' ',r'_', the_time)
print(the_time)

if save_results:
    zipObj = ZipFile(f"test_ukf_real{the_time}.zip", 'w')
    for i in plt.get_fignums():
        filename = f"fig_ukf_real{i}{the_time}.pdf"
        plt.figure(i)
        plt.savefig(filename)
        zipObj.write(filename)
        os.remove(filename)
    zipObj.close()



plt.show()