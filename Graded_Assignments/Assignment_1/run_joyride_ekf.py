"""
File name: run_joyride_ekf.py

Creation Date: So 11 Okt 2020

Description: Track joyride with EKF filter.

"""

# Standard Python Libraries
# -----------------------------------------------------------------------------------------

from typing import List

import scipy
import scipy.io
import scipy.stats

import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local Application Modules
# -----------------------------------------------------------------------------------------
# %% imports

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import estimationstatistics as estats
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda

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
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
time = loaded_data["time"].squeeze()
time=time-time[0]

# plot measurements close to the trajectory

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)


# %% setup and track

# sensor
sigma_z = 25
clutter_intensity = 1e-5
PD = 0.90
gate_size = 5

# dynamic models
sigma_a_CV = 1.45
sigma_a_CT = 1.45
sigma_omega = 0.00005 * np.pi



mean_init_CV = np.array([7000,3600, 0, 0, 0])
cov_init_CV = np.zeros((5, 5))
cov_init_CV[[0, 1], [0, 1]] = 2 * sigma_z ** 2
cov_init_CV[[2, 3], [2, 3]] = 10 ** 2
cov_init_CV[4,4] = 0.1

mean_init_CT = np.array([7096,3627, 0, 0, 0])
cov_init_CT = np.zeros((5, 5))
cov_init_CT[[0, 1], [0, 1]] = 2 * sigma_z ** 2
cov_init_CT[[2, 3], [2, 3]] = 10 ** 2
cov_init_CT[4,4] = 0.1



init_ekf_CV_state = GaussParams(mean_init_CV, cov_init_CV)
init_ekf_CT_state = GaussParams(mean_init_CT, cov_init_CT)

init_states = [init_ekf_CV_state, init_ekf_CT_state]


# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))

tracker_EKF_CV = pda.PDA(ekf_filters[0], clutter_intensity, PD, gate_size)
tracker_EKF_CT = pda.PDA(ekf_filters[1], clutter_intensity, PD, gate_size)

trackers = [tracker_EKF_CV, tracker_EKF_CT]


NEES_CV = np.zeros(K)
NEESpos_CV = np.zeros(K)
NEESvel_CV = np.zeros(K)
NEES_CT = np.zeros(K)
NEESpos_CT = np.zeros(K)
NEESvel_CT = np.zeros(K)

tracker_update_CV = init_ekf_CV_state
tracker_update_list_CV = []
tracker_predict_list_CV = []
tracker_estimate_list_CV = []

tracker_update_CT = init_ekf_CT_state
tracker_update_list_CT = []
tracker_predict_list_CT = []
tracker_estimate_list_CT = []

Ts = np.append([2.5],Ts)

for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
    
    current_Ts = Ts[k]
    tracker_predict_CV = trackers[0].predict(tracker_update_CV, current_Ts) 
    tracker_update_CV = trackers[0].update(Zk, tracker_predict_CV) 
    tracker_estimate_CV = trackers[0].estimate(tracker_update_CV) 

    tracker_predict_CT = trackers[1].predict(tracker_update_CT, current_Ts) 
    tracker_update_CT = trackers[1].update(Zk, tracker_predict_CT) 
    tracker_estimate_CT = trackers[1].estimate(tracker_update_CT) 

    NEES_CV[k] = estats.NEES(*tracker_estimate_CV, x_true_k, idxs=np.arange(4))
    NEESpos_CV[k] = estats.NEES(*tracker_estimate_CV, x_true_k, idxs=np.arange(2))
    NEESvel_CV[k] = estats.NEES(*tracker_estimate_CV, x_true_k, idxs=np.arange(2, 4))

    NEES_CT[k] = estats.NEES(*tracker_estimate_CT, x_true_k, idxs=np.arange(4))
    NEESpos_CT[k] = estats.NEES(*tracker_estimate_CT, x_true_k, idxs=np.arange(2))
    NEESvel_CT[k] = estats.NEES(*tracker_estimate_CT, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list_CV.append(tracker_predict_CV)
    tracker_update_list_CV.append(tracker_update_CV)
    tracker_estimate_list_CV.append(tracker_estimate_CV)

    tracker_predict_list_CT.append(tracker_predict_CT)
    tracker_update_list_CT.append(tracker_update_CT)
    tracker_estimate_list_CT.append(tracker_estimate_CT)


x_hat_CV = np.array([est.mean for est in tracker_estimate_list_CV])
x_hat_CT = np.array([est.mean for est in tracker_estimate_list_CT])

# calculate a performance metrics
poserr_CV = np.linalg.norm(x_hat_CV[:, :2] - Xgt[:, :2], axis=0)
velerr_CV = np.linalg.norm(x_hat_CV[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE_CV = np.sqrt(
    np.mean(poserr_CV ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE_CV = np.sqrt(np.mean(velerr_CV ** 2))

poserr_CT = np.linalg.norm(x_hat_CT[:, :2] - Xgt[:, :2], axis=0)
velerr_CT = np.linalg.norm(x_hat_CT[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE_CT = np.sqrt(
    np.mean(poserr_CT ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE_CT = np.sqrt(np.mean(velerr_CT ** 2))

peak_pos_deviation_CV = poserr_CV.max()
peak_vel_deviation_CV = velerr_CV.max()
peak_pos_deviation_CT = poserr_CT.max()
peak_vel_deviation_CT = velerr_CT.max()


# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos_CV = np.mean(NEESpos_CV)
ANEESvel_CV = np.mean(NEESvel_CV)
ANEES_CV = np.mean(NEES_CV)

ANEESpos_CT = np.mean(NEESpos_CT)
ANEESvel_CT = np.mean(NEESvel_CT)
ANEES_CT = np.mean(NEES_CT)

# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat_CV.T[:2], label=r"$\hat x$_CV")
axs3[0].plot(*Xgt.T[:2], label="$x$")
axs3[0].set_title(
        f"CV:RMSE(pos, vel) = ({posRMSE_CV:.3f}, {velRMSE_CV:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation_CV:.3f}, {peak_vel_deviation_CV:.3f})"
)
axs3[0].axis("equal")
axs3[1].plot(*x_hat_CT.T[:2], label=r"$\hat x$ CT")
axs3[1].plot(*Xgt.T[:2], label="$x$")
axs3[1].set_title(
        f"CT: RMSE(pos, vel) = ({posRMSE_CT:.3f}, {velRMSE_CT:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation_CT:.3f}, {peak_vel_deviation_CT:.3f})"
)
axs3[1].axis("equal")
bbox_chosen = {'facecolor': 'white', 'alpha': 0.5, 'pad': 5}
for i in range(0,len(x_hat_CV.T[:2][0]), 15):
    axs3[0].text(x_hat_CV.T[0][i], x_hat_CV.T[1][i], f"t: {round(time[i],1)}",  style='oblique',
        bbox=bbox_chosen)
    axs3[1].text(x_hat_CT.T[0][i], x_hat_CT.T[1][i], f"t: {round(time[i],1)}",  style='oblique',
        bbox=bbox_chosen)

# NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
axs4[0].plot(time, NEESpos_CV)
axs4[0].plot([0, time[-1]], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos_CV")
inCIpos = np.mean((CI2[0] <= NEESpos_CV) * (NEESpos_CV <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(time, NEESvel_CV)
axs4[1].plot([0, time[-1]], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel_CV")
inCIvel = np.mean((CI2[0] <= NEESvel_CV) * (NEESvel_CV <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(time, NEES_CV)
axs4[2].plot([0, time[-1]], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES_CV")
inCI = np.mean((CI2[0] <= NEES_CV) * (NEES_CV <= CI2[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos_CV = {ANEESpos_CV:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel_CV = {ANEESvel_CV:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES_CV = {ANEES_CV:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(time, np.linalg.norm(x_hat_CV[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(time, np.linalg.norm(x_hat_CV[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")


# %% plots
# trajectory
# probabilities

# NEES
fig7, axs7 = plt.subplots(3, sharex=True, num=7, clear=True)
axs7[0].plot(time, NEESpos_CT)
axs7[0].plot([0, time[-1]], np.repeat(CI2[None], 2, 0), "--r")
axs7[0].set_ylabel("NEES pos_CT")
inCIpos = np.mean((CI2[0] <= NEESpos_CT) * (NEESpos_CT <= CI2[1]))
axs7[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs7[1].plot(time, NEESvel_CT)
axs7[1].plot([0, time[-1]], np.repeat(CI2[None], 2, 0), "--r")
axs7[1].set_ylabel("NEES vel_CT")
inCIvel = np.mean((CI2[0] <= NEESvel_CT) * (NEESvel_CT <= CI2[1]))
axs7[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs7[2].plot(time, NEES_CT)
axs7[2].plot([0, time[-1]], np.repeat(CI4[None], 2, 0), "--r")
axs7[2].set_ylabel("NEES_CT")
inCI = np.mean((CI2[0] <= NEES_CT) * (NEES_CT <= CI2[1]))
axs7[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos_CT = {ANEESpos_CT:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel_CT = {ANEESvel_CT:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES_CT = {ANEES_CT:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# errors
fig8, axs8 = plt.subplots(2, num=8, clear=True)
axs8[0].plot(time, np.linalg.norm(x_hat_CT[:, :2] - Xgt[:, :2], axis=1))
axs8[0].set_ylabel("position error")

axs8[1].plot(time, np.linalg.norm(x_hat_CT[:, 2:4] - Xgt[:, 2:4], axis=1))
axs8[1].set_ylabel("velocity error")

plt.show()

