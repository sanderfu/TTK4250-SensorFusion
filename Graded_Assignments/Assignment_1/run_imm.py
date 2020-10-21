# %% imports
import numpy as np
import scipy
import scipy.io
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import imm
import ukf
import estimationstatistics as estats


# %% plot config check and style setup

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

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

# %% load data
use_pregen = True
# you can generate your own data if set to false
if use_pregen:
    data_filename = "data_for_imm.mat"
    loaded_data = scipy.io.loadmat(data_filename)
    Z = loaded_data["Z"].T
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
else:
    K = 100
    Ts = 2.5
    sigma_z = 2.25
    sigma_a = 0.7  # effective all the time
    sigma_omega = 5e-4 * np.pi  # effective only in turns

    init_x = [0, 0, 2, 0, 0]
    init_P = np.diag([25, 25, 3, 3, 0.0005]) ** 2
    # [Xgt, Z] = simulate_atc(q, r, K, init, false);
    raise NotImplementedError


fig1, ax1 = plt.subplots(num=1, clear=True)
ax1.plot(*Xgt.T[:2])
ax1.scatter(*Z.T[:2])


# %% tune single filters

# parameters
sigma_z = 3
sigma_a_CV = 0.3
sigma_a_CT = 0.1
sigma_omega = 0.002 * np.pi

# initial values
init_mean = np.array([2, 0, 0, 0, 0])
init_cov = np.diag([5, 5, 1, 1, 0.0005]) ** 2

init_state_CV = GaussParams(init_mean[:4], init_cov[:4, :4])  # get rid of turn rate
init_state_CT = GaussParams(init_mean, init_cov)  # same init otherwise
init_states = [init_state_CV, init_state_CT]

# create models
measurement_model_CV = measurementmodels.CartesianPosition(sigma_z)
measurement_model_CT = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)

# create filters
filters = []
filters.append(ukf.UKF(CV, measurement_model_CV))
filters.append(ukf.UKF(CT, measurement_model_CT))

# allocate
pred = []
upd = []
NIS = np.empty((2, K))
NEES_pred = np.empty((2, K))
NEES_upd = np.empty((2, K))
err_pred = np.empty((2, 2, K))  # (filters, vel/pos, time)
err_upd = np.empty((2, 2, K))  # (filters, vel/pos, time)

# per filter
for i, (ukf_filter, init) in enumerate(zip(filters, init_states)):
    # setup per filter
    updated = init
    ukfpred_list = []
    ukfupd_list = []

    # over time steps
    for k, (zk, x_gt_k) in enumerate(zip(Z, Xgt)):  # bypass any UKF.sequence problems
        # filtering
        predicted = ukf_filter.predict(updated, Ts)
        updated = ukf_filter.update(zk, predicted)

        # store per time
        ukfpred_list.append(predicted)
        ukfupd_list.append(updated)


    # stor per filter
    pred.append(ukfpred_list)
    upd.append(ukfupd_list)

    # extract means and covs for metric processing
    x_bar = np.array([p.mean for p in ukfpred_list])
    x_hat = np.array([u.mean for u in ukfupd_list])

    P_bar = np.array([p.cov for p in ukfpred_list])
    P_hat = np.array([u.cov for u in ukfupd_list])


    #err_pred[i, 0] = estats.distance_sequence_indexed(x_bar, Xgt, np.arange(2))
    #err_pred[i, 1] = estats.distance_sequence_indexed(x_bar, Xgt, np.arange(2, 4))
    #err_upd[i, 0] = estats.distance_sequence_indexed(x_hat, Xgt, np.arange(2))
    #err_upd[i, 1] = estats.distance_sequence_indexed(x_hat, Xgt, np.arange(2, 4))


# errors
import ipdb
ipdb.set_trace()


# plot
"""
fig2, axs2 = plt.subplots(2, 2, num=2, clear=True)
for axu, axl, u_s, rmse_pred, rmse_upd in zip(
    axs2[0], axs2[1], upd, RMSE_pred, RMSE_upd
):
    # ax.scatter(*Z.T)
    x = np.array([data.mean for data in u_s])
    axu.plot(*x.T[:2])
    rmsestr = ", ".join(f"{num:.3f}" for num in (*rmse_upd, *rmse_pred))
    axu.set_title(f"RMSE(p_u, v_u, p_pr, v_pr)|\n{rmsestr}|")
    axu.axis("equal")
    if x.shape[1] >= 5:
        axl.plot(np.arange(K) * Ts, x.T[4])
    axl.plot(np.arange(K) * Ts, Xgt[:, 4])

axs2[1, 0].set_ylabel(r"$\omega$")

fig3, axs3 = plt.subplots(1, 3, num=3, clear=True)

axs3[0].plot(np.arange(K) * Ts, NIS[0])
axs3[0].plot(np.arange(K) * Ts, NIS[1])
axs3[0].set_title("NIS")

axs3[1].plot(np.arange(K) * Ts, err_upd[:, 0].T)
# axs3[1].plot(np.arange(K) * Ts, err_upd[1, :, 0])
axs3[1].set_title("pos error")

axs3[2].plot(np.arange(K) * Ts, err_upd[:, 1].T)
# axs3[2].plot(np.arange(K) * Ts, err_upd[1, :, 1])
axs3[2].set_title("vel error")

"""
# %% tune IMM by only looking at the measurements
sigma_z = 3.0
sigma_a_CV = 0.2
sigma_a_CT = 0.1
sigma_omega = 0.002 * np.pi
PI = np.array([[0.95, 0.05], [0.05, 0.95]])
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"
# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ukf_filters = []
ukf_filters.append(ukf.UKF(CV, measurement_model))
ukf_filters.append(ukf.UKF(CT, measurement_model))
imm_filter = imm.IMM(ukf_filters, PI)

init_weights = np.array([0.5] * 2)
init_mean = [0] * 5
init_cov = np.diag(
    [1] * 5
)  # HAVE TO BE DIFFERENT: use intuition, eg. diag guessed distance to true values squared.
init_mode_states = [GaussParams(init_mean, init_cov)] * 2  # copy of the two modes
init_immstate = MixtureParameters(init_weights, init_mode_states)

imm_preds = []
imm_upds = []
imm_ests = []
updated_immstate = init_immstate
for zk in Z:
    predicted_immstate = imm_filter.predict(updated_immstate, Ts)
    updated_immstate = imm_filter.update(zk, predicted_immstate)
    estimate = imm_filter.estimate(updated_immstate)

    imm_preds.append(predicted_immstate)
    imm_upds.append(updated_immstate)
    imm_ests.append(estimate)

x_est = np.array([est.mean for est in imm_ests])
prob_est = np.array([upds.weights for upds in imm_upds])


# plot
fig4, axs4 = plt.subplots(2, 2, num=4, clear=True)
axs4[0, 0].plot(*x_est.T[:2], label="est", color="C0")
axs4[0, 0].scatter(*Z.T, label="z", color="C1")
axs4[0, 0].legend()
axs4[0, 1].plot(np.arange(K) * Ts, x_est[:, 4], label=r"$\omega$")
axs4[0, 1].legend()
axs4[1, 0].plot(np.arange(K) * Ts, prob_est, label=r"$Pr(s)$")
axs4[1, 0].legend()
plt.show()
