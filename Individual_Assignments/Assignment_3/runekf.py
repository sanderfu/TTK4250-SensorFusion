# %% Imports
from gaussparams import GaussParams
import measurmentmodels
import dynamicmodels
import ekf
import scipy
import scipy.stats
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# to see your plot config
print(f'matplotlib backend: {matplotlib.get_backend()}')
print(f'matplotlib config file: {matplotlib.matplotlib_fname()}')
print(f'matplotlib config dir: {matplotlib.get_configdir()}')
plt.close('all')

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ['science', 'grid', 'bright','no-latex']
    plt.style.use(plt_styles)
    print(f'pyplot using style set {plt_styles}')
except Exception as e:
    print(e)
    print('setting grid and only grid and legend manually')
    plt.rcParams.update({
        # set grid
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': 'k',
        'grid.alpha': 0.5,
        'grid.linewidth': 0.5,
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.fancybox': True,
        'legend.numpoints': 1,
        #Improve resolution
        
    })


# %% get and plot the data
data_path = 'data_for_ekf.mat'

# TODO: choose this for the last task
usePregen = True  # choose between own generated data and pre generated

if usePregen:
    loadData: dict = scipy.io.loadmat(data_path)
    K: int = int(loadData['K'])  # The number of time steps
    Ts: float = float(loadData['Ts'])  # The sampling time
    Xgt: np.ndarray = loadData['Xgt'].T  # ground truth
    Z: np.ndarray = loadData['Z'].T  # the measurements
else:
    from sample_CT_trajectory import sample_CT_trajectory
    np.random.seed(10)  # random seed can be set for repeatability

    # initial state distribution
    x0 = np.array([0, 0, 1, 1, 0])
    P0 = np.diag([50, 50, 10, 10, np.pi/4]) ** 2

    # model parameters to sample from # TODO for toying around
    sigma_a_true = 0.25
    sigma_omega_true = np.pi/15
    sigma_z_true = 3

    # sampling interval a length
    K = 1000
    Ts = 0.1

    # get data
    Xgt, Z = sample_CT_trajectory(
        K, Ts, x0, P0, sigma_a_true, sigma_omega_true, sigma_z_true)

# show ground truth and measurements
fig, ax = plt.subplots(num=1, clear=True,dpi=200)
ax.scatter(*Z.T, color='C0', marker='.')
ax.plot(*Xgt.T[:2], color='C1')
ax.set_title('Data')

# show turn rate
fig2, ax2 = plt.subplots(num=2, clear=True,dpi=200)
ax2.plot(Xgt.T[4])
ax2.set_xlabel('time step')
ax2.set_ylabel('turn rate')




# %% a: tune by hand and comment

# set parameters
sigma_a = 0.6  # Tuning DONE
sigma_z = 0.1  # Tuning DONE

#Comment: With these values the kalman filter output follows the ground truth quite accurately. By trusting the model more than the measurement,
# we get less noise in our output and this was in focus when tuning to follow the ground truth in a smooth manner.

# create the model and estimator object
dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measmod = measurmentmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(dynmod, measmod)
print(ekf_filter)  # make use of the @dataclass automatic repr

# initialize mean and covariance
# DONE: ArrayLike (list, np. array, tuple, ...) with 4 elements
x_bar_init = np.array([0,0,1,1])
"""
    Awaiting answer in forum on how to choose P0
"""
P_bar_init = np.diag([50,50,10,10]) ** 2  # DONE: ArrayLike with 4 x 4 elements, hint: np.diag 
init_ekfstate = ekf.GaussParams(x_bar_init, P_bar_init)

# estimate
ekfpred_list, ekfupd_list = ekf_filter.estimate_sequence(Z,init_ekfstate,Ts,)

# get statistics:
stats = ekf_filter.performance_stats_sequence(
    K, Z=Z, ekfpred_list=ekfpred_list, ekfupd_list=ekfupd_list, X_true=Xgt[:, :4],
    norm_idxs=[[0, 1], [2, 3]], norms=[2, 2]
)

print(f'keys in stats is {stats.dtype.names}')

# %% Calculate average performance metrics
# stats['dists_pred'] contains 2 norm of position and speed for each time index
# same for 'dists_upd'
# DONE: square stats['dists_pred'] -> take its mean over time -> take square root

RMSE_pred = [np.sqrt(pow(stats['dists_pred'][:,i],2).mean()) for i in range(stats['dists_pred'].shape[1])]
RMSE_upd = [np.sqrt(pow(stats['dists_upd'][:,i],2).mean()) for i in range(stats['dists_upd'].shape[1])]

fig3, ax3 = plt.subplots(num=3, clear=True,dpi=200)

ax3.plot(*Xgt.T[:2])
ax3.plot(*ekfupd_list.mean.T[:2])
RMSEs_str = ", ".join(f"{v:.2f}" for v in (*RMSE_pred, *RMSE_upd))
ax3.set_title(
    rf'$\sigma_a = {sigma_a}$, $\sigma_z= {sigma_z}$,' + f'\nRMSE(p_p, p_v, u_p, u_v) = ({RMSEs_str})')
# %% Task 5 b and c

# % parameters for the parameter grid
# TODO: pick reasonable values for grid search
# n_vals = 20  # is Ok, try lower to begin with for more speed (20*20*1000 = 400 000 KF steps)
n_vals = 5
sigma_a_low = 0.05
sigma_a_high = 10
sigma_z_low = 0.1
sigma_z_high = 100

# % set the grid on logscale(not mandatory)
sigma_a_list = np.logspace(
    np.log10(sigma_a_low), np.log10(sigma_a_high), n_vals, base=10
)
sigma_z_list = np.logspace(
    np.log10(sigma_z_low), np.log10(sigma_z_high), n_vals, base=10
)

dtype = stats.dtype  # assumes the last cell has been run without faults
stats_array = np.empty((n_vals, n_vals, K), dtype=dtype)
# %% run through the grid and estimate
# ? Should be more or less a copy of the above
for i, sigma_a in enumerate(sigma_a_list):
    dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)  # DONE
    # initialize mean and covariance
    # DONE: ArrayLike (list, np. array, tuple, ...) with 4 elements
    x_bar_init = np.array([0,0,0,0])
    P_bar_init = np.diag([sigma_a,sigma_a,sigma_a,sigma_a])
    init_ekfstate = ekf.GaussParams(x_bar_init, P_bar_init)
        
    for j, sigma_z in enumerate(sigma_z_list):
        
        measmod = measurmentmodels.CartesianPosition(sigma_z)  # DONE
        ekf_filter = ekf.EKF(dynmod,measmod)  # DONE

        ekfpred_list, ekfupd_list = ekf_filter.estimate_sequence(Z,init_ekfstate,Ts)  # DONE
        stats_array[i, j] = ekf_filter.performance_stats_sequence(K,Z=Z,ekfpred_list=ekfpred_list,ekfupd_list=ekfupd_list, X_true=Xgt[:,:4],norm_idxs=[[0,1],[2,3]],norms=[2,2]) # DONE
# %% calculate averages

# DONE, remember to use axis argument, see eg. stats_array['dists_pred'].shape
RMSE_pred = np.empty((n_vals, n_vals, 2))
for i in range(stats_array.shape[0]):
    for j in range(stats_array.shape[1]):
        RMSE_pred[i,j] = [np.sqrt(pow(stats_array[i,j]['dists_pred'][:,k],2).mean()) for k in range(stats_array[i,j]['dists_pred'].shape[1])]  # DONE
        
RMSE_upd = np.empty((n_vals, n_vals, 2))
for i in range(stats_array.shape[0]):
    for j in range(stats_array.shape[1]):
        RMSE_upd[i,j] = [np.sqrt(pow(stats_array[i,j]['dists_upd'][:,k],2).mean()) for k in range(stats_array[i,j]['dists_upd'].shape[1])]  # DONE
        
ANEES_pred = np.empty((n_vals, n_vals))
for i in range(stats_array.shape[0]):
    for j in range(stats_array.shape[1]):
        ANEES_pred[i,j] = stats_array[i,j]['NEESpred'].mean() # DONE (ANEES: mean of NEES over time)
        
ANEES_upd = np.empty((n_vals, n_vals))
for i in range(stats_array.shape[0]):
    for j in range(stats_array.shape[1]):
        ANEES_upd[i,j] = stats_array[i,j]['NEESupd'].mean() #DONE
        
ANIS = np.empty((n_vals, n_vals))
for i in range(stats_array.shape[0]):
    for j in range(stats_array.shape[1]):
        ANIS[i,j] = stats_array[i,j]['NIS'].mean() # DONE mean of NIS over time


# %% find confidence regions for NIS and plot
confprob = 0.95  # TODO number to use for confidence interval
CINIS = scipy.stats.chi2.interval(confprob,2*K,scale=1/K)  # TODO confidence intervall for NIS, hint: scipy.stats.chi2.interval
print(CINIS)

# plot
fig4 = plt.figure(4, clear=True,dpi=400)
ax4 = plt.gca(projection='3d')
ax4.plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                 ANIS, alpha=0.9)
ax4.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
            ANIS, [1, 1.5, *CINIS, 2.5, 3], offset=0)  # , extend3d=True, colors='yellow')
ax4.set_xlabel(r'$\sigma_a$')
ax4.set_ylabel(r'$\sigma_z$')
ax4.set_zlabel('ANIS')
ax4.set_zlim(0, 10)
ax4.view_init(30, 20)

# %% find confidence regions for NEES and plot
confprob = np.nan  # TODO
CINEES = np.nan  # TODO, not NIS now, but very similar
print(CINEES)

# plot
fig5 = plt.figure(5, clear=True)
ax5s = [fig5.add_subplot(1, 2, 1, projection='3d'),
        fig5.add_subplot(1, 2, 2, projection='3d')]
ax5s[0].plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                     ANEES_pred, alpha=0.9)
ax5s[0].contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                ANEES_pred, [3, 3.5, *CINEES, 4.5, 5], offset=0)
ax5s[0].set_xlabel(r'$\sigma_a$')
ax5s[0].set_ylabel(r'$\sigma_z$')
ax5s[0].set_zlabel('ANEES_pred')
ax5s[0].set_zlim(0, 50)
ax5s[0].view_init(40, 30)

ax5s[1].plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                     ANEES_upd, alpha=0.9)
ax5s[1].contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                ANEES_upd, [3, 3.5, *CINEES, 4.5, 5], offset=0)
ax5s[1].set_xlabel(r'$\sigma_a$')
ax5s[1].set_ylabel(r'$\sigma_z$')
ax5s[1].set_zlabel('ANEES_upd')
ax5s[1].set_zlim(0, 50)
ax5s[1].view_init(40, 30)

# %% see the intersection of NIS and NEESes
fig6, ax6 = plt.subplots(num=6, clear=True)
cont_upd = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                       ANEES_upd, CINEES, colors=['C0', 'C1'], labels='ANEESupd')
cont_pred = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                        ANEES_pred, CINEES, colors=['C2', 'C3'], labels='ANEESpred')
cont_nis = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                       ANIS, CINIS, colors=['C4', 'C5'], labels='NIS')

for cs, l in zip([cont_upd, cont_pred, cont_nis], ['NEESupd', 'NEESpred', 'NIS']):
    for c, hl in zip(cs.collections, ['low', 'high']):
        c.set_label(l + '_' + hl)
ax6.legend()
ax6.set_xlabel(r'$\sigma_a$')
ax6.set_ylabel(r'$\sigma_z$')

# %% show all the plots
plt.show()
