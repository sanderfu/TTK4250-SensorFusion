# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

save_results = True

try: # see if tqdm is available, otherwise define it as a dummy
    try: # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
        __IPYTHON__
        from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
except Exception as e:
    print(e)
    print(
        "install tqdm (conda install tqdm, or pip install tqdm) to get nice progress bars. "
    )

    def tqdm(iterable, *args, **kwargs):
        return iterable

from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

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
filename_to_load = "task_real.mat"
loaded_data = scipy.io.loadmat(filename_to_load)

do_corrections = True # TODO: set to false for the last task
if do_corrections:
    S_a = loaded_data['S_a']
    S_g = loaded_data['S_g']
else:
    # Only accounts for basic mounting directions
    S_a = S_g = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T
accuracy_GNSS = loaded_data['GNSSaccuracy'].ravel()

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration)
steps = 400000
gnss_steps = len(z_GNSS)

# %% Measurement noise
# Continous noise
discrete_gyro_noise_std = 1.3*4.36e-5  # (rad/s)/sqrt(Hz)
discrete_acc_noise_std = 1.3*1.167e-3  # (m/s**2)/sqrt(Hz)

#Based on eq. 10.70
cont_gyro_noise_std = discrete_gyro_noise_std * np.sqrt(1/dt) # (rad/s)
cont_acc_noise_std = discrete_acc_noise_std * np.sqrt(1/dt)  # (m/s**2)

# Bias values
rate_bias_driving_noise_std = 5e-5*4
cont_rate_bias_driving_noise_std = (
    (1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)

acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)

# Position and velocity measurement
p_std = 0.2*np.array([1, 1, 2.5])  # Measurement noise
R_GNSS = np.diag(p_std ** 2)

p_acc = 1e-16

p_gyro = 1e-8
def RGNSS(GNSSk):
    return (accuracy_GNSS[GNSSk]**2)*R_GNSS*np.array([0.005, 0.005, 0.015])


# %% Estimator
eskf = ESKF(
    cont_acc_noise_std,
    cont_gyro_noise_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a = S_a, # set the accelerometer correction matrix
    S_g = S_g, # set the gyro correction matrix,
    debug=False # False to avoid expensive debug checks
)


# %% Allocate
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

NIS = np.zeros(gnss_steps)
NIS_planar = np.zeros(gnss_steps)
NIS_altitude = np.zeros(gnss_steps)

# %% Initialise
x_pred[0, POS_IDX] = np.array([0, 0, 0]) # starting 5 metres above ground
x_pred[0, VEL_IDX] = np.array([0, 0, 0]) # starting at 20 m/s due north
x_pred[0, ATT_IDX] = np.array([
    np.cos(45 * np.pi / 180),
    0, 0,
    np.sin(45 * np.pi / 180)
])  # nose to east, right to south and belly down.

P_pred[0][POS_IDX**2] = (2e-1)**2*np.eye(3)
P_pred[0][VEL_IDX**2] = (3e-4)**2*np.eye(3)
P_pred[0][ERR_ATT_IDX**2] = (1e-1*(np.pi/30))**2 * np.eye(3) # error rotation vector (not quat)
P_pred[0][ERR_ACC_BIAS_IDX**2] = 0.02**2 *np.eye(3)
P_pred[0][ERR_GYRO_BIAS_IDX**2] = (1e-4)**2 * np.eye(3)

# %% Run estimation
N=steps
doGNSS: bool = True

GNSSk: int = 0  # keep track of current step in GNSS measurements

stationary = True
diff_to_change=1
GNSSk_start_moving = 0
k_start_moving = 0

import logging
import threading
import time


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

nis_threads = []
def nis_calculations(x, P, z, GNSSk,R):
    NIS[GNSSk] = eskf.NIS_GNSS_position(x, P, z, R, lever_arm)
    NIS_planar[GNSSk] = eskf.NIS_Planar(x, P, z, R, lever_arm)
    NIS_altitude[GNSSk] = eskf.NIS_Altitude(x, P, z, R, lever_arm)

for k in tqdm(range(N)):


    if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:

        # if stationary:
        #     if GNSSk>0 and abs(la.norm(z_GNSS[GNSSk])-la.norm(z_GNSS[GNSSk-1])>diff_to_change):
        #         stationary=False
        #         x_pred[k,POS_IDX]=z_GNSS[GNSSk]
        #         P_pred[k][POS_IDX**2]=RGNSS(GNSSk)
        #         GNSSk_start_moving = GNSSk
        #         k_start_moving = k
        #         print(x_pred[k])
        #         print("Leaving stationary state, starting ESKF, k=",k," GNSSk=",GNSSk)
        #     else:
        #         x_pred[k+1]=x_pred[k]
        #         P_pred[k+1]=P_pred[k]
        #         print("Staying in stationary state, k=",k)
        #         GNSSk += 1
        #         continue
        x_est[k,:], P_est[k] = eskf.update_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], RGNSS(GNSSk), lever_arm)
        nis_thread = threading.Thread(target=nis_calculations, args=(x_est[k],P_est[k], z_GNSS[GNSSk], GNSSk,RGNSS(GNSSk)))
        nis_thread.start()
        nis_threads.append(nis_thread)



        if eskf.debug:
            assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"

        GNSSk += 1
    else:

        # if stationary:
        #     x_pred[k+1]=x_pred[k]
        #     P_pred[k+1]=P_pred[k]
        #     continue
        # no updates, so let us take estimate = prediction
        x_est[k,:] = x_pred[k,:] #Done
        P_est[k] = P_pred[k]

    if k < N - 1:
        x_pred[k + 1,:], P_pred[k + 1] = eskf.predict(x_est[k], P_est[k], z_acceleration[k+1], z_gyroscope[k+1], dt)  #Done : Hint: measurements come from the the present and past, not the future

    if eskf.debug:
        assert np.all(np.isfinite(P_pred[k])), f"Not finite P_pred at index {k + 1}"


# %% Plots
for thr in nis_threads:
    thr.join()
fig1 = plt.figure(1)
ax = plt.axes(projection='3d')

ax.plot3D(x_est[k_start_moving:N, 1], x_est[k_start_moving:N, 0], -x_est[k_start_moving:N, 2])
ax.plot3D(z_GNSS[GNSSk_start_moving:GNSSk, 1], z_GNSS[GNSSk_start_moving:GNSSk, 0], -z_GNSS[GNSSk_start_moving:GNSSk, 2])
ax.set_xlabel('East [m]')
ax.set_xlabel('North [m]')
ax.set_xlabel('Altitude [m]')

plt.grid()

# state estimation
t = np.linspace(0, dt*(N-1), N)
eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])

fig2, axs2 = plt.subplots(5, 1)

axs2[0].plot(t, x_est[0:N, POS_IDX])
axs2[0].set(ylabel='NED position [m]')
axs2[0].legend(['North', 'East', 'Down'])
plt.grid()

axs2[1].plot(t, x_est[0:N, VEL_IDX])
axs2[1].set(ylabel='Velocities [m/s]')
axs2[1].legend(['North', 'East', 'Down'])
plt.grid()

axs2[2].plot(t, eul[0:N] * 180 / np.pi)
axs2[2].set(ylabel='Euler angles [deg]')
axs2[2].legend(['\phi', '\theta', '\psi'])
plt.grid()

axs2[3].plot(t, x_est[0:N, ACC_BIAS_IDX])
axs2[3].set(ylabel='Accl bias [m/s^2]')
axs2[3].legend(['x', 'y', 'z'])
plt.grid()

axs2[4].plot(t, x_est[0:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
axs2[4].set(ylabel='Gyro bias [deg/h]')
axs2[4].legend(['x', 'y', 'z'])
plt.grid()

fig2.suptitle('States estimates')

# %% Consistency
confprob = 0.95
CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))

fig3, ax = plt.subplots()

ax.plot(NIS[:GNSSk], label='NIS')
ax.plot(NIS_planar[:GNSSk], label='NIS_Planar')
ax.plot(NIS_altitude[:GNSSk], label='NIS_Altitude')
ax.plot(np.array([0, N-1]) * dt, (CI3@np.ones((1, 2))).T)
insideCI = np.mean((CI3[0] <= NIS) * (NIS <= CI3[1]))
ax.legend()
plt.title(f'NIS ({100 *  insideCI:.1f} inside {100 * confprob} confidence interval)')
plt.grid()

# %% box plots
fig4 = plt.figure()

gauss_compare = np.sum(np.random.randn(3, GNSSk)**2, axis=0)
plt.boxplot([NIS[0:GNSSk], gauss_compare], notch=True)
plt.legend(['NIS', 'gauss'])
plt.grid()

plt.show()

# %%
plt.show()
from zipfile import ZipFile
import datetime
import re
import os
the_time = str(datetime.datetime.now())
the_time = re.sub(r':',r';', the_time)
the_time = re.sub(r' ',r'_', the_time)
print(the_time)
zipObj = ZipFile(f"test_real{the_time}.zip", 'w')



# with open("tuning_parameters.txt", "w+") as f:

#     f.write(f"Steps:{steps}\n")
#     f.write(f"cont_gyro_noise_std:{cont_gyro_noise_std}\n")
#     f.write(f"cont_acc_noise_std :{cont_acc_noise_std}\n")
#     f.write(f"rate_std: {rate_std}\n")
#     f.write(f"acc_std: {acc_std}\n")
#     f.write(f"rate_bias_driving_noise_std:{rate_bias_driving_noise_std}\n")
#     f.write(f"cont_rate_bias_driving_noise_std:{cont_rate_bias_driving_noise_std}\n")
#     f.write(f"acc_bias_driving_noise_std:{acc_bias_driving_noise_std}\n")
#     f.write(f"cont_acc_bias_driving_noise_std :{cont_acc_bias_driving_noise_std }\n")
#     f.write(f"p_std:{p_std}\n")
#     f.write(f"R_GNSS:{R_GNSS}\n")
#     f.write(f"p_acc:{p_acc}\n")
#     f.write(f"p_gyro:{p_gyro}\n")
#     f.write(f"P_pred[0]:{P_pred[0]}\n")
#     f.write(f"x_pred[0]:{x_pred[0]}\n")

# zipObj.write("tuning_parameters.txt")

if save_results:
    zipObj.write("run_INS_real.py")
    for i in plt.get_fignums():
        filename = f"fig_real{i}{the_time}.pdf"
        plt.figure(i)
        plt.savefig(filename)
        zipObj.write(filename)
        os.remove(filename)
    zipObj.close()
