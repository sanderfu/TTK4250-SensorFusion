# %% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

rng = np.random.default_rng()
# %% trajectory generation
# scenario parameters
x0 = np.array([np.pi / 2, -np.pi / 100])
Ts = 0.05
K = round(10 / Ts)

# constants
g = 9.81
l = 1
a = g / l
d = 0.5  # dampening
S = 5

# disturbance PDF
process_noise_sampler = lambda: rng.uniform(-S, S)

# dynamic function
def modulo2pi(x, idx=0):
    xmod = x
    xmod[idx] = (xmod[idx] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return xmod


def pendulum_dynamics(x, a, d=0):  # continuous dynamics
    xdot = np.array([x[1], -d * x[1] - a * np.sin(x[0])])
    return xdot


def pendulum_dynamics_discrete(xk, vk, Ts, a, d=0):
    xkp1 = modulo2pi(xk + Ts * pendulum_dynamics(xk, a, d))  # euler discretize
    xkp1[1] += Ts * vk  #  zero orde hold noise
    return xkp1


# sample a trajectory
x = np.zeros((K, 2))
x[0] = x0
for k in range(K - 1):
    v = process_noise_sampler()
    x[k + 1] = pendulum_dynamics_discrete(x[k], v, Ts, a, d)


# vizualize
fig1, axs1 = plt.subplots(2, sharex=True, num=1, clear=True)
axs1[0].plot(x[:, 0])
axs1[0].set_ylabel(r"$\theta$")
axs1[0].set_ylim((-np.pi, np.pi))

axs1[1].plot(x[:, 1])
axs1[1].set_xlabel("Time step")
axs1[1].set_ylabel(r"$\dot \theta$")

# %% measurement generation

# constants
Ld = 4
Ll = 1
r = 0.25

# noise pdf
measurement_noise_sampler = lambda: rng.triangular(-r, 0, r)

# measurement function
def h(x, Ld, l, Ll):  # measurement function
    lcth = l * np.cos(x[0])
    lsth = l * np.sin(x[0])
    z = np.sqrt((Ld - lcth) ** 2 + (lsth - Ll) ** 2)  # 2norm
    return z


Z = np.zeros(K)
for k, xk in enumerate(x):
    wk = measurement_noise_sampler()
    Z[k] = h(x[k], Ld, l, Ll) + wk


# vizualize
fig2, ax2 = plt.subplots(num=2, clear=True)
ax2.plot(Z)
ax2.set_xlabel("Time step")
ax2.set_ylabel("z")

# %% Task: Estimate using a particle filter

# number of particles to use
N = 2000 # DONE

# initialize particles, pretend you do not know where the pendulum starts
"""
The point: 
    Create N particles that combined are distributed over all possible state combinations for the system.
    Use the uniform distribution to make sure we do not get any consentration of particles initially,
    as it would indicate that we are more confident that the state is at this concentration than elsewhere.
"""
px = np.array([
    rng.uniform(-np.pi, np.pi, N),  # x1 = theta
    rng.normal(size=N) * np.pi/4  # x2 = theta_dot
    ]).T

# initial weights
"""
We do not know which particle is good or bad, so set all to equal weight 1/N so that they sum to 1 (normalized)

"""
w = np.full((N,1), 1/N)

# allocate some space for resampling particles
pxn = np.zeros_like(px)

# PF transition PDF: SIR proposal, or something you would like to test
PF_dynamic_distribution = scipy.stats.uniform(loc=-S, scale=2 * S)
PF_measurement_distribution = scipy.stats.triang(c=0.5, loc=-r, scale=2 * r)

# initialize a figure for particle animation.
plt.ion()
fig4, ax4 = plt.subplots(num=4, clear=True)
plotpause = 0.01

sch_particles = ax4.scatter(np.nan, np.nan, marker=".", c="b", label=r"$\hat \theta^n$")
sch_true = ax4.scatter(np.nan, np.nan, c="r", marker="x", label=r"$\theta$")
ax4.set_ylim((-1.5 * l, 1.5 * l))
ax4.set_xlim((-1.5 * l, 1.5 * l))
ax4.set_xlabel("x")
ax4.set_ylabel("y")
th = ax4.set_title(f"theta mapped to x-y")
ax4.legend()

particle_out = np.zeros((K,2))

eps = np.finfo(float).eps
for k in range(K):
    print(f"k = {k}")
    # weight update
    # update weights using (5.39)
    for n in range(N):
        # Using (5.39) to perform weight update
        w[n] = PF_measurement_distribution.pdf(Z[k] - h(px[n], Ld, l, Ll))
    w = w + eps #Avoid round-off error
    w = w / np.sum(w)#normalize

    N_eff = 1 / np.sum(w ** 2)
    print(f"N_eff = {N_eff}")

    # resample using algorithm 3 p. 90
    # DONE: some pre calculations -> cumulative sum
    cumw = np.cumsum(w) # staircase cdf
    noise = rng.random((1, 1)) / N
    i = 0
    for n in range(N):
        # find a particle 'i' to pick
        # algorithm in the book, but there are other options as well
        u = n/N + noise
        while u>cumw[i]:
            i += 1
        pxn[n] = px[i] # resampled particles, instead of indicesout
    rng.shuffle(pxn,axis=0) # shuffle to maximize randomness
    w.fill(1 / N)
    # resetting all weights to 1/N, it's not mentioned in the book.

    # trajecory sample prediction
    # propose new particles using (5.38)
    for n in range(N):
        # process noise, hint: PF_dynamic_distribution.rvs
        vkn = PF_dynamic_distribution.rvs()
        # particle prediction/proposal according proposal density q = p (5.38)
        px[n] = pendulum_dynamics_discrete(pxn[n], vkn, Ts, a)

    # centroid of theta position of particles
    px_theta_centroid = np.mean(pxn[:, 0], axis=0)
    particle_out[k] = px_theta_centroid

    # plot
    show_plot = False
    if show_plot:
        sch_particles.set_offsets(np.c_[l * np.sin(pxn[:, 0]), -l * np.cos(pxn[:, 0])])
        sch_true.set_offsets(np.c_[l * np.sin(x[k, 0]), -l * np.cos(x[k, 0])])

        fig4.canvas.draw_idle()
        plt.show(block=False)
        plt.waitforbuttonpress(plotpause)
if show_plot:
    plt.waitforbuttonpress()

# %% plot particle centroid path
fig5, ax5 = plt.subplots(num=5, clear=True)
ax5.plot(particle_out[:,0], label='Particle theta mean')
ax5.plot(x[:,0], label='True theta')
ax5.set_xlabel("Time step")
ax5.set_ylabel(r"$\theta_{particles}$")
ax5.legend()
ax5.set_title('Particle mean vs true path')
plt.show()
plt.waitforbuttonpress()
