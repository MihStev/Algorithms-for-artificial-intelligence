import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import laplace as laplace_dist

# CPTs (0 = negative, 1 = positive)
# Dimension convention: P_XY[x, y] = P(X=x | Y=y)

P_A = np.array([0.3, 0.7])                          # [a]

P_BA = np.array([[0.3, 0.8],                         # [b, a]
                 [0.7, 0.2]])

P_CA = np.array([[0.4, 0.6],                         # [c, a]
                 [0.6, 0.4]])

P_DAB = np.array([[[0.1, 0.5],                       # [d, a, b]
                   [0.2, 0.8]],
                  [[0.9, 0.5],
                   [0.8, 0.2]]])

P_ECD = np.array([[[0.3, 0.8],                       # [e, c, d]
                   [0.6, 0.6]],
                  [[0.7, 0.2],
                   [0.4, 0.4]]])

P_FD = np.array([[0.2, 0.6],                         # [f, d]
                 [0.8, 0.4]])

# Task 1.1 — DAG

def draw_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
        ("D", "F"),
    ])

    pos = {
        "A": (0, 2),
        "B": (-1, 1),
        "C": (1, 1),
        "D": (0, 0),
        "E": (0, -1),
        "F": (1, -1),
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#4C9BE8", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=16, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        edge_color="#222222",
        width=2,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    ax.set_title("Bayesian Network — DAG", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("src/homework_2/dag.png", dpi=150)
    plt.show()


# Task 1.2a — Variable Elimination
# Elimination order C → A → D (justification in the report)

def variable_elimination():
    # Factors with evidence instantiated
    f_Ee_CD = P_ECD[0]    # P(e-|C,D),  shape [c, d]
    f_Ff_D  = P_FD[0]     # P(f-|D),    shape [d]

    # Step 1 — eliminate C
    # f1[a,d] = Sum_c P(C=c|A=a) . P(e-|C=c,D=d)
    f1 = np.einsum('ca,cd->ad', P_CA, f_Ee_CD)

    # Step 2 — eliminate A
    # f2[b,d] = Sum_a P(A=a).P(B=b|A=a).P(D=d|A=a,B=b).f1[a,d]
    f2 = np.einsum('a,ba,dab,ad->bd', P_A, P_BA, P_DAB, f1)

    # Step 3 — eliminate D
    # f3[b] = Sum_d f2[b,d] . P(f-|D=d)
    f3 = np.einsum('bd,d->b', f2, f_Ff_D)

    prob = f3 / f3.sum()
    return prob


# Task 1.2b — Rejection Sampling
# Topological sampling (A → B,C → D → E,F), reject if E!=e- or F!=f-

def rejection_sampling(N, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    a = rng.binomial(1, P_A[1],         N)       # P(a+)
    b = rng.binomial(1, P_BA[1, a])              # P(b+ | a)
    c = rng.binomial(1, P_CA[1, a])              # P(c+ | a)
    d = rng.binomial(1, P_DAB[1, a, b])          # P(d+ | a, b)
    e = rng.binomial(1, P_ECD[1, c, d])          # P(e+ | c, d)
    f = rng.binomial(1, P_FD[1, d])              # P(f+ | d)

    accepted = b[(e == 0) & (f == 0)]
    if len(accepted) == 0:
        return np.nan
    return accepted.mean()


# Task 1.2d — Gibbs Sampling
# Fix E=e-, F=f- and cyclically sample A,B,C,D from the conditional
# distributions given by the Markov blanket (formula next to each step below)

def gibbs_sampling(N, burn_in=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Random initialization of the hidden variables
    a, b, c, d = rng.integers(0, 2, size=4)

    samples_b = np.empty(N, dtype=int)

    for i in range(burn_in + N):
        # P(A | B=b, C=c, D=d)
        p_a = P_A * P_BA[b, :] * P_CA[c, :] * P_DAB[d, :, b]
        p_a /= p_a.sum()
        a = rng.choice(2, p=p_a)

        # P(B | A=a, D=d)
        p_b = P_BA[:, a] * P_DAB[d, a, :]
        p_b /= p_b.sum()
        b = rng.choice(2, p=p_b)

        # P(C | A=a, D=d, E=0)
        p_c = P_CA[:, a] * P_ECD[0, :, d]
        p_c /= p_c.sum()
        c = rng.choice(2, p=p_c)

        # P(D | A=a, B=b, C=c, E=0, F=0)
        p_d = P_DAB[:, a, b] * P_ECD[0, c, :] * P_FD[0, :]
        p_d /= p_d.sum()
        d = rng.choice(2, p=p_d)

        if i >= burn_in:
            samples_b[i - burn_in] = b

    return samples_b.mean()


# Task 1.2 — Monte Carlo loop (Nr=100)
# N_RS is chosen analytically (std_RS = sqrt(p(1-p) / (P(e-,f-)*N))), while
# N_Gibbs is chosen empirically (estimate std on N_cal=300 samples, then
# scale) — target: std ~= 0.025

def run_monte_carlo(Nr=100, target_std=0.025, seed=42):
    ve_prob   = variable_elimination()
    ve_exact  = ve_prob[1]

    # P(e-, f-) — normalization factor from VE (= acceptance rate for RS)
    f1 = np.einsum('ca,cd->ad', P_CA, P_ECD[0])
    f2 = np.einsum('a,ba,dab,ad->bd', P_A, P_BA, P_DAB, f1)
    f3 = np.einsum('bd,d->b', f2, P_FD[0])
    acc_rate = float(f3.sum())

    # N for RS (analytical)
    N_rs = int(np.ceil(ve_exact * (1 - ve_exact) / (acc_rate * target_std**2)))

    # N for Gibbs (empirical: estimate std on N_cal samples, then scale)
    rng_cal = np.random.default_rng(seed - 1)
    N_cal   = 300
    cal_est = np.array([gibbs_sampling(N_cal, burn_in=200, rng=rng_cal) for _ in range(50)])
    N_gibbs = int(np.ceil(N_cal * (cal_est.std() / target_std) ** 2))

    print(f"Target std           ~= {target_std:.3f}")
    print(f"P(e-, f-)           = {acc_rate:.4f}")
    print(f"N (Rejection)        = {N_rs}  (accepted ~= {int(N_rs * acc_rate)})")
    print(f"N (Gibbs)            = {N_gibbs}  (+burn-in 200)")

    rng = np.random.default_rng(seed)
    rs_est    = np.array([rejection_sampling(N_rs,    rng) for _ in range(Nr)])
    gibbs_est = np.array([gibbs_sampling(N_gibbs, burn_in=200, rng=rng) for _ in range(Nr)])

    _, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    configs = [
        (axes[0], rs_est,    f"Rejection Sampling  (N = {N_rs})"),
        (axes[1], gibbs_est, f"Gibbs Sampling  (N = {N_gibbs},  burn-in = 200)"),
    ]
    for ax, est, title in configs:
        mean, std = est.mean(), est.std()
        ax.hist(est, bins=20, color="#4C9BE8", edgecolor="white", alpha=0.85)
        ax.axvline(ve_exact, color="crimson", lw=2,   ls="--",
                   label=f"VE (exact) = {ve_exact:.4f}")
        ax.axvline(mean,     color="black",   lw=1.5, ls="-",
                   label=f"Mean = {mean:.4f},  Std = {std:.4f}")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Number of runs")
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Estimate  P(b+ | e-, f-)")
    plt.suptitle(f"Monte Carlo estimate  P(b+ | e-, f-),   Nr = {Nr}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("src/homework_2/mc_histograms.png", dpi=150)
    plt.show()

    print(f"\n{'Method':<32} {'Mean':>8} {'Std':>8} {'N':>6}")
    print("-" * 56)
    print(f"{'Rejection Sampling':<32} {rs_est.mean():>8.4f} {rs_est.std():>8.4f} {N_rs:>6}")
    print(f"{'Gibbs Sampling':<32} {gibbs_est.mean():>8.4f} {gibbs_est.std():>8.4f} {N_gibbs:>6}")
    print(f"{'VE (exact)':<32} {ve_exact:>8.4f}")


# TASK 2 — Particle filter

# Task 2.1 — Trajectory simulation
# v[t] ~ N(mu[t], (mu[t]/3)^2), mu[t] = 2^(-t/10) — the target decelerates
# exponentially, while the velocity's coefficient of variation stays
# constant (= 1/3) throughout the trajectory

T = 50

def simulate_trajectory(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    t   = np.arange(T)
    mu  = 2.0 ** (-t / 10.0)          # mean velocity, shape (T,)
    std = mu / 3.0                     # velocity std

    v = rng.normal(mu, std)            # v[t] ~ N(mu[t], (mu[t]/3)^2)
    x = np.zeros(T + 1)
    for k in range(T):
        x[k + 1] = x[k] + v[k]

    return x[:T], v                    # position x[0..49], velocity v[0..49]


def plot_trajectory(x, v):
    t = np.arange(T)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(t, x, color="#2563EB", lw=2)
    ax1.set_ylabel("Position  x[t]")
    ax1.set_title("True target trajectory")
    ax1.grid(alpha=0.3)

    ax2.plot(t, v, color="#16A34A", lw=2)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel("Velocity  v[t]")
    ax2.set_xlabel("Time  t")
    ax2.grid(alpha=0.3)

    # Show mu[t] as the reference mean value
    mu = 2.0 ** (-t / 10.0)
    ax2.plot(t, mu,  color="crimson", lw=1.5, ls="--", label="mu[t] = 2^(-t/10)")
    ax2.plot(t, -mu, color="crimson", lw=1.5, ls="--")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("src/homework_2/trajectory.png", dpi=150)
    plt.show()


# Task 2.2 — Generating observations
# e[t] = theta[t]*x[t] + n[t], theta ~ Bernoulli(0.9) (10% outliers when the
# sensor "cannot see" the target), n ~ Laplace(0, b(x)); b(x) = sqrt(|x|)/5
# — noise grows with the target's distance

def _meas_scale(x):
    """b(x[t]) = sqrt(|x[t]|) / 5  — Laplace noise scale."""
    return np.sqrt(np.abs(x)) / 5.0


def generate_observations(x_true, rng=None):
    """Returns (e, theta): observations and the Bernoulli variable realizations."""
    if rng is None:
        rng = np.random.default_rng()

    b     = _meas_scale(x_true)                  # shape (T,)
    theta = rng.binomial(1, 0.9, size=T)          # 1 = inlier, 0 = outlier
    n     = rng.laplace(0.0, np.maximum(b, 1e-9)) # Laplace noise
    e     = theta * x_true + n
    return e, theta


def observation_likelihood(e_t, x_particles):
    """Likelihood p(e[t] | x_i) for every particle — vectorized."""
    b   = np.maximum(_meas_scale(x_particles), 1e-9)
    lik = (0.9 * laplace_dist.pdf(e_t - x_particles, 0.0, b)
         + 0.1 * laplace_dist.pdf(e_t,               0.0, b))
    return lik


def plot_observations(x_true, e_obs, theta):
    t = np.arange(T)
    _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, x_true, color="#2563EB", lw=2, label="True position x[t]")

    inliers  = theta == 1
    outliers = theta == 0
    ax.scatter(t[inliers],  e_obs[inliers],  s=20, color="#16A34A", alpha=0.75,
               label=f"Observation e[t]  (theta=1, n={inliers.sum()})")
    if outliers.any():
        ax.scatter(t[outliers], e_obs[outliers], s=60, color="crimson",
                   marker="x", linewidths=2,
                   label=f"Outlier  (theta=0, n={outliers.sum()})")

    ax.set_xlabel("Time  t")
    ax.set_ylabel("Position")
    ax.set_title("True trajectory and noisy observations  (Task 2.2)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("src/homework_2/observations.png", dpi=150)
    plt.show()


# Task 2.3 — Particle filter
# We work in the log domain (log_weights): 50 successive multiplications of
# small probabilities would quickly underflow in standard arithmetic

def particle_filter(e_obs, N=300, strategy='every', rng=None):
    """
    Returns x_est (T,), particles_hist and weights_hist (for visualization).
    strategy: 'none' | 'every' | 'adaptive'
    """
    if rng is None:
        rng = np.random.default_rng()

    t_arr  = np.arange(T)
    mu_arr = 2.0 ** (-t_arr / 10.0)

    particles   = np.zeros(N)
    log_weights = np.zeros(N)   # log(1) — uniform, we normalize at every step

    x_est          = np.zeros(T)
    particles_hist = []          # for visualization (Task 2.4)
    weights_hist   = []

    for t in range(T):
        # 1. Update log-weights with observation e[t]
        lik         = observation_likelihood(e_obs[t], particles)
        log_weights = log_weights + np.log(lik + 1e-300)

        # Normalization: log-sum-exp trick (subtract max, exponentiate, normalize)
        log_weights = log_weights - log_weights.max()
        weights     = np.exp(log_weights)
        weights     = weights / weights.sum()

        # 2. Position estimate: x_est[t] = sum_i w_i * x_i
        x_est[t] = float(np.dot(weights, particles))

        # 3. Save for visualization (before resampling)
        particles_hist.append(particles.copy())
        weights_hist.append(weights.copy())

        # 4. Resampling
        Neff = 1.0 / float(np.dot(weights, weights))
        do_resample = (strategy == 'every' or
                      (strategy == 'adaptive' and Neff < N / 2))
        if do_resample:
            idx         = rng.choice(N, size=N, p=weights)
            particles   = particles[idx]
            log_weights = np.zeros(N)   # reset: all particles get equal weight

        # 5. Propagation according to the motion model
        if t < T - 1:
            v_i       = rng.normal(mu_arr[t], mu_arr[t] / 3.0, size=N)
            particles = particles + v_i

    return x_est, particles_hist, weights_hist


def plot_pf_results(x_true, v_true, e_obs, x_est):
    t = np.arange(T)

    # Velocity estimated as the finite difference of the estimated position: v_est[t] ~ x_est[t]-x_est[t-1]
    v_est      = np.empty(T)
    v_est[:-1] = np.diff(x_est)
    v_est[-1]  = v_est[-2]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top panel — position
    ax1.scatter(t, e_obs, s=14, color="#94A3B8", alpha=0.65,
                label="Observations e[t]", zorder=1)
    ax1.plot(t, x_true, color="#2563EB", lw=2,
             label="True position x[t]", zorder=2)
    ax1.plot(t, x_est,  color="crimson",  lw=2, ls="--",
             label="PF estimate  x_est[t]", zorder=3)
    ax1.set_ylabel("Position")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Particle filter — position and velocity  (Task 2.3)")

    # Bottom panel — velocity
    ax2.plot(t, v_true, color="#2563EB", lw=2, label="True velocity v[t]")
    ax2.plot(t, v_est,  color="crimson",  lw=2, ls="--",
             label="Estimated velocity (delta x_est)")
    ax2.set_ylabel("Velocity")
    ax2.set_xlabel("Time  t")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("src/homework_2/particle_filter.png", dpi=150)
    plt.show()


# Task 2.4 — Particle visualization
# We show the state BEFORE resampling — that's when the weights carry
# information (after resampling they are all ~= 1/N, so the distribution is
# less informative)

def plot_particles(x_true, e_obs, particles_hist, weights_hist,
                   time_steps=(2, 8, 15, 25, 35, 49)):
    n   = len(time_steps)
    N   = len(particles_hist[0])

    # constrained_layout=True: matplotlib automatically computes the spacing
    # (incl. suptitle and colorbars) — no manual subplots_adjust needed
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n),
                              constrained_layout=True)

    fig.suptitle("Particle visualization before resampling  (Task 2.4)",
                 fontsize=13, fontweight="bold")

    for idx, (ax, t) in enumerate(zip(axes, time_steps)):
        parts = particles_hist[t]
        w     = weights_hist[t]
        Neff  = 1.0 / float(np.dot(w, w))

        sizes = (w / w.max()) * 220
        sizes = np.maximum(sizes, 4)

        rng_jit = np.random.default_rng(t + 100)
        jitter  = rng_jit.uniform(-0.30, 0.30, size=len(parts))

        sc = ax.scatter(parts, jitter, s=sizes, c=w,
                        cmap="Blues", alpha=0.75, edgecolors="none",
                        vmin=0, vmax=w.max(), zorder=2)

        ax.axvline(x_true[t], color="#2563EB", lw=2.5, zorder=3,
                   label=f"x_true = {x_true[t]:.2f}")
        ax.axvline(e_obs[t],  color="#16A34A", lw=2.0, ls="--", zorder=3,
                   label=f"e_obs  = {e_obs[t]:.2f}")

        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xlabel("Position" if idx == n - 1 else "", fontsize=10)
        ax.set_title(f"t = {t}  |  Neff = {Neff:.1f} / {N}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85,
                  borderpad=0.5, handlelength=1.2)
        ax.grid(axis="x", alpha=0.3)

        fig.colorbar(sc, ax=ax, fraction=0.014, pad=0.008, label="$w_i$")

    plt.savefig("src/homework_2/particles_viz.png", dpi=150)
    plt.show()


# Task 2.5 — Monte Carlo RMSE analysis
# Fair comparison: the same trajectory and observations for all 3 strategies
# in each MC iteration, so RMSE differences come only from resampling

def run_mc_particle_filter(Nr=500, N=100, seed=7):
    rng_master = np.random.default_rng(seed)
    # Pre-generate all seeds: [traj+obs, none, every, adaptive] per iteration
    all_seeds = rng_master.integers(0, 2**31, size=(Nr, 4))

    strategies = ('none', 'every', 'adaptive')
    rmse       = np.zeros((3, Nr))

    for r in range(Nr):
        rng_r    = np.random.default_rng(int(all_seeds[r, 0]))
        x_r, _  = simulate_trajectory(rng_r)
        e_r, _  = generate_observations(x_r, rng_r)

        for i, s in enumerate(strategies):
            rng_pf       = np.random.default_rng(int(all_seeds[r, i + 1]))
            x_est, _, _  = particle_filter(e_r, N=N, strategy=s, rng=rng_pf)
            rmse[i, r]   = np.sqrt(np.mean((x_est - x_r) ** 2))

        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{Nr} simulations done...")

    return rmse   # shape (3, Nr)


def plot_mc_rmse(rmse):
    labels = ['No resampling\n(none)',
              'Every step\n(every)',
              'Adaptive\n(Neff < N/2)']
    colors = ['#94A3B8', '#2563EB', '#16A34A']
    data   = [rmse[0], rmse[1], rmse[2]]

    _, ax = plt.subplots(figsize=(9, 5))

    vp = ax.violinplot(data, positions=[1, 2, 3],
                       showmedians=True, showextrema=True, widths=0.6)
    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(0.55)
        body.set_edgecolor(c)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        vp[part].set_color('#111111')
        vp[part].set_linewidth(1.5)

    # Marker for the mean value
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        ax.scatter(i, d.mean(), color=c, s=90, zorder=5,
                   edgecolors='black', linewidths=0.8)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Position RMSE")
    ax.set_title("MC RMSE analysis for 3 resampling strategies  (Nr=500, N=100)",
                 fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("src/homework_2/mc_rmse.png", dpi=150)
    plt.show()

    print(f"\n{'Strategy':<22} {'Mean':>8} {'Std':>8} {'Median':>8} {'Max':>8}")
    print("-" * 54)
    for name, d in zip(['No resampling', 'Every step', 'Adaptive'], data):
        print(f"{name:<22} {d.mean():>8.4f} {d.std():>8.4f} {np.median(d):>8.4f} {d.max():>8.4f}")


if __name__ == "__main__":
    draw_dag()

    ve_prob = variable_elimination()
    print("--- Variable Elimination ---------------------")
    print(f"P(b- | e-, f-) = {ve_prob[0]:.6f}")
    print(f"P(b+ | e-, f-) = {ve_prob[1]:.6f}")

    print("\n--- Monte Carlo (Nr=100) ----------------------")
    run_monte_carlo(Nr=100, target_std=0.025, seed=42)

    print("\n--- Task 2.1 — Trajectory ---------------------")
    rng2 = np.random.default_rng(0)
    x_true, v_true = simulate_trajectory(rng2)
    print(f"x[0]  = {x_true[0]:.3f},  x[49] = {x_true[49]:.3f}")
    print(f"v[0]  = {v_true[0]:.3f},  v[49] = {v_true[49]:.3f}")
    plot_trajectory(x_true, v_true)

    print("\n--- Task 2.2 — Observations --------------------")
    # Separate seed for observations — change OBS_SEED to test different realizations
    OBS_SEED = 0
    rng_obs = np.random.default_rng(OBS_SEED)
    e_obs, theta = generate_observations(x_true, rng_obs)
    n_outliers = int((theta == 0).sum())
    print(f"Outliers (theta=0): {n_outliers} / {T}  (expected ~{int(T * 0.1)}, seed={OBS_SEED})")
    print(f"e[0]  = {e_obs[0]:.3f},  e[49] = {e_obs[49]:.3f}")
    plot_observations(x_true, e_obs, theta)

    print("\n--- Task 2.3 — Particle filter -----------------")
    rng3 = np.random.default_rng(1)
    x_est_pf, particles_hist, weights_hist = particle_filter(
        e_obs, N=300, strategy='every', rng=rng3)
    rmse_pf = float(np.sqrt(np.mean((x_est_pf - x_true) ** 2)))
    print(f"Position RMSE (N=300, resample every step): {rmse_pf:.4f}")
    plot_pf_results(x_true, v_true, e_obs, x_est_pf)

    print("\n--- Task 2.4 — Particle visualization ----------")
    plot_particles(x_true, e_obs, particles_hist, weights_hist,
                   time_steps=(2, 11, 22, 35, 49))

    print("\n--- Task 2.5 — MC RMSE analysis (Nr=500) -------")
    rmse_mc = run_mc_particle_filter(Nr=500, N=100, seed=7)
    plot_mc_rmse(rmse_mc)
